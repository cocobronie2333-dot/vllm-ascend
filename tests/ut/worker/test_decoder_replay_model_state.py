# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config, set_current_vllm_config

from vllm_ascend.worker.v2.model_states.default import AscendModelState


def test_decoder_metadata_preserves_prefix_floor():
    state = AscendModelState.__new__(AscendModelState)
    state.decoder_replay = SimpleNamespace(window=32)
    state._decoder_replay_num_tokens = state._decoder_replay_num_actual_tokens = 16
    state._decoder_replay_dropped_before = SimpleNamespace(copy_to_gpu=lambda size: None, gpu=torch.zeros(2))
    state._decoder_replay_offsets = SimpleNamespace(
        gpu=torch.tensor([0, 16]), cpu=torch.tensor([0, 16]), np=np.array([0, 16])
    )
    state._decoder_replay_start = torch.empty(1, dtype=torch.int32)
    state._decoder_replay_rows = torch.empty(16, dtype=torch.int64)
    state._decoder_replay_positions = torch.arange(64, 80)
    state._decoder_replay_source_slot_mappings = torch.empty((1, 16), dtype=torch.int64)
    state._decoder_replay_slot_mappings = torch.empty((1, 16), dtype=torch.int64)
    state.slot_mappings = [torch.arange(16)]
    state.block_tables = [torch.zeros((1, 8), dtype=torch.int32)]
    original = SimpleNamespace(
        query_start_loc=torch.tensor([0, 16]),
        seq_lens=torch.tensor([80]),
        positions=torch.arange(64, 80),
        is_prefilling=torch.tensor([True]),
        replay_start=torch.tensor([64]),
    )
    state._decoder_replay_common_metadata = {0: original}
    builder = Mock()
    state._decoder_replay_builders = [(0, ["layer"], builder)]

    def gather(*args):
        # This untrimmed chunk would otherwise admit unwritten history at 33.
        args[9].fill_(33)

    with (
        patch("vllm_ascend.ops.triton.decoder_replay.gather_replay_batch", side_effect=gather),
        patch("vllm_ascend.ops.rope_dsv4.get_cos_and_sin_dsa", return_value=None),
        patch(
            "vllm_ascend.worker.v2.model_states.default.replace",
            side_effect=lambda obj, **kwargs: SimpleNamespace(**(vars(obj) | kwargs)),
        ),
    ):
        state._build_decoder_replay_metadata(SimpleNamespace(num_tokens=16), np.array([16]))
    assert builder.build.call_args.kwargs["replay_start"].tolist() == [64]
    assert builder.build.call_args.args[1].replay_start.tolist() == [64]


@pytest.mark.parametrize("raises", [False, True])
def test_replay_context_continues_moe_layer_cursor(raises):
    state = AscendModelState.__new__(AscendModelState)
    state._decoder_replay_builders = [(0, ["layer"], None)]
    state._decoder_replay_num_tokens = 128
    state._decoder_replay_slot_mappings = torch.zeros((1, 128), dtype=torch.int64)
    state._decoder_replay_metadata = {}
    state._decoder_replay_counts = torch.tensor([128, 1])
    state.vllm_config = object()
    outer = SimpleNamespace(moe_layer_index=21, is_padding=None)
    inner = SimpleNamespace(moe_layer_index=0)
    module = "vllm_ascend.worker.v2.model_states.default."
    with (
        patch(module + "get_forward_context", side_effect=[outer, inner]),
        patch(module + "set_forward_context", return_value=nullcontext()) as context,
    ):
        try:
            with state._decoder_replay_context():
                assert inner.moe_layer_index == 21
                inner.moe_layer_index = 40
                if raises:
                    raise RuntimeError("decoder failed")
        except RuntimeError:
            assert raises
    assert outer.moe_layer_index == 40
    assert context.call_args.kwargs["num_tokens_across_dp"] is state._decoder_replay_counts


@pytest.mark.parametrize("prompt_logprobs", [False, True])
def test_decoder_replay_preserves_decode_and_prompt_logprob_rows(prompt_logprobs):
    state = AscendModelState.__new__(AscendModelState)
    state.decoder_replay = SimpleNamespace(window=128)
    state.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1))
    state._decoder_replay_builders = [(0, ["layer"], None)]
    state._decoder_replay_common_metadata = {0: object()}
    state.prompt_logprobs_worker = SimpleNamespace(uses_prompt_logprobs=np.array([False, prompt_logprobs, False]))
    state._decoder_replay_offsets = SimpleNamespace(np=np.empty(4, dtype=np.int32))
    state._decoder_replay_dropped_before = SimpleNamespace(np=np.empty(4, dtype=np.int32))
    state._decoder_replay_rows = torch.empty(500, dtype=torch.int64)
    state._build_decoder_replay_metadata = Mock(return_value={})
    batch = SimpleNamespace(
        num_reqs=3,
        num_tokens=428,
        idx_mapping_np=np.arange(3),
        query_start_loc_np=np.array([0, 1, 301, 428]),
        is_prefilling_np=np.array([False, True, True]),
    )
    with patch("vllm_ascend.worker.v2.model_states.default.enable_dsa_cp", return_value=False):
        state._prepare_decoder_replay(batch, CUDAGraphMode.NONE, [], None)
    if prompt_logprobs:
        assert state.decoder_replay.rows is None
        state._build_decoder_replay_metadata.assert_not_called()
    else:
        kept_lens = state._build_decoder_replay_metadata.call_args.args[1]
        np.testing.assert_array_equal(kept_lens, [1, 128, 127])
        assert state.decoder_replay.num_actual_rows == 256
        assert state.decoder_replay.rows.numel() == 256
        np.testing.assert_array_equal(state._decoder_replay_offsets.np, [0, 1, 129, 256])
        np.testing.assert_array_equal(state._decoder_replay_dropped_before.np, [0, 0, 172, 172])


def test_replay_lazy_builder_initialization_scopes_config():
    state = AscendModelState.__new__(AscendModelState)
    state.decoder_replay = SimpleNamespace(window=128)
    state.vllm_config = VllmConfig()
    state._decoder_replay_builders = []
    outer_config = VllmConfig()

    def initialize(attn_groups, cache_config):
        assert get_current_vllm_config() is state.vllm_config

    state._initialize_decoder_replay_metadata = initialize
    with set_current_vllm_config(outer_config):
        state._prepare_decoder_replay(SimpleNamespace(), CUDAGraphMode.FULL, [], None)
        assert get_current_vllm_config() is outer_config


@pytest.mark.parametrize("idle", [False, True])
def test_idle_dummy_joins_replay_agreement_but_profile_dummy_does_not(idle):
    state = AscendModelState.__new__(AscendModelState)
    state.device = torch.device("cpu")
    state.decoder_replay = SimpleNamespace(window=128)
    state.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=2))
    state._decoder_replay_builders = [(0, ["layer"], None)]
    state._decoder_replay_common_metadata = {0: object()}
    state._decoder_replay_rows = torch.empty(4, dtype=torch.int64)
    state._decoder_replay_slot_mappings = torch.full((2, 4), 12345, dtype=torch.int64)
    state.attn_metadata = {"layer": object()}
    batch = SimpleNamespace(is_dummy=True, num_reqs=1, num_tokens=1, num_tokens_after_padding=1)

    def all_reduce(agreed, **kwargs):
        agreed[0, 1] = 1  # The active peer trims a long prefill.
        agreed[1, 1] = 128

    module = "vllm_ascend.worker.v2.model_states.default."
    with (
        patch(module + "should_skip_allreduce_across_dp_group", return_value=False),
        patch(module + "ring_state_update_skipped", return_value=idle),
        patch(module + "enable_dsa_cp", return_value=False),
        patch(module + "get_dp_group", return_value=SimpleNamespace(rank_in_group=0, cpu_group=object())),
        patch(module + "dist.all_reduce", side_effect=all_reduce) as collective,
        patch.object(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor),
    ):
        state._prepare_decoder_replay(batch, CUDAGraphMode.NONE, [], None)
    if idle:
        collective.assert_called_once()
        assert state.decoder_replay.rows.tolist() == [0]
        assert state.decoder_replay.num_actual_rows == 1
        assert state._decoder_replay_counts.tolist() == [1, 128]
        assert state._decoder_replay_metadata is state.attn_metadata
        assert state._decoder_replay_slot_mappings[:, :1].tolist() == [[-1], [-1]]
    else:
        collective.assert_not_called()
        assert state.decoder_replay.rows is None
