# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.models.deepseek_v41 import decoder_replay_compat
from vllm_ascend.models.deepseek_v41.decoder_replay_layers import AscendDecoderReplayLayers, make_decoder_replay


def test_replay_without_rows_preserves_inputs():
    hidden = torch.arange(12).reshape(6, 2)
    replay = decoder_replay_compat.DecoderReplayLayers(2, lambda *states: states, [])
    outputs = replay(hidden, None)
    assert replay.window == 2
    assert outputs[0] is hidden
    assert outputs[1] is None


def test_vendored_replay_compacts_buffers_and_scatters_outputs():
    hidden = torch.arange(12).reshape(6, 2)
    positions = torch.arange(6)
    buffer = positions.clone()
    replay = decoder_replay_compat.DecoderReplayLayers(2, lambda x, p, optional: (x + 1, p), [buffer])
    replay.rows = torch.tensor([4, 5])
    with patch.object(decoder_replay_compat, "override_forward_context", return_value=nullcontext()):
        actual, actual_positions = replay(hidden, positions, None)
    assert torch.equal(buffer[:2], positions[4:])
    assert torch.equal(actual[:4], torch.zeros_like(hidden[:4]))
    assert torch.equal(actual[4:], hidden[4:] + 1)
    assert torch.equal(actual_positions[4:], positions[4:])


def test_ascend_replay_excludes_padding_rows_from_scatter():
    hidden = torch.arange(12).reshape(6, 2)
    positions = torch.arange(6)
    buffer = positions.clone()
    replay = AscendDecoderReplayLayers(2, lambda x, optional, p: (x + 1, p), [buffer])
    replay.rows = torch.tensor([4, 5, 0])
    replay.num_actual_rows = 2
    replay.context_factory = nullcontext
    with patch("vllm_ascend.models.deepseek_v41.decoder_replay_layers.enable_dsa_cp", return_value=False):
        actual, actual_positions = replay(hidden, None, positions)
    assert torch.equal(buffer[:3], positions[replay.rows])
    assert torch.equal(actual[:4], torch.zeros_like(hidden[:4]))
    assert torch.equal(actual[4:], hidden[4:] + 1)
    assert torch.equal(actual_positions[:4], torch.zeros_like(positions[:4]))


def test_packed_replay_compacts_sparse_lengths_with_indices():
    buffers = [torch.arange(6) + offset for offset in (0, 10, 20, 30)]
    model = SimpleNamespace(
        config=SimpleNamespace(kv_source_layer_ids=[2], engram_layer_ids=[1], sliding_window=2),
        end_layer=5,
        use_sequence_parallel=False,
        topk_indices_buffer=buffers[0],
        candidate_indices_buffer=buffers[1],
        candidate_lengths_buffer=buffers[2],
        topk_lengths_buffer=buffers[3],
        _run_replay_layers=lambda hidden, optional, positions: (hidden,),
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(swa_bounded_replay=True, enable_prefix_caching=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
        speculative_config=None,
        lora_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )
    replay = make_decoder_replay(model, config)
    assert replay is not None
    replay.rows = torch.tensor([4, 5])
    replay.num_actual_rows = 2
    replay.context_factory = nullcontext
    with patch("vllm_ascend.models.deepseek_v41.decoder_replay_layers.enable_dsa_cp", return_value=False):
        replay(torch.arange(6), None, torch.arange(6))
    for buffer, offset in zip(buffers, (0, 10, 20, 30)):
        assert buffer[:2].tolist() == [offset + 4, offset + 5]
