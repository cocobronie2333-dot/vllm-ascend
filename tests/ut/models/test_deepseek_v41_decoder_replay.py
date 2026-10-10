# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.models.deepseek_v41.decoder_replay_layers import AscendDecoderReplayLayers
from vllm_ascend.models.deepseek_v41.upstream_decoder_replay_layers import DecoderReplayLayers


def test_decoder_replay_inherits_copied_upstream_base():
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )
    replay = AscendDecoderReplayLayers(config, 128, lambda hidden, _ids, _positions: (hidden + 1,), [None, None])
    assert isinstance(replay, DecoderReplayLayers)
    hidden = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    ids = torch.arange(4)
    positions = torch.arange(4)

    (full_output,) = replay(hidden, ids, positions)
    torch.testing.assert_close(full_output, hidden + 1)

    replay.rows = torch.tensor([1, 3])
    replay.num_actual_rows = 2
    replay.context_factory = nullcontext
    (trimmed_output,) = replay(hidden, ids, positions)
    torch.testing.assert_close(trimmed_output, torch.tensor([[0.0], [2.0], [0.0], [4.0]]))
