# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.attention.dsa_v41 import build_decoder_replay_swa_indices, resolve_replay_start


@pytest.mark.parametrize("prefix,decoder,expected", [(0, 384, 384), (64, 33, 64), (64, 80, 80)])
def test_composed_replay_indices_use_the_stricter_floor(prefix, decoder, expected):
    floor = resolve_replay_start(torch.tensor([prefix]), torch.tensor([decoder]))
    positions = torch.arange(expected, expected + 4)
    indices, lengths = build_decoder_replay_swa_indices(
        positions, 128, torch.tensor([0, 4]), torch.tensor([expected + 4]), floor
    )
    assert lengths[:, 0].tolist() == [1, 2, 3, 4]
    assert torch.equal((indices >= 0).sum(dim=-1), lengths)
    assert indices[0, 0, 0] == expected
    assert torch.all(indices[0, 0, 1:] == -1)


def test_single_replay_boundary_is_preserved():
    floor = torch.tensor([64, 384])
    assert resolve_replay_start(None, floor) is floor
    assert resolve_replay_start(floor, None) is floor
    assert resolve_replay_start(None, None) is None


def test_trimmed_prefill_excludes_unwritten_keys_and_preserves_decode_history():
    window = 4
    positions = torch.tensor([10, 97, 98, 99, 100])
    indices, lengths = build_decoder_replay_swa_indices(
        positions, window, torch.tensor([0, 1, 5]), torch.tensor([11, 101]), torch.tensor([7, 97])
    )
    assert lengths.tolist() == [[4], [1], [2], [3], [4]]
    assert indices[:, 0].tolist() == [
        [7, 8, 9, 10],
        [97, -1, -1, -1],
        [97, 98, -1, -1],
        [97, 98, 99, -1],
        [97, 98, 99, 100],
    ]
