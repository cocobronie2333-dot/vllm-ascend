# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_ascend.worker.v2.input_batch import AscendInputBatch


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("has_prefill", [False, True])
def test_prefill_decode_kwargs_match_input_batch_api(modern, has_prefill):
    @dataclass
    class LegacyBatch:
        has_prefill: bool

    @dataclass
    class ModernBatch(LegacyBatch):
        prefill_runs_as_decode_np: object = None
        decode_graph_eligible: bool = False

    batch_type = ModernBatch if modern else LegacyBatch
    state = SimpleNamespace(has_prefill=has_prefill)
    kwargs = AscendInputBatch.prefill_decode_kwargs.__func__(batch_type, state)
    batch = batch_type(has_prefill=has_prefill, **kwargs)
    if modern:
        assert batch.prefill_runs_as_decode_np is None
        assert batch.decode_graph_eligible is (not has_prefill)
    else:
        assert kwargs == {}


def test_prefill_decode_kwargs_preserve_upstream_classification():
    @dataclass
    class ModernBatch:
        prefill_runs_as_decode_np: object
        decode_graph_eligible: bool

    rows = np.array([True, False])
    state = SimpleNamespace(has_prefill=True, prefill_runs_as_decode_np=rows, decode_graph_eligible=True)
    kwargs = AscendInputBatch.prefill_decode_kwargs.__func__(ModernBatch, state)
    assert kwargs["prefill_runs_as_decode_np"] is rows
    assert kwargs["decode_graph_eligible"] is True
