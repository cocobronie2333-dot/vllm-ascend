# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 FP8 tiling must retain its advantage over the capped SIMD fallback."""

import statistics
from contextlib import ExitStack
from dataclasses import replace

import pytest
import torch

from vllm_ascend.device.hardware_profile import EngramUvaBackend
from vllm_ascend.models.deepseek_v41.engram import npu

GRAPH_CALLS = 64
SAMPLES = 9
TABLE_ROWS = 262144
HEADS = 24
WIDTH = 256
MAX_LATENCY_RATIO = 1.05


@pytest.mark.parametrize("tokens", [8, 128, 384])
@pytest.mark.parametrize("vectorcore_divisor", [2, 4])
def test_host_uva_tiled_simd_latency(tokens, vectorcore_divisor, monkeypatch):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU required")
    torch.npu.set_device(0)
    profile = npu.get_current_hardware_profile()
    if profile.engram_uva_backend != EngramUvaBackend.TILED_SIMD:
        pytest.skip("A3 tiled SIMD profile required")
    device = torch.device("npu:0")
    with ExitStack() as stack:
        codes = npu.HostUvaBuffer((TABLE_ROWS, WIDTH), torch.float8_e4m3fn, device)
        stack.callback(codes.close)
        scales = npu.HostUvaBuffer((TABLE_ROWS, WIDTH // npu.SCALE_GROUP), torch.uint8, device)
        stack.callback(scales.close)
        generator = torch.Generator().manual_seed(7)
        codes.tensor.copy_(torch.randn(codes.tensor.shape, generator=generator).to(torch.float8_e4m3fn))
        scales.tensor.fill_(127)
        ids_cpu = torch.randint(TABLE_ROWS, (tokens, HEADS), generator=generator, dtype=torch.int32)
        ids = ids_cpu.to(device)
        expected = codes.tensor.float()[ids_cpu.long()].reshape(-1, WIDTH).bfloat16()
        graphs = []
        outputs = []
        for backend in (EngramUvaBackend.SIMD, EngramUvaBackend.TILED_SIMD):
            selected_profile = replace(profile, engram_uva_backend=backend)
            monkeypatch.setattr(npu, "get_current_hardware_profile", lambda profile=selected_profile: profile)
            output = torch.empty_like(expected, device=device)
            outputs.append(output)

            def lookup(output=output):
                npu.gather_dequantize_host_uva(
                    codes, scales, ids, local_heads=HEADS, output=output, vectorcore_divisor=vectorcore_divisor
                )

            for _ in range(5):
                lookup()
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(GRAPH_CALLS):
                    lookup()
            for _ in range(3):
                graph.replay()
            torch.npu.synchronize()
            graphs.append(graph)
        samples = [[], []]
        for repeat in range(SAMPLES):
            for index in (repeat % 2, (repeat + 1) % 2):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                graphs[index].replay()
                end.record()
                end.synchronize()
                samples[index].append(start.elapsed_time(end) * 1000 / GRAPH_CALLS)
        rowwise_us, tiled_us = (statistics.median(values) for values in samples)
        assert tiled_us <= rowwise_us * MAX_LATENCY_RATIO, (rowwise_us, tiled_us)
