# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify device dispatch without launching architecture-specific kernels."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.deepseek_v41.engram import npu
from vllm_ascend.ops.triton import triton_utils


@pytest.mark.parametrize("device_type", list(AscendDeviceType))
@pytest.mark.parametrize("tokens", [0, 1, 2, 4, 5, 8, 9, 17, 31, 32, 128])
@pytest.mark.parametrize("width", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.int8, torch.bfloat16])
def test_host_uva_dispatch(device_type, tokens, width, dtype, monkeypatch):
    monkeypatch.setattr(npu, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    monkeypatch.setattr(triton_utils, "init_device_properties_triton", lambda: None)
    names = (
        "_engram_host_uva_gather_dequant_kernel",
        "_engram_host_uva_gather_dequant_fp8_small_kernel",
        "_engram_host_uva_gather_dequant_fp8_token_kernel",
        "_engram_host_uva_gather_dequant_fp8_simd_kernel",
    )
    kernels = [MagicMock() for _ in names]
    for name, kernel in zip(names, kernels):
        monkeypatch.setattr(npu, name, kernel)
    codes = SimpleNamespace(tensor=torch.empty((64, width), dtype=dtype), ptrs=object())
    scales = None if dtype == torch.bfloat16 else SimpleNamespace(ptrs=object())
    ids = torch.zeros((tokens, 24), dtype=torch.int32)
    output = torch.empty((tokens * 24, width), dtype=torch.bfloat16)
    assert npu.gather_dequantize_host_uva(codes, scales, ids, local_heads=24, output=output) is output
    if not tokens:
        assert all(not kernel.__getitem__.called for kernel in kernels)
        return
    optimized = (
        device_type in (AscendDeviceType.A3, AscendDeviceType.A5) and dtype == torch.float8_e4m3fn and width == 256
    )
    selected = 0
    if optimized:
        if device_type == AscendDeviceType.A3:
            selected = 3 if tokens >= npu.UVA_FP8_SIMD_MIN_TOKENS else 0
        else:
            selected = 1 if tokens < npu.UVA_FP8_ROW_TILE_MIN_TOKENS else 2
    for index, kernel in enumerate(kernels):
        assert kernel.__getitem__.call_count == (1 if index == selected else 0)
    grid = kernels[selected].__getitem__.call_args.args[0]
    assert grid == (8,)
    kwargs = kernels[selected].__getitem__.return_value.call_args.kwargs
    if device_type != AscendDeviceType.A5:
        assert kwargs["compile_mode"] == "simd"
        if selected != 3:
            assert kwargs["NATIVE_FP8"] is False
        else:
            assert kwargs["BLOCK_ROWS"] == (4 if tokens <= 4 else 8 if tokens < 32 else 16)
    elif selected == 2:
        assert kwargs["compile_mode"] == "simt_only"
