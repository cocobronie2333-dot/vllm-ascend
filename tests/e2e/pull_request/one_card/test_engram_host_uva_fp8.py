# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Numerical comparison of tiled and row-wise HOST_UVA FP8 lookups."""

from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.device.hardware_profile import EngramUvaBackend
from vllm_ascend.models.deepseek_v41.engram import npu

pytestmark = pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="NPU required")


@pytest.fixture(scope="module", autouse=True)
def initialize_npu():
    # HostUvaBuffer uses ACL directly, before a tensor allocation could lazily
    # initialize the context in a standalone single-operator test process.
    torch.npu.set_device(0)


@pytest.fixture(params=["device", "simd", "simd_tiled"], autouse=True)
def lookup_backend(request, monkeypatch):
    # On 950 also exercise the portable software-FP8 path. This verifies its
    # numerics, not A3 hardware performance or A3 compiler code generation.
    if request.param != "device":
        backend = EngramUvaBackend.TILED_SIMD if request.param == "simd_tiled" else EngramUvaBackend.SIMD
        profile = replace(npu.get_current_hardware_profile(), engram_uva_backend=backend)
        monkeypatch.setattr(npu, "get_current_hardware_profile", lambda: profile)


@pytest.mark.parametrize(
    "num_tokens,local_heads",
    [
        (0, 24),
        (1, 24),
        (2, 24),
        (4, 24),
        (5, 24),
        (6, 24),
        (7, 24),
        (8, 24),
        (9, 24),
        (10, 24),
        (17, 24),
        (31, 24),
        (32, 24),
        (33, 24),
        (127, 24),
        (128, 24),
        (129, 24),
        (384, 24),
        (1024, 24),
        (9, 3),
        (128, 3),
        (384, 3),
        (5, 1),
        (7, 6),
        (9, 12),
    ],
)
@pytest.mark.parametrize("width", [256])
@pytest.mark.parametrize("force_tiled", [False, True])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_host_uva_fp8_token_tiles_match_rowwise_lookup(
    num_tokens, local_heads, width, force_tiled, ids_dtype, monkeypatch
):
    # Cross pointer-table chunks without allocating a multi-gigabyte host table.
    monkeypatch.setattr(npu, "CHUNK_ROWS", 32)
    # Cover both the normal dispatch (including short unrolled lookups) and
    # partial row tiles too small to select the tiled path in production.
    if force_tiled:
        monkeypatch.setattr(npu, "UVA_FP8_MIN_TILES_PER_PROGRAM", 0)
        monkeypatch.setattr(npu, "UVA_FP8_ROW_TILE_MIN_TOKENS", 0)
        monkeypatch.setattr(npu, "UVA_FP8_SIMD_MIN_TOKENS", 0)
    table_rows = 64
    vocab_start = 11
    head_start = 2
    pad_heads = local_heads + 2
    device = torch.device("npu:0")
    codes_data = torch.linspace(-2, 2, table_rows * width, dtype=torch.float32).reshape(table_rows, width)
    codes_data = codes_data.to(torch.float8_e4m3fn)
    scale_data = (124 + torch.arange(table_rows * (width // npu.SCALE_GROUP)) % 5).to(torch.uint8)
    scale_data = scale_data.reshape(table_rows, width // npu.SCALE_GROUP)

    with ExitStack() as stack:
        codes = npu.HostUvaBuffer((table_rows, width), torch.float8_e4m3fn, device)
        stack.callback(codes.close)
        scales = npu.HostUvaBuffer(scale_data.shape, torch.uint8, device)
        stack.callback(scales.close)
        codes.tensor.copy_(codes_data)
        scales.tensor.copy_(scale_data)

        ids_data = torch.arange(num_tokens * (local_heads + head_start), dtype=ids_dtype)
        ids_data = (ids_data % table_rows + vocab_start).reshape(num_tokens, local_heads + head_start)
        if num_tokens:
            ids_data[0, head_start] = -1
            ids_data[-1, head_start + local_heads - 1] = vocab_start + table_rows
        # Preserve contiguous heads but exercise a non-contiguous token stride.
        ids = torch.empty((num_tokens * 2, local_heads + head_start), dtype=ids_dtype, device=device)[::2]
        ids.copy_(ids_data)
        output_shape = (num_tokens * pad_heads, width)
        tiled = torch.full(output_shape, -3, dtype=torch.bfloat16, device=device)
        rowwise = torch.full_like(tiled, -3)

        npu.gather_dequantize_host_uva(
            codes,
            scales,
            ids,
            head_start=head_start,
            local_heads=local_heads,
            pad_heads=pad_heads,
            output=tiled,
            vocab_start=vocab_start,
            vocab_end=vocab_start + table_rows,
        )
        expected = torch.full(output_shape, -3, dtype=torch.bfloat16)
        selected_ids = ids_data[:, head_start:] - vocab_start
        owned = (selected_ids >= 0) & (selected_ids < table_rows)
        safe_ids = selected_ids.clamp(0, table_rows - 1).long()
        decoded = codes_data.float().unflatten(-1, (-1, npu.SCALE_GROUP))
        decoded *= torch.ldexp(torch.ones_like(scale_data, dtype=torch.float32), scale_data.int() - 127).unsqueeze(-1)
        selected = decoded.flatten(-2)[safe_ids].bfloat16()
        selected[~owned] = 0
        expected.view(num_tokens, pad_heads, width)[:, :local_heads] = selected
        torch.testing.assert_close(tiled.cpu(), expected, rtol=0, atol=0)
        # Compare against the capped row-wise schedule used before token tiling.
        rows = num_tokens * local_heads
        if not rows:
            return
        npu._engram_host_uva_gather_dequant_kernel[(min(rows, npu.UVA_MAX_PROGRAMS),)](
            codes.ptrs,
            scales.ptrs,
            ids,
            rowwise,
            rows,
            vocab_start,
            vocab_start + table_rows,
            ids.stride(0),
            CHUNK=npu.CHUNK_ROWS,
            WIDTH=width,
            GROUP=npu.SCALE_GROUP,
            HEAD_START=head_start,
            LOCAL_HEADS=local_heads,
            PAD_HEADS=pad_heads,
            QUANTIZED=True,
            MXFP8=True,
            num_warps=4,
        )
        torch.npu.synchronize()
        assert torch.equal(tiled, rowwise)


@pytest.mark.parametrize("width", [256])
@pytest.mark.parametrize("num_tokens", [1, 8, 9, 128])
def test_host_uva_fp8_all_encodings(width, num_tokens):
    """Exercise FP8 subnormals/NaNs and E8M0 underflow/overflow on the NPU."""
    local_heads = 24
    table_rows = 256
    device = torch.device("npu:0")
    # Every FP8 byte with every E8M0 byte; unlike a random normal-distribution
    # input, this also guards the SIMT software decoder's special values.
    codes_data = torch.arange(width, dtype=torch.int32).to(torch.uint8).expand(table_rows, -1).contiguous()
    codes_data = codes_data.view(torch.float8_e4m3fn)
    scale_data = torch.arange(table_rows, dtype=torch.uint8)[:, None].expand(-1, width // npu.SCALE_GROUP).contiguous()
    with ExitStack() as stack:
        codes = npu.HostUvaBuffer(codes_data.shape, torch.float8_e4m3fn, device)
        stack.callback(codes.close)
        scales = npu.HostUvaBuffer(scale_data.shape, torch.uint8, device)
        stack.callback(scales.close)
        codes.tensor.copy_(codes_data)
        scales.tensor.copy_(scale_data)
        decoded_scale = torch.ldexp(torch.ones_like(scale_data, dtype=torch.float32), scale_data.int() - 127)
        decoded_scale[scale_data == 255] = float("nan")
        expected = codes_data.float().unflatten(-1, (-1, npu.SCALE_GROUP)) * decoded_scale.unsqueeze(-1)
        expected = expected.flatten(-2).bfloat16()
        for offset in range(0, table_rows, num_tokens * local_heads):
            ids_data = torch.arange(num_tokens * local_heads, dtype=torch.int32).reshape(num_tokens, local_heads)
            ids_data = (ids_data + offset) % table_rows
            result = npu.gather_dequantize_host_uva(codes, scales, ids_data.to(device), local_heads=local_heads)
            reference = expected[ids_data.long()].reshape(-1, width)
            torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 17, 129])
def test_host_uva_vector_storage(dtype, num_tokens, monkeypatch):
    """A3 model storage uses INT8/FP32 or unquantized BF16 tables."""
    monkeypatch.setattr(npu, "CHUNK_ROWS", 32)
    width, local_heads, table_rows = 256, 24, 64
    device = torch.device("npu:0")
    data = (torch.arange(table_rows * width).reshape(table_rows, width) % 256 - 128).to(dtype)
    ids_cpu = torch.arange(num_tokens * local_heads, dtype=torch.int32).reshape(num_tokens, local_heads) % table_rows
    ids_cpu[0, 0] = -1
    ids_cpu[-1, -1] = table_rows
    owned = (ids_cpu >= 0) & (ids_cpu < table_rows)
    with ExitStack() as stack:
        codes = npu.HostUvaBuffer(data.shape, dtype, device)
        stack.callback(codes.close)
        codes.tensor.copy_(data)
        scales = None
        decoded = data.float()
        if dtype == torch.int8:
            scale_data = torch.arange(table_rows * (width // npu.SCALE_GROUP)).reshape(table_rows, -1) % 5 + 1
            scale_data = scale_data.float() / 64
            scales = npu.HostUvaBuffer(scale_data.shape, torch.float32, device)
            stack.callback(scales.close)
            scales.tensor.copy_(scale_data)
            decoded = (decoded.unflatten(-1, (-1, npu.SCALE_GROUP)) * scale_data.unsqueeze(-1)).flatten(-2)
        expected = decoded[ids_cpu.clamp(0, table_rows - 1).long()].bfloat16()
        expected[~owned] = 0
        result = npu.gather_dequantize_host_uva(codes, scales, ids_cpu.to(device), local_heads=local_heads)
        torch.testing.assert_close(result.cpu(), expected.reshape(-1, width), rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [1, 8, 128])
def test_host_uva_fp8_disjoint_chunks(num_tokens, monkeypatch):
    """SIMD tile offsets must support separate allocations and negative deltas."""
    chunk_rows, chunks, width, local_heads = 32, 3, 256, 24
    monkeypatch.setattr(npu, "CHUNK_ROWS", chunk_rows)
    device = torch.device("npu:0")
    with ExitStack() as stack:
        code_chunks, scale_chunks = [], []
        for index in range(chunks):
            codes = npu.HostUvaBuffer((chunk_rows, width), torch.float8_e4m3fn, device)
            stack.callback(codes.close)
            scales = npu.HostUvaBuffer((chunk_rows, width // npu.SCALE_GROUP), torch.uint8, device)
            stack.callback(scales.close)
            codes.tensor.fill_(index + 1)
            scales.tensor.fill_(127 + index)
            code_chunks.append(codes)
            scale_chunks.append(scales)
        # CPU synchronization is outside the operator: deliberately make the
        # scalar origin the largest address so later chunks have negative offsets.
        code_chunks.sort(key=lambda chunk: int(chunk.ptrs.cpu()[0]), reverse=True)
        scale_chunks.sort(key=lambda chunk: int(chunk.ptrs.cpu()[0]), reverse=True)
        code_data = torch.cat([chunk.tensor for chunk in code_chunks])
        scale_data = torch.cat([chunk.tensor for chunk in scale_chunks])
        codes = SimpleNamespace(tensor=code_data, ptrs=torch.cat([chunk.ptrs for chunk in code_chunks]))
        scales = SimpleNamespace(ptrs=torch.cat([chunk.ptrs for chunk in scale_chunks]))
        ids_cpu = torch.arange(num_tokens * local_heads, dtype=torch.int64).reshape(num_tokens, local_heads)
        ids_cpu = (ids_cpu * chunk_rows + ids_cpu // chunks) % (chunks * chunk_rows)
        scale = torch.ldexp(torch.ones_like(scale_data, dtype=torch.float32), scale_data.int() - 127)
        decoded = (code_data.float().unflatten(-1, (-1, npu.SCALE_GROUP)) * scale[..., None]).flatten(-2)
        expected = decoded[ids_cpu].reshape(-1, width).bfloat16()
        result = npu.gather_dequantize_host_uva(codes, scales, ids_cpu.to(device), local_heads=local_heads)
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
