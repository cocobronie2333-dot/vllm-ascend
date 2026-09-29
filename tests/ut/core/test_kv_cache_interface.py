# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
)
from vllm_ascend.models.deepseek_v41.cache_config import is_deepseek_v41_cache
from vllm_ascend.models.qwen4_exp.cache_config import is_qwen4_exp_cache


def _mla_spec():
    return AscendMLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    ("version", "detect"),
    [("deepseek_v41", is_deepseek_v41_cache), ("qwen4_exp", is_qwen4_exp_cache)],
)
def test_model_version_detection_accepts_raw_and_grouped_specs(version, detect):
    spec = AscendMLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        model_version=version,
    )
    uniform = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"renamed": spec})
    group = KVCacheGroupSpec(layer_names=["renamed"], kv_cache_spec=uniform)
    for items in ({"renamed": spec}, [spec], [uniform], [group], iter([_mla_spec(), group])):
        assert detect(items)
    other_detect = is_qwen4_exp_cache if version == "deepseek_v41" else is_deepseek_v41_cache
    assert not other_detect([group])
    assert not detect([_mla_spec()])
    assert not detect([])


def test_get_storage_block_size_and_dcp_memory():
    spec = _mla_spec()
    # On main, storage_block_size is an optional dataclass field and may be
    # None. Ascend derives physical rows from block_size / compression ratio.
    expected = spec.block_size // get_kv_cache_compression_ratio(spec)
    assert get_storage_block_size(spec) == expected

    uniform = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"layer": spec})
    assert get_storage_block_size(uniform) == expected

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
    )
    assert spec.max_memory_usage_bytes(vllm_config) > 0


def test_sliding_window_mla_storage_and_page_size():
    spec = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=64,
    )
    assert spec.storage_block_size == 16
    assert spec.real_page_size_bytes == 16 * 128 * 2
