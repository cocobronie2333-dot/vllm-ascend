from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.core.qwen4_exp_kv_cache_layout import (
    CIRCULAR_STATE,
    COMPRESSED_KEY,
    CONV_STATE,
    KEY_OR_SSM,
    VALUE_OR_PLE,
    Qwen4ExpKVCachePlanner,
    make_plane_view,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _get_qwen4_exp_kv_cache_config,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

_DEFAULT_LAYER_TYPES = tuple(["linear_attention"] * 3 + ["qwen_sparse_attention"]) * 12


def _planner_config(layer_types: tuple[str, ...] = _DEFAULT_LAYER_TYPES) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(layer_types=list(layer_types)),
        )
    )


def _specs(
    *,
    layer_types: tuple[str, ...] = _DEFAULT_LAYER_TYPES,
    include_mtp: bool = False,
    include_ple: bool = True,
) -> dict[str, KVCacheSpec]:
    main = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    raw = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=130,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    compressed = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    gdn = MambaSpec(
        block_size=128,
        shapes=((1, 1280), (1, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
    )
    ple = MambaSpec(
        block_size=128,
        shapes=((10240,),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
        tp_replicated=True,
    )
    specs: dict[str, KVCacheSpec] = {}
    for index, layer_type in enumerate(layer_types):
        if layer_type == "qwen_sparse_attention":
            source = f"model.layers.{index}.self_attn"
            specs[f"{source}.indexer.compressed_key_cache"] = compressed
            specs[f"{source}.indexer.raw_key_cache"] = raw
            specs[f"{source}.attn"] = main
        elif layer_type == "linear_attention":
            specs[f"model.layers.{index}.linear_attn"] = gdn
        else:
            raise ValueError(f"Unsupported test layer type: {layer_type}")
    if include_mtp:
        source = "model.mtp.layers.48.self_attn"
        specs[f"{source}.indexer.compressed_key_cache"] = compressed
        specs[f"{source}.indexer.raw_key_cache"] = raw
        specs[f"{source}.attn"] = main
    if include_ple:
        specs["model.ple"] = ple
    return specs


def _groups(*, include_mtp: bool = False):
    return Qwen4ExpKVCachePlanner(_planner_config()).get_kv_cache_groups(_specs(include_mtp=include_mtp))


def test_standard_topology_has_six_semantic_groups() -> None:
    groups = _groups()
    assert len(groups) == 6
    assert all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in groups
    )
    qsa_specs = groups[0].kv_cache_spec.kv_cache_specs
    assert {
        type(spec) for spec in qsa_specs.values()
    } == {FullAttentionSpec, MLAAttentionSpec}
    assert sum(name.endswith(".attn") for name in groups[0].layer_names) == 12
    assert sum(name.endswith(".compressed_key_cache") for name in groups[0].layer_names) == 12
    assert [len(group.layer_names) for group in groups[1:4]] == [12, 12, 12]
    assert groups[1].layer_names[:2] == ["model.layers.0.linear_attn", "model.layers.4.linear_attn"]
    assert groups[2].layer_names[:2] == ["model.layers.1.linear_attn", "model.layers.5.linear_attn"]
    assert groups[3].layer_names[:2] == ["model.layers.2.linear_attn", "model.layers.6.linear_attn"]
    assert groups[4].layer_names == ["model.ple"]
    assert all(
        isinstance(spec, MambaSpec)
        for group in groups[1:5]
        for spec in group.kv_cache_spec.kv_cache_specs.values()
    )
    assert all(name.endswith(".raw_key_cache") for name in groups[5].layer_names)
    assert all(
        isinstance(spec, CircularBufferSpec)
        for spec in groups[5].kv_cache_spec.kv_cache_specs.values()
    )
    assert groups[5].enable_kv_transfer is False


def test_gdn_topology_lanes_are_derived_not_fixed() -> None:
    layer_types = tuple(["linear_attention"] * 2 + ["qwen_sparse_attention"]) * 12
    specs = _specs(layer_types=layer_types)
    groups = Qwen4ExpKVCachePlanner(_planner_config(layer_types)).get_kv_cache_groups(specs)
    assert len(groups) == 5
    assert [len(group.layer_names) for group in groups[1:3]] == [12, 12]


def test_incomplete_topology_tail_does_not_require_gdn_qsa_divisibility() -> None:
    layer_types = tuple(["linear_attention"] * 2 + ["qwen_sparse_attention"]) * 4 + ("linear_attention",)
    specs = _specs(layer_types=layer_types)
    groups = Qwen4ExpKVCachePlanner(_planner_config(layer_types)).get_kv_cache_groups(specs)
    assert [len(group.layer_names) for group in groups[1:3]] == [5, 4]


def test_qsa_only_reduced_model_does_not_require_gdn_or_ple() -> None:
    layer_types = ("qwen_sparse_attention",)
    specs = _specs(layer_types=layer_types, include_ple=False)
    planner = Qwen4ExpKVCachePlanner(_planner_config(layer_types))
    groups = planner.get_kv_cache_groups(specs)
    assert len(groups) == 2
    plan = planner.build_physical_plan(groups, num_blocks=2, layout=KVCacheLayout.LBNHC)
    assert plan is not None
    assert [plane.name for plane in plan.planes] == [
        VALUE_OR_PLE,
        KEY_OR_SSM,
        COMPRESSED_KEY,
        CIRCULAR_STATE,
    ]


@pytest.mark.parametrize("layout", [KVCacheLayout.BLNHC, KVCacheLayout.LBNHC])
def test_plane_geometry_follows_resolved_layout(layout: KVCacheLayout) -> None:
    plan = Qwen4ExpKVCachePlanner().build_physical_plan(_groups(), num_blocks=3, layout=layout)
    assert plan is not None
    assert [plane.name for plane in plan.planes] == [
        VALUE_OR_PLE,
        KEY_OR_SSM,
        CONV_STATE,
        COMPRESSED_KEY,
        CIRCULAR_STATE,
    ]
    assert [plane.page_size_bytes for plane in plan.planes] == [
        128 * 256 * 2,
        128 * 256 * 2,
        1280 * 2,
        (128 // 4) * 128 * 2,
        8 * 130 * 2,
    ]
    assert plan.slot_count == 12
    assert plan.plane(VALUE_OR_PLE).page_size_bytes // 2 - 10240 == 22528
    assert plan.plane(KEY_OR_SSM).page_size_bytes // 2 - 128 * 128 == 128 * 128
    for plane in plan.planes:
        if layout.is_block_outermost:
            assert plane.layer_stride == plane.page_size_bytes
            assert plane.block_stride == plane.slot_count * plane.page_size_bytes
        else:
            assert plane.layer_stride == 3 * plane.page_size_bytes
            assert plane.block_stride == plane.page_size_bytes


def test_blnhc_view_uses_block_outermost_stride() -> None:
    plan = Qwen4ExpKVCachePlanner().build_physical_plan(_groups(), num_blocks=3, layout=KVCacheLayout.BLNHC)
    assert plan is not None
    owner = plan.owner("model.layers.3.self_attn.attn")
    recipe = next(v for v in plan.owner_views(owner.layer_name) if v.component == "k")
    plane = plan.plane(recipe.plane_name)
    backing = torch.zeros(plane.size, dtype=torch.int8)
    view = make_plane_view(
        backing,
        plane=plane,
        slot=recipe.slot,
        dtype=recipe.dtype,
        item_shape=recipe.item_shape,
    )
    assert view.shape == (3, 128, 1, 256)
    assert view.stride(0) == plane.block_stride // 2
    assert not view.is_contiguous()


def test_mtp_extends_qsa_and_ring_without_changing_gdn_lanes() -> None:
    groups = _groups(include_mtp=True)
    assert len(groups) == 6
    assert groups[0].is_eagle_group
    assert groups[5].is_eagle_group
    plan = Qwen4ExpKVCachePlanner().build_physical_plan(groups, num_blocks=2, layout=KVCacheLayout.LBNHC)
    assert plan is not None
    assert plan.slot_count == 13
    assert plan.ring_slot_count == 13
    assert plan.owner("model.mtp.layers.48.self_attn.indexer.raw_key_cache").group_id == 5


def test_config_uses_resolved_blnhc_descriptor_geometry() -> None:
    cache_config = SimpleNamespace(
        num_gpu_blocks_override=3,
        prefix_cache_retention_interval=None,
        get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLNHC,
    )
    config = _get_qwen4_exp_kv_cache_config(SimpleNamespace(cache_config=cache_config), _groups(), 10**9)
    assert config is not None
    assert config.num_blocks == 3
    assert config.kv_cache_layout == "BLNHC"
    plan = Qwen4ExpKVCachePlanner().build_physical_plan(config.kv_cache_groups, 3, KVCacheLayout.BLNHC)
    assert plan is not None
    assert config.kv_cache_tensors[0].block_stride == plan.plane(KEY_OR_SSM).block_stride


def test_config_memory_budget_counts_ring_buffer_once() -> None:
    cache_config = SimpleNamespace(
        num_gpu_blocks_override=None,
        prefix_cache_retention_interval=None,
        get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
    )
    vllm_config = SimpleNamespace(cache_config=cache_config)
    groups = _groups()
    unit_plan = Qwen4ExpKVCachePlanner().build_physical_plan(
        groups, 1, KVCacheLayout.LBNHC
    )
    assert unit_plan is not None
    bytes_per_block = unit_plan.normal_backing_size + unit_plan.ring_backing_size

    config = _get_qwen4_exp_kv_cache_config(
        vllm_config, groups, available_memory=3 * bytes_per_block
    )

    assert config is not None
    assert config.num_blocks == 3


def test_runner_allocates_each_plane_once_and_materializes_recipes() -> None:
    cache_config = SimpleNamespace(
        num_gpu_blocks_override=3,
        prefix_cache_retention_interval=None,
        get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
    )
    vllm_config = SimpleNamespace(
        cache_config=cache_config,
        kv_transfer_config=None,
    )
    config = _get_qwen4_exp_kv_cache_config(vllm_config, _groups(), 10**9)
    assert config is not None
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    runner.vllm_config = vllm_config
    runner.use_sparse = False
    runner.use_compress = False
    runner.use_hybrid_blocks = False
    runner.runner_only_attn_layers = set()
    runner.sparse_kv_offload_enabled = False
    runner._kv_cache_spec_attn_group_iterator = lambda: [
        SimpleNamespace(backend=SimpleNamespace(), kv_cache_spec=g.kv_cache_spec, layer_names=g.layer_names)
        for g in config.kv_cache_groups
    ]
    raw = runner._allocate_kv_cache_tensors(config)
    caches = runner._reshape_kv_cache_tensors(config, raw)
    source = "model.layers.3.self_attn"
    qsa = raw[f"{source}.attn"]
    gdn = raw["model.layers.0.linear_attn"]
    assert isinstance(qsa, tuple) and isinstance(gdn, tuple)
    pointers = {
        tensor.untyped_storage().data_ptr()
        for value in raw.values()
        for tensor in (value if isinstance(value, tuple) else (value,))
    }
    assert len(pointers) == 5
    assert caches[f"{source}.attn"][0].shape == (3, 128, 1, 256)
    assert caches["model.ple"][0].shape == (3, 10240)


def test_mismatched_qsa_sources_fail_fast() -> None:
    specs = _specs()
    del specs["model.layers.47.self_attn.indexer.raw_key_cache"]
    with pytest.raises(ValueError, match="one-to-one"):
        Qwen4ExpKVCachePlanner().get_kv_cache_groups(specs)
