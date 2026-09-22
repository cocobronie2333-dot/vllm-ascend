from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_ascend.core.qwen4_exp_kv_cache_layout import (
    GDN,
    PLE,
    QSA_RAW,
    TENSOR1,
    TENSOR2,
    TENSOR3,
    TENSOR4,
    build_qwen4_exp_kv_cache_layout,
    make_contiguous_plane_view,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _get_qwen4_exp_kv_cache_config,
    _merge_ple_into_last_gdn_group,
    _merge_qsa_composite_groups,
    _merge_shattered_gdn_groups,
    _order_qwen4_exp_cache_groups,
    _prepare_qsa_composite_groups,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _specs(*, include_mtp: bool = False) -> dict[str, KVCacheSpec]:
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
    for index in range(12):
        source = f"model.layers.{index * 4 + 3}.self_attn"
        specs[f"{source}.indexer.compressed_key_cache"] = compressed
        specs[f"{source}.indexer.raw_key_cache"] = raw
        specs[f"{source}.attn"] = main
    if include_mtp:
        source = "model.mtp.layers.48.self_attn"
        specs[f"{source}.indexer.compressed_key_cache"] = compressed
        specs[f"{source}.indexer.raw_key_cache"] = raw
        specs[f"{source}.attn"] = main
    for index in range(36):
        specs[f"model.layers.{index}.linear_attn"] = gdn
    specs["model.ple"] = ple
    return specs


def _five_groups(*, include_mtp: bool = False) -> list[KVCacheGroupSpec]:
    specs = _specs(include_mtp=include_mtp)
    groups = [KVCacheGroupSpec([name], spec, is_eagle_group=".mtp." in name) for name, spec in specs.items()]
    groups = _merge_shattered_gdn_groups(specs, groups)
    owners = _prepare_qsa_composite_groups(specs)
    assert owners is not None
    groups = _merge_qsa_composite_groups(groups, specs, *owners)
    groups = _merge_ple_into_last_gdn_group(groups, specs)
    return _order_qwen4_exp_cache_groups(groups, specs)


def test_qwen4_exp_group_ids_are_qsa_three_gdn_and_ring() -> None:
    groups = _five_groups()
    assert len(groups) == 5

    assert sum(name.endswith(".attn") for name in groups[0].layer_names) == 12
    assert sum(name.endswith(".indexer.compressed_key_cache") for name in groups[0].layer_names) == 12
    assert all(len(groups[index].layer_names) == 12 for index in (1, 2))
    assert len(groups[3].layer_names) == 13
    assert groups[3].layer_names[-1] == "model.ple"
    assert len(groups[4].layer_names) == 12
    assert all(name.endswith(".indexer.raw_key_cache") for name in groups[4].layer_names)

    for ordinal, group_id in enumerate((1, 2, 3)):
        expected = [f"model.layers.{index}.linear_attn" for index in range(ordinal, 36, 3)]
        assert groups[group_id].layer_names[:12] == expected


def test_qwen4_exp_layout_has_four_planes_and_independent_ring() -> None:
    layout = build_qwen4_exp_kv_cache_layout(_five_groups(), num_blocks=3)
    assert layout is not None
    assert layout.slot_count == 12
    assert layout.ring_slot_count == 12
    assert [plane.name for plane in layout.planes] == [
        TENSOR1,
        TENSOR2,
        TENSOR3,
        TENSOR4,
    ]
    assert [plane.page_size_bytes for plane in layout.planes] == [
        128 * 256 * 2,
        128 * 256 * 2,
        1280 * 2,
        32 * 128 * 2,
    ]
    assert layout.plane(TENSOR1).page_size_bytes // 2 - 10240 == 22528
    assert layout.plane(TENSOR2).page_size_bytes - (128 * 128 * 2) == 128 * 128 * 2
    assert layout.ring_page_size_bytes == 8 * 130 * 2
    assert layout.ring_backing_size == (layout.ring_slot_count * layout.ring_slot_backing_size)

    ple = layout.owner("model.ple")
    assert ple.role == PLE
    assert ple.group_id == 3
    assert ple.slot == 0
    gdn_groups = {owner.group_id for owner in layout.owners if owner.role == GDN}
    assert gdn_groups == {1, 2, 3}
    assert {owner.group_id for owner in layout.owners if owner.role == QSA_RAW} == {4}


def test_mtp_adds_one_qsa_and_ring_slot_and_propagates_eagle() -> None:
    groups = _five_groups(include_mtp=True)
    assert len(groups) == 5
    assert groups[0].is_eagle_group
    assert groups[4].is_eagle_group
    layout = build_qwen4_exp_kv_cache_layout(groups, num_blocks=2)
    assert layout is not None
    assert layout.slot_count == 13
    assert layout.ring_slot_count == 13
    assert layout.owner("model.mtp.layers.48.self_attn.attn").slot == 12
    assert layout.owner("model.mtp.layers.48.self_attn.indexer.raw_key_cache").group_id == 4


def test_packed_views_are_bounded_and_ring_storage_is_independent() -> None:
    layout = build_qwen4_exp_kv_cache_layout(_five_groups(), num_blocks=3)
    assert layout is not None
    planes = {
        plane.name: torch.zeros(layout.plane_backing_size(plane.name), dtype=torch.int8) for plane in layout.planes
    }
    ring = torch.zeros(layout.ring_backing_size, dtype=torch.int8)
    pointers = {backing.untyped_storage().data_ptr() for backing in planes.values()}
    assert len(pointers) == 4
    assert ring.untyped_storage().data_ptr() not in pointers

    qsa_source = "model.layers.3.self_attn"
    qsa_owner = layout.owner(f"{qsa_source}.attn")
    k_cache = make_contiguous_plane_view(
        planes[TENSOR2],
        dtype=torch.bfloat16,
        num_blocks=3,
        item_shape=(128, 1, 256),
        storage_offset=qsa_owner.slot * layout.plane(TENSOR2).size,
    )
    v_cache = make_contiguous_plane_view(
        planes[TENSOR1],
        dtype=torch.bfloat16,
        num_blocks=3,
        item_shape=(128, 1, 256),
        storage_offset=qsa_owner.slot * layout.plane(TENSOR1).size,
    )
    raw_owner = layout.owner(f"{qsa_source}.indexer.raw_key_cache")
    raw_cache = make_contiguous_plane_view(
        ring,
        dtype=torch.bfloat16,
        num_blocks=3,
        item_shape=(1, 8, 130),
        storage_offset=raw_owner.slot * layout.ring_slot_backing_size,
    )
    assert k_cache.shape == (3, 128, 1, 256)
    assert v_cache.shape == (3, 128, 1, 256)
    assert raw_cache.shape == (3, 1, 8, 130)
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    assert raw_cache.is_contiguous()
    assert raw_cache.untyped_storage().data_ptr() == ring.untyped_storage().data_ptr()
    assert raw_cache.untyped_storage().data_ptr() not in pointers


def test_planner_describes_normal_and_ring_backings_separately() -> None:
    config = _get_qwen4_exp_kv_cache_config(
        SimpleNamespace(
            cache_config=SimpleNamespace(
                num_gpu_blocks_override=3,
                prefix_cache_retention_interval=None,
            )
        ),
        _five_groups(),
        10**9,
    )
    assert config is not None
    assert config.num_blocks == 3
    assert len(config.kv_cache_groups) == 5
    layout = build_qwen4_exp_kv_cache_layout(config.kv_cache_groups, 3)
    assert layout is not None
    normal = config.kv_cache_tensors[:6]
    ring = config.kv_cache_tensors[6]
    assert [tensor.size for tensor in normal] == [
        layout.plane_backing_size(name)
        for name in (
            TENSOR2,
            TENSOR4,
            TENSOR3,
            TENSOR3,
            TENSOR3,
            TENSOR1,
        )
    ]
    assert ring.size == layout.ring_backing_size
    assert ring.layers == config.kv_cache_groups[4].layer_names


def test_runner_allocates_independent_ring_and_materializes_views() -> None:
    config = _get_qwen4_exp_kv_cache_config(
        SimpleNamespace(
            cache_config=SimpleNamespace(
                num_gpu_blocks_override=3,
                prefix_cache_retention_interval=None,
            )
        ),
        _five_groups(),
        10**9,
    )
    assert config is not None
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        cache_config=SimpleNamespace(),
    )
    runner.use_sparse = False
    runner.use_compress = False
    runner.use_hybrid_blocks = False
    runner.runner_only_attn_layers = set()
    runner.sparse_kv_offload_enabled = False
    runner._kv_cache_spec_attn_group_iterator = lambda: [
        SimpleNamespace(
            backend=SimpleNamespace(),
            kv_cache_spec=group.kv_cache_spec,
            layer_names=group.layer_names,
        )
        for group in config.kv_cache_groups
    ]

    raw = runner._allocate_kv_cache_tensors(config)
    caches = runner._reshape_kv_cache_tensors(config, raw)
    source = "model.layers.3.self_attn"
    normal = raw[f"{source}.attn"]
    ring = raw[f"{source}.indexer.raw_key_cache"]
    assert isinstance(normal, tuple)
    assert isinstance(ring, torch.Tensor)
    normal_pointers = {backing.untyped_storage().data_ptr() for backing in normal}
    assert len(normal_pointers) == 2
    assert ring.untyped_storage().data_ptr() not in normal_pointers
    gdn = raw["model.layers.0.linear_attn"]
    ple = raw["model.ple"]
    compressed = raw[f"{source}.indexer.compressed_key_cache"]
    assert isinstance(gdn, tuple)
    assert isinstance(ple, torch.Tensor)
    assert isinstance(compressed, torch.Tensor)
    assert gdn[1].untyped_storage().data_ptr() == normal[0].untyped_storage().data_ptr()
    assert ple.untyped_storage().data_ptr() == normal[1].untyped_storage().data_ptr()
    four_plane_pointers = {
        normal[0].untyped_storage().data_ptr(),
        normal[1].untyped_storage().data_ptr(),
        gdn[0].untyped_storage().data_ptr(),
        compressed.untyped_storage().data_ptr(),
    }
    assert len(four_plane_pointers) == 4
    assert caches[f"{source}.attn"][0].shape == (3, 128, 1, 256)
    assert caches[f"{source}.attn"][1].shape == (3, 128, 1, 256)
    assert caches[f"{source}.indexer.raw_key_cache"].shape == (3, 1, 8, 130)
    assert caches["model.ple"][0].shape == (3, 10240)


def test_qsa_source_sets_must_match() -> None:
    groups = _five_groups()
    groups[4].layer_names[-1] = "model.layers.46.self_attn.indexer.raw_key_cache"
    try:
        build_qwen4_exp_kv_cache_layout(groups, num_blocks=2)
    except (KeyError, ValueError) as error:
        assert "raw" in str(error).lower() or "source-layer mapping" in str(error)
    else:
        raise AssertionError("mismatched QSA owners were accepted")
