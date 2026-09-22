from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import regex as re
import torch
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import get_kv_cache_compression_ratio

QSA_MAIN: Final = "qsa_main"
QSA_RAW: Final = "qsa_raw"
QSA_COMPRESSED: Final = "qsa_compressed"
GDN: Final = "gdn"
PLE: Final = "ple"
HIDDEN: Final = "hidden"

TENSOR1: Final = "tensor1"
TENSOR2: Final = "tensor2"
TENSOR3: Final = "tensor3"
TENSOR4: Final = "tensor4"


@dataclass(frozen=True)
class CacheOwner:
    layer_name: str
    spec: KVCacheSpec
    role: str
    slot: int
    group_id: int


@dataclass(frozen=True)
class TensorPlane:
    name: str
    page_size_bytes: int
    num_blocks: int

    @property
    def size(self) -> int:
        return self.page_size_bytes * self.num_blocks


@dataclass(frozen=True)
class Qwen4ExpKVCacheLayout:
    """Four packed tensor planes plus an independent QSA ring backing."""

    planes: tuple[TensorPlane, ...]
    owners: tuple[CacheOwner, ...]
    slot_count: int
    ring_slot_count: int
    ring_slot_backing_size: int
    ring_page_size_bytes: int
    alignment: int
    num_blocks: int

    def plane(self, name: str) -> TensorPlane:
        return next(plane for plane in self.planes if plane.name == name)

    def owner(self, layer_name: str) -> CacheOwner:
        return next(owner for owner in self.owners if owner.layer_name == layer_name)

    @property
    def normal_backing_size(self) -> int:
        return sum(self.plane_backing_size(plane.name) for plane in self.planes)

    def plane_backing_size(self, name: str) -> int:
        return self.slot_count * self.plane(name).size

    @property
    def ring_backing_size(self) -> int:
        return self.ring_slot_count * self.ring_slot_backing_size

    def plane_name_for_owner(self, layer_name: str) -> str:
        return {
            QSA_MAIN: TENSOR2,
            QSA_COMPRESSED: TENSOR4,
            GDN: TENSOR3,
            PLE: TENSOR1,
        }[self.owner(layer_name).role]


def _group_member_specs(group: KVCacheGroupSpec) -> dict[str, KVCacheSpec]:
    if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
        return group.kv_cache_spec.kv_cache_specs
    return {name: group.kv_cache_spec for name in group.layer_names}


def _classify_spec(layer_name: str, spec: KVCacheSpec) -> str | None:
    if isinstance(spec, HiddenStateCacheSpec):
        return HIDDEN
    if layer_name.endswith(".raw_key_cache") and isinstance(spec, CircularBufferSpec):
        return QSA_RAW
    if (
        layer_name.endswith(".compressed_key_cache")
        and isinstance(spec, MLAAttentionSpec)
        and get_kv_cache_compression_ratio(spec) > 1
    ):
        return QSA_COMPRESSED
    if layer_name.endswith(".attn") and isinstance(spec, FullAttentionSpec):
        return QSA_MAIN
    if isinstance(spec, MambaSpec):
        if len(spec.shapes) == 2:
            return GDN
        if len(spec.shapes) == 1:
            return PLE
    return None


def _qsa_source_name(layer_name: str, role: str) -> str:
    suffix = {
        QSA_MAIN: ".attn",
        QSA_RAW: ".indexer.raw_key_cache",
        QSA_COMPRESSED: ".indexer.compressed_key_cache",
    }[role]
    if not layer_name.endswith(suffix):
        raise ValueError(f"Invalid {role} owner name: {layer_name}")
    return layer_name[: -len(suffix)]


def _layer_sort_key(layer_name: str) -> tuple[int, str]:
    matches = re.findall(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if not matches:
        raise ValueError(f"Cannot derive source layer index from {layer_name}")
    return int(matches[-1]), layer_name


def _state_bytes(spec: MambaSpec, index: int) -> int:
    return math.prod(spec.shapes[index]) * get_dtype_size(spec.dtypes[index])


def _qsa_main_page_bytes(spec: FullAttentionSpec) -> tuple[int, int]:
    dtype_size = get_dtype_size(spec.dtype)
    tokens_heads = spec.block_size * spec.num_kv_heads
    return (
        tokens_heads * spec.head_size * dtype_size,
        tokens_heads * spec.head_size_v * dtype_size,
    )


def _real_page_size_bytes(spec: KVCacheSpec) -> int:
    return int(getattr(spec, "real_page_size_bytes", spec.page_size_bytes))


def build_qwen4_exp_kv_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
    num_blocks: int,
) -> Qwen4ExpKVCacheLayout | None:
    if num_blocks <= 0:
        raise ValueError("Qwen4Exp layout requires a positive block count")

    members: dict[str, list[tuple[str, KVCacheSpec, int]]] = {
        QSA_MAIN: [],
        QSA_RAW: [],
        QSA_COMPRESSED: [],
        GDN: [],
        PLE: [],
        HIDDEN: [],
    }
    unknown: list[tuple[str, KVCacheSpec]] = []
    for group_id, group in enumerate(kv_cache_groups):
        specs = _group_member_specs(group)
        for layer_name in group.layer_names:
            spec = specs[layer_name]
            role = _classify_spec(layer_name, spec)
            if role is None:
                unknown.append((layer_name, spec))
                continue
            members[role].append((layer_name, spec, group_id))

    if not members[QSA_RAW]:
        return None
    required = (QSA_MAIN, QSA_RAW, QSA_COMPRESSED, GDN, PLE)
    missing = [role for role in required if not members[role]]
    if missing or unknown:
        details = [(name, type(spec).__name__) for name, spec in unknown]
        raise ValueError(f"Qwen4Exp packed cache layout is incomplete: missing={missing}, unsupported={details}")

    qsa_by_role: dict[str, dict[str, tuple[str, KVCacheSpec, int]]] = {}
    for role in (QSA_MAIN, QSA_RAW, QSA_COMPRESSED):
        by_source = {_qsa_source_name(name, role): (name, spec, group_id) for name, spec, group_id in members[role]}
        if len(by_source) != len(members[role]):
            raise ValueError(f"Duplicate {role} owner")
        qsa_by_role[role] = by_source
    source_sets = {role: set(owners) for role, owners in qsa_by_role.items()}
    if len({frozenset(sources) for sources in source_sets.values()}) != 1:
        raise ValueError(f"QSA main/raw/compressed owners do not form a one-to-one source-layer mapping: {source_sets}")
    qsa_sources = sorted(source_sets[QSA_MAIN], key=_layer_sort_key)

    qsa_group_ids = {group_id for role in (QSA_MAIN, QSA_COMPRESSED) for _, _, group_id in members[role]}
    raw_group_ids = {group_id for _, _, group_id in members[QSA_RAW]}
    if len(qsa_group_ids) != 1 or len(raw_group_ids) != 1:
        raise ValueError("QSA main/compressed and raw owners need one group each")
    if qsa_group_ids == raw_group_ids:
        raise ValueError("QSA RingBuffer must have an independent cache group")
    if qsa_group_ids != {0} or raw_group_ids != {4}:
        raise ValueError(
            f"Qwen4Exp requires QSA at group 0 and RingBuffer at group 4; got qsa={qsa_group_ids}, ring={raw_group_ids}"
        )

    gdn_group_ids = sorted({group_id for _, _, group_id in members[GDN]})
    if gdn_group_ids != [1, 2, 3]:
        raise ValueError(f"Qwen4Exp packed cache requires GDN groups 1, 2 and 3; got {gdn_group_ids}")
    ple_group_ids = {group_id for _, _, group_id in members[PLE]}
    if ple_group_ids != {gdn_group_ids[-1]}:
        raise ValueError("PLE must share the third GDN cache group")

    gdn_by_group: dict[int, list[tuple[str, MambaSpec]]] = {}
    for name, spec, group_id in members[GDN]:
        assert isinstance(spec, MambaSpec)
        gdn_by_group.setdefault(group_id, []).append((name, spec))
    for entries in gdn_by_group.values():
        entries.sort(key=lambda item: _layer_sort_key(item[0]))
    gdn_counts = {len(entries) for entries in gdn_by_group.values()}
    if gdn_counts != {12}:
        raise ValueError(f"Qwen4Exp requires 12 GDN owners per group, got {gdn_counts}")

    main_specs = [spec for _, spec, _ in members[QSA_MAIN] if isinstance(spec, FullAttentionSpec)]
    gdn_specs = [spec for _, spec, _ in members[GDN] if isinstance(spec, MambaSpec)]
    ple_specs = [spec for _, spec, _ in members[PLE] if isinstance(spec, MambaSpec)]
    compressed_specs = [spec for _, spec, _ in members[QSA_COMPRESSED]]
    raw_specs = [spec for _, spec, _ in members[QSA_RAW]]
    main_pages = [_qsa_main_page_bytes(spec) for spec in main_specs]
    plane_sizes = (
        max(
            max(v_bytes for _, v_bytes in main_pages),
            max(_state_bytes(spec, 0) for spec in ple_specs),
        ),
        max(
            max(k_bytes for k_bytes, _ in main_pages),
            max(_state_bytes(spec, 1) for spec in gdn_specs),
        ),
        max(_state_bytes(spec, 0) for spec in gdn_specs),
        max(_real_page_size_bytes(spec) for spec in compressed_specs),
    )
    ring_page_size = max(_real_page_size_bytes(spec) for spec in raw_specs)
    dtype_sizes = {
        get_dtype_size(dtype)
        for role in required
        for _, spec, _ in members[role]
        for dtype in (spec.dtypes if isinstance(spec, MambaSpec) else (spec.dtype,))
    }
    alignment = math.lcm(16, *dtype_sizes)
    planes = [
        TensorPlane(name, page_size, num_blocks)
        for name, page_size in zip((TENSOR1, TENSOR2, TENSOR3, TENSOR4), plane_sizes, strict=True)
    ]
    ring_slot_backing_size = round_up(ring_page_size * num_blocks, alignment)

    owners: list[CacheOwner] = []
    for slot, source in enumerate(qsa_sources):
        for role in (QSA_MAIN, QSA_COMPRESSED, QSA_RAW):
            name, spec, group_id = qsa_by_role[role][source]
            owners.append(CacheOwner(name, spec, role, slot, group_id))
    for group_id in gdn_group_ids:
        owners.extend(
            CacheOwner(name, spec, GDN, slot, group_id) for slot, (name, spec) in enumerate(gdn_by_group[group_id])
        )
    owners.extend(CacheOwner(name, spec, PLE, 0, group_id) for name, spec, group_id in members[PLE])
    owners.extend(
        CacheOwner(name, spec, HIDDEN, slot, group_id) for slot, (name, spec, group_id) in enumerate(members[HIDDEN])
    )
    return Qwen4ExpKVCacheLayout(
        planes=tuple(planes),
        owners=tuple(owners),
        slot_count=max(len(qsa_sources), 12),
        ring_slot_count=len(qsa_sources),
        ring_slot_backing_size=ring_slot_backing_size,
        ring_page_size_bytes=ring_page_size,
        alignment=alignment,
        num_blocks=num_blocks,
    )


def make_contiguous_plane_view(
    backing: torch.Tensor,
    *,
    dtype: torch.dtype,
    num_blocks: int,
    item_shape: tuple[int, ...],
    storage_offset: int,
) -> torch.Tensor:
    dtype_size = get_dtype_size(dtype)
    if storage_offset % dtype_size:
        raise ValueError("Packed cache offset must align to the view dtype")
    required_bytes = num_blocks * math.prod(item_shape) * dtype_size
    if storage_offset + required_bytes > backing.numel():
        raise ValueError("Packed cache view exceeds its backing allocation")
    result = backing[storage_offset : storage_offset + required_bytes].view(dtype).view(num_blocks, *item_shape)
    assert result.is_contiguous()
    return result


__all__ = [
    "GDN",
    "HIDDEN",
    "PLE",
    "QSA_COMPRESSED",
    "QSA_MAIN",
    "QSA_RAW",
    "Qwen4ExpKVCacheLayout",
    "TENSOR1",
    "TENSOR2",
    "TENSOR3",
    "TENSOR4",
    "build_qwen4_exp_kv_cache_layout",
    "make_contiguous_plane_view",
]
