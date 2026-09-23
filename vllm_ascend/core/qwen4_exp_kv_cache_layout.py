from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import regex as re
import torch
from vllm.config import VllmConfig
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.core.kv_cache_interface import get_kv_cache_compression_ratio

QSA_MAIN: Final = "qsa_main"
QSA_RAW: Final = "qsa_raw"
QSA_COMPRESSED: Final = "qsa_compressed"
GDN: Final = "gdn"
PLE: Final = "ple"
HIDDEN: Final = "hidden"

VALUE_OR_PLE: Final = "value_or_ple"
KEY_OR_SSM: Final = "key_or_ssm"
CONV_STATE: Final = "conv_state"
COMPRESSED_KEY: Final = "compressed_key"
CIRCULAR_STATE: Final = "circular_state"


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
    slot_count: int
    num_blocks: int
    layout: KVCacheLayout

    @property
    def size(self) -> int:
        return self.page_size_bytes * self.slot_count * self.num_blocks

    @property
    def layer_stride(self) -> int:
        return self.page_size_bytes if self.layout.is_block_outermost else self.num_blocks * self.page_size_bytes

    @property
    def block_stride(self) -> int:
        return self.slot_count * self.page_size_bytes if self.layout.is_block_outermost else self.page_size_bytes

    def storage_offset(self, slot: int) -> int:
        if not 0 <= slot < self.slot_count:
            raise ValueError(f"Invalid {self.name} slot {slot}/{self.slot_count}")
        return slot * self.layer_stride


@dataclass(frozen=True)
class CacheComponentView:
    owner_name: str
    component: str
    plane_name: str
    slot: int
    dtype: torch.dtype
    item_shape: tuple[int, ...]


@dataclass(frozen=True)
class Qwen4ExpKVCacheLayout:
    """Four normal planes plus an independent circular-state plane."""

    planes: tuple[TensorPlane, ...]
    owners: tuple[CacheOwner, ...]
    views: tuple[CacheComponentView, ...]
    alignment: int
    num_blocks: int
    layout: KVCacheLayout

    def plane(self, name: str) -> TensorPlane:
        return next(p for p in self.planes if p.name == name)

    def owner(self, layer_name: str) -> CacheOwner:
        return next(o for o in self.owners if o.layer_name == layer_name)

    def owner_views(self, layer_name: str) -> tuple[CacheComponentView, ...]:
        return tuple(v for v in self.views if v.owner_name == layer_name)

    @property
    def normal_backing_size(self) -> int:
        return sum(p.size for p in self.planes if p.name != CIRCULAR_STATE)

    @property
    def ring_backing_size(self) -> int:
        return self.plane(CIRCULAR_STATE).size

    @property
    def ring_page_size_bytes(self) -> int:
        return self.plane(CIRCULAR_STATE).page_size_bytes

    @property
    def ring_slot_count(self) -> int:
        return self.plane(CIRCULAR_STATE).slot_count

    @property
    def ring_slot_backing_size(self) -> int:
        return self.plane(CIRCULAR_STATE).layer_stride

    @property
    def slot_count(self) -> int:
        return max(p.slot_count for p in self.planes if p.name != CIRCULAR_STATE)

    def plane_backing_size(self, name: str) -> int:
        return self.plane(name).size


def _classify(name: str, spec: KVCacheSpec) -> str | None:
    if isinstance(spec, HiddenStateCacheSpec):
        return HIDDEN
    if name.endswith(".raw_key_cache") and isinstance(spec, CircularBufferSpec):
        return QSA_RAW
    if (
        name.endswith(".compressed_key_cache")
        and isinstance(spec, MLAAttentionSpec)
        and get_kv_cache_compression_ratio(spec) > 1
    ):
        return QSA_COMPRESSED
    if name.endswith(".attn") and isinstance(spec, FullAttentionSpec):
        return QSA_MAIN
    if isinstance(spec, MambaSpec):
        return GDN if len(spec.shapes) == 2 else PLE if len(spec.shapes) == 1 else None
    return None


def _source(name: str, role: str) -> str:
    suffix = {
        QSA_MAIN: ".attn",
        QSA_RAW: ".indexer.raw_key_cache",
        QSA_COMPRESSED: ".indexer.compressed_key_cache",
    }[role]
    if not name.endswith(suffix):
        raise ValueError(f"Invalid {role} owner: {name}")
    return name[: -len(suffix)]


def _sort_key(name: str) -> tuple[int, str]:
    match = re.findall(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return (int(match[-1]), name) if match else (2**31 - 1, name)


def _members(group: KVCacheGroupSpec) -> dict[str, KVCacheSpec]:
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.kv_cache_specs
    return dict.fromkeys(group.layer_names, spec)


def _uniform(role: str, specs: dict[str, KVCacheSpec]) -> UniformTypeKVCacheSpecs:
    result = UniformTypeKVCacheSpecs.from_specs(specs)
    if result is None:
        raise ValueError(
            f"Qwen4Exp cache group is not uniform: role={role}, "
            f"specs={[(n, type(s).__name__) for n, s in specs.items()]}"
        )
    return result


def _state_bytes(spec: MambaSpec, index: int) -> int:
    return math.prod(spec.shapes[index]) * get_dtype_size(spec.dtypes[index])


class Qwen4ExpKVCachePlanner:
    """Own Qwen4Exp logical grouping and component-aware physical planning."""

    def __init__(self, vllm_config: VllmConfig | None = None):
        self.vllm_config = vllm_config

    @staticmethod
    def is_applicable(specs: dict[str, KVCacheSpec]) -> bool:
        return any(_classify(n, s) == QSA_RAW for n, s in specs.items())

    def get_kv_cache_groups(self, specs: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
        if not self.is_applicable(specs):
            raise ValueError("Qwen4Exp planner received non-Qwen cache specs")
        qsa: dict[str, dict[str, str]] = {QSA_MAIN: {}, QSA_COMPRESSED: {}, QSA_RAW: {}}
        for name, spec in specs.items():
            role = _classify(name, spec)
            if role in qsa:
                qsa[role][_source(name, role)] = name
        source_sets = {role: set(names) for role, names in qsa.items()}
        if len({frozenset(v) for v in source_sets.values()}) != 1:
            raise ValueError(f"QSA owners do not form one-to-one sources: {source_sets}")
        sources = sorted(source_sets[QSA_MAIN], key=_sort_key)
        for source in sources:
            main = specs[qsa[QSA_MAIN][source]]
            compressed = specs[qsa[QSA_COMPRESSED][source]]
            raw = specs[qsa[QSA_RAW][source]]
            assert isinstance(main, FullAttentionSpec)
            assert isinstance(compressed, MLAAttentionSpec)
            assert isinstance(raw, CircularBufferSpec)
            ratio = get_kv_cache_compression_ratio(compressed)
            if main.block_size != compressed.block_size:
                raise ValueError("QSA main/compressed block sizes differ")
            if main.block_size % 128 or main.block_size % ratio or raw.block_size % ratio:
                raise ValueError("QSA block sizes violate kernel/compression alignment")

        main_names = [qsa[QSA_MAIN][s] for s in sources]
        compressed_names = [qsa[QSA_COMPRESSED][s] for s in sources]
        raw_names = [qsa[QSA_RAW][s] for s in sources]
        groups: list[KVCacheGroupSpec] = []
        combined = {n: specs[n] for n in [*main_names, *compressed_names]}
        if uniform := UniformTypeKVCacheSpecs.from_specs(combined):
            groups.append(KVCacheGroupSpec(list(combined), uniform, is_eagle_group=any(".mtp." in n for n in combined)))
        else:
            for role, names in ((QSA_MAIN, main_names), (QSA_COMPRESSED, compressed_names)):
                subset = {n: specs[n] for n in names}
                groups.append(
                    KVCacheGroupSpec(names, _uniform(role, subset), is_eagle_group=any(".mtp." in n for n in names))
                )

        gdn_names = sorted((n for n, s in specs.items() if _classify(n, s) == GDN), key=_sort_key)
        target_qsa = sum(".mtp." not in s for s in sources)
        if not target_qsa or len(gdn_names) % target_qsa:
            raise ValueError(f"Cannot derive GDN lanes: target_qsa={target_qsa}, gdn={len(gdn_names)}")
        lane_count = len(gdn_names) // target_qsa
        for lane in range(lane_count):
            names = gdn_names[lane::lane_count]
            subset = {n: specs[n] for n in names}
            groups.append(KVCacheGroupSpec(names, _uniform(f"gdn_{lane}", subset)))

        ple_names = sorted((n for n, s in specs.items() if _classify(n, s) == PLE), key=_sort_key)
        if not ple_names:
            raise ValueError("Qwen4Exp requires a PLE cache owner")
        groups.append(KVCacheGroupSpec(ple_names, _uniform(PLE, {n: specs[n] for n in ple_names})))
        groups.append(
            KVCacheGroupSpec(
                raw_names,
                _uniform(QSA_RAW, {n: specs[n] for n in raw_names}),
                is_eagle_group=any(".mtp." in n for n in raw_names),
                enable_kv_transfer=False,
            )
        )
        groups.extend(KVCacheGroupSpec([n], s) for n, s in specs.items() if _classify(n, s) == HIDDEN)
        assigned = {n for g in groups for n in g.layer_names}
        unknown = [(n, type(s).__name__) for n, s in specs.items() if n not in assigned]
        if unknown:
            raise ValueError(f"Unsupported Qwen4Exp cache owners: {unknown}")
        return groups

    def _layout(self) -> KVCacheLayout:
        if self.vllm_config is None:
            raise ValueError("VllmConfig is required to resolve KVCacheLayout")
        return self.vllm_config.cache_config.get_resolved_kv_cache_layout()

    def build_physical_plan(
        self,
        groups: list[KVCacheGroupSpec],
        num_blocks: int,
        layout: KVCacheLayout | None = None,
    ) -> Qwen4ExpKVCacheLayout | None:
        if num_blocks <= 0:
            raise ValueError("Qwen4Exp requires a positive block count")
        layout = layout or self._layout()
        if not layout.is_block_compact:
            raise ValueError(f"Qwen4Exp requires block-compact layout, got {layout.name}")
        by_role: dict[str, list[tuple[str, KVCacheSpec, int]]] = {
            r: [] for r in (QSA_MAIN, QSA_RAW, QSA_COMPRESSED, GDN, PLE, HIDDEN)
        }
        unknown = []
        for group_id, group in enumerate(groups):
            specs = _members(group)
            for name in group.layer_names:
                role = _classify(name, specs[name])
                (by_role[role] if role else unknown).append((name, specs[name], group_id))
        if not by_role[QSA_RAW]:
            return None
        missing = [r for r in (QSA_MAIN, QSA_RAW, QSA_COMPRESSED, GDN, PLE) if not by_role[r]]
        if missing or unknown:
            raise ValueError(f"Incomplete Qwen4Exp layout: missing={missing}, unknown={unknown}")

        qsa: dict[str, dict[str, tuple[str, KVCacheSpec, int]]] = {}
        for role in (QSA_MAIN, QSA_COMPRESSED, QSA_RAW):
            qsa[role] = {_source(n, role): (n, s, g) for n, s, g in by_role[role]}
        if len({frozenset(v) for v in (set(x) for x in qsa.values())}) != 1:
            raise ValueError("QSA physical owners do not form one-to-one sources")
        sources = sorted(qsa[QSA_MAIN], key=_sort_key)
        gdn_groups: dict[int, list[tuple[str, MambaSpec]]] = {}
        for n, s, g in by_role[GDN]:
            assert isinstance(s, MambaSpec)
            gdn_groups.setdefault(g, []).append((n, s))
        for entries in gdn_groups.values():
            entries.sort(key=lambda x: _sort_key(x[0]))
        ple = sorted(by_role[PLE], key=lambda x: _sort_key(x[0]))
        slots = max(len(sources), len(ple), *(len(v) for v in gdn_groups.values()))

        mains = [s for _, s, _ in by_role[QSA_MAIN]]
        gdns = [s for _, s, _ in by_role[GDN]]
        ples = [s for _, s, _ in by_role[PLE]]
        assert all(isinstance(s, FullAttentionSpec) for s in mains)
        assert all(isinstance(s, MambaSpec) for s in [*gdns, *ples])
        main_pages = [
            (
                s.block_size * s.num_kv_heads * s.head_size * get_dtype_size(s.dtype),
                s.block_size * s.num_kv_heads * s.head_size_v * get_dtype_size(s.dtype),
            )
            for s in mains
            if isinstance(s, FullAttentionSpec)
        ]
        alignment = math.lcm(
            16,
            *(
                get_dtype_size(d)
                for role in (QSA_MAIN, QSA_RAW, QSA_COMPRESSED, GDN, PLE)
                for _, s, _ in by_role[role]
                for d in (s.dtypes if isinstance(s, MambaSpec) else (s.dtype,))
            ),
        )
        pages = {
            VALUE_OR_PLE: max(max(v for _, v in main_pages), max(_state_bytes(s, 0) for s in ples)),
            KEY_OR_SSM: max(max(k for k, _ in main_pages), max(_state_bytes(s, 1) for s in gdns)),
            CONV_STATE: max(_state_bytes(s, 0) for s in gdns),
            COMPRESSED_KEY: max(
                s.block_size * s.num_kv_heads * s.head_size * get_dtype_size(s.dtype)
                for _, s, _ in by_role[QSA_COMPRESSED]
                if isinstance(s, MLAAttentionSpec)
            ),
            CIRCULAR_STATE: max(
                int(getattr(s, "real_page_size_bytes", s.page_size_bytes)) for _, s, _ in by_role[QSA_RAW]
            ),
        }
        planes = tuple(
            TensorPlane(n, round_up(w, alignment), len(sources) if n == CIRCULAR_STATE else slots, num_blocks, layout)
            for n, w in pages.items()
        )
        owners: list[CacheOwner] = []
        views: list[CacheComponentView] = []
        for slot, source in enumerate(sources):
            for role in (QSA_MAIN, QSA_COMPRESSED, QSA_RAW):
                n, s, g = qsa[role][source]
                owners.append(CacheOwner(n, s, role, slot, g))
                if role == QSA_MAIN:
                    assert isinstance(s, FullAttentionSpec)
                    views += [
                        CacheComponentView(
                            n, "k", KEY_OR_SSM, slot, s.dtype, (s.block_size, s.num_kv_heads, s.head_size)
                        ),
                        CacheComponentView(
                            n, "v", VALUE_OR_PLE, slot, s.dtype, (s.block_size, s.num_kv_heads, s.head_size_v)
                        ),
                    ]
                elif role == QSA_COMPRESSED:
                    assert isinstance(s, MLAAttentionSpec)
                    views.append(
                        CacheComponentView(
                            n,
                            "compressed_k",
                            COMPRESSED_KEY,
                            slot,
                            s.dtype,
                            (s.num_kv_heads, s.num_states, s.head_size),
                        )
                    )
                else:
                    assert isinstance(s, CircularBufferSpec)
                    views.append(
                        CacheComponentView(
                            n, "raw_k", CIRCULAR_STATE, slot, s.dtype, (s.num_kv_heads, s.block_size, s.head_size)
                        )
                    )
        for group_id, entries in sorted(gdn_groups.items()):
            for slot, (n, s) in enumerate(entries):
                owners.append(CacheOwner(n, s, GDN, slot, group_id))
                views += [
                    CacheComponentView(n, "conv", CONV_STATE, slot, s.dtypes[0], tuple(s.shapes[0])),
                    CacheComponentView(n, "ssm", KEY_OR_SSM, slot, s.dtypes[1], tuple(s.shapes[1])),
                ]
        for slot, (n, s, g) in enumerate(ple):
            assert isinstance(s, MambaSpec)
            owners.append(CacheOwner(n, s, PLE, slot, g))
            views.append(CacheComponentView(n, "ple", VALUE_OR_PLE, slot, s.dtypes[0], tuple(s.shapes[0])))
        owners += [CacheOwner(n, s, HIDDEN, i, g) for i, (n, s, g) in enumerate(by_role[HIDDEN])]
        return Qwen4ExpKVCacheLayout(planes, tuple(owners), tuple(views), alignment, num_blocks, layout)

    def make_kv_cache_tensors(self, plan: Qwen4ExpKVCacheLayout) -> list[KVCacheTensor]:
        anchors = {
            QSA_MAIN: KEY_OR_SSM,
            QSA_COMPRESSED: COMPRESSED_KEY,
            GDN: KEY_OR_SSM,
            PLE: VALUE_OR_PLE,
            QSA_RAW: CIRCULAR_STATE,
        }
        result = []
        for role, plane_name in anchors.items():
            role_owners = [o for o in plan.owners if o.role == role]
            for group_id in sorted({o.group_id for o in role_owners}):
                owners = sorted((o for o in role_owners if o.group_id == group_id), key=lambda o: o.slot)
                plane = plan.plane(plane_name)
                result.append(
                    KVCacheTensor(plane.size, [o.layer_name for o in owners], plane.layer_stride, plane.block_stride, 0)
                )
        result += [
            KVCacheTensor(
                o.spec.page_size_bytes * plan.num_blocks,
                [o.layer_name],
                o.spec.page_size_bytes * plan.num_blocks,
                o.spec.page_size_bytes,
                0,
            )
            for o in plan.owners
            if o.role == HIDDEN
        ]
        return result


def build_qwen4_exp_kv_cache_layout(
    groups: list[KVCacheGroupSpec],
    num_blocks: int,
    layout: KVCacheLayout = KVCacheLayout.LBNHC,
) -> Qwen4ExpKVCacheLayout | None:
    return Qwen4ExpKVCachePlanner().build_physical_plan(groups, num_blocks, layout)


def make_plane_view(
    backing: torch.Tensor, *, plane: TensorPlane, slot: int, dtype: torch.dtype, item_shape: tuple[int, ...]
) -> torch.Tensor:
    dtype_size = get_dtype_size(dtype)
    offset = plane.storage_offset(slot)
    if offset % dtype_size or plane.block_stride % dtype_size:
        raise ValueError("Packed cache geometry is not dtype-aligned")
    if math.prod(item_shape) * dtype_size > plane.page_size_bytes or backing.numel() < plane.size:
        raise ValueError("Packed cache view exceeds its plane")
    inner, stride = [], 1
    for size in reversed(item_shape):
        inner.append(stride)
        stride *= size
    return torch.as_strided(
        backing.view(dtype),
        (plane.num_blocks, *item_shape),
        (plane.block_stride // dtype_size, *reversed(inner)),
        offset // dtype_size,
    )


def make_contiguous_plane_view(
    backing: torch.Tensor, *, dtype: torch.dtype, num_blocks: int, item_shape: tuple[int, ...], storage_offset: int
) -> torch.Tensor:
    dtype_size = get_dtype_size(dtype)
    required = num_blocks * math.prod(item_shape) * dtype_size
    if storage_offset % dtype_size or storage_offset + required > backing.numel():
        raise ValueError("Packed cache view exceeds its backing allocation")
    return backing[storage_offset : storage_offset + required].view(dtype).view(num_blocks, *item_shape)


__all__ = [
    "CIRCULAR_STATE",
    "COMPRESSED_KEY",
    "CONV_STATE",
    "GDN",
    "HIDDEN",
    "KEY_OR_SSM",
    "PLE",
    "QSA_COMPRESSED",
    "QSA_MAIN",
    "QSA_RAW",
    "Qwen4ExpKVCacheLayout",
    "Qwen4ExpKVCachePlanner",
    "VALUE_OR_PLE",
    "build_qwen4_exp_kv_cache_layout",
    "make_contiguous_plane_view",
    "make_plane_view",
]
