"""Which instruction-edit weights a tier uses.

16 GB prefers our r256 mirror when that mirror is published, then the QuantFunc
r256 SVDQuant pack, then QuantFunc r128, then the FP8 pack with group offload.
24 GB stays on the FP8 pack. ``CANVAS_EDIT_ROUTE=fp8`` selects that pack on
16 GB as well, which is how a machine check compares the two routes.
"""
from __future__ import annotations

import os
from typing import assert_never

from canvas.models.modes import Tier
from canvas.models.registry import Catalog, ModelSpec
from canvas.models.snapshots import snapshot_ready

FP8_EDIT_ID = "qwen-image-edit-2511-fp8"
MIRROR_R256_ID = "qwen-image-edit-2511-int4-r256-mirror"
QUANTFUNC_R256_ID = "qwen-image-edit-2511-int4-r256"
QUANTFUNC_R128_ID = "qwen-image-edit-2511-int4-r128"
PREFERENCE_16 = (MIRROR_R256_ID, QUANTFUNC_R256_ID, QUANTFUNC_R128_ID, FP8_EDIT_ID)
ENV_EDIT_ROUTE = "CANVAS_EDIT_ROUTE"

_override: str | None = None


def set_edit_route_override(value: str | None) -> None:
    """Force the next instruction-edit selection. ``None`` follows the tier."""
    global _override
    _override = value


def edit_route_override() -> str:
    """``fp8`` when a caller asked for the FP8 pack, otherwise empty."""
    if _override:
        return _override
    return os.environ.get(ENV_EDIT_ROUTE, "").strip().casefold()


def select_edit_route(catalog: Catalog, tier: Tier) -> ModelSpec:
    """The edit pack for this tier. A pending mirror is skipped."""
    if tier is Tier.GB24 or edit_route_override() == "fp8":
        return catalog.by_id(FP8_EDIT_ID)
    if tier is Tier.GB16:
        for model_id in PREFERENCE_16:
            spec = catalog.by_id(model_id)
            if _usable(spec):
                return spec
        return catalog.by_id(FP8_EDIT_ID)
    assert_never(tier)


def edit_route_report(catalog: Catalog, tier: Tier, root) -> dict[str, object]:
    """The selected route and whether its pinned files are on disk."""
    spec = select_edit_route(catalog, tier)
    return {
        "id": spec.id,
        "quant": spec.quant,
        "rank": spec.variant,
        "licence": spec.licence,
        "revision": spec.revision,
        "on_disk": snapshot_ready(spec, root) is not None,
        "fallback": FP8_EDIT_ID if spec.quant == "int4" else "",
    }


def int4_weight_name(spec: ModelSpec) -> str:
    """The single INT4 transformer file on this route."""
    names = [
        name
        for name in spec.files
        if name.endswith(".safetensors") and "/" not in name and "text-encoder" not in name
    ]
    if len(names) != 1:
        raise ValueError("the INT4 route needs one transformer file")
    return names[0]


def _usable(spec: ModelSpec) -> bool:
    if spec.mirror_status == "pending" or spec.status != "ready":
        return False
    if spec.licence.casefold() not in {"apache-2.0", "mit"}:
        return False
    if spec.quant == "int4":
        return spec.pinned()
    return True
