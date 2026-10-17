"""Aspect ratios mapped to each model's native frames.

Native frames stay near 1–2 megapixels. A 4K request uses the ~2K frame and
an upscale whose long side is 3840. Nothing in this table is a 4K generation.
"""
from __future__ import annotations

from typing import assert_never

from canvas.models.modes import Mode
from canvas.models.plan import BaseNotSelected
from canvas.models.registry import Catalog, FrameSize, RegistryError

ASPECT_GROUPS = (
    ("Landscape", ("16:9", "21:9", "3:2", "4:3")),
    ("Square", ("1:1",)),
    ("Portrait", ("9:16", "2:3", "3:4")),
)
UI_ASPECTS = tuple(aspect for _group, aspects in ASPECT_GROUPS for aspect in aspects)
FOUR_K_LONG = 3840


def family_for(mode: Mode, catalog: Catalog) -> str:
    """Which size table a mode uses. Image modes stay on Z-Image."""
    if mode is Mode.TEXT:
        family = catalog.default_text_base
        if family is None:
            raise BaseNotSelected()
        return family
    if mode is Mode.TEXT_CREATIVE or mode is Mode.RESTYLE or mode is Mode.INPAINT or mode is Mode.OUTPAINT:
        return "z-image-turbo"
    if mode is Mode.POSE or mode is Mode.DEPTH or mode is Mode.EDGE:
        return "z-image-turbo"
    assert_never(mode)


def frame_size(catalog: Catalog, family: str, aspect: str, *, twok: bool = False) -> FrameSize:
    """One registered frame. ``twok`` is the source for a 4K upscale."""
    if catalog.sizes is None or family not in catalog.sizes:
        raise RegistryError(f"{family} has no sizes")
    table = catalog.sizes[family]
    band = table.twok if twok else table.native
    if aspect not in band:
        raise RegistryError(f"unknown aspect {aspect}")
    return band[aspect]


def upscale_factor(frame: FrameSize) -> int:
    """2 or 4, so the long side becomes 3840. A 4K frame is rejected."""
    long_side = max(frame.width, frame.height)
    if long_side < 1 or FOUR_K_LONG % long_side:
        raise RegistryError("the native size does not scale to 4K")
    factor = FOUR_K_LONG // long_side
    if factor not in (2, 4):
        raise RegistryError("4K is an upscale, not a direct generation size")
    return factor


def closest_aspect(width: int, height: int) -> str:
    """The picker ratio nearest to an input image."""
    if width < 1 or height < 1:
        raise RegistryError("image size must be positive")
    target = width / height
    best = "1:1"
    best_gap = abs(target - 1.0)
    for aspect in UI_ASPECTS:
        ratio_w, ratio_h = _ratio(aspect)
        gap = abs(target - (ratio_w / ratio_h))
        if gap < best_gap:
            best = aspect
            best_gap = gap
    return best


def _ratio(aspect: str) -> tuple[int, int]:
    width, _, height = aspect.partition(":")
    if not width.isdigit() or not height.isdigit():
        raise RegistryError(f"unknown aspect {aspect}")
    return int(width), int(height)
