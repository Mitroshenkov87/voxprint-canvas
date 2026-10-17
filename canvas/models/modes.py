"""Generation modes and the VRAM tier that may run them."""
from __future__ import annotations

from enum import Enum
from typing import assert_never

from canvas.infra.vram import get_vram_budget_gb
from canvas.safety.inputs import ImageRole


class Tier(Enum):
    """15.5 GB, or 23.5 GB and above."""

    GB16 = "16gb"
    GB24 = "24gb"


class Mode(Enum):
    """One picture operation the window can run."""

    TEXT = "text"
    TEXT_CREATIVE = "text-creative"
    RESTYLE = "restyle"
    INPAINT = "inpaint"
    OUTPAINT = "outpaint"
    POSE = "pose"
    DEPTH = "depth"
    EDGE = "edge"


# Full speed headroom is the suite rule, clamped to the 2-3 GB band used at install.
HEADROOM_FLOOR_GB = 2.0
HEADROOM_CAP_GB = 3.0
HEADROOM_RATIO = 0.08
TIER_24_MIN_GB = 23.5
TIER_16_MIN_GB = 15.5


def headroom_gb(total_gb: float) -> float:
    """GB left unused: at least 2, at most 3, and at least 8 % of the card."""
    total = max(0.0, float(total_gb))
    return min(HEADROOM_CAP_GB, max(HEADROOM_FLOOR_GB, HEADROOM_RATIO * total))


def select_tier(total_gb: float) -> Tier | None:
    """23.5 GB and above uses the larger tier. 15.5 GB uses the smaller one. Below 15.5 GB has no tier.

    A VRAM cap from ``CANVAS_VRAM_CAP_GB`` or ``--vram-cap-gb`` replaces the
    card size when it is lower. A 24 GB card capped at 16 GB takes the 16 GB schedule.
    """
    budget = get_vram_budget_gb(float(total_gb))
    total = float(total_gb) if budget is None else float(budget)
    if total >= TIER_24_MIN_GB:
        return Tier.GB24
    if total >= TIER_16_MIN_GB:
        return Tier.GB16
    return None


def visible_modes(tier: Tier) -> tuple[Mode, ...]:
    """Modes the UI may offer. Creative stays on 24 GB."""
    shared = (
        Mode.TEXT,
        Mode.RESTYLE,
        Mode.INPAINT,
        Mode.OUTPAINT,
        Mode.POSE,
        Mode.DEPTH,
        Mode.EDGE,
    )
    if tier is Tier.GB16:
        return shared
    if tier is Tier.GB24:
        return shared + (Mode.TEXT_CREATIVE,)
    assert_never(tier)


def required_roles(mode: Mode) -> tuple[ImageRole, ...]:
    """Images a mode needs. Text needs none. Every supplied image is still screened."""
    if mode is Mode.TEXT or mode is Mode.TEXT_CREATIVE:
        return ()
    if mode is Mode.RESTYLE:
        return (ImageRole.INIT,)
    if mode is Mode.INPAINT or mode is Mode.OUTPAINT:
        return (ImageRole.INIT, ImageRole.MASK)
    if mode is Mode.POSE:
        return (ImageRole.POSE,)
    if mode is Mode.DEPTH:
        return (ImageRole.DEPTH,)
    if mode is Mode.EDGE:
        return (ImageRole.EDGE,)
    assert_never(mode)


def mode_group(mode: Mode) -> str:
    """``text`` or ``image`` for the two UI tabs."""
    if mode is Mode.TEXT or mode is Mode.TEXT_CREATIVE:
        return "text"
    if mode is Mode.RESTYLE or mode is Mode.INPAINT or mode is Mode.OUTPAINT:
        return "image"
    if mode is Mode.POSE or mode is Mode.DEPTH or mode is Mode.EDGE:
        return "image"
    assert_never(mode)
