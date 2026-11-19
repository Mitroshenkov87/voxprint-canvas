"""Choose the 16 GB schedule from the VRAM budget.

Group offload runs the text encoder and frees it, keeps one transformer block
group on the GPU, then decodes with the VAE. The planning peak is 14.5 GB at
2048×1152 and 2048×2048. At 23.5 GB and above the weights stay resident.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from canvas.models.modes import TIER_24_MIN_GB, Mode

PEAK_TARGET_GB = 14.5

_image_seconds: float | None = None


class Placement(Enum):
    """Where the weights sit for one job."""

    GROUP_OFFLOAD = "group-offload"
    RESIDENT = "resident"


@dataclass(frozen=True)
class ScheduleChoice:
    """The placement for one budget. ``peak_target_gb`` is the planning ceiling."""

    placement: Placement
    peak_target_gb: float


def choose_schedule(budget_gb: float, mode: Mode, ram_gb: float) -> ScheduleChoice:
    """Pick group offload under the 24 GB line, or the resident path at 23.5 GB and above."""
    del mode, ram_gb
    budget = float(budget_gb)
    if budget >= TIER_24_MIN_GB:
        return ScheduleChoice(Placement.RESIDENT, 24.0)
    return ScheduleChoice(Placement.GROUP_OFFLOAD, PEAK_TARGET_GB)


def note_image_seconds(seconds: float) -> None:
    """Remember the last diffusion call so the command can report seconds per image."""
    global _image_seconds
    _image_seconds = float(seconds)


def last_image_seconds() -> float | None:
    """Seconds of the last diffusion call, or None when none has run."""
    return _image_seconds


def reset_image_seconds() -> None:
    """Clear the remembered diffusion time."""
    global _image_seconds
    _image_seconds = None
