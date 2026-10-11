"""Choose the 16 GB schedule from the VRAM budget and the system RAM.

Plan B is group offload: the text encoder runs and is freed, the transformer
keeps one block group on the GPU, then the VAE decodes. The planning peak is
14.5 GB at 2048×1152 and 2048×2048. Qwen-Image-Edit keeps its FP8 weights in
system RAM, so a machine under 40 GB is refused before any weight is loaded.

Plan A, the Nunchaku SVDQuant INT4 generator, is selected only when the
registry row is enabled. Until then this module never returns it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import assert_never

import psutil

from canvas.models.modes import TIER_24_MIN_GB, Mode
from canvas.pipeline.bf16_schedule import GpuLedger, VramRefused
from canvas.pipeline.vram_probe import PeakVramProbe

PEAK_TARGET_GB = 14.5
QWEN_MIN_RAM_GB = 40.0
NUNCHAKU_PEAK_GB = 12.0

# FP8 Qwen2.5-VL weights plus the image-token activations for one encode.
# The encoder is freed before the transformer block group is loaded.
EDIT_ENCODER_GB = 13.1
EDIT_BLOCK_GB = 2.4
EDIT_ACT_PER_MP_GB = 0.6
EDIT_VAE_GB = 1.1

_image_seconds: float | None = None


class Placement(Enum):
    """Where the weights sit for one job."""

    GROUP_OFFLOAD = "group-offload"
    RESIDENT = "resident"
    NUNCHAKU_INT4 = "nunchaku-int4"


class SystemMemoryRefused(Exception):
    """Instruction edit would exhaust system RAM. The message names the requirement."""

    def __init__(self, have_gb: float, need_gb: float = QWEN_MIN_RAM_GB) -> None:
        self.have_gb = float(have_gb)
        self.need_gb = float(need_gb)
        shown = max(0, int(self.have_gb))
        super().__init__(
            f"Instruction edit needs {self.need_gb:.0f} GB of system memory. This PC has {shown} GB."
        )


@dataclass(frozen=True)
class ScheduleChoice:
    """The placement for one budget. ``peak_target_gb`` is the planning ceiling."""

    placement: Placement
    peak_target_gb: float


@dataclass(frozen=True)
class EditTrace:
    """How one 16 GB instruction edit used the GPU."""

    peak_gb: float
    tiled: bool


def system_ram_gb() -> float:
    """Installed system RAM in GB."""
    return float(psutil.virtual_memory().total) / (1024 ** 3)


def choose_schedule(
    budget_gb: float,
    mode: Mode,
    ram_gb: float,
    *,
    nunchaku_ready: bool = False,
) -> ScheduleChoice:
    """Pick Plan B under the 24 GB line, or the resident path at 23.5 GB and above.

    Instruction edit is refused when system RAM is under 40 GB, on either tier.
    The INT4 generator is used only when ``nunchaku_ready`` is true, and never
    for instruction edit or Creative.
    """
    if mode is Mode.INSTRUCTION and float(ram_gb) < QWEN_MIN_RAM_GB:
        raise SystemMemoryRefused(ram_gb)
    budget = float(budget_gb)
    if budget >= TIER_24_MIN_GB:
        return ScheduleChoice(Placement.RESIDENT, 24.0)
    if nunchaku_ready and _nunchaku_mode(mode):
        return ScheduleChoice(Placement.NUNCHAKU_INT4, NUNCHAKU_PEAK_GB)
    return ScheduleChoice(Placement.GROUP_OFFLOAD, PEAK_TARGET_GB)


def edit_group_peak_gb(width: int, height: int) -> float:
    """Planning peak for one 16 GB instruction edit. The encoder is the tall stage."""
    megapixels = max(0.0, float(width) * float(height)) / 1_000_000.0
    denoise = EDIT_BLOCK_GB + EDIT_ACT_PER_MP_GB * megapixels
    return max(EDIT_ENCODER_GB, denoise, EDIT_VAE_GB)


def run_edit_16gb(
    *,
    width: int,
    height: int,
    ledger: GpuLedger,
    probe: PeakVramProbe | None = None,
) -> EditTrace:
    """Encode, free the encoder, denoise one block group, then a tiled VAE decode."""
    start = len(ledger.history)
    try:
        _encode(ledger, probe)
        _denoise(width, height, ledger, probe)
        _decode(ledger, probe)
    except VramRefused:
        ledger.drop_all()
        raise
    if ledger.held:
        ledger.drop_all()
    samples = ledger.history[start - 1 :] if ledger.history else [ledger.used_gb]
    return EditTrace(max(samples), True)


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


def _nunchaku_mode(mode: Mode) -> bool:
    """Z-Image-Turbo modes the INT4 generator can replace. Creative and edit stay out."""
    if mode is Mode.TEXT or mode is Mode.RESTYLE or mode is Mode.INPAINT or mode is Mode.OUTPAINT:
        return True
    if mode is Mode.POSE or mode is Mode.DEPTH or mode is Mode.EDGE:
        return True
    if mode is Mode.TEXT_CREATIVE or mode is Mode.INSTRUCTION:
        return False
    assert_never(mode)


def _mark(probe: PeakVramProbe | None, stage: str) -> None:
    if probe is not None:
        probe.mark(stage)


def _encode(ledger: GpuLedger, probe: PeakVramProbe | None) -> None:
    if not ledger.take("encoder", EDIT_ENCODER_GB):
        raise VramRefused()
    _mark(probe, "encode-gpu-start")
    _mark(probe, "encode-gpu-end")
    ledger.drop("encoder")
    _mark(probe, "encode-free")


def _denoise(width: int, height: int, ledger: GpuLedger, probe: PeakVramProbe | None) -> None:
    megapixels = max(0.0, float(width) * float(height)) / 1_000_000.0
    if not ledger.take("transformer", EDIT_BLOCK_GB + EDIT_ACT_PER_MP_GB * megapixels):
        raise VramRefused()
    _mark(probe, "denoise-start")
    _mark(probe, "denoise-end")
    ledger.drop("transformer")
    _mark(probe, "transformer-free")


def _decode(ledger: GpuLedger, probe: PeakVramProbe | None) -> None:
    if not ledger.take("vae", EDIT_VAE_GB):
        raise VramRefused()
    _mark(probe, "decode-start")
    _mark(probe, "decode-tiled")
    _mark(probe, "decode-end")
    ledger.drop("vae")
    _mark(probe, "vae-free")
