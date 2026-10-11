"""NVIDIA GPU probe. There is no CPU path and no path for cards older than sm_89."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from canvas.app.info import MIN_COMPUTE_CAPABILITY


@dataclass(frozen=True)
class GpuInfo:
    """The first NVIDIA GPU this process can see."""

    name: str
    total_gb: float
    free_gb: float
    compute_major: int
    compute_minor: int
    source: str

    @property
    def supported(self) -> bool:
        """True for RTX 40 series (compute capability 8.9) or newer."""
        return (self.compute_major, self.compute_minor) >= MIN_COMPUTE_CAPABILITY


def _parse_cc(value: str) -> tuple[int, int]:
    major, minor = value.split(".", 1)
    return int(major), int(minor)


def vram_cap_gb() -> float | None:
    """``CANVAS_VRAM_CAP_GB`` when it is a positive number of gigabytes."""
    raw = os.environ.get("CANVAS_VRAM_CAP_GB", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def apply_vram_cap(info: GpuInfo, cap_gb: float | None = None) -> GpuInfo:
    """Limit reported memory to ``cap_gb`` and reserve only that fraction on the device.

    Tier detection reads ``total_gb``, so a 24 GB card capped at 16 GB is the 16 GB tier.
    """
    cap = vram_cap_gb() if cap_gb is None else cap_gb
    if cap is None or info.total_gb <= 0:
        return info
    limited = min(info.total_gb, cap)
    _limit_fraction(limited / info.total_gb)
    return GpuInfo(
        info.name,
        limited,
        min(info.free_gb, limited),
        info.compute_major,
        info.compute_minor,
        info.source,
    )


def _limit_fraction(fraction: float) -> None:
    """Apply ``torch.cuda.set_per_process_memory_fraction`` when CUDA is present.

    torch is an optional extra. Importing it here is not a circular import.
    A second call, or a machine without CUDA, leaves the process as it is.
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    setter = getattr(torch.cuda, "set_per_process_memory_fraction", None)
    if setter is None:
        return
    try:
        setter(fraction)
    except (RuntimeError, ValueError):
        return


def probe_gpu() -> GpuInfo | None:
    """Read the first NVIDIA GPU. ``VOXPRINT_FAKE_VRAM`` is ``name,total_gb,free_gb,cc`` for tests."""
    info = _probe_gpu()
    if info is None:
        return None
    return apply_vram_cap(info)


def _probe_gpu() -> GpuInfo | None:
    """The uncapped GPU reading."""
    fake = os.environ.get("VOXPRINT_FAKE_VRAM", "").strip()
    if fake:
        name, total, free, cc = [part.strip() for part in fake.split(",")]
        major, minor = _parse_cc(cc)
        return GpuInfo(name, float(total), float(free), major, minor, "fake")
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            creationflags=flags,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    row = [part.strip() for part in proc.stdout.splitlines()[0].split(",")]
    if len(row) < 4:
        return None
    major, minor = _parse_cc(row[3])
    return GpuInfo(row[0], float(row[1]) / 1024.0, float(row[2]) / 1024.0, major, minor, "nvidia-smi")
