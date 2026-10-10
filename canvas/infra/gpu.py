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


def probe_gpu() -> GpuInfo | None:
    """Read the first NVIDIA GPU. ``VOXPRINT_FAKE_VRAM`` is ``name,total_gb,free_gb,cc`` for tests."""
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
