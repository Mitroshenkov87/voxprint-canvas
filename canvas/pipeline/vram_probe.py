"""Peak-VRAM marks for the generate-then-upscale chain.

A 16 GB card can pass ``sample_used_mib`` later. The chain only calls ``mark``.
"""
from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable


class PeakVramProbe:
    def __init__(self, sample: Callable[[], int | None] | None = None) -> None:
        self.sample = sample
        self.marks: list[tuple[str, int | None]] = []

    def mark(self, stage: str) -> None:
        used = None if self.sample is None else self.sample()
        self.marks.append((stage, used))


def sample_used_mib() -> int | None:
    """Current GPU memory in MiB, or None when nvidia-smi is not available."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()
    if not line:
        return None
    try:
        return int(line[0].strip())
    except ValueError:
        return None
