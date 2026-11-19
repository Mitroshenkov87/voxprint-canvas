"""Process VRAM budget shared by tier selection and the memory cap.

``CANVAS_VRAM_CAP_GB`` and ``--vram-cap-gb`` are the same number. Another part
of startup applies ``torch.cuda.set_per_process_memory_fraction`` from it.
This module only reads the cap and reports the bytes PyTorch has allocated.
"""
from __future__ import annotations

import os

ENV_VRAM_CAP = "CANVAS_VRAM_CAP_GB"
_CLI_CAP: float | None = None


def set_vram_cap_gb(value: float | None) -> None:
    """Remember a cap from the command line for this process. ``None`` clears it."""
    global _CLI_CAP
    if value is None:
        _CLI_CAP = None
        os.environ.pop(ENV_VRAM_CAP, None)
        return
    cap = float(value)
    if cap <= 0:
        raise ValueError("VRAM cap must be positive")
    _CLI_CAP = cap
    os.environ[ENV_VRAM_CAP] = _format_cap(cap)


def get_vram_budget_gb(physical_gb: float | None = None) -> float | None:
    """GB this process may plan for.

    A cap from the command line or ``CANVAS_VRAM_CAP_GB`` limits the physical
    card. With no cap, the physical size is returned. ``None`` when neither
    value is known.
    """
    cap = _read_cap()
    if physical_gb is None:
        return cap
    physical = float(physical_gb)
    if cap is None:
        return physical
    return min(physical, cap)


def peak_allocated_bytes() -> int | None:
    """``torch.cuda.max_memory_allocated`` for this process, or None without CUDA."""
    try:
        import torch
    except ImportError:
        return None
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        return None
    return int(cuda.max_memory_allocated())


def reset_peak_allocated() -> None:
    """Start a new peak sample. A machine without CUDA is unchanged."""
    try:
        import torch
    except ImportError:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        return
    # A Photoshoot phase spans many stage samples. It keeps the larger peak.
    from canvas.infra.phase_vram import fold_peak

    fold_peak()
    cuda.reset_peak_memory_stats()


def _read_cap() -> float | None:
    if _CLI_CAP is not None:
        return _CLI_CAP
    raw = os.environ.get(ENV_VRAM_CAP, "").strip()
    if not raw:
        return None
    try:
        cap = float(raw)
    except ValueError:
        return None
    if cap <= 0:
        return None
    return cap


def _format_cap(cap: float) -> str:
    if cap == int(cap):
        return str(int(cap))
    return str(cap)
