"""Free CUDA memory before another suite app uses the GPU.

torch is imported inside the functions that touch the device. That package is
an optional extra, so unit tests can import the pipeline without it. This is
not a circular import.
"""
from __future__ import annotations

import gc
from typing import Any


def release_gate_weights(gate: object) -> None:
    """Drop a gate's GPU weights, then return the freed blocks to the driver."""
    release = getattr(gate, "release_weights", None)
    if callable(release):
        release()
    release_cuda_cache()


def release_cuda_cache() -> None:
    """Collect dropped models and return their CUDA blocks to the driver."""
    gc.collect()
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    ipc_collect = getattr(torch.cuda, "ipc_collect", None)
    if ipc_collect is not None:
        ipc_collect()


def free_vram_gb() -> float | None:
    """Free GPU memory in GB, or None when this process has no CUDA device."""
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return float(free) / (1024 ** 3)


def _torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch
