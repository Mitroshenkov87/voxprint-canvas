"""Per-phase CUDA peaks and the release check between Photoshoot phases.

A series runs in three phases: ``prep`` (writer, prompt checks, text encoder),
``render`` (the transformer, loaded once) and ``decode`` (the VAE, once for all
frames). Each phase resets ``torch.cuda.max_memory_allocated`` when it starts
and logs its peak when it ends. Stages inside a phase reset the same counter
for their own lines, so every reset folds the current peak into the open
phase first.

Torch is optional. Without CUDA every reading is 0 and nothing is logged as
over the limit.
"""
from __future__ import annotations

import gc
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from canvas.infra.errors import note_error
from canvas.infra.stage_timer import log_detail

# What the previous phase may leave allocated before the next model loads.
RESIDUAL_LIMIT_BYTES = 512 * 1024**2
_MIB = 1024**2

_lock = threading.Lock()
_open: list["Phase"] = []


class ResidualVram(Exception):
    """A phase finished and its models still hold VRAM above the limit."""

    code = "vram.residual"

    def __init__(self, phase: str, allocated: int) -> None:
        super().__init__(f"{phase} left {allocated} bytes allocated on the GPU")
        self.phase = phase
        self.allocated = allocated


class Phase:
    """One open phase. ``finish`` logs its peak once."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.peak = 0
        self.started = time.perf_counter()
        self.done = False

    def finish(self) -> int:
        """Close the phase and log ``phase <name> peak``. Returns the peak bytes."""
        if self.done:
            return self.peak
        self.done = True
        fold_peak()
        with _lock:
            if self in _open:
                _open.remove(self)
        cuda = _cuda()
        allocated = 0 if cuda is None else _counter(cuda, "memory_allocated")
        elapsed = max(0, int((time.perf_counter() - self.started) * 1000))
        log_detail(
            "phase", _token(self.name), "cuda" if cuda is not None else "cpu",
            f"peak={self.peak} peak_mib={self.peak // _MIB} allocated={allocated} ms={elapsed}",
        )
        return self.peak


def start_phase(name: str) -> Phase:
    """Reset the CUDA peak and open ``name``. The caller ends it with ``finish``."""
    fold_peak()
    cuda = _cuda()
    if cuda is not None:
        _reset(cuda)
    current = Phase(name)
    with _lock:
        _open.append(current)
    return current


def _cuda() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return None
    try:
        if not bool(cuda.is_available()):
            return None
    except (TypeError, RuntimeError, AttributeError):
        return None
    return cuda


def _counter(cuda: Any, name: str) -> int:
    fn = getattr(cuda, name, None)
    if not callable(fn):
        return 0
    try:
        return max(0, int(fn()))
    except (TypeError, ValueError, RuntimeError):
        return 0


def fold_peak() -> None:
    """Add the current allocator peak to every open phase. Call before a reset."""
    with _lock:
        if not _open:
            return
    cuda = _cuda()
    if cuda is None:
        return
    peak = _counter(cuda, "max_memory_allocated")
    with _lock:
        for phase in _open:
            phase.peak = max(phase.peak, peak)


def _reset(cuda: Any) -> None:
    fn = getattr(cuda, "reset_peak_memory_stats", None)
    if callable(fn):
        try:
            fn()
        except (TypeError, RuntimeError, AttributeError):
            return


@contextmanager
def phase(name: str) -> Iterator[Phase]:
    """Measure one phase: reset the peak, run, then log ``phase <name> peak``."""
    current = start_phase(name)
    try:
        yield current
    finally:
        current.finish()


def release_residual(name: str) -> int:
    """Collect, empty the allocator, and return the bytes still allocated."""
    gc.collect()
    cuda = _cuda()
    allocated = reserved = 0
    if cuda is not None:
        for fn_name in ("synchronize", "empty_cache"):
            fn = getattr(cuda, fn_name, None)
            if callable(fn):
                try:
                    fn()
                except (TypeError, RuntimeError):
                    continue
        allocated = _counter(cuda, "memory_allocated")
        reserved = _counter(cuda, "memory_reserved")
    over = allocated > RESIDUAL_LIMIT_BYTES
    log_detail(
        "phase", _token(name), "cuda" if cuda is not None else "cpu",
        f"release allocated={allocated} reserved={reserved} limit={RESIDUAL_LIMIT_BYTES} over={int(over)}",
    )
    return allocated


def require_released(name: str) -> int:
    """Stop when ``name`` left more than ``RESIDUAL_LIMIT_BYTES`` allocated.

    Raises:
        ResidualVram: The next phase would start with that memory taken.
    """
    allocated = release_residual(name)
    if allocated > RESIDUAL_LIMIT_BYTES:
        note_error(f"{_token(name)}.residual")
        raise ResidualVram(name, allocated)
    return allocated


def _token(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name.lower())
    return cleaned[:32] or "phase"
