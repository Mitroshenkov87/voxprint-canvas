"""Whether this process may still use its CUDA context.

rc.34/35: two heavy jobs on two cards ran out of memory, ``empty_cache`` then
raised ``AcceleratorError``, and the next garbage collection freed tensors that
belonged to the broken context. The driver aborted the interpreter (exit 3).

After a fault like that this process never touches CUDA again. :func:`mark_lost`
keeps every object reachable from the failing exception alive, freezes the
garbage collector so nothing that exists now is freed by a collection, and
turns automatic collection off. The job ends with error code ``vram`` and the
process leaves with :data:`EXIT_CUDA_LOST` through :func:`exit_now`, which skips
the interpreter's teardown and with it every finalizer. A worker host starts a
fresh process.

torch is never imported here.
"""
from __future__ import annotations

import gc
import os
import sys
import threading

from canvas.infra.errors import note_error
from canvas.infra.stage_timer import log_detail

# Distinct from 0 (clean), 1 (job failed), 2 (usage) and 3 (abort / no GPU).
EXIT_CUDA_LOST = 75

_FAULT_TYPES = frozenset({"AcceleratorError", "CudaError", "CUDAError"})
_FAULT_TEXT = (
    "cuda error",
    "cudaerror",
    "illegal memory access",
    "unspecified launch failure",
    "device-side assert",
    "cublas_status",
    "cudnn_status",
    "misaligned address",
    "an illegal instruction",
)

_lock = threading.Lock()
_lost = False
_reason = ""
# Objects whose finalizers must not run while the context is broken.
_graveyard: list[object] = []


def is_cuda_fault(exc: BaseException | None) -> bool:
    """True for an error that leaves the CUDA context unusable (not a plain out-of-memory)."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ in _FAULT_TYPES:
            return True
        text = str(exc).lower()
        if any(marker in text for marker in _FAULT_TEXT):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def lost() -> bool:
    """True once CUDA failed in a way this process cannot recover from."""
    return _lost


def reason() -> str:
    """The error type that marked the context lost, or ``""``."""
    return _reason


def mark_lost(cause: BaseException | str, *keep: object) -> None:
    """Stop every later CUDA call in this process and keep ``cause`` and ``keep`` alive.

    Safe to call more than once. Never raises.
    """
    global _lost, _reason
    with _lock:
        _graveyard.append(cause)
        _graveyard.extend(keep)
        if _lost:
            return
        _lost = True
        _reason = cause if isinstance(cause, str) else type(cause).__name__
    try:
        gc.disable()
        gc.freeze()
    except Exception:  # noqa: BLE001 - this path must not raise
        pass
    try:
        note_error("cuda.lost")
        log_detail("cuda", "lost", "cuda", f"pid={os.getpid()} cause={_token(_reason)}")
    except Exception:  # noqa: BLE001
        pass


def exit_now(code: int = EXIT_CUDA_LOST) -> None:
    """Flush the standard streams and leave without running finalizers or atexit hooks.

    The RAM ledger entry is removed first, so other jobs stop counting this one.
    """
    try:
        from canvas.infra import ram_ledger

        ram_ledger.withdraw()
    except Exception:  # noqa: BLE001
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    _exit(code)


def _exit(code: int) -> None:
    """Tests replace this."""
    os._exit(code)


def reset() -> None:
    """Forget a lost context. Tests only: a real process exits instead."""
    global _lost, _reason
    with _lock:
        _lost = False
        _reason = ""
        _graveyard.clear()
    try:
        gc.unfreeze()
        gc.enable()
    except Exception:  # noqa: BLE001
        pass


def _token(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in value)
    return cleaned[:40] or "unknown"
