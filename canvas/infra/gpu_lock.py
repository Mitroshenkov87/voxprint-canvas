"""GPU lock, one per CUDA device (``%TEMP%/voxprint-gpu-<device>.lock``).

Adapted from Voxprint AI Movie Dubber ``dubber/infra/gpu_lock.py``
(Apache-2.0, Copyright Aleksandr Mitroshenkov).

The file is JSON ``{owner, pid, job, started, eta}`` with ISO-8601 UTC times.
It is stale when the recorded pid is gone or when ``eta`` is more than two hours
in the past.

The device is the physical GPU this process uses: the UUID torch reports for
the current CUDA device, which follows ``CUDA_VISIBLE_DEVICES``. Two processes
on different GPUs hold different files and run side by side; two on the same
GPU exclude each other. A live holder of the older machine-wide
``voxprint-gpu.lock`` from another suite app still counts as busy.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from ctypes import wintypes
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from canvas.infra.errors import note_error

OWNER = "canvas"
LOCK_NAME = "voxprint-gpu.lock"
STALE_AFTER = timedelta(hours=2)
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87
# Lock files a whole command holds. ``release`` leaves them in place until ``pinned`` exits.
_PINNED: set[str] = set()


class GpuLockTimeout(TimeoutError):
    """Another suite app holds the GPU lock."""


def lock_path(device: str | None = None) -> Path:
    """The lock file for one CUDA device. ``VOXPRINT_GPU_LOCK`` overrides it with one fixed path.

    Args:
        device: A device key. None is the device this process uses.
    """
    override = os.environ.get("VOXPRINT_GPU_LOCK", "").strip()
    if override:
        return Path(override)
    key = device_key() if device is None else _safe_key(device)
    return Path(tempfile.gettempdir()) / f"voxprint-gpu-{key}.lock"


def legacy_lock_path() -> Path:
    """The older machine-wide suite lock. Canvas reads it and never writes it."""
    return Path(tempfile.gettempdir()) / LOCK_NAME


def device_key() -> str:
    """A file-name key for the physical GPU this process uses.

    The torch UUID of the current CUDA device when torch can see one. Otherwise
    the first ``CUDA_VISIBLE_DEVICES`` entry, else ``0``.
    """
    uuid = _cuda_uuid()
    if uuid:
        return _safe_key(uuid)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return _safe_key(visible or "0")


@lru_cache(maxsize=1)
def _cuda_uuid() -> str | None:
    """The UUID of the current CUDA device. Read once per process."""
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
    except (RuntimeError, AssertionError, AttributeError):
        return None
    uuid = getattr(props, "uuid", None)
    if not uuid:
        return None
    text = str(uuid)
    return text if text.startswith("GPU-") else f"GPU-{text}"


def forget_device_key() -> None:
    """Drop the remembered card after CUDA_VISIBLE_DEVICES changed."""
    clear = getattr(_cuda_uuid, "cache_clear", None)
    if clear is not None:
        clear()


def _safe_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())[:64]
    return key or "0"


def _foreign_legacy_holder(path: Path, owner: str) -> dict[str, Any] | None:
    """A live holder of the machine-wide lock from another app, if any."""
    if os.environ.get("VOXPRINT_GPU_LOCK", "").strip() or path == legacy_lock_path():
        return None
    current = read_lock(legacy_lock_path())
    if not current or current.get("owner") == owner or is_stale(current):
        return None
    return current


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(value: object) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def _windows_pid_alive(pid: int) -> bool:
    """Ask the kernel whether ``pid`` is running.

    ``os.kill(pid, 0)`` is Ctrl+C on Windows, not an existence probe, and it
    raises ``WinError 87`` for a process outside this console.
    """
    if sys.platform != "win32":
        return False
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetLastError.restype = wintypes.DWORD
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return int(kernel32.GetLastError()) == _ERROR_ACCESS_DENIED
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return int(kernel32.GetLastError()) != _ERROR_INVALID_PARAMETER
        return int(code.value) == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _pid_alive(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, (int, str)):
        return False
    try:
        value = int(pid)
    except ValueError:
        return False
    if value <= 0:
        return False
    if sys.platform == "win32":
        return _windows_pid_alive(value)
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def read_lock(path: Path) -> dict[str, Any] | None:
    """The lock object, None when the file is absent, or ``{}`` when it is unreadable."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except OSError:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def is_stale(data: dict[str, Any], now: datetime | None = None) -> bool:
    """True when the pid is gone or ``eta`` is more than two hours ago."""
    moment = now or _now()
    if not _pid_alive(data.get("pid")):
        return True
    eta = _parse(data.get("eta")) or _parse(data.get("started"))
    if eta is None:
        return True
    return moment - eta > STALE_AFTER


def _remove_stale(path: Path, seen: dict[str, Any]) -> bool:
    """Remove the lock file ``seen`` describes. False when it changed or cannot be removed.

    A second process may have taken over between the read and the removal; its
    fresh lock is left alone. A takeover from a dead owner is logged once.
    """
    now = read_lock(path)
    if now is None:
        return True
    if now != seen:
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if seen.get("pid") != os.getpid():
        note_error(f"gpu_lock: took over a stale lock (pid {seen.get('pid')})")
    return True


def try_acquire(job: str, eta_s: float, path: Path, owner: str = OWNER) -> dict[str, Any] | None:
    """Create the lock. Return None on success, otherwise the holder's data."""
    foreign = _foreign_legacy_holder(path, owner)
    if foreign is not None:
        return foreign
    current = read_lock(path)
    if current is not None:
        mine = current.get("owner") == owner and current.get("pid") == os.getpid()
        if mine and str(path) in _PINNED:
            return None
        if not mine and not is_stale(current):
            return current
        if not _remove_stale(path, current):
            return read_lock(path) or {"owner": "?"}
    now = _now()
    payload = {
        "owner": owner,
        "pid": os.getpid(),
        "job": job,
        "started": _iso(now),
        "eta": _iso(now + timedelta(seconds=max(60.0, eta_s))),
    }
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    try:
        if os.name == "nt":
            os.rename(tmp, path)
        else:
            os.link(tmp, path)
            tmp.unlink()
    except (FileExistsError, PermissionError):
        tmp.unlink(missing_ok=True)
        return read_lock(path) or {"owner": "?"}
    return None


def prepare_startup_lock() -> str | None:
    """Drop a lock whose owner is gone.

    Returns ``busy`` when a live process still holds the lock.
    """
    path = lock_path()
    if _foreign_legacy_holder(path, OWNER) is not None:
        return "busy"
    current = read_lock(path)
    if not current:
        return None
    if is_stale(current):
        return None if _remove_stale(path, current) else "busy"
    if current.get("owner") == OWNER and current.get("pid") == os.getpid():
        return None
    return "busy"


def release(path: Path, owner: str = OWNER, *, force: bool = False) -> None:
    """Remove the lock when this process owns it.

    A pinned lock stays until its command ends. ``force`` removes it anyway,
    for the Enlarge hand-off, which takes it back afterwards.
    """
    if str(path) in _PINNED and not force:
        return
    current = read_lock(path)
    if current is not None and current.get("owner") == owner and current.get("pid") in (None, os.getpid()):
        try:
            path.unlink()
        except OSError:
            pass


@contextmanager
def gpu_job(job: str, eta_s: float = 600.0, path: Path | None = None, timeout: float = 0.0) -> Iterator[None]:
    """Hold the lock for one job. Re-entrant for this process."""
    target = path or lock_path()
    current = read_lock(target)
    if current and current.get("owner") == OWNER and current.get("pid") == os.getpid():
        yield
        return
    started = time.monotonic()
    while True:
        holder = try_acquire(job, eta_s, target)
        if holder is None:
            break
        if time.monotonic() - started > timeout:
            raise GpuLockTimeout("the GPU is busy")
        time.sleep(0.01)
    try:
        yield
    finally:
        release(target)


@contextmanager
def pinned(job: str, eta_s: float = 600.0, path: Path | None = None) -> Iterator[Path]:
    """Hold this device's lock for a whole command, from before any weights load.

    Inner ``try_acquire`` calls for the same file succeed and inner ``release``
    calls leave it held, so a second process on the same GPU cannot slip in
    between pictures.

    Raises:
        GpuLockTimeout: Another live process holds the lock.
    """
    target = path or lock_path()
    holder = try_acquire(job, eta_s, target)
    if holder is not None:
        raise GpuLockTimeout("the GPU is busy")
    _PINNED.add(str(target))
    try:
        yield target
    finally:
        _PINNED.discard(str(target))
        release(target)
