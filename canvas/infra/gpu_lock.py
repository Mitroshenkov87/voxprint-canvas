"""Shared GPU lock (``%TEMP%/voxprint-gpu.lock``).

Adapted from Voxprint AI Movie Dubber ``dubber/infra/gpu_lock.py``
(Apache-2.0, Copyright Aleksandr Mitroshenkov).

The file is JSON ``{owner, pid, job, started, eta}`` with ISO-8601 UTC times.
It is stale when the recorded pid is gone or when ``eta`` is more than two hours
in the past.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from ctypes import wintypes
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

OWNER = "canvas"
LOCK_NAME = "voxprint-gpu.lock"
STALE_AFTER = timedelta(hours=2)
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87


class GpuLockTimeout(TimeoutError):
    """Another suite app holds the GPU lock."""


def lock_path() -> Path:
    """The suite lock file. ``VOXPRINT_GPU_LOCK`` overrides the temp path."""
    override = os.environ.get("VOXPRINT_GPU_LOCK", "").strip()
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / LOCK_NAME


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


def try_acquire(job: str, eta_s: float, path: Path, owner: str = OWNER) -> dict[str, Any] | None:
    """Create the lock. Return None on success, otherwise the holder's data."""
    current = read_lock(path)
    if current is not None:
        mine = current.get("owner") == owner and current.get("pid") == os.getpid()
        if not mine and not is_stale(current):
            return current
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except PermissionError:
            return current
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


def release(path: Path, owner: str = OWNER) -> None:
    """Remove the lock when this process owns it."""
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
