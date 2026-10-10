"""Shared GPU lock (``%TEMP%/voxprint-gpu.lock``).

Adapted from Voxprint AI Movie Dubber ``dubber/infra/gpu_lock.py``
(Apache-2.0, Copyright Aleksandr Mitroshenkov).

The file is JSON ``{owner, pid, job, started, eta}`` with ISO-8601 UTC times.
It is stale when the recorded pid is gone or when ``eta`` is more than two hours
in the past.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

OWNER = "canvas"
LOCK_NAME = "voxprint-gpu.lock"
STALE_AFTER = timedelta(hours=2)


class GpuLockTimeout(TimeoutError):
    pass


def lock_path() -> Path:
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


def _pid_alive(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, (int, str)):
        return False
    try:
        value = int(pid)
    except ValueError:
        return False
    if value <= 0:
        return False
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
