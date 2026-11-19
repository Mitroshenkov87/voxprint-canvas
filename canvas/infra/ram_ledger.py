"""RAM each running Canvas job still expects to take, shared across processes.

Every job writes ``voxprint-ram-<pid>.json`` next to the per-card GPU lock files
(the temp folder, or ``VOXPRINT_RAM_LEDGER``). The file holds the peak RAM the
job planned for. A reader subtracts the job's current RAM use from that peak,
so free RAM is not counted twice. A file whose process is gone is ignored and
removed.

Planning is serialized across processes by :func:`plan_lock`, so two jobs that
start together on two cards cannot both read an empty ledger and both plan the
full tier.
"""
from __future__ import annotations

import atexit
import json
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from canvas.infra.gpu_lock import _pid_alive

_PREFIX = "voxprint-ram-"
_PLAN_LOCK = "voxprint-ram-plan.lock"
# Planning reads a few files and psutil. Longer than this means the holder is stuck.
_PLAN_LOCK_WAIT_S = 10.0
_registered = False


def folder() -> Path:
    """Where the ledger files live."""
    override = os.environ.get("VOXPRINT_RAM_LEDGER", "").strip()
    return Path(override) if override else Path(tempfile.gettempdir())


def _own() -> Path:
    return folder() / f"{_PREFIX}{os.getpid()}.json"


def publish(peak_bytes: int) -> None:
    """Record the peak RAM this process plans to reach. A write failure is ignored."""
    global _registered
    path = _own()
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps({"pid": os.getpid(), "peak": max(0, int(peak_bytes))}), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        return
    if not _registered:
        _registered = True
        atexit.register(withdraw)


def withdraw() -> None:
    """Remove this process's entry."""
    try:
        _own().unlink(missing_ok=True)
    except OSError:
        return


@contextmanager
def plan_lock(timeout: float = _PLAN_LOCK_WAIT_S) -> Iterator[bool]:
    """Hold the cross-process planning lock while a job reads the ledger and publishes its plan.

    Yields True when the lock is held. A lock left by a finished process is
    taken over. When the lock cannot be taken within ``timeout`` (or the folder
    is not writable) the body still runs and False is yielded.
    """
    path = folder() / _PLAN_LOCK
    held = False
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if _stale_lock(path):
                continue
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
            continue
        except OSError:
            break
        try:
            os.write(handle, str(os.getpid()).encode("ascii"))
        finally:
            os.close(handle)
        held = True
        break
    try:
        yield held
    finally:
        if held:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _stale_lock(path: Path) -> bool:
    """Remove the plan lock when its process is gone. True when it was removed."""
    try:
        text = path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return True
    except (OSError, UnicodeError):
        return False
    if not text.isdigit():
        # A lock being written right now has no pid yet.
        return False
    pid = int(text)
    if pid != os.getpid() and _pid_alive(pid):
        return False
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def others_pending() -> int:
    """RAM other live Canvas jobs still expect to take: their planned peak minus what they use now."""
    total = 0
    try:
        entries = list(folder().glob(f"{_PREFIX}*.json"))
    except OSError:
        return 0
    for path in entries:
        entry = _read(path)
        if entry is None:
            continue
        pid, peak = entry
        if pid == os.getpid():
            continue
        if not _pid_alive(pid):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        total += max(0, peak - process_bytes(pid))
    return total


def process_bytes(pid: int) -> int:
    """Resident RAM of one process. 0 when it cannot be read."""
    try:
        import psutil

        return max(0, int(psutil.Process(pid).memory_info().rss))
    except Exception:  # noqa: BLE001 - psutil raises its own types for a gone or denied process
        return 0


def _read(path: Path) -> tuple[int, int] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    pid = data.get("pid")
    peak = data.get("peak")
    if isinstance(pid, bool) or not isinstance(pid, int) or isinstance(peak, bool) or not isinstance(peak, int):
        return None
    return pid, peak
