"""RAM each running Canvas job still expects to take, shared across processes.

Every job writes ``voxprint-ram-<pid>.json`` next to the per-card GPU lock files
(the temp folder, or ``VOXPRINT_RAM_LEDGER``). The file holds the peak RAM the
job planned for. A reader subtracts the job's current RAM use from that peak,
so free RAM is not counted twice. A file whose process is gone is ignored and
removed.
"""
from __future__ import annotations

import atexit
import json
import os
import tempfile
from pathlib import Path

from canvas.infra.gpu_lock import _pid_alive

_PREFIX = "voxprint-ram-"
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
