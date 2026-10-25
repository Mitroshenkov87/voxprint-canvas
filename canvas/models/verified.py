"""Remember a verified file hash so a later check can skip the read.

The shared manifest stores one record per path. The record is reused when the
path, size, and modification time are unchanged. ``--selftest --deep`` ignores
the record and hashes the file again.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from canvas.infra.manifest import ManifestError, read_extra, update_extra
from canvas.infra.paths import shared_dir
from canvas.models.pin import PinError, require_pin, verify_file

VERIFIED_KEY = "verified"


class VerifiedCache:
    """In-memory view of the manifest hash cache. Call ``flush`` to write it."""

    def __init__(self, records: dict[str, Any], deep: bool, shared: Path | None) -> None:
        self.records = records
        self.deep = deep
        self.shared = shared
        self.dirty = False

    @classmethod
    def open(cls, deep: bool) -> VerifiedCache:
        """Load the cache. A missing or unreadable manifest starts empty."""
        try:
            shared = shared_dir()
            raw = read_extra(shared, VERIFIED_KEY)
        except (ManifestError, OSError):
            return cls({}, deep, None)
        records = raw if isinstance(raw, dict) else {}
        return cls(dict(records), deep, shared)

    def matches(self, path: Path, expected: str) -> bool:
        """True when the file matches ``expected``. An unchanged file is not read again."""
        try:
            pin = require_pin(expected)
            stat = path.stat()
        except (PinError, OSError):
            return False
        key = path.resolve().as_posix()
        size = int(stat.st_size)
        mtime = int(stat.st_mtime_ns)
        if not self.deep:
            row = self.records.get(key)
            if (
                isinstance(row, dict)
                and row.get("size") == size
                and row.get("mtime_ns") == mtime
                and row.get("sha256") == pin
            ):
                return True
        try:
            verify_file(path, pin)
        except PinError:
            if key in self.records:
                self.records.pop(key, None)
                self.dirty = True
            return False
        self.records[key] = {"size": size, "mtime_ns": mtime, "sha256": pin}
        self.dirty = True
        return True

    def flush(self) -> None:
        """Write the cache back to the shared manifest when a check changed it."""
        if not self.dirty or self.shared is None:
            return
        try:
            update_extra(self.shared, VERIFIED_KEY, self.records)
        except (ManifestError, OSError):
            return
        self.dirty = False


_process_lock = threading.Lock()
_process: VerifiedCache | None = None


def process_cache() -> VerifiedCache:
    """One cache for this process, so a series hashes each weight file once."""
    global _process
    with _process_lock:
        if _process is None:
            _process = VerifiedCache.open(False)
        return _process


def flush_process_cache() -> None:
    """Write the process cache when a check changed it."""
    with _process_lock:
        if _process is not None:
            _process.flush()
