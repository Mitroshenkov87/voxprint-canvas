"""SHA-256 pins. An empty or mismatched pin refuses the download and the load."""
from __future__ import annotations

import hashlib
from pathlib import Path

from canvas.app.i18n import text as catalog_text


class PinError(Exception):
    """A weight file has no pin or the bytes do not match it."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code

    def __str__(self) -> str:
        return catalog_text("error.verify")


def require_pin(value: str) -> str:
    """A 64-character hex SHA-256, or ``PinError``."""
    text = value.strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise PinError("unpinned")
    return text


def verify_bytes(data: bytes, expected: str) -> None:
    """Raise when ``data`` does not hash to ``expected``."""
    pin = require_pin(expected)
    got = hashlib.sha256(data).hexdigest()
    if got != pin:
        raise PinError("hash_mismatch")


def verify_file(path: Path, expected: str) -> None:
    """Raise when the file on disk does not hash to ``expected``.

    The file is read in chunks so a multi-gigabyte weight is not loaded whole.
    """
    pin = require_pin(expected)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    if digest.hexdigest() != pin:
        raise PinError("hash_mismatch")


def resolved_sha256(lfs_oid: str = "", sidecar: str = "") -> str:
    """SHA-256 for one published file.

    Hugging Face LFS objects use the LFS oid, which is the content SHA-256.
    Files stored directly in git use the ``.sha256`` sidecar. When both are
    present they have to be the same digest.
    """
    oid = lfs_oid.strip().lower()
    side = sidecar.strip().lower()
    if oid and side and oid != side:
        raise PinError("hash_mismatch")
    return require_pin(oid or side)
