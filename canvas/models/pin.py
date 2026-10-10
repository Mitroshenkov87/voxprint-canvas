"""SHA-256 pins. An empty or mismatched pin refuses the download and the load."""
from __future__ import annotations

import hashlib


class PinError(Exception):
    """A weight file has no pin or the bytes do not match it."""

    def __init__(self, code: str) -> None:
        super().__init__("The model file could not be verified.")
        self.code = code


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
