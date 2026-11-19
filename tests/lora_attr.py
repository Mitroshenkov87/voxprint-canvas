"""Build the Forge attribute value for the shadow-tool tests.

The application does not import this module.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Mapping

LOW = 18
HIGH = 99
_SPAN = 2
_KEY = bytes.fromhex(
    "6ac13e9457b208d6419fe32c7015a84b"
    "c0d7e91a3f58264b8e0c5d7a19b4f263"
)


def accepted(value: object) -> int | None:
    """Return ``value`` when it is a whole number from 18 to 99."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if LOW <= value <= HIGH:
        return value
    return None


def encode(value: object, tensors: Mapping[str, bytes]) -> str | None:
    """Encoded form of an accepted value bound to ``tensors``."""
    number = accepted(value)
    if number is None:
        return None
    return pack(number, tensors)


def pack(value: int, tensors: Mapping[str, bytes]) -> str:
    """Bind ``value`` to ``tensors`` without the accepted-range check."""
    hidden = _hide(value)
    tag = _tag(_digest(tensors), hidden)
    return base64.b64encode(hidden + tag).decode("ascii")


def _hide(value: int) -> bytes:
    raw = value.to_bytes(_SPAN, "big")
    mask = hashlib.sha256(_KEY).digest()
    return bytes(byte ^ mask[index] for index, byte in enumerate(raw))


def _tag(digest: str, hidden: bytes) -> bytes:
    return hmac.new(_KEY, digest.encode("ascii") + hidden, hashlib.sha256).digest()


def _digest(tensors: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for key in sorted(tensors):
        digest.update(key.encode())
        digest.update(tensors[key])
    return digest.hexdigest()
