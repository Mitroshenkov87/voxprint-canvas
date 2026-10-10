"""Pixel values that have passed or not yet passed the content checks.

``RawImage`` is unreviewed. ``ScreenedInput`` is the only value a generator may
accept. ``SafeImage`` is the only value the UI may show or save. Both released
types are constructed only by ``canvas.safety.gate``.
"""
from __future__ import annotations

import inspect

_OUTPUT_TOKEN = object()
_INPUT_TOKEN = object()


def _issued_by(expected: str) -> bool:
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        return False
    caller = frame.f_back.f_back
    module = inspect.getmodule(caller)
    if module is None or module.__name__ != "canvas.safety.gate":
        return False
    return caller.f_code.co_name == expected


def _wipe(buf: bytearray) -> None:
    for index in range(len(buf)):
        buf[index] = 0
    buf.clear()


class RawImage:
    """Unreviewed RGB pixels. Wipe discards them in place."""

    __slots__ = ("width", "height", "mode", "_buf")

    def __init__(self, width: int, height: int, data: bytes, mode: str = "RGB") -> None:
        if width < 1 or height < 1:
            raise ValueError("image size must be positive")
        if mode != "RGB":
            raise ValueError("only RGB images are accepted")
        if len(data) != width * height * 3:
            raise ValueError("RGB byte count does not match the image size")
        self.width = width
        self.height = height
        self.mode = mode
        self._buf = bytearray(data)

    def to_bytes(self) -> bytes:
        """A copy of the pixel bytes."""
        return bytes(self._buf)

    def wipe(self) -> None:
        """Zero the pixel buffer."""
        _wipe(self._buf)
        self.width = 0
        self.height = 0


class ScreenedFrame:
    """One input image the checks have already accepted."""

    __slots__ = ("role", "width", "height", "_buf")

    def __init__(self, role: str, width: int, height: int, data: bytes) -> None:
        self.role = role
        self.width = width
        self.height = height
        self._buf = bytearray(data)

    def to_bytes(self) -> bytes:
        """A copy of the pixel bytes."""
        return bytes(self._buf)

    def wipe(self) -> None:
        """Zero the pixel buffer."""
        _wipe(self._buf)
        self.width = 0
        self.height = 0


class ScreenedInput:
    """A prompt plus input images the generator is allowed to see."""

    __slots__ = ("prompt", "mode", "frames")

    def __init__(self, token: object, prompt: str, mode: str, frames: tuple[ScreenedFrame, ...]) -> None:
        if token is not _INPUT_TOKEN or not _issued_by("issue_input"):
            raise RuntimeError("ScreenedInput is issued only by the content checks")
        self.prompt = prompt
        self.mode = mode
        self.frames = frames

    def roles(self) -> tuple[str, ...]:
        """Role name of each input image."""
        return tuple(frame.role for frame in self.frames)

    def wipe(self) -> None:
        """Clear the prompt and the input images."""
        self.prompt = ""
        self.mode = ""
        for frame in self.frames:
            frame.wipe()


class SafeImage:
    """An output image the UI may show and the encoder may save."""

    __slots__ = ("width", "height", "mode", "data")

    def __init__(self, token: object, width: int, height: int, mode: str, data: bytes) -> None:
        if token is not _OUTPUT_TOKEN or not _issued_by("issue"):
            raise RuntimeError("SafeImage is issued only by the content checks")
        self.width = width
        self.height = height
        self.mode = mode
        self.data = data
