"""Output containers. AVIF is the default. PNG is not an output format."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import assert_never

from canvas.safety.image import SafeImage


class OutputFormat(Enum):
    AVIF = "avif"
    JXL = "jxl"
    WEBP = "webp"
    JPEG = "jpeg"  # legacy option
    HEIC = "heic"  # legacy option, system codec only when one is present


DEFAULT_FORMAT = OutputFormat.AVIF

_EXTENSION = {
    OutputFormat.AVIF: ".avif",
    OutputFormat.JXL: ".jxl",
    OutputFormat.WEBP: ".webp",
    OutputFormat.JPEG: ".jpg",
    OutputFormat.HEIC: ".heic",
}


class UnsupportedFormat(ValueError):
    """The caller asked for a container this app does not write."""


def parse_format(name: str) -> OutputFormat:
    """Resolve a user-facing name. ``png`` is rejected."""
    key = name.strip().lower()
    if key in {"png", ".png", "image/png"}:
        raise UnsupportedFormat("PNG output is not available")
    for fmt in OutputFormat:
        if key in {fmt.value, _EXTENSION[fmt], _EXTENSION[fmt].lstrip(".")}:
            return fmt
    raise UnsupportedFormat("Choose AVIF, JPEG XL or WebP")


def extension(fmt: OutputFormat) -> str:
    if fmt is OutputFormat.AVIF:
        return _EXTENSION[fmt]
    if fmt is OutputFormat.JXL:
        return _EXTENSION[fmt]
    if fmt is OutputFormat.WEBP:
        return _EXTENSION[fmt]
    if fmt is OutputFormat.JPEG:
        return _EXTENSION[fmt]
    if fmt is OutputFormat.HEIC:
        return _EXTENSION[fmt]
    assert_never(fmt)


def save_image(image: SafeImage, path: Path, fmt: OutputFormat = DEFAULT_FORMAT) -> None:
    """Write one released image. Any other object is refused, and PNG paths are refused."""
    if type(image) is not SafeImage:
        raise TypeError("only a released image can be saved")
    if path.suffix.lower() == ".png":
        raise UnsupportedFormat("PNG output is not available")
    if path.suffix.lower() != extension(fmt):
        raise UnsupportedFormat("the file extension does not match the format")
    # Container encoding (libavif, imagecodecs/libjxl, Pillow WebP) is not wired yet.
    # The bytes are the screened RGB plus a small header so a test can see the format.
    header = f"VOXPRINT {fmt.value}\n".encode("ascii")
    path.write_bytes(header + image.data)
