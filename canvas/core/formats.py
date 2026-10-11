"""Output containers. AVIF is the default. PNG is not an output format.

Pillow 11.3 and newer writes AVIF and WebP. Pillow itself is HPND. Its wheels
include libavif (BSD-2-Clause) for AVIF. JPEG XL goes through imagecodecs
(BSD-3-Clause). pillow-jxl-plugin is not used.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import assert_never

import numpy as np
from PIL import Image
from imagecodecs import jpegxl_encode

from canvas.safety.image import SafeImage


class OutputFormat(Enum):
    """Containers the app can write."""

    AVIF = "avif"
    JXL = "jxl"
    WEBP = "webp"
    JPEG = "jpeg"  # legacy option
    HEIC = "heic"  # legacy option, system codec only when one is present


DEFAULT_FORMAT = OutputFormat.AVIF

# Visually lossless defaults. AVIF stays 4:4:4. JPEG XL distance 1.0 matches that size.
_AVIF_QUALITY = 80
_AVIF_SPEED = 6
_AVIF_SUBSAMPLING = "4:4:4"
_WEBP_QUALITY = 90
_JPEG_QUALITY = 95
_JXL_DISTANCE = 1.0

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
    """File suffix for one output format, including the dot."""
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
    picture = Image.frombytes("RGB", (image.width, image.height), image.data)
    if fmt is OutputFormat.AVIF:
        picture.save(
            path,
            format="AVIF",
            quality=_AVIF_QUALITY,
            subsampling=_AVIF_SUBSAMPLING,
            speed=_AVIF_SPEED,
        )
        return
    if fmt is OutputFormat.WEBP:
        picture.save(path, format="WEBP", quality=_WEBP_QUALITY)
        return
    if fmt is OutputFormat.JPEG:
        picture.save(path, format="JPEG", quality=_JPEG_QUALITY)
        return
    if fmt is OutputFormat.JXL:
        pixels = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, 3)
        path.write_bytes(jpegxl_encode(pixels, distance=_JXL_DISTANCE))
        return
    if fmt is OutputFormat.HEIC:
        try:
            picture.save(path, format="HEIF")
        except (KeyError, OSError) as exc:
            raise UnsupportedFormat("HEIC needs a system codec") from exc
        return
    assert_never(fmt)
