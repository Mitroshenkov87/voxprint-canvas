"""Pictures handed to Enlarge and to Forge.

The 4K handoff into Enlarge is a lossless PNG. The Forge hand-off defaults to
PNG and can be AVIF, JPEG or WebP. ``write_rgb`` remains for tests that pass a
raw buffer.
"""
from __future__ import annotations

from pathlib import Path

import imagecodecs
import numpy as np
from PIL import Image

from canvas.core.formats import EXPORT_FORMAT, FORGE_EXPORTS, OutputFormat, UnsupportedFormat, save_image
from canvas.safety.image import RawImage, SafeImage

_PREFIX = b"VOXPRINT rgb "


def write_rgb(path: Path, width: int, height: int, data: bytes) -> None:
    """Write one RGB image for Enlarge to read."""
    if len(data) != width * height * 3:
        raise ValueError("RGB byte count does not match the image size")
    path.write_bytes(_PREFIX + f"{width} {height}\n".encode("ascii") + data)


def read_rgb(path: Path) -> RawImage:
    """Read an RGB image written by ``write_rgb``."""
    raw = path.read_bytes()
    if not raw.startswith(_PREFIX):
        raise ValueError("the enlarged image could not be read")
    header, _, body = raw.partition(b"\n")
    parts = header.decode("ascii").split()
    if len(parts) != 4:
        raise ValueError("the enlarged image could not be read")
    width, height = int(parts[2]), int(parts[3])
    return RawImage(width, height, body)


def handoff_image(image: SafeImage, path: Path, fmt: OutputFormat = EXPORT_FORMAT) -> None:
    """Write one picture for Forge. PNG is the default. AVIF stays available.

    Forge opens PNG, JPEG and WebP. Any other container is refused.
    """
    if fmt not in FORGE_EXPORTS:
        raise UnsupportedFormat("Forge export accepts PNG, JPEG, WebP or AVIF")
    save_image(image, path, fmt)


def write_png(path: Path, width: int, height: int, data: bytes) -> None:
    """Write a lossless PNG. Six bytes per pixel is 16-bit RGB; three is 8-bit."""
    pixels = width * height
    if width < 1 or height < 1:
        raise ValueError("image size must be positive")
    if len(data) == pixels * 6:
        samples = np.frombuffer(data, dtype="<u2").reshape(height, width, 3)
        path.write_bytes(imagecodecs.png_encode(np.ascontiguousarray(samples), level=0))
        return
    if len(data) != pixels * 3:
        raise ValueError("RGB byte count does not match the image size")
    Image.frombytes("RGB", (width, height), data).save(path, format="PNG")


def read_picture(path: Path) -> RawImage:
    """Decode the picture Enlarge wrote. The gate reviews 8-bit RGB."""
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            body = rgb.tobytes()
    except (OSError, ValueError) as exc:
        raise ValueError("the enlarged image could not be read") from exc
    return RawImage(width, height, body)
