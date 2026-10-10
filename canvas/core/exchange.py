"""RGB file passed to Enlarge until the container codecs are wired."""
from __future__ import annotations

from pathlib import Path

from canvas.safety.image import RawImage

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
