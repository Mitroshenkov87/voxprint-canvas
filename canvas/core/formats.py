"""Output containers. AVIF is the picture default. Forge export defaults to PNG.

Pillow 11.3 and newer writes AVIF, PNG and WebP. Pillow itself is HPND. Its
wheels include libavif (BSD-2-Clause) for AVIF. JPEG XL goes through imagecodecs
(BSD-3-Clause). pillow-jxl-plugin is not used. Forge opens PNG, JPEG and WebP.
AVIF stays available as an export option.

A PNG can carry text fields with how the picture was made (seed, steps, the
LoRA, the prompt). Other containers are written without them.
"""
from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import assert_never

from canvas.safety.image import SafeImage


class OutputFormat(Enum):
    """Containers the app can write."""

    AVIF = "avif"
    JXL = "jxl"
    WEBP = "webp"
    JPEG = "jpeg"  # legacy option
    HEIC = "heic"  # legacy option, system codec only when one is present
    PNG = "png"


DEFAULT_FORMAT = OutputFormat.AVIF
# Photoshoot series and the Forge hand-off. Forge opens PNG, JPEG and WebP.
EXPORT_FORMAT = OutputFormat.PNG
FORGE_EXPORTS = frozenset({
    OutputFormat.PNG, OutputFormat.JPEG, OutputFormat.WEBP, OutputFormat.AVIF,
})

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
    OutputFormat.PNG: ".png",
}
_EXPORT_NAMES = {
    "png": OutputFormat.PNG,
    "jpg": OutputFormat.JPEG,
    "jpeg": OutputFormat.JPEG,
    "webp": OutputFormat.WEBP,
    "avif": OutputFormat.AVIF,
}


class UnsupportedFormat(ValueError):
    """The caller asked for a container this app does not write."""


def parse_format(name: str) -> OutputFormat:
    """Resolve a user-facing picture name. PNG is written lossless."""
    key = name.strip().lower()
    if key == "image/png":
        return OutputFormat.PNG
    for fmt in OutputFormat:
        if key in {fmt.value, _EXTENSION[fmt], _EXTENSION[fmt].lstrip(".")}:
            return fmt
    raise UnsupportedFormat("Choose AVIF, JPEG XL, WebP, JPEG or PNG")


def parse_export_format(name: str) -> OutputFormat:
    """Resolve a Forge export name. PNG is the default when the caller omits it.

    Forge opens PNG, JPEG and WebP. AVIF stays available.
    """
    key = name.strip().lower().lstrip(".")
    if key in {"image/png"}:
        key = "png"
    if key in {"image/jpeg"}:
        key = "jpeg"
    if key in {"image/webp"}:
        key = "webp"
    found = _EXPORT_NAMES.get(key)
    if found is None:
        raise UnsupportedFormat("Choose PNG, JPEG, WebP or AVIF")
    return found


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
    if fmt is OutputFormat.PNG:
        return _EXTENSION[fmt]
    assert_never(fmt)


def _png_text(metadata: Mapping[str, str] | None):  # -> PngInfo | None
    """PNG text chunks for one picture. None when nothing is recorded.

    Values are written as UTF-8 bytes so a prompt in any language round-trips.
    """
    if not metadata:
        return None
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    for key, value in metadata.items():
        info.add_text(str(key), str(value).encode("utf-8"))
    return info


def save_image(image: SafeImage, path: Path, fmt: OutputFormat = DEFAULT_FORMAT, *,
               metadata: Mapping[str, str] | None = None) -> None:
    """Write one released image. Any other object is refused.

    PNG is lossless and carries ``metadata`` as text chunks; other containers
    are written without it. The picture default is still AVIF, and a path whose
    extension does not match the format is refused.
    """
    if type(image) is not SafeImage:
        raise TypeError("only a released image can be saved")
    # Imported here so parsing a format name stays free of the image stack.
    from PIL import Image

    if fmt is OutputFormat.PNG:
        if path.suffix.lower() != extension(fmt):
            raise UnsupportedFormat("the file extension does not match the format")
        picture = Image.frombytes("RGB", (image.width, image.height), image.data)
        picture.save(path, format="PNG", pnginfo=_png_text(metadata))
        return
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
        import numpy as np
        from imagecodecs import jpegxl_encode

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
