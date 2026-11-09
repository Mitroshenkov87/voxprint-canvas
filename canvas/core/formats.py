"""Output containers. WebP is the picture default and the Forge export default.

Pillow 11.3 and newer writes AVIF, PNG and WebP. Pillow itself is HPND. Its
wheels include libavif (BSD-2-Clause) for AVIF. JPEG XL goes through imagecodecs
(BSD-3-Clause). pillow-jxl-plugin is not used. Forge opens PNG, JPEG and WebP.
AVIF stays available as an export option, and PNG stays an option too.

PNG, WebP, AVIF and JPEG are written by the shared ``voxprint_common.image``
writer: it encodes in memory, publishes the bytes atomically, and records
metadata as XMP (an iTXt chunk keyed ``XML:com.adobe.xmp`` on PNG; an XMP
packet plus a small EXIF copy on WebP, AVIF and JPEG). Metadata is off unless
the caller passes a non-empty mapping. JPEG XL goes through imagecodecs, which
takes no metadata, so JXL pictures are written without it; HEIF pictures carry
none either.
"""
from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, assert_never

from voxprint_common.image import read_metadata as _common_read_metadata
from voxprint_common.image import save_image as _common_save_image

from canvas.safety.image import SafeImage


class OutputFormat(Enum):
    """Containers the app can write."""

    AVIF = "avif"
    JXL = "jxl"
    WEBP = "webp"
    JPEG = "jpeg"  # legacy option
    HEIC = "heic"  # legacy option, system codec only when one is present
    PNG = "png"


DEFAULT_FORMAT = OutputFormat.WEBP
# Photoshoot series and the Forge hand-off. Forge opens PNG, JPEG and WebP.
EXPORT_FORMAT = OutputFormat.WEBP
FORGE_EXPORTS = frozenset({
    OutputFormat.PNG, OutputFormat.JPEG, OutputFormat.WEBP, OutputFormat.AVIF,
})

# Canvas keeps its historical AVIF and JPEG qualities. WebP quality comes from
# the shared writer: lossless for masters, its lossy default for pictures.
_AVIF_QUALITY = 80
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

# Containers the shared writer saves, and the suffixes it reads metadata back.
_COMMON_FORMATS = {
    OutputFormat.PNG: "png",
    OutputFormat.WEBP: "webp",
    OutputFormat.AVIF: "avif",
    OutputFormat.JPEG: "jpeg",
}
_COMMON_SUFFIXES = frozenset({".png", ".webp", ".avif", ".jpg", ".jpeg"})


class UnsupportedFormat(ValueError):
    """The caller asked for a container this app does not write."""


def parse_format(name: str) -> OutputFormat:
    """Resolve a user-facing picture name. WebP is the default; PNG is written lossless."""
    key = name.strip().lower()
    if key == "image/png":
        return OutputFormat.PNG
    for fmt in OutputFormat:
        if key in {fmt.value, _EXTENSION[fmt], _EXTENSION[fmt].lstrip(".")}:
            return fmt
    raise UnsupportedFormat("Choose AVIF, JPEG XL, WebP, JPEG or PNG")


def parse_export_format(name: str) -> OutputFormat:
    """Resolve a Forge export name. WebP is the default when the caller omits it.

    Forge opens PNG, JPEG and WebP. PNG stays an option, and AVIF too.
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


def _setting(fmt: OutputFormat) -> dict[str, Any]:
    """The shared save format for one Canvas container.

    WebP and PNG leave the quality unset so the save purpose decides (lossless
    for masters, the shared lossy default for pictures); AVIF and JPEG keep the
    historical Canvas qualities.
    """
    setting: dict[str, Any] = {"format": _COMMON_FORMATS[fmt]}
    if fmt is OutputFormat.AVIF:
        setting["quality"] = _AVIF_QUALITY
    elif fmt is OutputFormat.JPEG:
        setting["quality"] = _JPEG_QUALITY
    return setting


def _decimal_int(value: Any) -> Any:
    """``value`` as an int when it is a decimal string, otherwise unchanged."""
    if isinstance(value, str):
        text = value.strip()
        digits = text[1:] if text[:1] in ("+", "-") else text
        if digits and digits.isdecimal():
            return int(text)
    return value


def _common_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Map Canvas picture metadata onto the shared schema. None when empty.

    The historical flat mapping records every field as a string and names one
    LoRA ``lora_name``/``lora_sha256``/``lora_strength``; the shared schema
    types ``seed`` and ``steps`` as integers and lists every LoRA as
    ``{"name", "strength", "sha256", "base"}``. Both shapes are accepted so
    existing callers keep working; LoRA fields that were not provided are not
    invented. Fields the shared schema does not know are dropped.
    """
    if not metadata:
        return None
    document: dict[str, Any] = {}
    prompt = metadata.get("prompt")
    if prompt is not None:
        document["prompt"] = str(prompt)
    seed = _decimal_int(metadata.get("seed"))
    if seed is not None:
        document["seed"] = seed
    steps = _decimal_int(metadata.get("steps"))
    if steps is not None:
        document["steps"] = steps
    base = metadata.get("base", metadata.get("base_model"))
    if base is not None:
        document["base"] = str(base)
    strength = metadata.get("lora_strength")
    if strength is not None:
        document["lora_strength"] = strength
    loras = metadata.get("loras")
    if loras:
        document["loras"] = loras
        return document
    name = metadata.get("lora_name")
    digest = metadata.get("lora_sha256")
    lora: dict[str, Any] = {}
    if name is not None:
        lora["name"] = str(name)
    if strength is not None:
        lora["strength"] = strength
    if digest is not None:
        lora["sha256"] = str(digest)
    if lora:
        document["loras"] = [lora]
    return document


def _canvas_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize shared metadata back to Canvas's flat string mapping.

    A picture saved without metadata (every shared field empty) reads back as
    an empty mapping. One LoRA reads back under the historical ``lora_name``,
    ``lora_strength`` and ``lora_sha256`` keys; several read back as the shared
    ``loras`` list.
    """
    fields: dict[str, Any] = {}
    if metadata.get("prompt") is not None:
        fields["prompt"] = str(metadata["prompt"])
    if metadata.get("seed") is not None:
        fields["seed"] = str(metadata["seed"])
    if metadata.get("steps") is not None:
        fields["steps"] = str(metadata["steps"])
    base = metadata.get("base_model")
    if base is None:
        base = metadata.get("base")
    if base is not None:
        fields["base_model"] = str(base)
    if metadata.get("lora_strength") is not None:
        fields["lora_strength"] = str(metadata["lora_strength"])
    loras = metadata.get("loras") or []
    if len(loras) == 1 and isinstance(loras[0], Mapping):
        lora = loras[0]
        if lora.get("name") is not None:
            fields["lora_name"] = str(lora["name"])
        if lora.get("strength") is not None:
            fields["lora_strength"] = str(lora["strength"])
        if lora.get("sha256") is not None:
            fields["lora_sha256"] = str(lora["sha256"])
        if lora.get("base") is not None:
            fields["lora_base"] = str(lora["base"])
    elif loras:
        fields["loras"] = [
            {key: (None if value is None else str(value)) for key, value in lora.items()}
            for lora in loras
            if isinstance(lora, Mapping)
        ]
    return fields


def save_image(image: SafeImage, path: Path, fmt: OutputFormat = DEFAULT_FORMAT, *,
               metadata: Mapping[str, Any] | None = None, lossless: bool = False,
               purpose: str | None = None) -> None:
    """Write one released image. Any other object is refused.

    PNG, WebP, AVIF and JPEG go through the shared voxprint-common writer.
    ``metadata`` is written only when it is a non-empty mapping, in the
    historical flat string mapping or the shared schema (see
    :func:`_common_metadata`). ``purpose`` picks the WebP default: ``"master"``
    — also the legacy ``lossless=True`` used by series and Forge hand-offs —
    is lossless WebP; ``"picture"`` (the default) is lossy at the shared
    default quality. An explicit ``purpose`` wins over ``lossless``. JPEG XL
    goes through imagecodecs and HEIF through a system codec; both carry no
    metadata. A path whose extension does not match the format is refused
    either way.
    """
    if type(image) is not SafeImage:
        raise TypeError("only a released image can be saved")
    # Imported here so parsing a format name stays free of the image stack.
    from PIL import Image

    if path.suffix.lower() != extension(fmt):
        raise UnsupportedFormat("the file extension does not match the format")
    picture = Image.frombytes("RGB", (image.width, image.height), image.data)
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
    if purpose is None:
        purpose = "master" if lossless else "picture"
    elif purpose not in ("master", "picture"):
        raise ValueError("purpose must be 'master' or 'picture'")
    try:
        _common_save_image(
            picture,
            path,
            _setting(fmt),
            metadata=_common_metadata(metadata),
            write_metadata=bool(metadata),
            purpose=purpose,
        )
    except RuntimeError as exc:
        # The shared writer raises RuntimeError when the Pillow build lacks the
        # codec; report it like a failed encode so encoder probes (selftest)
        # mark the format missing instead of crashing.
        raise OSError(str(exc)) from exc


def read_metadata(path: Path) -> dict[str, Any]:
    """Read the fields ``save_image`` recorded for one picture.

    WebP, AVIF, JPEG and PNG go through the shared reader and are normalized
    back to Canvas's flat string mapping: a picture saved without metadata
    reads back as an empty mapping, and one LoRA reads back as
    ``lora_name``/``lora_strength``/``lora_sha256`` (see
    :func:`_canvas_metadata`). JPEG XL and HEIF carry no metadata and read
    back empty.
    """
    if path.suffix.lower() not in _COMMON_SUFFIXES:
        return {}
    return _canvas_metadata(_common_read_metadata(path))
