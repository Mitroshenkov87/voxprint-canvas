"""Output containers. WebP is the picture default and the Forge export default.

Pillow 11.3 and newer writes AVIF, PNG and WebP. Pillow itself is HPND. Its
wheels include libavif (BSD-2-Clause) for AVIF. JPEG XL goes through imagecodecs
(BSD-3-Clause). pillow-jxl-plugin is not used. Forge opens PNG, JPEG and WebP.
AVIF stays available as an export option, and PNG stays an option too.

Every picture can carry text fields describing how it was made (seed, steps,
the LoRA, the prompt). PNG stores them as text chunks. WebP, AVIF and JPEG
store the same fields in an XMP packet and in EXIF: ImageDescription holds the
prompt, Software names the app and UserComment holds the whole record as JSON.
JPEG XL goes through imagecodecs, which takes no metadata, so JXL pictures are
written without it; HEIF pictures carry none either.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import assert_never
from xml.etree import ElementTree
from xml.sax.saxutils import escape

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

# Metadata: one rdf:Description with namespaced attributes per field.
_METADATA_NS = "https://voxprint.app/ns/picture/1.0/"
_RDF_DESCRIPTION = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"
_XML_EXTRA = {'"': "&quot;"}

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


def _png_text(metadata: Mapping[str, str] | None):  # -> PngInfo | None
    """PNG text chunks for one picture. None when nothing is recorded.

    Values are iTXt (UTF-8) chunks so a prompt in any language round-trips.
    """
    if not metadata:
        return None
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    for key, value in metadata.items():
        info.add_itxt(str(key), str(value))
    return info


def _xmp_packet(metadata: Mapping[str, str]) -> bytes:
    """One XMP packet recording every field as a voxprint attribute.

    Values are XML-escaped and the packet is UTF-8, so any prompt round-trips.
    """
    attributes = "".join(
        f' voxprint:{key}="{escape(str(value), _XML_EXTRA)}"'
        for key, value in metadata.items()
    )
    body = (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="VOXPrint Canvas">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        f'  <rdf:Description rdf:about="" xmlns:voxprint="{_METADATA_NS}"{attributes}/>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    )
    return body.encode("utf-8")


def _exif_block(metadata: Mapping[str, str]):  # -> Image.Exif | None
    """EXIF block for one picture. None when nothing is recorded.

    ImageDescription (0x010E) holds the prompt and Software (0x0131) names
    the app. UserComment (0x9286) holds the whole record as JSON, encoded
    with the EXIF ``UNICODE`` prefix.
    """
    if not metadata:
        return None
    from PIL import Image

    record = dict(metadata)
    block = Image.Exif()
    block[0x010E] = str(record.get("prompt", ""))
    block[0x0131] = "VOXPrint Canvas"
    block[0x9286] = b"UNICODE\0" + json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-16")
    return block


def _xmp_exif(metadata: Mapping[str, str] | None):  # -> dict[str, object]
    """Save kwargs carrying metadata. Empty when nothing is recorded.

    The single metadata writer for non-PNG containers. XMP is the primary
    record (read_metadata reads it); EXIF is a convenience copy. Swap this
    for the shared voxprint-common helper when it ships.
    """
    if not metadata:
        return {}
    return {"xmp": _xmp_packet(metadata), "exif": _exif_block(metadata)}


def save_image(image: SafeImage, path: Path, fmt: OutputFormat = DEFAULT_FORMAT, *,
               metadata: Mapping[str, str] | None = None, lossless: bool = False) -> None:
    """Write one released image. Any other object is refused.

    PNG is lossless and carries ``metadata`` as text chunks. WebP, AVIF and
    JPEG carry the same fields in an XMP packet plus an EXIF block. JPEG XL
    goes through imagecodecs, which takes no metadata, so JXL pictures are
    written without it; HEIF pictures carry none either. A path whose
    extension does not match the format is refused either way. ``lossless``
    applies to WebP only; series and Forge hand-offs set it.
    """
    if type(image) is not SafeImage:
        raise TypeError("only a released image can be saved")
    # Imported here so parsing a format name stays free of the image stack.
    from PIL import Image

    if path.suffix.lower() != extension(fmt):
        raise UnsupportedFormat("the file extension does not match the format")
    picture = Image.frombytes("RGB", (image.width, image.height), image.data)
    if fmt is OutputFormat.PNG:
        picture.save(path, format="PNG", pnginfo=_png_text(metadata))
        return
    extra = _xmp_exif(metadata)
    if fmt is OutputFormat.AVIF:
        picture.save(
            path,
            format="AVIF",
            quality=_AVIF_QUALITY,
            subsampling=_AVIF_SUBSAMPLING,
            speed=_AVIF_SPEED,
            **extra,
        )
        return
    if fmt is OutputFormat.WEBP:
        if lossless:
            picture.save(path, format="WEBP", lossless=True, quality=100, method=4, **extra)
        else:
            picture.save(path, format="WEBP", quality=_WEBP_QUALITY, **extra)
        return
    if fmt is OutputFormat.JPEG:
        picture.save(path, format="JPEG", quality=_JPEG_QUALITY, **extra)
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


def read_metadata(path: Path) -> dict[str, str]:
    """Read the fields ``save_image`` recorded for one picture.

    PNG text chunks are read back as written. WebP, AVIF and JPEG carry an
    XMP packet instead, which is parsed here. Containers written without
    metadata (and pictures saved without any) read back as an empty mapping.
    """
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        packet = image.info.get("xmp")
        if packet is None:
            text = getattr(image, "text", None)
            if not text:
                return {}
            return {str(key): str(value) for key, value in text.items()}
    if isinstance(packet, str):
        packet = packet.encode("utf-8")
    prefix = f"{{{_METADATA_NS}}}"
    fields: dict[str, str] = {}
    for element in ElementTree.fromstring(packet).iter():
        if element.tag != _RDF_DESCRIPTION:
            continue
        for attr, value in element.attrib.items():
            if attr.startswith(prefix):
                fields[attr[len(prefix):]] = value
    return fields
