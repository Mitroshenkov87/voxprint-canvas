"""AVIF is the default. PNG is not an output format."""
from pathlib import Path

import numpy as np
import pytest

from canvas.core.formats import DEFAULT_FORMAT, OutputFormat, UnsupportedFormat, parse_format, save_image
from canvas.safety.gate import issue
from canvas.safety.image import RawImage
from tests.fakes import avif_file, rgb

# A 1920×1088 frame of 6px tiles plus grain. The old AVIF default (quality 75,
# 4:2:0) lands under this floor. Quality 80, 4:4:4, speed 6 stays above it.
_DETAIL_FLOOR = 700_000


def test_parse_and_default():
    assert DEFAULT_FORMAT is OutputFormat.AVIF
    assert parse_format("AVIF") is OutputFormat.AVIF
    assert parse_format("jxl") is OutputFormat.JXL
    assert parse_format("webp") is OutputFormat.WEBP
    assert parse_format("jpeg") is OutputFormat.JPEG
    assert parse_format("heic") is OutputFormat.HEIC
    with pytest.raises(UnsupportedFormat):
        parse_format("png")
    with pytest.raises(UnsupportedFormat):
        parse_format("image/png")


def test_save_requires_a_released_image(tmp_path: Path):
    image = issue(rgb())
    path = tmp_path / "picture.avif"
    save_image(image, path)
    body = path.read_bytes()
    assert avif_file(body)
    with pytest.raises(TypeError):
        save_image(rgb(), tmp_path / "nope.avif")  # type: ignore[arg-type]
    with pytest.raises(UnsupportedFormat):
        save_image(image, tmp_path / "picture.png")
    with pytest.raises(UnsupportedFormat):
        save_image(image, tmp_path / "picture.jpg", OutputFormat.AVIF)


def _detail_frame() -> RawImage:
    width, height = 1920, 1088
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:height, 0:width]
    tile = 6
    base = np.stack(
        [
            ((xx // tile * 37 + yy // tile * 17) & 255).astype(np.uint8),
            ((xx // tile * 11 + yy // tile * 53) & 255).astype(np.uint8),
            ((xx // tile * 29 + yy // tile * 7) & 255).astype(np.uint8),
        ],
        axis=-1,
    )
    grain = rng.integers(0, 24, size=base.shape, dtype=np.uint8)
    pixels = np.clip(base.astype(np.int16) - 12 + grain, 0, 255).astype(np.uint8)
    return RawImage(width, height, pixels.tobytes())


def test_default_2k_encodes_stay_above_the_size_floor(tmp_path: Path):
    image = issue(_detail_frame())
    for fmt, name in (
        (OutputFormat.AVIF, "frame.avif"),
        (OutputFormat.WEBP, "frame.webp"),
        (OutputFormat.JXL, "frame.jxl"),
    ):
        path = tmp_path / name
        save_image(image, path, fmt)
        size = path.stat().st_size
        assert size > _DETAIL_FLOOR, f"{fmt.value} default encode is {size} bytes"


def test_webp_jpeg_xl_and_jpeg_are_real_files(tmp_path: Path):
    image = issue(rgb(8, 8))
    webp = tmp_path / "picture.webp"
    save_image(image, webp, OutputFormat.WEBP)
    raw = webp.read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"
    jpeg = tmp_path / "picture.jpg"
    save_image(image, jpeg, OutputFormat.JPEG)
    assert jpeg.read_bytes()[:2] == b"\xff\xd8"
    jxl = tmp_path / "picture.jxl"
    save_image(image, jxl, OutputFormat.JXL)
    body = jxl.read_bytes()
    assert body[:2] == b"\xff\x0a" or body[:12] == bytes.fromhex("0000000c4a584c20")
    heic = tmp_path / "picture.heic"
    try:
        save_image(image, heic, OutputFormat.HEIC)
    except UnsupportedFormat as exc:
        assert "system codec" in str(exc)
    else:
        assert heic.stat().st_size > 16
