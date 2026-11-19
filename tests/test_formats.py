"""WebP is the default. Canvas never writes PNG; a PNG name means WebP."""
from pathlib import Path

import numpy as np
import pytest

from canvas.core.formats import (
    DEFAULT_FORMAT,
    EXPORT_FORMAT,
    OutputFormat,
    UnsupportedFormat,
    parse_export_format,
    parse_format,
    save_image,
)
from canvas.core.exchange import handoff_image
from canvas.safety.gate import issue
from canvas.safety.image import RawImage
from tests.fakes import avif_file, rgb

# A 1920×1088 frame of 6px tiles plus grain. The old AVIF default (quality 75,
# 4:2:0) lands under this floor. Quality 80, 4:4:4, speed 6 stays above it.
_DETAIL_FLOOR = 700_000


def test_parse_and_default():
    assert DEFAULT_FORMAT is OutputFormat.WEBP
    assert parse_format("AVIF") is OutputFormat.AVIF
    assert parse_format("jxl") is OutputFormat.JXL
    assert parse_format("webp") is OutputFormat.WEBP
    assert parse_format("jpeg") is OutputFormat.JPEG
    assert parse_format("heic") is OutputFormat.HEIC
    assert parse_format(".png") is OutputFormat.WEBP
    assert parse_format("PNG") is OutputFormat.WEBP
    assert parse_format("image/png") is OutputFormat.WEBP
    with pytest.raises(UnsupportedFormat):
        parse_format(".bmp")


def test_png_is_never_written(tmp_path: Path):
    image = issue(rgb(8, 8))
    with pytest.raises(UnsupportedFormat):
        save_image(image, tmp_path / "picture.png", OutputFormat.PNG)
    assert not (tmp_path / "picture.png").exists()
    with pytest.raises(UnsupportedFormat):
        handoff_image(image, tmp_path / "frame.png", OutputFormat.PNG)


def test_a_png_path_becomes_webp_with_a_warning(tmp_path: Path):
    from canvas.core.formats import picture_path

    moved, warning = picture_path(tmp_path / "shot.PNG")
    assert moved == tmp_path / "shot.webp"
    assert warning is not None and "WebP" in warning
    assert picture_path(tmp_path / "shot.webp") == (tmp_path / "shot.webp", None)
    assert picture_path(tmp_path / "shot.jpg") == (tmp_path / "shot.jpg", None)
    assert picture_path(tmp_path / "shot") == (tmp_path / "shot.webp", None)


def test_save_requires_a_released_image(tmp_path: Path):
    image = issue(rgb())
    path = tmp_path / "picture.avif"
    save_image(image, path, OutputFormat.AVIF)
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


def test_forge_export_defaults_to_webp_and_still_accepts_avif(tmp_path: Path):
    image = issue(rgb(8, 8))
    assert EXPORT_FORMAT is OutputFormat.WEBP
    assert parse_export_format("png") is OutputFormat.WEBP
    assert parse_export_format("jpg") is OutputFormat.JPEG
    assert parse_export_format("JPEG") is OutputFormat.JPEG
    assert parse_export_format("webp") is OutputFormat.WEBP
    assert parse_export_format("avif") is OutputFormat.AVIF
    with pytest.raises(UnsupportedFormat):
        parse_export_format("jxl")
    webp = tmp_path / "frame.webp"
    handoff_image(image, webp)
    assert webp.read_bytes()[8:12] == b"WEBP"
    from PIL import Image

    with Image.open(webp) as reread:
        assert reread.convert("RGB").tobytes() == image.data  # series/Forge WebP is lossless
    avif = tmp_path / "frame.avif"
    handoff_image(image, avif, OutputFormat.AVIF)
    assert avif_file(avif.read_bytes())
    with pytest.raises(UnsupportedFormat):
        handoff_image(image, tmp_path / "frame.jxl", OutputFormat.JXL)


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


def test_gui_save_as_keeps_the_webp_container(tmp_path: Path):
    from canvas.ui.shell_screen import save_as_target

    source = tmp_path / "pic.webp"
    assert save_as_target(source, tmp_path / "mine.png") == tmp_path / "mine.webp"
    assert save_as_target(source, tmp_path / "mine") == tmp_path / "mine.webp"
    assert save_as_target(source, tmp_path / "mine.WEBP") == tmp_path / "mine.WEBP"


def test_gui_picture_and_export_settings_offer_no_png():
    from canvas.ui.shell_layout import _settings

    settings = _settings()
    for key in ("canvas.export", "canvas.format"):
        spec = settings.get(key)
        assert spec.default == "webp"
        assert "png" not in {choice.value for choice in spec.choices}
