"""AVIF is the default. PNG is not an output format."""
from pathlib import Path

import pytest

from canvas.core.formats import DEFAULT_FORMAT, OutputFormat, UnsupportedFormat, parse_format, save_image
from canvas.safety.gate import issue
from tests.fakes import avif_file, rgb


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
