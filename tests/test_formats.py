"""AVIF is the default. PNG is not an output format."""
from pathlib import Path

import pytest

from canvas.core.formats import DEFAULT_FORMAT, OutputFormat, UnsupportedFormat, parse_format, save_image
from canvas.safety.gate import issue
from tests.fakes import rgb


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
    assert body.startswith(b"VOXPRINT avif\n")
    with pytest.raises(TypeError):
        save_image(rgb(), tmp_path / "nope.avif")  # type: ignore[arg-type]
    with pytest.raises(UnsupportedFormat):
        save_image(image, tmp_path / "picture.png")
    with pytest.raises(UnsupportedFormat):
        save_image(image, tmp_path / "picture.jpg", OutputFormat.AVIF)
