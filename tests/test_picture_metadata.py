"""Metadata round-trips for the containers ``save_image`` writes."""
from __future__ import annotations

from pathlib import Path

import pytest

from canvas.safety.gate import issue
from tests.fakes import rgb
from canvas.core.formats import (
    DEFAULT_FORMAT,
    EXPORT_FORMAT,
    OutputFormat,
    extension,
    parse_export_format,
    read_metadata,
    save_image,
)

_METADATA = {
    "prompt": "a red bicycle by a lake, watercolor",
    "seed": "42",
    "steps": "8",
    "base_model": "z-image-turbo",
    "lora_name": "Kettle Style",
    "lora_strength": "0.8",
    "lora_sha256": "ab" * 32,
}

_CONTAINERS = (
    OutputFormat.PNG,
    OutputFormat.WEBP,
    OutputFormat.AVIF,
    OutputFormat.JPEG,
)


def _avif_available() -> bool:
    from PIL import features

    return features.check("avif")


def _require_container(fmt: OutputFormat) -> None:
    if fmt is OutputFormat.AVIF and not _avif_available():
        pytest.skip("Pillow was built without AVIF support")


def _released_image(monkeypatch: pytest.MonkeyPatch):
    return issue(rgb(8, 6))


def _save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fmt: OutputFormat,
    metadata: dict[str, str] | None,
) -> Path:
    path = tmp_path / f"picture{extension(fmt)}"
    save_image(_released_image(monkeypatch), path, fmt, metadata=metadata)
    return path


@pytest.mark.parametrize("fmt", _CONTAINERS)
def test_metadata_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fmt: OutputFormat
) -> None:
    _require_container(fmt)
    path = _save(tmp_path, monkeypatch, fmt, dict(_METADATA))
    assert read_metadata(path) == _METADATA


@pytest.mark.parametrize("fmt", _CONTAINERS)
def test_metadata_unicode_and_escaping_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fmt: OutputFormat
) -> None:
    _require_container(fmt)
    metadata = {"prompt": "кот на окне", "note": '<bikes> & "boats"'}
    path = _save(tmp_path, monkeypatch, fmt, metadata)
    assert read_metadata(path) == metadata


@pytest.mark.parametrize("fmt", _CONTAINERS)
def test_picture_without_metadata_reads_back_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fmt: OutputFormat
) -> None:
    _require_container(fmt)
    path = _save(tmp_path, monkeypatch, fmt, None)
    assert read_metadata(path) == {}


def test_picture_defaults_are_webp() -> None:
    assert DEFAULT_FORMAT is OutputFormat.WEBP
    assert EXPORT_FORMAT is OutputFormat.WEBP


def test_explicit_export_names_still_resolve() -> None:
    assert parse_export_format("png") is OutputFormat.PNG
    assert parse_export_format("jpg") is OutputFormat.JPEG
    assert parse_export_format("jpeg") is OutputFormat.JPEG
    assert parse_export_format("webp") is OutputFormat.WEBP
    assert parse_export_format("avif") is OutputFormat.AVIF
