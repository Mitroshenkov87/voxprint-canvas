"""Metadata round-trips for the containers ``save_image`` writes."""
from __future__ import annotations

import os
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
from canvas.models.modes import Mode
from canvas.ui.shell_layout import PictureJob, metadata_enabled

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


def _job(save_metadata: bool = False) -> PictureJob:
    return PictureJob(
        "a red bicycle by a lake, watercolor",
        "",
        Mode.TEXT,
        "1:1",
        "native",
        OutputFormat.PNG,
        42,
        8,
        1.0,
        "euler",
        1,
        save_metadata=save_metadata,
    )


def test_default_picture_job_has_metadata_off() -> None:
    job = PictureJob(
        "a red bicycle by a lake, watercolor", "", Mode.TEXT, "1:1", "native",
        OutputFormat.PNG, 42, 8, 1.0, "euler", 1,
    )
    assert job.save_metadata is False


def test_job_with_metadata_off_saves_a_clean_picture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from canvas.worker_jobs import _picture_metadata

    job = _job()
    fields = _picture_metadata(job)
    assert fields == {}
    path = tmp_path / "picture.png"
    save_image(_released_image(monkeypatch), path, job.fmt, metadata=fields)
    assert read_metadata(path) == {}


def test_job_with_metadata_on_writes_the_recipe_into_the_picture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from canvas.worker_jobs import _picture_metadata

    job = _job(save_metadata=True)
    fields = _picture_metadata(job)
    assert fields["prompt"] == job.prompt
    assert fields["seed"] == "42"
    assert fields["steps"] == "8"
    path = tmp_path / "picture.png"
    save_image(_released_image(monkeypatch), path, job.fmt, metadata=fields)
    assert read_metadata(path) == fields


def test_metadata_enabled_env_wins_over_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    for raw in ("1", "true", "TRUE", "yes", "on", " On "):
        monkeypatch.setenv("VOXPRINT_CANVAS_SAVE_METADATA", raw)
        assert metadata_enabled(False) is True
    for raw in ("0", "false", "no", "off", "anything"):
        monkeypatch.setenv("VOXPRINT_CANVAS_SAVE_METADATA", raw)
        assert metadata_enabled(True) is False


def test_metadata_enabled_falls_back_to_the_setting_then_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXPRINT_CANVAS_SAVE_METADATA", raising=False)
    assert metadata_enabled(True) is True
    assert metadata_enabled(False) is False
    assert metadata_enabled(None) is False
    assert metadata_enabled("yes") is False


def test_save_metadata_setting_exists_and_defaults_off() -> None:
    from voxprint_common.ui.settings import SettingType

    from canvas.ui.shell_layout import _settings

    spec = _settings().get("canvas.save_metadata")
    assert spec is not None
    assert spec.type is SettingType.BOOL
    assert spec.default is False
    assert spec.expert is False


def test_save_metadata_flag_sets_the_env(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from canvas.app.main import main

    monkeypatch.delenv("VOXPRINT_CANVAS_SAVE_METADATA", raising=False)
    assert main(["--version", "--save-metadata"]) == 0
    assert os.environ["VOXPRINT_CANVAS_SAVE_METADATA"] == "1"
    capsys.readouterr()
