"""The installer adopts each legacy folder, then downloads only what is missing."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from canvas.models.install_fetch import fetch
from canvas.models.registry import load_catalog
from canvas.models.required import required_specs
from canvas.models.snapshots import weight_dir


def _turbos() -> list:
    catalog = load_catalog()
    return [spec for spec in required_specs(catalog) if spec.repo == "Tongyi-MAI/Z-Image-Turbo"]


def test_one_adopt_per_legacy_directory_then_a_download(tmp_path: Path, monkeypatch, capsys) -> None:
    turbos = _turbos()
    assert len(turbos) >= 2
    root = tmp_path / "models"
    legacy = weight_dir(root, turbos[0].repo)
    legacy.mkdir(parents=True)
    marker = legacy / "keep.txt"
    marker.write_text("stay", encoding="utf-8")
    calls: list[list[str]] = []

    def run(command: list[str], check: bool = False) -> SimpleNamespace:
        del check
        calls.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("canvas.models.install_fetch.subprocess.run", run)
    monkeypatch.setattr("canvas.models.install_fetch._binary", lambda: Path("/venv/voxprint-mdl"))
    monkeypatch.setattr(
        "canvas.models.install_fetch._planned_ids",
        lambda manifest, models: [spec.id for spec in turbos],
    )
    monkeypatch.setattr("canvas.models.install_fetch.view_complete", lambda spec, models: False)
    code = fetch(tmp_path / "mdl.json", root=root)
    assert code == 0
    adopt = [command for command in calls if "adopt" in command]
    download = [command for command in calls if "download" in command]
    assert len(adopt) == 1
    assert len(download) == 1
    assert calls.index(adopt[0]) < calls.index(download[0])
    assert adopt[0][1:6] == ["adopt", "--app", "canvas", "--from", os.fspath(legacy)]
    for spec in turbos:
        assert spec.id in download[0]
    assert marker.read_text(encoding="utf-8") == "stay"
    out = capsys.readouterr().out
    for spec in turbos:
        assert f"id={spec.id} action=adopted" in out


def test_a_failed_adopt_is_not_downloaded(tmp_path: Path, monkeypatch, capsys) -> None:
    turbos = _turbos()
    writer = next(spec for spec in required_specs(load_catalog()) if spec.id == "qwen3-4b-writer")
    root = tmp_path / "models"
    legacy = weight_dir(root, turbos[0].repo)
    legacy.mkdir(parents=True)
    calls: list[list[str]] = []

    def run(command: list[str], check: bool = False) -> SimpleNamespace:
        del check
        calls.append(list(command))
        code = 2 if "adopt" in command else 0
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr("canvas.models.install_fetch.subprocess.run", run)
    monkeypatch.setattr("canvas.models.install_fetch._binary", lambda: Path("/venv/voxprint-mdl"))
    monkeypatch.setattr(
        "canvas.models.install_fetch._planned_ids",
        lambda manifest, models: [spec.id for spec in turbos] + [writer.id],
    )
    monkeypatch.setattr("canvas.models.install_fetch.view_complete", lambda spec, models: False)
    code = fetch(tmp_path / "mdl.json", root=root)
    assert code == 0
    download = [command for command in calls if "download" in command]
    assert len(download) == 1
    assert writer.id in download[0]
    for spec in turbos:
        assert spec.id not in download[0]
    assert legacy.is_dir()
    out = capsys.readouterr().out
    for spec in turbos:
        assert f"id={spec.id} action=failed reason=exit_2" in out


def test_nothing_left_to_download_exits_zero(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], check: bool = False) -> SimpleNamespace:
        del check
        calls.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("canvas.models.install_fetch.subprocess.run", run)
    monkeypatch.setattr("canvas.models.install_fetch._binary", lambda: Path("/venv/voxprint-mdl"))
    monkeypatch.setattr(
        "canvas.models.install_fetch._planned_ids",
        lambda manifest, models: ["qwen3-4b-writer"],
    )
    monkeypatch.setattr("canvas.models.install_fetch.view_complete", lambda spec, models: True)
    code = fetch(tmp_path / "mdl.json", root=tmp_path / "models")
    assert code == 0
    assert calls == []


def test_a_missing_binary_does_not_download(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("canvas.models.install_fetch._binary", lambda: None)
    monkeypatch.setattr("canvas.models.install_fetch.subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("download")))
    monkeypatch.setattr(
        "canvas.models.install_fetch._planned_ids",
        lambda manifest, models: ["qwen3-4b-writer"],
    )
    monkeypatch.setattr("canvas.models.install_fetch.view_complete", lambda spec, models: False)
    code = fetch(tmp_path / "mdl.json", root=tmp_path / "empty-models")
    assert code == 0
    assert "id=qwen3-4b-writer action=failed reason=voxprint_mdl_missing" in capsys.readouterr().out
