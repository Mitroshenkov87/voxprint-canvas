"""``--out`` is checked, and its folder created, before any model loads."""
import json
import os
import sys
from pathlib import Path

import pytest

from canvas.app.main import main
from canvas.app.platform import GpuCheck, OsCheck
from canvas.infra.gpu import GpuInfo
from canvas.infra.paths import OutputFolderError, prepare_output_folder
from canvas.models.modes import Tier


def test_missing_parents_are_created(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    assert prepare_output_folder(target) == target
    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_a_file_in_the_way_fails_fast(tmp_path: Path) -> None:
    blocker = tmp_path / "taken"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(OutputFolderError, match="is a file"):
        prepare_output_folder(blocker)
    with pytest.raises(OutputFolderError, match="cannot be created"):
        prepare_output_folder(blocker / "below")


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions as a normal user")
def test_a_read_only_folder_fails_fast(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(OutputFolderError, match="not writable"):
            prepare_output_folder(locked)
    finally:
        locked.chmod(0o700)


def _ready_cli(monkeypatch, tmp_path: Path, calls: list[str]) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr("canvas.app.main.probe_gpu", lambda: GpuInfo("RTX 4090", 23.99, 20, 8, 9, "fake"))
    monkeypatch.setattr("canvas.app.main.check_gpu", lambda info: GpuCheck(True, "", Tier.GB24))

    def install(*args: object, **kwargs: object) -> None:
        calls.append("install")
        raise OSError("stopped before the models")

    monkeypatch.setattr("canvas.app.main.install_supporting", install)


def test_generate_creates_the_output_folder_before_the_models(tmp_path: Path, monkeypatch, capsys) -> None:
    calls: list[str] = []
    _ready_cli(monkeypatch, tmp_path, calls)
    out = tmp_path / "new" / "deeper" / "picture.png"
    code = main(["generate", "--prompt", "a red boat", "--out", str(out), "--json"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["error"] == "stopped before the models"
    assert out.parent.is_dir()
    assert calls == ["install"]


def test_generate_with_a_bad_output_folder_stops_before_the_models(tmp_path: Path, monkeypatch, capsys) -> None:
    calls: list[str] = []
    _ready_cli(monkeypatch, tmp_path, calls)
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    code = main(["generate", "--prompt", "a red boat", "--out", str(blocker / "picture.avif"), "--json"])
    assert code == 2
    body = json.loads(capsys.readouterr().out)
    assert body["ok"] is False and body["code"] == "output_folder"
    assert "is a file" in body["error"]
    assert calls == []


def test_photoshoot_with_a_bad_output_folder_stops_before_the_models(tmp_path: Path, monkeypatch, capsys) -> None:
    calls: list[str] = []
    _ready_cli(monkeypatch, tmp_path, calls)
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    code = main(["photoshoot", "--prompt", "a red boat", "--out", str(blocker), "--json"])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["code"] == "output_folder"
    assert calls == []
