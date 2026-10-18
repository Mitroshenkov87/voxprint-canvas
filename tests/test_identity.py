"""The release identity is read from BUILD.json."""
import json
from pathlib import Path

from canvas.app.identity import build_number, release_title, splash_line, version, version_label
from canvas.app.main import main

ROOT = Path(__file__).resolve().parents[1]


def test_version_label_and_release_title_come_from_build_json():
    assert version() == "0.1.0"
    assert version_label({}) == '0.1.0 RC · build 13 "Tavnit"'
    assert splash_line({}) == '0.1.0 RC build 13 "Tavnit"'
    assert release_title({}) == 'VOXPrint AI Canvas 0.1.0 RC · build 13 "Tavnit"'
    assert release_title({"GITHUB_REF_NAME": "v0.1.0-rc.14"}) == "VOXPrint AI Canvas v0.1.0-rc.14 build 13"
    assert "Tavnit" not in release_title({"GITHUB_REF_NAME": "v0.1.0-rc.14"})
    assert release_title({"GITHUB_REF_NAME": "v1.2.3-rc.4", "VOXPRINT_BUILD": "11"}) == (
        "VOXPrint AI Canvas v1.2.3-rc.4 build 11"
    )
    assert release_title({"GITHUB_REF_NAME": "v0.1.0"}) == 'VOXPrint AI Canvas 0.1.0 RC · build 13 "Tavnit"'
    template = (ROOT / "docs" / "RELEASE-NOTES-TEMPLATE.md").read_text(encoding="utf-8")
    assert "VOXPrint AI Canvas vX.Y.Z-rc.N build B" in template
    data = json.loads((ROOT / "BUILD.json").read_text(encoding="utf-8"))
    packaged = json.loads((ROOT / "canvas" / "app" / "BUILD.json").read_text(encoding="utf-8"))
    assert packaged == data
    assert data["version"] == "0.1.0"
    assert data["channel"] == "RC"
    assert data["build"] == 13
    assert data["codename"] == "Tavnit"
    assert '"canvas.app" = ["BUILD.json", "messages.json"]' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{data["version"]}"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    iss = (ROOT / "installer" / "windows" / "voxprint-canvas.iss").read_text(encoding="utf-8")
    assert f'#define MyAppVersion "{data["version"]}"' in iss


def test_build_number_comes_from_build_json(monkeypatch):
    monkeypatch.setattr(
        "canvas.app.identity.load_build",
        lambda: {"version": "0.1.0", "channel": "RC", "offset": 5, "codename": "Yotzer"},
    )
    assert build_number({"GITHUB_RUN_NUMBER": "3"}) == 0
    assert build_number({"VOXPRINT_BUILD": "9", "GITHUB_RUN_NUMBER": "3"}) == 9
    assert build_number({}) == 0
    assert version_label({}) == '0.1.0 RC · build 0 "Yotzer"'
    monkeypatch.setattr(
        "canvas.app.identity.load_build",
        lambda: {"version": "0.1.0", "channel": "RC", "build": 4, "offset": 5, "codename": "Yotzer"},
    )
    assert build_number({"GITHUB_RUN_NUMBER": "99"}) == 4


def test_version_flag_prints_the_release_identity(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == version_label()
