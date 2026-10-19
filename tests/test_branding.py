"""Installer and branding standard: one Start menu entry with the app icon."""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from canvas.app.branding import APP_USER_MODEL_ID, icon_path, set_app_user_model_id

ROOT = Path(__file__).resolve().parents[1]
ISS = (ROOT / "installer" / "windows" / "voxprint-canvas.iss").read_text(encoding="utf-8")


def _section(name: str) -> list[str]:
    match = re.search(rf"^\[{name}\]\n(.*?)(?=^\[|\Z)", ISS, re.MULTILINE | re.DOTALL)
    assert match is not None, name
    return [line for line in match.group(1).splitlines() if line.strip() and not line.startswith(";")]


def test_icon_has_every_frame_size() -> None:
    with Image.open(icon_path()) as icon:
        assert sorted(icon.info["sizes"]) == [(n, n) for n in (16, 24, 32, 48, 64, 128, 256)]
    with Image.open(ROOT / "assets" / "voxprint-canvas.png") as master:
        assert master.size == (1024, 1024)


def test_start_menu_has_one_entry_on_the_launcher() -> None:
    icons = _section("Icons")
    assert len(icons) == 1
    line = icons[0]
    assert 'Name: "{group}\\{#MyAppName}"' in line
    assert 'Filename: "{app}\\voxprint-canvas.exe"' in line
    assert 'IconFilename: "{app}\\assets\\voxprint-canvas.ico"' in line
    assert f'AppUserModelID: "{APP_USER_MODEL_ID}"' in line
    assert "venv\\Scripts" not in line


def test_old_shortcuts_are_removed_and_the_group_is_left_to_inno() -> None:
    removed = "\n".join(_section("InstallDelete"))
    assert "Uninstall {#MyAppName}.lnk" in removed
    assert "Guide (English).lnk" in removed and "(русский).lnk" in removed
    assert "[UninstallDelete]" not in ISS or "{group}" not in "\n".join(_section("UninstallDelete"))


def test_setup_and_uninstall_entry_use_the_icon() -> None:
    setup = "\n".join(_section("Setup"))
    assert "SetupIconFile=..\\..\\assets\\voxprint-canvas.ico" in setup
    assert "UninstallDisplayName={#MyAppName}" in setup
    assert "UninstallDisplayIcon={app}\\assets\\voxprint-canvas.ico" in setup
    assert '#define MyAppPublisher "Aleksandr Mitroshenkov"' in ISS
    files = "\n".join(_section("Files"))
    assert 'Source: "voxprint-canvas.exe"; DestDir: "{app}"' in files
    assert 'Source: "..\\..\\assets\\voxprint-canvas.ico"; DestDir: "{app}\\assets"' in files
    assert "/win32icon:" in ISS and "launcher.cs" in ISS


def test_launcher_runs_the_app_module() -> None:
    source = (ROOT / "installer" / "windows" / "launcher.cs").read_text(encoding="utf-8")
    assert 'Module = "canvas.app.main"' in source
    assert 'Title = "VOXPrint AI Canvas"' in source
    assert '"pythonw.exe"' in source and '"-I -m "' in source


def test_app_user_model_id_is_a_no_op_off_windows(monkeypatch) -> None:
    monkeypatch.setattr("canvas.app.branding.sys.platform", "linux")
    assert set_app_user_model_id() is False


def test_make_icon_redraws_the_committed_files(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_icon", ROOT / "tools" / "make_icon.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(tmp_path)
    for name in ("voxprint-canvas.png", "voxprint-canvas.ico"):
        assert (tmp_path / name).read_bytes() == (ROOT / "assets" / name).read_bytes()
