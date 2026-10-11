"""Suite install paths, Enlarge lookup order, and silent upgrade."""
from pathlib import Path

from canvas.app.info import APP_NAME
from canvas.infra.discovery import AppInstall, find_app
from canvas.infra.layout import (
    CLI_EXE,
    GUI_EXE,
    SUITE_NAME,
    enlarge_executable,
    linux_canvas_dir,
    migrate,
    windows_canvas_dir,
)


def test_display_name_and_install_paths():
    assert APP_NAME == "VOXPrint AI Canvas"
    assert windows_canvas_dir(Path("C:/Program Files")) == Path("C:/Program Files") / SUITE_NAME / "Canvas"
    assert linux_canvas_dir() == Path("/opt/voxprint-ai-suite/canvas")
    assert enlarge_executable("win32", Path("C:/Program Files")) == (
        Path("C:/Program Files") / SUITE_NAME / "Enlarge" / "voxprint-enlarge-cli.exe"
    )
    assert enlarge_executable("linux") == Path("/opt/voxprint-ai-suite/enlarge/voxprint-enlarge-cli")
    iss = Path("installer/windows/voxprint-canvas.iss").read_text(encoding="utf-8")
    assert 'MyAppName "VOXPrint AI Canvas"' in iss
    assert 'MySuiteName "VOXPrint AI Suite"' in iss
    assert r"DefaultDirName={autopf}\{#MySuiteName}\Canvas" in iss
    assert "DefaultGroupName={#MySuiteName}" in iss
    assert "UsePreviousAppDir=no" in iss
    assert GUI_EXE in iss
    assert CLI_EXE in iss
    assert "voxprint-canvas.exe" in iss
    readme = Path("README.md").read_text(encoding="utf-8")
    assert readme.startswith("# VOXPrint AI Canvas\n")
    assert "C:\\Program Files\\VOXPrint AI Suite\\Canvas\\" in readme
    assert "/opt/voxprint-ai-suite/canvas/" in readme
    assert "%LOCALAPPDATA%\\Voxprint" in readme
    assert "~/.local/share/voxprint" in readme
    notes = Path("docs/RELEASE-NOTES-TEMPLATE.md").read_text(encoding="utf-8")
    assert 'VOXPrint AI Canvas X.Y.Z RC · build N "Yotzer"' in notes


def test_suite_registry_beats_the_default_enlarge_path(tmp_path, monkeypatch):
    monkeypatch.delenv("CANVAS_ENLARGE_EXE", raising=False)
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "empty"))
    recorded = AppInstall(["/suite/registry/voxprint-enlarge-cli"], "1", "voxprint-enlarge")
    monkeypatch.setattr("canvas.infra.discovery._from_suite_registry", lambda _pascal, _kebab: recorded)
    default = tmp_path / "voxprint-enlarge-cli"
    default.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr("canvas.infra.discovery.enlarge_executable", lambda: default)
    found = find_app("Enlarge", "enlarge")
    assert found is recorded


def test_upgrade_keeps_user_data_and_removes_the_old_program(tmp_path):
    user = tmp_path / "Local" / "Voxprint"
    shared = user / "shared"
    shared.mkdir(parents=True)
    manifest = shared / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    old = user / "Canvas"
    tier = old / "tier.txt"
    tier.parent.mkdir()
    tier.write_text("24gb", encoding="utf-8")
    venv = old / "venv"
    venv.mkdir()
    (venv / "python.exe").write_text("old", encoding="utf-8")
    menu = tmp_path / "Start Menu" / "Voxprint"
    menu.mkdir(parents=True)
    (menu / "Voxprint AI Canvas.lnk").write_text("old", encoding="utf-8")
    (menu / "Uninstall Voxprint AI Canvas.lnk").write_text("old", encoding="utf-8")
    (menu / "VOXPrint AI Enlarge.lnk").write_text("keep", encoding="utf-8")
    new = tmp_path / "Program Files" / "VOXPrint AI Suite" / "Canvas"
    new.mkdir(parents=True)
    (new / "keep.txt").write_text("new", encoding="utf-8")
    migrate(
        new,
        registry=[{"InstallLocation": str(old), "Inno Setup: Icon Group": "Voxprint"}],
        user_data=user,
        shortcut_dirs=[menu],
        legacy_files=[],
        platform="win32",
    )
    assert not venv.exists()
    assert tier.read_text(encoding="utf-8") == "24gb"
    assert manifest.read_text(encoding="utf-8") == "{}\n"
    assert user.is_dir()
    assert shared.is_dir()
    assert not (menu / "Voxprint AI Canvas.lnk").exists()
    assert not (menu / "Uninstall Voxprint AI Canvas.lnk").exists()
    assert (menu / "VOXPrint AI Enlarge.lnk").read_text(encoding="utf-8") == "keep"
    assert (new / "keep.txt").read_text(encoding="utf-8") == "new"


def test_upgrade_removes_an_old_program_folder_outside_user_data(tmp_path):
    user = tmp_path / "Local" / "Voxprint"
    (user / "shared").mkdir(parents=True)
    (user / "shared" / "manifest.json").write_text("{}\n", encoding="utf-8")
    old = tmp_path / "Old Programs" / "Canvas"
    (old / "venv").mkdir(parents=True)
    new = tmp_path / "Program Files" / "VOXPrint AI Suite" / "Canvas"
    new.mkdir(parents=True)
    migrate(
        new,
        registry=[{"InstallLocation": str(old)}],
        user_data=user,
        shortcut_dirs=[],
        legacy_files=[],
        platform="win32",
    )
    assert not old.exists()
    assert (user / "shared" / "manifest.json").is_file()


def test_upgrade_never_deletes_the_data_root(tmp_path):
    user = tmp_path / "Voxprint"
    (user / "shared").mkdir(parents=True)
    (user / "shared" / "manifest.json").write_text("{}\n", encoding="utf-8")
    (user / "notes.txt").write_text("keep", encoding="utf-8")
    new = tmp_path / "Program Files" / "VOXPrint AI Suite" / "Canvas"
    new.mkdir(parents=True)
    migrate(
        new,
        registry=[{"InstallLocation": str(user)}, {"InstallLocation": str(user / "shared")}],
        user_data=user,
        shortcut_dirs=[],
        legacy_files=[],
        platform="win32",
    )
    assert (user / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert (user / "shared" / "manifest.json").is_file()


def test_linux_upgrade_drops_the_old_launcher_and_keeps_the_manifest(tmp_path):
    user = tmp_path / "share" / "voxprint"
    (user / "shared").mkdir(parents=True)
    (user / "shared" / "manifest.json").write_text("{}\n", encoding="utf-8")
    old = user / "canvas"
    (old / "venv").mkdir(parents=True)
    (old / "tier.txt").write_text("16gb", encoding="utf-8")
    home = tmp_path / "home"
    launcher = home / ".local" / "bin" / "voxprint-canvas"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(old / "venv" / "python")
    desktop = home / ".local" / "share" / "applications" / "voxprint-canvas.desktop"
    desktop.parent.mkdir(parents=True)
    desktop.write_text(f"Exec={old}/venv/bin/voxprint-canvas\n", encoding="utf-8")
    new = Path("/opt/voxprint-ai-suite/canvas")
    migrate(new, registry=[], user_data=user, shortcut_dirs=[], platform="linux", home=home)
    assert not (old / "venv").exists()
    assert (old / "tier.txt").read_text(encoding="utf-8") == "16gb"
    assert (user / "shared" / "manifest.json").is_file()
    assert not launcher.exists()
    assert not desktop.exists()
