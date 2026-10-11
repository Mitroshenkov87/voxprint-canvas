"""Install locations for VOXPrint AI Canvas.

The window, the Start menu, and the installer use the suite display name.
Package names, the repository slug, URLs, and per-user data paths stay
lower-case. An upgrade removes the previous program folder and its shortcuts.
It leaves the per-user data folder and the shared manifest in place.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from canvas.infra.paths import voxprint_home

if sys.platform == "win32":
    import winreg
else:
    winreg = None

SUITE_NAME = "VOXPrint AI Suite"
LINUX_PREFIX = Path("/opt/voxprint-ai-suite")
APP_ID = "{8F3C1A2E-6B44-4C1A-9E2D-7A0C5B1E4D33}"
UNINSTALL_SUBKEY = (
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall" + "\\" + APP_ID + "_is1"
)
OLD_START_MENU = "Voxprint"
OLD_SHORTCUTS = (
    "Voxprint AI Canvas.lnk",
    "Uninstall Voxprint AI Canvas.lnk",
)
GUI_EXE = "voxprint-canvas.exe"
CLI_EXE = "voxprint-canvas-cli.exe"
ENLARGE_EXE = "voxprint-enlarge-cli.exe"
ENLARGE_LINUX = "voxprint-enlarge-cli"
_APP_PAYLOAD = frozenset({
    "venv",
    "canvas",
    "tools",
    "scripts",
    "installer",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "NOTICE",
    "BUILD.json",
    "constraints.txt",
    "install.ps1",
    "install.sh",
    "voxprint-canvas.cmd",
    "uv.exe",
    "__pycache__",
})


@dataclass(frozen=True)
class PreviousInstall:
    """An older Canvas install recorded by Inno Setup."""

    install_dir: Path
    icon_group: str


@dataclass(frozen=True)
class Removal:
    """One filesystem change an upgrade may perform."""

    path: Path
    kind: str


def windows_canvas_dir(program_files: Path | None = None) -> Path:
    """``C:\\Program Files\\VOXPrint AI Suite\\Canvas``."""
    root = Path(r"C:\Program Files") if program_files is None else program_files
    return root / SUITE_NAME / "Canvas"


def linux_canvas_dir() -> Path:
    """``/opt/voxprint-ai-suite/canvas``."""
    return LINUX_PREFIX / "canvas"


def enlarge_executable(platform: str | None = None, program_files: Path | None = None) -> Path:
    """Default Enlarge CLI for 4K. The suite manifest and PATH are separate steps."""
    system = sys.platform if platform is None else platform
    if system == "win32":
        root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) if program_files is None else program_files
        return root / SUITE_NAME / "Enlarge" / ENLARGE_EXE
    return LINUX_PREFIX / "enlarge" / ENLARGE_LINUX


def previous_from_registry(values: Mapping[str, str]) -> PreviousInstall | None:
    """Read one Inno uninstall record. An empty install location is not an install."""
    location = str(values.get("InstallLocation", "")).strip().rstrip("\\/")
    if not location:
        return None
    group = str(values.get("Inno Setup: Icon Group", "")).strip() or OLD_START_MENU
    return PreviousInstall(Path(location), group)


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _protected(path: Path, user_data: Path) -> bool:
    """The data root, the shared folder, and anything inside shared stay."""
    resolved = path.resolve()
    home = user_data.resolve()
    shared = home / "shared"
    if resolved == home or resolved == shared.resolve() or _is_inside(resolved, shared):
        return True
    return False


def plan_migration(
    *,
    previous: Sequence[PreviousInstall],
    new_install: Path,
    user_data: Path,
    shortcut_dirs: Sequence[Path] = (),
    legacy_files: Sequence[Path] = (),
) -> tuple[Removal, ...]:
    """Removals for one upgrade.

    A previous program folder outside the data directory is removed.
    A previous folder inside the data directory loses the shipped app files
    and keeps ``tier.txt``. Shortcuts from the old Start-menu group are removed.
    The data root and ``shared/`` are never listed.
    """
    planned: list[Removal] = []
    new_resolved = new_install.resolve()
    for item in previous:
        old = item.install_dir
        try:
            old_resolved = old.resolve()
        except OSError:
            old_resolved = old
        if old_resolved == new_resolved or _protected(old, user_data):
            continue
        if _is_inside(old, user_data):
            for name in sorted(_APP_PAYLOAD):
                child = old / name
                if child.exists():
                    kind = "tree" if child.is_dir() else "file"
                    planned.append(Removal(child, kind))
            continue
        if old.exists():
            planned.append(Removal(old, "tree" if old.is_dir() else "file"))
    for folder in shortcut_dirs:
        if not folder.is_dir() or _protected(folder, user_data):
            continue
        for name in OLD_SHORTCUTS:
            link = folder / name
            if link.is_file():
                planned.append(Removal(link, "file"))
        planned.append(Removal(folder, "dir-if-empty"))
    for path in legacy_files:
        if (path.is_symlink() or path.exists()) and not _protected(path, user_data):
            planned.append(Removal(path, "tree" if path.is_dir() and not path.is_symlink() else "file"))
    return tuple(planned)


def apply_migration(removals: Sequence[Removal], *, user_data: Path) -> list[Path]:
    """Perform removals. A protected path is skipped even if it was planned."""
    removed: list[Path] = []
    for item in removals:
        if _protected(item.path, user_data):
            continue
        if item.kind == "file":
            if item.path.is_file() or item.path.is_symlink():
                item.path.unlink()
                removed.append(item.path)
            continue
        if item.kind == "tree":
            if item.path.is_symlink() or item.path.is_file():
                item.path.unlink()
                removed.append(item.path)
            elif item.path.is_dir():
                _remove_tree(item.path)
                removed.append(item.path)
            continue
        if item.kind == "dir-if-empty":
            if item.path.is_dir() and not any(item.path.iterdir()):
                item.path.rmdir()
                removed.append(item.path)
            continue
        raise ValueError(f"unknown removal {item.kind}")
    return removed


def _remove_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


def _windows_registry_values() -> list[dict[str, str]]:
    if winreg is None:
        return []
    found: list[dict[str, str]] = []
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, UNINSTALL_SUBKEY) as key:
                values: dict[str, str] = {}
                index = 0
                while True:
                    try:
                        name, value, _typ = winreg.EnumValue(key, index)
                    except OSError:
                        break
                    if isinstance(value, str):
                        values[name] = value
                    index += 1
                if values:
                    found.append(values)
        except OSError:
            continue
    return found


def _windows_shortcut_dirs(group: str) -> list[Path]:
    folders: list[Path] = []
    appdata = os.environ.get("APPDATA", "").strip()
    programdata = os.environ.get("PROGRAMDATA", "").strip()
    for base in (appdata, programdata):
        if not base:
            continue
        folders.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / group)
    return folders


def linux_legacy_files(user_data: Path, home: Path) -> list[Path]:
    """Old per-user launcher and desktop entry that still point at the data tree."""
    old = user_data / "canvas"
    found: list[Path] = []
    marker = str(old)
    launcher = home / ".local" / "bin" / "voxprint-canvas"
    if launcher.is_symlink():
        target = os.readlink(launcher)
        if marker in target:
            found.append(launcher)
    desktop = home / ".local" / "share" / "applications" / "voxprint-canvas.desktop"
    if desktop.is_file():
        text = desktop.read_text(encoding="utf-8", errors="replace")
        if marker in text:
            found.append(desktop)
    return found


def migrate(
    new_install: Path,
    *,
    registry: Sequence[Mapping[str, str]] | None = None,
    user_data: Path | None = None,
    shortcut_dirs: Sequence[Path] | None = None,
    legacy_files: Sequence[Path] | None = None,
    platform: str | None = None,
    home: Path | None = None,
) -> list[Path]:
    """Remove the previous install. User data and the shared manifest stay."""
    system = sys.platform if platform is None else platform
    data = voxprint_home() if user_data is None else user_data
    records = list(_windows_registry_values() if registry is None and system == "win32" else (registry or ()))
    previous = [item for item in (previous_from_registry(values) for values in records) if item is not None]
    if system != "win32":
        old_app = data / "canvas"
        if old_app.exists() and old_app.resolve() != new_install.resolve():
            previous.append(PreviousInstall(old_app, OLD_START_MENU))
    groups = {item.icon_group for item in previous}
    groups.add(OLD_START_MENU)
    shortcuts = list(shortcut_dirs) if shortcut_dirs is not None else []
    if shortcut_dirs is None and system == "win32":
        for group in groups:
            shortcuts.extend(_windows_shortcut_dirs(group))
    extras = list(legacy_files) if legacy_files is not None else []
    if legacy_files is None and system != "win32":
        extras.extend(linux_legacy_files(data, Path.home() if home is None else home))
    planned = plan_migration(
        previous=previous,
        new_install=new_install,
        user_data=data,
        shortcut_dirs=shortcuts,
        legacy_files=extras,
    )
    return apply_migration(planned, user_data=data)


def main(argv: list[str] | None = None) -> int:
    """``python -m canvas.infra.layout migrate --install-dir <dir>``."""
    parser = argparse.ArgumentParser(prog="python -m canvas.infra.layout")
    parser.add_argument("command", choices=("migrate",))
    parser.add_argument("--install-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "migrate":
        migrate(args.install_dir)
        return 0
    raise ValueError(args.command)
