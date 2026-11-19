"""Folders shared by the Voxprint programs.

Adapted from Voxprint AI Movie Dubber ``dubber/infra/shared_paths.py``
(Apache-2.0, Copyright Aleksandr Mitroshenkov).
"""
from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from voxprint_common import paths as _common_paths


def voxprint_home() -> Path:
    """The suite data home from voxprint-common.

    ``%LOCALAPPDATA%\\VOXPrint AI Suite`` on Windows and
    ``~/.local/share/voxprint-ai-suite`` elsewhere. ``VOXPRINT_HOME`` replaces
    it. A legacy ``Voxprint`` folder is moved there on first use.
    """
    root = _common_paths.home()
    root.mkdir(parents=True, exist_ok=True)
    return root


def shared_dir() -> Path:
    """Versioned suite folder: ``<data home>/shared``."""
    path = voxprint_home() / "shared"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    """The suite weights folder, from :func:`voxprint_common.paths.models_dir`.

    ``VOXPRINT_MODELS_DIR``, then ``models_dir`` in ``suite.json``, then the
    legacy ``state/models_dir.txt``, otherwise ``<data home>/models``. The MDL
    client downloads into this same folder.
    """
    path = _common_paths.models_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolved_models_dir() -> tuple[Path, str]:
    """The weights folder and how it was chosen.

    The first absolute path wins, matching :func:`voxprint_common.paths.models_dir`:

    - ``VOXPRINT_MODELS_DIR``
    - ``models_dir`` in ``suite.json``
    - the legacy ``state/models_dir.txt``
    - otherwise ``<data home>/models`` (``default``)
    """
    env = _common_paths._absolute_env("VOXPRINT_MODELS_DIR")
    if env is not None:
        return env, "VOXPRINT_MODELS_DIR"
    suite = _common_paths._suite_models_dir()
    if suite is not None:
        return suite, "suite.json"
    legacy = _common_paths._legacy_models_dir()
    if legacy is not None:
        return legacy, "models_dir.txt"
    return _common_paths.home() / "models", "default"


def lora_library_dir() -> Path:
    """Imported LoRAs: ``<data home>/canvas/lora-library``."""
    path = voxprint_home() / "canvas" / "lora-library"
    path.mkdir(parents=True, exist_ok=True)
    return path


PROJECTS_FOLDER = "VOXPrint Projects"
"""Suite projects folder inside the user's Documents folder (suite rule 5)."""

APP_PROJECTS_FOLDER = "Canvas"
"""This app's subfolder, its display name, matching ``Model Forge``."""


def _user_dirs_documents(home: Path, config_home: Path) -> Path | None:
    """``XDG_DOCUMENTS_DIR`` from ``user-dirs.dirs``, or None when unset, relative or the home itself.

    The file is shell syntax: ``XDG_DOCUMENTS_DIR="$HOME/Документы"``. Only ``$HOME``
    is expanded, as xdg-user-dirs does. The value may hold spaces and non-ASCII names.
    """
    try:
        text = (config_home / "user-dirs.dirs").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for line in text.splitlines():
        name, sep, value = line.strip().partition("=")
        if not sep or name.strip() != "XDG_DOCUMENTS_DIR":
            continue
        try:
            words = shlex.split(value, comments=True)
        except ValueError:
            return None
        if len(words) != 1:
            return None
        raw = words[0]
        if raw.startswith("$HOME"):
            raw = str(home) + raw[len("$HOME"):]
        elif raw.startswith("${HOME}"):
            raw = str(home) + raw[len("${HOME}"):]
        path = Path(raw)
        if not path.is_absolute() or path == home:
            return None
        return path
    return None


def _windows_documents() -> Path | None:
    """The Documents known folder (FOLDERID_Documents), which follows OneDrive and moved folders."""
    try:
        import ctypes
        import uuid
        from ctypes import wintypes

        class _Guid(ctypes.Structure):
            _fields_ = [("data", ctypes.c_ubyte * 16)]

        guid = _Guid()
        ctypes.memmove(guid.data, uuid.UUID("{FDD39AD0-238F-46AF-ADB4-6C85480369C7}").bytes_le, 16)
        out = ctypes.c_wchar_p()
        shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
        if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(out)) != 0:
            return None
        try:
            return Path(out.value) if out.value else None
        finally:
            ctypes.windll.ole32.CoTaskMemFree(ctypes.cast(out, wintypes.LPVOID))  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        return None


def documents_dir(*, platform: str | None = None, environ: dict[str, str] | None = None,
                  home: Path | None = None) -> Path:
    """The user's Documents folder, by the platform's own lookup, never a hard-coded name.

    - Linux and other XDG systems: ``XDG_DOCUMENTS_DIR`` from the environment, then
      ``$XDG_CONFIG_HOME/user-dirs.dirs`` (``~/.config``), then ``~/Documents``.
      On a Russian desktop this is ``~/Документы``.
    - Windows: the Documents known folder, then ``%USERPROFILE%\\Documents``.
    - macOS: ``~/Documents``.
    """
    plat = sys.platform if platform is None else platform
    env = os.environ if environ is None else environ
    base = Path.home() if home is None else home
    if plat == "win32":
        found = _windows_documents() if platform is None else None
        if found is not None:
            return found
        profile = env.get("USERPROFILE")
        return (Path(profile) if profile else base) / "Documents"
    if plat == "darwin":
        return base / "Documents"
    direct = env.get("XDG_DOCUMENTS_DIR", "")
    if direct and Path(direct).is_absolute() and Path(direct) != base:
        return Path(direct)
    config = env.get("XDG_CONFIG_HOME", "")
    config_home = Path(config) if config and Path(config).is_absolute() else base / ".config"
    found = _user_dirs_documents(base, config_home)
    return found if found is not None else base / "Documents"


def projects_dir(*, create: bool = True) -> Path:
    """Default home of Canvas pictures, series and batches: ``<Documents>/VOXPrint Projects/Canvas``.

    ``VOXPRINT_PROJECTS_DIR`` (an absolute path) replaces ``<Documents>/VOXPrint Projects``.
    An explicit ``--out`` always wins over this default. Pictures saved before this
    default existed stay where they are.
    """
    override = os.environ.get("VOXPRINT_PROJECTS_DIR", "")
    root = Path(override) if override and Path(override).is_absolute() else documents_dir() / PROJECTS_FOLDER
    path = root / APP_PROJECTS_FOLDER
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


class OutputFolderError(OSError):
    """The output folder cannot be created or written."""


def prepare_output_folder(folder: Path) -> Path:
    """Create ``folder`` with its parents and prove it is writable.

    Commands call this before any model loads, so a bad ``--out`` stops in
    seconds instead of after a picture was made.

    Raises:
        OutputFolderError: The folder cannot be created or written.
    """
    target = Path(folder)
    if target.exists() and not target.is_dir():
        raise OutputFolderError(f"The output folder is a file: {target}")
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputFolderError(f"The output folder cannot be created: {target} ({exc.strerror or exc})") from exc
    probe = target / f".canvas-write-check-{os.getpid()}"
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        raise OutputFolderError(f"The output folder is not writable: {target} ({exc.strerror or exc})") from exc
    return target
