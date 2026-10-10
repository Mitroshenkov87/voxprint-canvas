"""Folders shared by the Voxprint programs.

Adapted from Voxprint AI Movie Dubber ``dubber/infra/shared_paths.py``
(Apache-2.0, Copyright Aleksandr Mitroshenkov).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def voxprint_home() -> Path:
    """``%LOCALAPPDATA%\\Voxprint`` on Windows, ``~/.local/share/voxprint`` elsewhere."""
    override = os.environ.get("VOXPRINT_HOME", "").strip()
    if override:
        root = Path(override)
    elif sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        root = Path(base) / "Voxprint"
    else:
        xdg = os.environ.get("XDG_DATA_HOME", "").strip()
        base = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".local" / "share"
        root = base / "voxprint"
    root.mkdir(parents=True, exist_ok=True)
    return root


def models_dir() -> Path:
    """Shared model folder. ``VOXPRINT_MODELS_DIR`` overrides it when it is absolute."""
    override = os.environ.get("VOXPRINT_MODELS_DIR", "").strip()
    if override:
        path = Path(os.path.expandvars(os.path.expanduser(override)))
        if path.is_absolute():
            path.mkdir(parents=True, exist_ok=True)
            return path
    path = voxprint_home() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path
