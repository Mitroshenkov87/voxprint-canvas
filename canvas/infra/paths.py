"""Folders shared by the Voxprint programs.

Adapted from Voxprint AI Movie Dubber ``dubber/infra/shared_paths.py``
(Apache-2.0, Copyright Aleksandr Mitroshenkov).
"""
from __future__ import annotations

import os
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
    """Shared model folder. ``VOXPRINT_MODELS_DIR`` overrides it when it is absolute."""
    override = os.environ.get("VOXPRINT_MODELS_DIR", "").strip()
    if override:
        path = Path(os.path.expandvars(os.path.expanduser(override)))
        if path.is_absolute():
            path.mkdir(parents=True, exist_ok=True)
            return path
    path = shared_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path
