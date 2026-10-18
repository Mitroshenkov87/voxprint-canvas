"""Folders shared by the Voxprint programs.

Adapted from Voxprint AI Movie Dubber ``dubber/infra/shared_paths.py``
(Apache-2.0, Copyright Aleksandr Mitroshenkov).
"""
from __future__ import annotations

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
