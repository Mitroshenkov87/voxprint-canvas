"""Which apps use the shared suite folder.

``manifest.json`` in the shared directory is schema 1. ``register`` records
this app when the file is empty. ``unregister`` drops this app and deletes a
component only when nothing else still refs it.
"""
from __future__ import annotations

from pathlib import Path

from canvas.infra.manifest import APP_ID, apps_in, register_app, unregister_app

USER_KEY = APP_ID


def read_users(root: Path) -> dict[str, bool]:
    """Apps recorded in the manifest. A missing file is an empty map."""
    return {app: True for app in apps_in(root)}


def register(root: Path, key: str = USER_KEY) -> dict[str, bool]:
    """Record this app in the shared manifest."""
    register_app(root, key)
    return read_users(root)


def unregister(root: Path, key: str = USER_KEY) -> list[str]:
    """Drop this app. Returns the apps that still ref a component."""
    unregister_app(root, key)
    return sorted(read_users(root))
