"""Which apps use the shared suite folder.

The file format other apps should adopt is ``manifest.json`` in the shared
directory. ``register`` adds this app to every resource, creating a suite
entry when the folder is empty. ``unregister`` drops this app and deletes a
resource only when its refcount reaches zero.
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
    """Drop this app. Returns the apps that still use a resource."""
    unregister_app(root, key)
    return sorted(read_users(root))
