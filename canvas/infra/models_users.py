"""Refcount of programs that use the shared models folder.

Adapted from Voxprint AI Audiobook Builder ``infra/models_users.py``
(Apache-2.0, Copyright Aleksandr Mitroshenkov). The file format is
``{"canvas": true}`` in ``<models>/.users.json``.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

USER_KEY = "canvas"
USERS_FILE = ".users.json"


def users_path(root: Path) -> Path:
    return root / USERS_FILE


def read_users(root: Path) -> dict[str, bool]:
    try:
        data = json.loads(users_path(root).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): bool(value) for key, value in data.items()}


def _write(users: dict[str, bool], root: Path) -> None:
    path = users_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(users, indent=1, sort_keys=True), encoding="utf-8")
    try:
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.01 * (attempt + 1))
        raise PermissionError(f"cannot replace {path}")
    finally:
        if tmp.exists():
            tmp.unlink()


def register(root: Path, key: str = USER_KEY) -> dict[str, bool]:
    users = read_users(root)
    if users.get(key) is True:
        return users
    users[key] = True
    _write(users, root)
    return users


def unregister(root: Path, key: str = USER_KEY) -> list[str]:
    users = read_users(root)
    if key in users:
        users.pop(key)
        _write(users, root)
    return sorted(name for name, used in users.items() if used)
