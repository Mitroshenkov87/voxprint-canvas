"""Release identity. ``BUILD.json`` is the only copy of version, channel, build and codename."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def build_file() -> Path:
    """Packaged ``BUILD.json`` beside this module, then the repository copy."""
    here = Path(__file__).resolve()
    candidates = [here.parent / "BUILD.json"]
    if len(here.parents) > 2:
        candidates.append(here.parents[2] / "BUILD.json")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("BUILD.json")


def load_build() -> dict[str, Any]:
    """The ``BUILD.json`` object."""
    data = json.loads(build_file().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("BUILD.json must be an object")
    return data


def version() -> str:
    """``X.Y.Z`` from ``BUILD.json``."""
    return str(load_build()["version"])


def build_number(env: Mapping[str, str] | None = None) -> int:
    """``build`` from ``BUILD.json``. ``VOXPRINT_BUILD`` wins. The Actions run number is ignored."""
    source: Mapping[str, str] = os.environ if env is None else env
    override = str(source.get("VOXPRINT_BUILD", "")).strip()
    if override.isdigit():
        return int(override)
    raw = load_build().get("build", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def version_label(env: Mapping[str, str] | None = None) -> str:
    """``0.1.0 RC · build N "Yotzer"``."""
    data = load_build()
    channel = str(data.get("channel") or "").strip()
    head = f"{version()} {channel}".strip()
    label = f"{head} · build {build_number(env)}"
    codename = str(data.get("codename") or "").strip()
    if codename:
        label += f' "{codename}"'
    return label


def release_title(env: Mapping[str, str] | None = None) -> str:
    """GitHub release title. The app name is the product name; the rest is ``BUILD.json``."""
    from canvas.app.info import APP_NAME

    return f"{APP_NAME} {version_label(env)}"
