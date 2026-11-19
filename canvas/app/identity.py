"""Release identity. ``BUILD.json`` is the only copy of version, channel, build and codename."""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Written here so the release workflow can load this file without Qt.
_PRODUCT_NAME = "VOXPrint AI Canvas"
_RC_TAG = re.compile(r"^v\d+\.\d+\.\d+-rc\.\d+$")


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


def _release_head(data: Mapping[str, Any]) -> str:
    """``0.1.0-rc.28`` when ``rc`` is set, otherwise ``0.1.0 RC``."""
    base = str(data.get("version") or "").strip() or version()
    rc = data.get("rc")
    if isinstance(rc, int) and not isinstance(rc, bool) and rc > 0:
        return f"{base}-rc.{rc}"
    channel = str(data.get("channel") or "").strip()
    return f"{base} {channel}".strip()


def version_label(env: Mapping[str, str] | None = None) -> str:
    """``0.1.0-rc.28 · build N "Tavnit"``."""
    data = load_build()
    label = f"{_release_head(data)} · build {build_number(env)}"
    codename = str(data.get("codename") or "").strip()
    if codename:
        label += f' "{codename}"'
    return label


def splash_line(env: Mapping[str, str] | None = None) -> str:
    """``0.1.0-rc.28 build N "Codename"`` for ``--version`` and the startup window."""
    data = load_build()
    label = f"{_release_head(data)} build {build_number(env)}"
    codename = str(data.get("codename") or "").strip()
    if codename:
        label += f' "{codename}"'
    return label


def release_title(env: Mapping[str, str] | None = None) -> str:
    """GitHub release title.

    A release-candidate tag is ``VOXPrint AI Canvas v<version> build <N>``.
    The codename in ``BUILD.json`` is reserved for the final release.
    ``GITHUB_REF_NAME`` supplies the tag. This function does not import the UI stack.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    ref = str(source.get("GITHUB_REF_NAME", "")).strip()
    number = build_number(env)
    if _RC_TAG.fullmatch(ref):
        return f"{_PRODUCT_NAME} {ref} build {number}"
    return f"{_PRODUCT_NAME} {version_label(env)}"


_TAG_PARTS = re.compile(r"^v(\d+\.\d+\.\d+)(?:-rc\.(\d+))?$")


def tag_mismatch(tag: str, data: Mapping[str, Any]) -> str | None:
    """Why ``tag`` does not name the release in ``data``. None when they match.

    ``vX.Y.Z-rc.N`` needs ``version`` X.Y.Z and ``rc`` N. A plain ``vX.Y.Z``
    needs ``version`` X.Y.Z and no ``rc``.
    """
    match = _TAG_PARTS.fullmatch(tag.strip())
    if match is None:
        return f"tag {tag!r} is not vX.Y.Z or vX.Y.Z-rc.N"
    tag_version, tag_rc = match.group(1), match.group(2)
    build_version = str(data.get("version") or "").strip()
    if build_version != tag_version:
        return f"tag {tag} has version {tag_version}, BUILD.json has {build_version or 'none'}"
    rc = data.get("rc")
    build_rc = rc if isinstance(rc, int) and not isinstance(rc, bool) and rc > 0 else None
    wanted = None if tag_rc is None else int(tag_rc)
    if build_rc != wanted:
        shown = "none" if build_rc is None else str(build_rc)
        return f"tag {tag} has rc {tag_rc or 'none'}, BUILD.json has rc {shown}"
    return None


def check_release_tag(tag: str, paths: Sequence[Path]) -> list[str]:
    """One line per ``BUILD.json`` in ``paths`` that does not match ``tag``. Empty when all match."""
    problems: list[str] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        reason = tag_mismatch(tag, data) if isinstance(data, dict) else "BUILD.json must be an object"
        if reason is not None:
            problems.append(f"{path}: {reason}")
    return problems
