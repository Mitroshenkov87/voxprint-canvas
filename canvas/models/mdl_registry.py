"""Read the Canvas model registry without a hard-coded id list.

A discovered install manifest wins. Otherwise the file shipped in this package
is used. Placeholders stay unavailable until Model Forge replaces the file.
"""
from __future__ import annotations

from pathlib import Path

from voxprint_common.mdl.catalog import CatalogError, Model, discover, load_manifest

_SHIPPED = Path(__file__).resolve().parents[1] / "mdl.json"


def shipped_manifest() -> Path:
    """Return the registry file packaged with Canvas."""
    return _SHIPPED


def manifest_path() -> Path:
    """Return the install manifest when one is registered, else the shipped file."""
    try:
        installed = discover()
    except CatalogError:
        installed = []
    for app in installed:
        if app.id != "canvas":
            continue
        candidate = app.path / "mdl.json"
        if candidate.is_file():
            return candidate
    return shipped_manifest()


def load_models() -> tuple[Model, ...]:
    """Return every model row in the registry.

    Raises:
        CatalogError: The manifest cannot be read.
    """
    return load_manifest(manifest_path()).models


def selectable() -> tuple[Model, ...]:
    """Return models a person can pick. Unavailable placeholders are left out."""
    return tuple(model for model in load_models() if not model.unavailable)


def enlarge_manifest_path() -> Path | None:
    """Return Enlarge's ``mdl.json`` when the MDL registry lists it."""
    try:
        installed = discover()
    except CatalogError:
        return None
    for app in installed:
        if app.id != "enlarge":
            continue
        candidate = app.path / "mdl.json"
        if candidate.is_file():
            return candidate
    return None


def enlarge_download_bytes() -> int | None:
    """Return the Enlarge download size in bytes.

    The figure is the sum of file sizes in the registered Enlarge manifest.
    The same digest is counted once. ``None`` means the registry has no size.
    """
    path = enlarge_manifest_path()
    if path is None:
        return None
    try:
        manifest = load_manifest(path)
    except CatalogError:
        return None
    if manifest.app != "enlarge":
        return None
    seen: set[str] = set()
    total = 0
    for model in manifest.models:
        if model.unavailable:
            continue
        for item in model.files:
            if item.size <= 0 or item.sha256 in seen:
                continue
            seen.add(item.sha256)
            total += item.size
    if total <= 0:
        return None
    return total
