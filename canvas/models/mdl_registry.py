"""Read the Canvas model registry without a hard-coded id list.

A registered install manifest wins when it carries real pins. Otherwise the
bundled repo-root ``mdl.json`` (the file the installer copies) is used. The
placeholder stub packaged with the Canvas module is the last resort.
"""
from __future__ import annotations

import json
from pathlib import Path

from voxprint_common.mdl.catalog import REGISTRY_NAME, CatalogError, Model, discover, install_dir, load_manifest, register_app, state_dir, unregister_app

APP_ID = "canvas"
_SHIPPED = Path(__file__).resolve().parents[1] / "mdl.json"
_BUNDLED = Path(__file__).resolve().parents[2] / "mdl.json"


def shipped_manifest() -> Path:
    """Return the registry file packaged with Canvas."""
    return _SHIPPED


def bundled_manifest() -> Path:
    """Return the real pinned registry that ships beside the package (repo-root ``mdl.json``)."""
    return _BUNDLED


def has_real_pins(path: Path) -> bool:
    """True when ``path`` loads and lists at least one available model with pinned files."""
    try:
        manifest = load_manifest(path)
    except (CatalogError, OSError, ValueError):
        return False
    return any(not model.unavailable and model.files for model in manifest.models)


def _install_manifest() -> Path | None:
    """The real pinned manifest in the default install folder, if any."""
    try:
        candidate = install_dir(APP_ID) / "mdl.json"
    except CatalogError:
        return None
    return candidate if candidate.is_file() and has_real_pins(candidate) else None


def manifest_path() -> Path:
    """Return the registry file to use.

    Order: the manifest registered for Canvas, the default install folder, the
    bundled repo-root ``mdl.json``. A candidate that is only a placeholder is
    skipped. The packaged placeholder stub is used only when none has pins.
    """
    try:
        installed = discover()
    except CatalogError:
        installed = []
    for app in installed:
        if app.id != APP_ID:
            continue
        candidate = _manifest_file(app)
        if candidate is not None and has_real_pins(candidate):
            return candidate
    found = _install_manifest()
    if found is not None:
        return found
    if _BUNDLED.is_file() and has_real_pins(_BUNDLED):
        return _BUNDLED
    return shipped_manifest()


def registry_path() -> Path:
    """Return ``<home>/state/installed-apps.json``."""
    return state_dir() / REGISTRY_NAME


def _app_folder() -> Path | None:
    """The folder holding the real pinned ``mdl.json`` that Canvas should register."""
    found = _install_manifest()
    if found is not None:
        return found.parent
    if _BUNDLED.is_file() and has_real_pins(_BUNDLED):
        return _BUNDLED.parent
    return None


DISPLAY_NAME = "VOXPrint AI Canvas"


def _manifest_file(app: object, *, require_pins: bool = True) -> Path | None:
    """Return the manifest for an ``InstalledApp``.

    voxprint-common 0.7.4 may set ``path`` to the ``mdl.json`` file itself.
    Older registry rows, and folder-scan results, still point at the install
    directory. ``install_path`` is tried last when present.
    """
    candidates: list[Path] = []
    raw = Path(getattr(app, "path"))
    if raw.is_file():
        candidates.append(raw)
    else:
        candidates.append(raw / "mdl.json")
    install = getattr(app, "install_path", None)
    if install is not None:
        candidates.append(Path(install) / "mdl.json")
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_file():
            continue
        if require_pins and not has_real_pins(candidate):
            continue
        return candidate
    return None


def _descriptor(folder: Path) -> Path | None:
    """Return ``folder/mdl.json`` when that file exists."""
    candidate = folder / "mdl.json"
    return candidate if candidate.is_file() else None


def _row_is_current(row: object, folder: Path) -> bool:
    """True when ``row`` already records Canvas at ``folder`` with our display name."""
    if not isinstance(row, dict) or row.get("id") != APP_ID:
        return False
    if row.get("path") != str(folder):
        return False
    if (row.get("display_name") or "") != DISPLAY_NAME:
        return False
    wanted = _descriptor(folder)
    stored = row.get("descriptor")
    if wanted is None:
        return not stored
    return stored == str(wanted)


def register_install(folder: Path | None = None) -> bool:
    """Record Canvas in ``installed-apps.json`` so MDL discovery lists it.

    Uses ``voxprint_common.mdl.catalog.register_app`` (idempotent). Other apps
    in the file are kept. A missing file is created. A row that already has
    the right path and display name is left alone. An unreadable registry is
    left untouched.

    Args:
        folder: Install folder that contains the pinned ``mdl.json``.
            Defaults to the install folder, else the bundled copy.

    Returns:
        True when the file was written.
    """
    target = folder if folder is not None else _app_folder()
    if target is None:
        return False
    path = registry_path()
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, ValueError):
            return False
        if not isinstance(loaded, dict) or not isinstance(loaded.get("apps", []), list):
            return False
        if any(_row_is_current(row, target) for row in loaded.get("apps", [])):
            return False
    try:
        register_app(APP_ID, target, display_name=DISPLAY_NAME, descriptor=_descriptor(target))
    except CatalogError:
        return False
    return True


def unregister_install() -> bool:
    """Remove Canvas from ``installed-apps.json``.

    Returns:
        True when a row was removed. A missing file or a missing row is False.
        A damaged registry is left untouched and returns False.
    """
    try:
        return unregister_app(APP_ID)
    except CatalogError:
        return False


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
        candidate = _manifest_file(app, require_pins=False)
        if candidate is not None:
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
