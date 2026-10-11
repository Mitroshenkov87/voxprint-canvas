"""Shared suite manifest.

Other Voxprint apps can use the same file. It lives at ``manifest.json`` inside
the shared directory. A resource records its files, their SHA-256 hashes, and
the apps that use it. ``refcount`` is the number of those apps. Uninstall
deletes a resource only when that count reaches zero.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from canvas.models.pin import PinError, verify_file

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
APP_ID = "canvas"


class ManifestError(ValueError):
    """The shared manifest could not be read."""


@dataclass(frozen=True)
class Resource:
    """One shared file set."""

    kind: str
    path: str
    nbytes: int
    files: dict[str, str]
    apps: tuple[str, ...]
    source: str = ""

    @property
    def refcount(self) -> int:
        """How many apps still use this resource."""
        return len(self.apps)


def manifest_path(shared: Path) -> Path:
    """``manifest.json`` inside the shared directory."""
    return shared / MANIFEST_NAME


def load_manifest(shared: Path) -> dict[str, Resource]:
    """Resources in the manifest. A missing file is an empty map."""
    path = manifest_path(shared)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ManifestError("manifest.json could not be read") from exc
    if not isinstance(data, dict) or int(data.get("version") or 0) != MANIFEST_VERSION:
        raise ManifestError("manifest.json has an unknown version")
    raw = data.get("resources")
    if not isinstance(raw, dict):
        raise ManifestError("manifest.json has no resources")
    resources: dict[str, Resource] = {}
    for key, row in raw.items():
        if not isinstance(row, dict):
            raise ManifestError("a resource entry is not an object")
        apps = tuple(str(item) for item in row.get("apps") or [])
        files = row.get("files") or {}
        if not isinstance(files, dict):
            raise ManifestError("a resource file map is not an object")
        resources[str(key)] = Resource(
            kind=str(row.get("kind") or ""),
            path=_safe_relative(str(row.get("path") or ".")),
            nbytes=int(row.get("bytes") or 0),
            files={str(name): str(digest) for name, digest in files.items()},
            apps=apps,
            source=str(row.get("source") or ""),
        )
    return resources


def save_manifest(shared: Path, resources: dict[str, Resource]) -> None:
    """Write the manifest atomically."""
    body = {
        "version": MANIFEST_VERSION,
        "resources": {
            key: {
                "kind": resource.kind,
                "path": resource.path,
                "bytes": resource.nbytes,
                "files": resource.files,
                "apps": list(resource.apps),
                "refcount": resource.refcount,
                "source": resource.source,
            }
            for key, resource in sorted(resources.items())
        },
    }
    path = manifest_path(shared)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
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


def apps_in(shared: Path) -> tuple[str, ...]:
    """Every app named on at least one resource."""
    found: list[str] = []
    for resource in load_manifest(shared).values():
        for app in resource.apps:
            if app not in found:
                found.append(app)
    return tuple(found)


def register_app(shared: Path, app: str = APP_ID) -> None:
    """Add ``app`` to every resource, or create an empty suite entry."""
    resources = load_manifest(shared)
    if not resources:
        resources = {
            "suite": Resource(kind="suite", path=".", nbytes=0, files={}, apps=(app,), source=""),
        }
    else:
        resources = {
            key: resource if app in resource.apps else _with_app(resource, app)
            for key, resource in resources.items()
        }
    save_manifest(shared, resources)


def register_resource(
    shared: Path,
    resource_id: str,
    *,
    kind: str,
    path: str,
    nbytes: int,
    files: dict[str, str],
    app: str = APP_ID,
    source: str = "",
) -> None:
    """Record one resource and add ``app`` to it."""
    resources = load_manifest(shared)
    current = resources.get(resource_id)
    apps = (app,) if current is None else tuple(dict.fromkeys((*current.apps, app)))
    resources[resource_id] = Resource(
        kind=kind,
        path=_safe_relative(path),
        nbytes=nbytes,
        files=dict(files),
        apps=apps,
        source=source,
    )
    save_manifest(shared, resources)


def unregister_app(shared: Path, app: str = APP_ID) -> list[str]:
    """Drop ``app``. Delete a resource, and its files, only at refcount zero."""
    resources = load_manifest(shared)
    kept: dict[str, Resource] = {}
    removed: list[str] = []
    for key, resource in resources.items():
        apps = tuple(item for item in resource.apps if item != app)
        if resource.apps and not apps:
            _delete_files(shared, resource)
            removed.append(key)
            continue
        kept[key] = Resource(
            resource.kind, resource.path, resource.nbytes, resource.files, apps, resource.source,
        )
    save_manifest(shared, kept)
    return removed


def broken_files(shared: Path, resource: Resource) -> list[str]:
    """File names whose bytes are missing or do not match the recorded hash."""
    base = _resource_dir(shared, resource)
    broken: list[str] = []
    for name, digest in resource.files.items():
        path = base / name
        if not path.is_file():
            broken.append(name)
            continue
        try:
            verify_file(path, digest)
        except (PinError, OSError):
            broken.append(name)
    return broken


def _with_app(resource: Resource, app: str) -> Resource:
    return Resource(
        resource.kind, resource.path, resource.nbytes, resource.files, resource.apps + (app,), resource.source,
    )


def _safe_relative(path: str) -> str:
    if not path or path.startswith(("/", "\\")) or ".." in Path(path).parts:
        raise ManifestError("a resource path must stay inside the shared folder")
    return Path(path).as_posix()


def _resource_dir(shared: Path, resource: Resource) -> Path:
    base = (shared / resource.path).resolve()
    root = shared.resolve()
    if base != root and root not in base.parents:
        raise ManifestError("a resource path must stay inside the shared folder")
    return base


def _delete_files(shared: Path, resource: Resource) -> None:
    base = _resource_dir(shared, resource)
    root = shared.resolve()
    for name in resource.files:
        target = (base / name).resolve()
        if target != root and root not in target.parents:
            continue
        if target.is_file():
            target.unlink()
        parent = target.parent
        while parent != root and parent != base.parent:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
