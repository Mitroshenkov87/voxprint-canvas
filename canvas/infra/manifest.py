"""Shared suite manifest, SHARED-RUNTIME schema 1.

The file is ``manifest.json`` in the shared directory. Another suite app may
already own it. Canvas reads that shape, keeps keys it does not use, and
writes the same shape back. Package ``voxprint-common`` will own this helper;
until then the implementation stays here and ``FUTURE_IMPORT`` names the swap.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

# Local SHARED-RUNTIME schema 1. The future import is ``voxprint_common.runtime``.
FUTURE_IMPORT = "voxprint_common.runtime"
SHARED_RUNTIME_SCHEMA = 1
SCHEMA = SHARED_RUNTIME_SCHEMA
MANIFEST_NAME = "manifest.json"
LOCK_NAME = "manifest.lock"
TMP_NAME = "manifest.json.tmp"
APP_ID = "canvas"
LOCK_TIMEOUT_S = 30.0
_KINDS = frozenset({"runtime", "wheels", "model", "tool"})
_KIND_FROM_OLD = {"pack": "model", "suite": "tool", "wheels": "wheels", "runtime": "runtime", "model": "model", "tool": "tool"}
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_KNOWN = ("kind", "version", "path", "sha256", "installed_at", "refs")


class ManifestError(ValueError):
    """The shared manifest could not be read or updated."""


@dataclass(frozen=True)
class Component:
    """One shared install."""

    kind: str
    version: str
    path: str
    sha256: str
    installed_at: str
    refs: tuple[str, ...]
    extra: dict[str, Any] = field(default_factory=dict)


def manifest_path(shared: Path) -> Path:
    """``manifest.json`` inside the shared directory."""
    return shared / MANIFEST_NAME


def lock_sha256(files: dict[str, str]) -> str:
    """SHA-256 of a lock listing ``name digest`` lines in sorted order."""
    body = "".join(f"{name} {digest}\n" for name, digest in sorted(files.items()))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@contextmanager
def manifest_lock(shared: Path):
    """Exclusive lock on ``manifest.lock``. Raises if it is not free in time."""
    shared.mkdir(parents=True, exist_ok=True)
    fd = os.open(shared / LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        _acquire(fd, LOCK_TIMEOUT_S)
        yield
    finally:
        _release(fd)
        os.close(fd)


def load_manifest(shared: Path) -> dict[str, Component]:
    """Components in the manifest. A missing file is an empty map.

    An old Canvas file (``version`` plus ``resources``) is converted and
    written back as schema 1.
    """
    path = manifest_path(shared)
    if not path.is_file():
        return {}
    with manifest_lock(shared):
        components, extra, migrated = _read_document(shared)
        if migrated:
            _write_unlocked(shared, components, extra)
        return components


def apps_in(shared: Path) -> tuple[str, ...]:
    """Every app named on at least one component."""
    found: list[str] = []
    for component in load_manifest(shared).values():
        for app in component.refs:
            if app not in found:
                found.append(app)
    return tuple(found)


def register_app(shared: Path, app: str = APP_ID) -> None:
    """Record ``app`` when the manifest is empty. Existing components stay as they are."""
    def change(components: dict[str, Component]) -> None:
        if components:
            return
        components["canvas"] = Component(
            kind="tool",
            version="0.1.0",
            path=".",
            sha256=hashlib.sha256(b"").hexdigest(),
            installed_at=_now(),
            refs=(app,),
        )

    _mutate(shared, change)


def register_component(
    shared: Path,
    component_id: str,
    *,
    kind: str,
    version: str,
    path: str,
    sha256: str,
    app: str = APP_ID,
) -> str:
    """Record a component and add ``app`` to its refs.

    The same id, version, and sha256 reuse the existing path. A different
    version or sha256 is installed beside it and never replaces a component
    another app still refs. The return value is the id that was written.
    """
    _require_id(component_id)
    if kind not in _KINDS:
        raise ManifestError("component kind is not valid")
    digest = _require_sha(sha256)
    relative = _safe_relative(path)

    def change(components: dict[str, Component]) -> str:
        existing = components.get(component_id)
        if existing is not None and existing.version == version and existing.sha256 == digest:
            if app not in existing.refs:
                components[component_id] = replace(existing, refs=existing.refs + (app,))
            return component_id
        target_id = component_id
        target_path = relative
        if existing is not None:
            target_id = _side_id(components, component_id, digest)
            if relative == existing.path:
                target_path = _side_path(components, relative, digest)
        components[target_id] = Component(
            kind=kind,
            version=version,
            path=target_path,
            sha256=digest,
            installed_at=_now(),
            refs=(app,),
        )
        return target_id

    return _mutate(shared, change)


def unregister_app(shared: Path, app: str = APP_ID) -> list[str]:
    """Drop ``app``. Delete a component only when its refs become empty."""
    def change(components: dict[str, Component]) -> list[str]:
        removed: list[str] = []
        for key, component in list(components.items()):
            if app not in component.refs:
                continue
            refs = tuple(item for item in component.refs if item != app)
            if refs:
                components[key] = replace(component, refs=refs)
                continue
            removed.append(key)
        for key in removed:
            component = components.pop(key)
            _delete_tree(shared, component, components)
        return removed

    return _mutate(shared, change)


def _mutate[T](shared: Path, change: Callable[[dict[str, Component]], T]) -> T:
    with manifest_lock(shared):
        components, extra, _migrated = _read_document(shared)
        result = change(components)
        _write_unlocked(shared, components, extra)
        return result


def _read_document(shared: Path) -> tuple[dict[str, Component], dict[str, Any], bool]:
    path = manifest_path(shared)
    if not path.is_file():
        return {}, {}, False
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ManifestError("manifest.json could not be read") from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest.json has an unknown version")
    schema = data.get("schema")
    raw_components = data.get("components")
    if type(schema) is int and schema == SCHEMA and isinstance(raw_components, dict):
        extra = {key: value for key, value in data.items() if key not in ("schema", "components")}
        return _parse_components(raw_components), extra, False
    if "version" in data and "resources" in data:
        components, extra = _migrate(data)
        return components, extra, True
    raise ManifestError("manifest.json has an unknown version")


def _migrate(data: dict[str, Any]) -> tuple[dict[str, Component], dict[str, Any]]:
    raw = data.get("resources")
    if not isinstance(raw, dict):
        raise ManifestError("manifest.json has no resources")
    installed_at = _now()
    components: dict[str, Component] = {}
    for key, row in raw.items():
        if not isinstance(row, dict):
            raise ManifestError("a resource entry is not an object")
        component_id = str(key)
        _require_id(component_id)
        files = row.get("files") or {}
        if not isinstance(files, dict):
            raise ManifestError("a resource file map is not an object")
        source = str(row.get("source") or "")
        apps = tuple(dict.fromkeys(str(item) for item in row.get("apps") or []))
        old_kind = str(row.get("kind") or "")
        components[component_id] = Component(
            kind=_KIND_FROM_OLD.get(old_kind, "tool"),
            version=source or "0",
            path=_safe_relative(str(row.get("path") or ".")),
            sha256=lock_sha256({str(name): str(digest) for name, digest in files.items()}),
            installed_at=installed_at,
            refs=apps,
        )
    extra = {
        key: value
        for key, value in data.items()
        if key not in ("version", "resources", "schema", "components")
    }
    return components, extra


def _parse_components(raw: dict[str, Any]) -> dict[str, Component]:
    components: dict[str, Component] = {}
    for key, row in raw.items():
        if not isinstance(row, dict):
            raise ManifestError("a component entry is not an object")
        refs_raw = row.get("refs") or []
        if not isinstance(refs_raw, list):
            raise ManifestError("component refs are not a list")
        extra = {name: value for name, value in row.items() if name not in _KNOWN}
        components[str(key)] = Component(
            kind=str(row.get("kind") or ""),
            version=str(row.get("version") or ""),
            path=_safe_relative(str(row.get("path") or ".")),
            sha256=str(row.get("sha256") or ""),
            installed_at=str(row.get("installed_at") or ""),
            refs=tuple(dict.fromkeys(str(item) for item in refs_raw)),
            extra=extra,
        )
    return components


def _write_unlocked(shared: Path, components: dict[str, Component], extra: dict[str, Any]) -> None:
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "components": {key: _dump(components[key]) for key in sorted(components)},
    }
    for key in sorted(extra):
        if key in ("schema", "components", "version", "resources"):
            continue
        body[key] = extra[key]
    path = manifest_path(shared)
    tmp = shared / TMP_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(body, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.01 * (attempt + 1))
        raise ManifestError("manifest.json could not be replaced")
    finally:
        if tmp.exists():
            tmp.unlink()


def _dump(component: Component) -> dict[str, Any]:
    body: dict[str, Any] = {
        "kind": component.kind,
        "version": component.version,
        "path": component.path,
        "sha256": component.sha256,
        "installed_at": component.installed_at,
        "refs": list(component.refs),
    }
    for key in sorted(component.extra):
        if key not in body:
            body[key] = component.extra[key]
    return body


def _delete_tree(shared: Path, component: Component, remaining: dict[str, Component]) -> None:
    if component.path in ("", "."):
        return
    for other in remaining.values():
        if other.path == component.path:
            return
    root = shared.resolve()
    target = (shared / component.path).resolve()
    if target == root or root not in target.parents:
        return
    if target.is_dir():
        shutil.rmtree(target)
    elif target.is_file():
        target.unlink()


def _side_id(components: dict[str, Component], base: str, sha256: str) -> str:
    suffix = sha256[:8]
    candidate = f"{base}-{suffix}"
    number = 2
    while candidate in components:
        candidate = f"{base}-{suffix}-{number}"
        number += 1
    _require_id(candidate)
    return candidate


def _side_path(components: dict[str, Component], path: str, sha256: str) -> str:
    suffix = sha256[:8]
    stem = path if path not in ("", ".") else "side"
    candidate = f"{stem}-{suffix}"
    used = {item.path for item in components.values()}
    number = 2
    while candidate in used:
        candidate = f"{stem}-{suffix}-{number}"
        number += 1
    return _safe_relative(candidate)


def _require_id(component_id: str) -> None:
    if not _ID_RE.fullmatch(component_id):
        raise ManifestError("component id is not valid")


def _require_sha(sha256: str) -> str:
    if not _SHA_RE.fullmatch(sha256):
        raise ManifestError("sha256 must be 64 hex characters")
    return sha256


def _safe_relative(path: str) -> str:
    raw = path.replace("\\", "/")
    if not raw or raw.startswith("/") or (len(raw) >= 2 and raw[1] == ":"):
        raise ManifestError("a component path must stay inside the shared folder")
    if ".." in Path(raw).parts:
        raise ManifestError("a component path must stay inside the shared folder")
    return Path(raw).as_posix()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _acquire(fd: int, timeout: float) -> None:
    if sys.platform == "win32":
        _prepare_windows_lock(fd)
    deadline = time.monotonic() + timeout
    while True:
        try:
            if sys.platform == "win32":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise ManifestError("manifest.lock timed out") from None
            time.sleep(0.05)


def _prepare_windows_lock(fd: int) -> None:
    os.lseek(fd, 0, os.SEEK_SET)
    if os.fstat(fd).st_size == 0:
        os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)


def _release(fd: int) -> None:
    try:
        if sys.platform == "win32":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        return
