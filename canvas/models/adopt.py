"""Adopt a legacy ``<org>--<name>`` folder into the MDL blob store.

Matching files are hard-linked into ``blobs/<sha256>``. Nothing in the legacy
folder is deleted. A file that is missing, mismatched, or on another volume is
fetched through MDL. The loader still opens only a view.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from voxprint_common.mdl.catalog import CatalogError, load_manifest

from canvas.infra.stage_timer import log_detail
from canvas.models.mdl_fetch import MANIFEST, _download, plan_blobs
from canvas.models.pin import PinError, verify_file
from canvas.models.registry import Catalog, ModelSpec, RegistryError, load_catalog
from canvas.models.snapshots import SnapshotError, weight_dir
from canvas.models.views import ViewError, blob_path, blobs_present, ensure_view

# The check models, both Z-Image-Turbo packs, and the scenario writer.
_GATE_ROLES = ("prompt-guard", "zero-shot", "face-detector", "age", "nsfw", "age-fine")
_MODEL_IDS = ("z-image-turbo-bf16-16gb", "z-image-turbo-bf16", "qwen3-4b-writer")
_done: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class AdoptResult:
    """One model's adopt counts. ``ok`` does not include files fetched afterwards."""

    model_id: str
    files: int
    ok: int
    fetched: int
    fetched_paths: tuple[str, ...] = ()


def required_specs(catalog: Catalog | None = None) -> tuple[ModelSpec, ...]:
    """Models adopted when their view is incomplete and a legacy folder exists."""
    chosen = catalog or load_catalog()
    specs: list[ModelSpec] = []
    for role in _GATE_ROLES:
        rows = chosen.by_role(role)
        if len(rows) == 1:
            specs.append(rows[0])
    for model_id in _MODEL_IDS:
        try:
            specs.append(chosen.by_id(model_id))
        except RegistryError:
            continue
    return tuple(specs)


def gate_specs(catalog: Catalog | None = None) -> tuple[ModelSpec, ...]:
    """Content-check models, one per role. An empty result means one role is missing."""
    chosen = catalog or load_catalog()
    found: list[ModelSpec] = []
    for role in _GATE_ROLES:
        rows = chosen.by_role(role)
        if len(rows) != 1:
            return ()
        found.append(rows[0])
    return tuple(found)


def pins_for(spec: ModelSpec) -> dict[str, str]:
    """SHA-256 for each catalog path. The MDL manifest wins when it lists the path."""
    catalog = {path: digest.lower() for path, digest in spec.files.items()}
    if not catalog:
        return {}
    try:
        manifest = load_manifest(MANIFEST)
    except (CatalogError, OSError, ValueError):
        return catalog
    listed = next((model for model in manifest.models if model.id == spec.id and model.files), None)
    if listed is None:
        return catalog
    published = {item.path: item.sha256.lower() for item in listed.files}
    return {path: published.get(path, digest) for path, digest in catalog.items()}


def mdl_adopt_command(legacy: Path) -> list[str]:
    """The voxprint-common 0.5.1 adopt command for one legacy folder."""
    return [
        sys.executable,
        "-m",
        "voxprint_common.mdl",
        "adopt",
        "--app",
        "canvas",
        "--from",
        os.fspath(legacy),
    ]


def adopt_cli(spec: ModelSpec, legacy: Path, root: Path) -> AdoptResult:
    """TODO: remove in rc.29, use voxprint-mdl adopt."""
    return _local_adopt(spec, legacy, root)


def adopt_required(root: Path, catalog: Catalog | None = None) -> tuple[AdoptResult, ...]:
    """Adopt each required model that has no complete view and a legacy folder.

    A model that already has a view is left alone. One failure does not stop
    the others. The legacy folder is never removed.
    """
    results: list[AdoptResult] = []
    for spec in required_specs(catalog):
        key = (os.fspath(root.resolve()), spec.id)
        if key in _done:
            continue
        if view_complete(spec, root):
            _done.add(key)
            continue
        legacy = weight_dir(root, spec.repo)
        if not legacy.is_dir():
            _done.add(key)
            continue
        started = time.perf_counter()
        try:
            result = adopt_cli(spec, legacy, root)
        except (OSError, RuntimeError, SnapshotError, ViewError):
            _log(spec.id, 0, 0, 0, _ms(started))
            continue
        _log(spec.id, result.files, result.ok, result.fetched, _ms(started))
        _done.add(key)
        results.append(result)
    return tuple(results)


def _mdl_adopt(spec: ModelSpec, legacy: Path, root: Path) -> AdoptResult:
    """Run ``voxprint-mdl adopt`` for ``legacy``. rc.29 makes this the one-line body of :func:`adopt_cli`."""
    del root
    completed = subprocess.run(mdl_adopt_command(legacy), check=False)
    if completed.returncode != 0:
        raise SnapshotError("voxprint-mdl adopt failed")
    return AdoptResult(spec.id, files=0, ok=0, fetched=0)


def _local_adopt(spec: ModelSpec, legacy: Path, root: Path) -> AdoptResult:
    """Hard-link files whose bytes match the pin. Leave ``legacy`` in place."""
    pins = pins_for(spec)
    ok = 0
    need: list[str] = []
    for path, digest in pins.items():
        blob = blob_path(root, digest)
        if _matches(blob, digest):
            ok += 1
            continue
        source = legacy / path
        if _matches(source, digest):
            try:
                blob.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, blob)
            except OSError:
                need.append(path)
                continue
            ok += 1
            continue
        need.append(path)
    fetched = _fetch(spec, root, tuple(need)) if need else 0
    if pins and all(_matches(blob_path(root, digest), digest) for digest in pins.values()):
        ensure_view(root, spec.id, pins)
    return AdoptResult(spec.id, len(pins), ok, fetched, tuple(need))


def view_complete(spec: ModelSpec, root: Path) -> bool:
    """True when every pin is a blob and a view can be built. The bytes are not hashed."""
    pins = pins_for(spec)
    if not pins or not blobs_present(root, pins):
        return False
    try:
        return ensure_view(root, spec.id, pins) is not None
    except ViewError:
        return False


def _fetch(spec: ModelSpec, root: Path, names: tuple[str, ...]) -> int:
    """Fetch ``names`` into the blob store through MDL. Tests replace this."""
    if not names:
        return 0
    wanted = set(names)
    planned = [item for item in plan_blobs((spec,), root) if item.name in wanted]
    if len(planned) != len(wanted):
        raise SnapshotError("a file is not pinned in mdl.json")
    _download(planned)
    return len(planned)


def _matches(path: Path, digest: str) -> bool:
    if not path.is_file():
        return False
    try:
        verify_file(path, digest)
    except (OSError, PinError):
        return False
    return True


def _log(model_id: str, files: int, ok: int, fetched: int, ms: int) -> None:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in model_id)[:80]
    log_detail("model", "adopt", "cpu", f"id={safe} files={files} ok={ok} fetched={fetched} ms={ms}")


def _ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))
