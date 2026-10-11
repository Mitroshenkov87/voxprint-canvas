"""Install, offer, and repair the shared model packs.

The default pack is Z-Image-Turbo plus the models that ship with it. Optional
packs stay out of the setup and are offered, with a size and an estimate,
when a picture needs them.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from canvas.infra.manifest import (
    APP_ID,
    ManifestError,
    Resource,
    broken_files,
    load_manifest,
    register_resource,
)
from canvas.infra.paths import models_dir, shared_dir
from canvas.infra.runtime import (
    WHEEL_SOURCE,
    file_sha256,
    hash_tree,
    interpreter_marker,
    python_home,
    refill_wheels,
    restore_runtime,
    wheel_dir,
)
from canvas.models.loader import plan_specs
from canvas.models.plan import CallPlan
from canvas.models.registry import Catalog, ModelSpec, RegistryError
from canvas.models.snapshots import SnapshotError, download_snapshot, snapshot_ready, weight_dir

_RATE = 8 * 1024 * 1024
_SAFETY_ROLES = ("prompt-guard", "zero-shot", "face-detector", "age", "nsfw")
OPTIONAL_IDS = (
    "z-image-turbo-fun-controlnet-union-bf16-lite",
    "z-image-turbo-fun-controlnet-union-bf16",
    "z-image-creative-bf16",
    "qwen-image-edit-2511-fp8",
)


class DownloadDeclined(Exception):
    """The user left an optional pack uninstalled."""

    def __init__(self) -> None:
        super().__init__("The download was cancelled.")


def default_specs(catalog: Catalog) -> tuple[ModelSpec, ...]:
    """Z-Image-Turbo, once, and the models installed with it."""
    specs = [catalog.by_id("z-image-turbo-bf16-16gb")]
    for role in _SAFETY_ROLES:
        found = catalog.by_role(role)
        if len(found) != 1:
            raise RegistryError("a required model is missing")
        specs.append(found[0])
    return tuple(specs)


def is_optional(spec: ModelSpec) -> bool:
    """True for a pack the setup does not install."""
    return spec.id in OPTIONAL_IDS


def pack_title(spec: ModelSpec) -> str:
    """A short name safe to show before a download."""
    if spec.id.endswith("-lite"):
        return "ControlNet lite"
    if spec.role == "control":
        return "ControlNet"
    if spec.id == "z-image-creative-bf16":
        return "Creative Z-Image"
    if spec.id == "qwen-image-edit-2511-fp8":
        return "Qwen-Image-Edit-2511"
    if spec.repo.endswith("Z-Image-Turbo"):
        return "Z-Image-Turbo"
    return spec.id


def pack_notice(spec: ModelSpec) -> str:
    """Size and estimated time, before the download starts."""
    total = spec.nbytes
    seconds = 1 if total <= 0 else max(1, (total + _RATE - 1) // _RATE)
    return f"{pack_title(spec)} is {format_bytes(total)}. Estimated time {format_duration(seconds)}."


def format_bytes(nbytes: int) -> str:
    """A short size, in GB, MB, KB, or bytes."""
    if nbytes >= 1_000_000_000:
        return f"{nbytes / 1_000_000_000:.1f} GB"
    if nbytes >= 1_000_000:
        return f"{nbytes / 1_000_000:.1f} MB"
    if nbytes >= 1_000:
        return f"{nbytes / 1_000:.1f} KB"
    return f"{nbytes} B"


def format_duration(seconds: int) -> str:
    """A short estimate in minutes or seconds."""
    if seconds >= 60:
        minutes = max(1, round(seconds / 60))
        unit = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {unit}"
    unit = "second" if seconds == 1 else "seconds"
    return f"{seconds} {unit}"


def install_supporting(catalog: Catalog, progress: Callable[[str], None]) -> None:
    """Download the models that ship with the default pack, other than the generator."""
    for spec in default_specs(catalog):
        if spec.role == "generator":
            continue
        install_spec(spec, progress, confirm=None)


def install_defaults(catalog: Catalog, progress: Callable[[str], None]) -> None:
    """Download the default packs during setup."""
    for spec in default_specs(catalog):
        progress(f"Installing {pack_title(spec)}")
        install_spec(spec, progress, confirm=None)


def install_spec(
    spec: ModelSpec,
    progress: Callable[[str], None],
    confirm: Callable[[str], bool] | None,
) -> None:
    """Download one pack when it is missing, then record it."""
    root = models_dir()
    if snapshot_ready(spec, root) is None:
        if is_optional(spec):
            notice = pack_notice(spec)
            progress(notice)
            if confirm is not None and not confirm(notice):
                raise DownloadDeclined()
        download_snapshot(spec, root, progress=progress)
    remember(spec)


def ensure_plan(
    catalog: Catalog,
    plan: CallPlan,
    progress: Callable[[str], None],
    confirm: Callable[[str], bool] | None,
) -> None:
    """Download the packs one plan needs. Optional packs ask first."""
    for spec in plan_specs(catalog, plan):
        install_spec(spec, progress, confirm)


def repair(catalog: Catalog, progress: Callable[[str], None]) -> list[str]:
    """Replace missing or corrupt shared files and return the resource ids fixed."""
    shared = shared_dir()
    try:
        resources = load_manifest(shared)
    except ManifestError as exc:
        progress(str(exc))
        return []
    fixed: list[str] = []
    for resource_id, resource in resources.items():
        if not broken_files(shared, resource):
            continue
        progress(f"Repairing {resource_id}")
        _repair_one(catalog, resource_id, resource, progress)
        fixed.append(resource_id)
    return fixed


def remember(spec: ModelSpec, app: str = APP_ID) -> None:
    """Record a pack that is already on disk."""
    folder = weight_dir(models_dir(), spec.repo)
    try:
        relative = _relative_pack(folder)
    except SnapshotError:
        return
    register_resource(
        shared_dir(),
        spec.id,
        kind="pack",
        path=relative,
        nbytes=spec.nbytes,
        files=dict(spec.files),
        app=app,
        source="",
    )


def record_runtime(app: str = APP_ID) -> None:
    """Record the suite interpreter when the setup installed one."""
    marker = interpreter_marker()
    if marker is None:
        return
    home = python_home()
    relative = marker.relative_to(home).as_posix()
    register_resource(
        shared_dir(),
        "python",
        kind="runtime",
        path="python",
        nbytes=marker.stat().st_size,
        files={relative: file_sha256(marker)},
        app=app,
        source="python",
    )


def record_wheels(app: str = APP_ID) -> None:
    """Record the cached CUDA wheels when that folder has files."""
    files = hash_tree(wheel_dir())
    if not files:
        return
    total = sum((wheel_dir() / name).stat().st_size for name in files)
    register_resource(
        shared_dir(),
        "torch-cu128",
        kind="wheels",
        path="wheels",
        nbytes=total,
        files=files,
        app=app,
        source=WHEEL_SOURCE,
    )


def _repair_one(
    catalog: Catalog, resource_id: str, resource: Resource, progress: Callable[[str], None],
) -> None:
    kind = resource.kind
    if kind == "pack":
        try:
            spec = catalog.by_id(resource_id)
        except RegistryError:
            progress(f"Leaving {resource_id} unchanged")
            return
        download_snapshot(spec, models_dir(), progress=progress)
        remember(spec)
        return
    if kind == "wheels":
        refill_wheels(progress)
        record_wheels()
        return
    if kind == "runtime":
        restore_runtime(progress)
        record_runtime()
        return
    if kind == "suite":
        return
    progress(f"Leaving {resource_id} unchanged")


def _relative_pack(folder: Path) -> str:
    shared = shared_dir().resolve()
    try:
        return folder.resolve().relative_to(shared).as_posix()
    except ValueError as exc:
        raise SnapshotError("the model folder is outside the shared directory") from exc
