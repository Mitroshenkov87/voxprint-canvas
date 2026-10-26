"""Install, offer, and repair the shared model packs.

The default pack is Z-Image-Turbo plus the models that ship with it. Optional
packs stay out of the setup and are offered, with a size and an estimate,
when a picture needs them.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from canvas.app.i18n import plural_form, text
from canvas.infra.manifest import (
    APP_ID,
    Component,
    ManifestError,
    load_manifest,
    lock_sha256,
    register_component,
)
from canvas.infra.paths import models_dir, shared_dir
from canvas.infra.runtime import (
    TRT_ID,
    TRT_PATH,
    python_supports_tensorrt,
    refill_wheels,
    restore_runtime,
    restore_tensorrt,
)
from canvas.models.adopt import adopt_required
from canvas.models.features import withheld_from_setup
from canvas.models.loader import plan_specs
from canvas.models.plan import CallPlan
from canvas.models.registry import Catalog, ModelSpec, RegistryError
from canvas.models.views import is_view
from canvas.models.snapshots import (
    SnapshotError,
    snapshot_ready,
    snapshot_repo,
    weight_dir,
)
from canvas.models.verified import VerifiedCache

_RATE = 8 * 1024 * 1024
_SAFETY_ROLES = ("prompt-guard", "zero-shot", "face-detector", "age", "nsfw", "age-fine")
OPTIONAL_IDS = (
    "z-image-turbo-fun-controlnet-union-bf16-lite",
    "z-image-turbo-fun-controlnet-union-bf16",
    "z-image-creative-bf16",
)


class DownloadDeclined(Exception):
    """The user left an optional pack uninstalled."""

    def __init__(self) -> None:
        super().__init__(text("error.cancelled"))

    def __str__(self) -> str:
        return text("error.cancelled")


def default_specs(catalog: Catalog) -> tuple[ModelSpec, ...]:
    """The models installed with the default pack, then Z-Image-Turbo."""
    specs: list[ModelSpec] = []
    for role in _SAFETY_ROLES:
        found = catalog.by_role(role)
        if len(found) != 1:
            raise RegistryError("a required model is missing")
        specs.append(found[0])
    specs.append(catalog.by_id("z-image-turbo-bf16-16gb"))
    return tuple(specs)


def is_optional(spec: ModelSpec) -> bool:
    """True for a pack the setup does not install."""
    return spec.id in OPTIONAL_IDS


def pack_title(spec: ModelSpec) -> str:
    """A short name safe to show before a download."""
    if spec.id.endswith("-lite"):
        return text("pack.controlnet_lite")
    if spec.role == "control":
        return text("pack.controlnet")
    if spec.id == "z-image-creative-bf16":
        return text("pack.creative")
    if spec.repo.endswith("Z-Image-Turbo"):
        return text("pack.turbo")
    return spec.id


def pack_notice(spec: ModelSpec) -> str:
    """Size and estimated time, before the download starts."""
    total = spec.nbytes
    seconds = 1 if total <= 0 else max(1, (total + _RATE - 1) // _RATE)
    return text("download.notice", title=pack_title(spec), size=format_bytes(total), duration=format_duration(seconds))


def format_bytes(nbytes: int) -> str:
    """A short size, in GB, MB, KB, or bytes."""
    if nbytes >= 1_000_000_000:
        return text("unit.gb", value=f"{nbytes / 1_000_000_000:.1f}")
    if nbytes >= 1_000_000:
        return text("unit.mb", value=f"{nbytes / 1_000_000:.1f}")
    if nbytes >= 1_000:
        return text("unit.kb", value=f"{nbytes / 1_000:.1f}")
    return text("unit.b", value=nbytes)


def format_duration(seconds: int) -> str:
    """A short estimate in minutes or seconds."""
    if seconds >= 60:
        count = max(1, round(seconds / 60))
        kind = "minute"
    else:
        count = seconds
        kind = "second"
    unit = text(f"unit.{kind}.{plural_form(count)}")
    return text("unit.span", count=count, unit=unit)


def install_supporting(catalog: Catalog, progress: Callable[[str], None]) -> None:
    """Record supporting packs that are already on disk. VOXPrint MDL fetches weights."""
    del progress
    for spec in default_specs(catalog):
        if spec.role == "generator":
            continue
        install_spec(spec, lambda _message: None, confirm=None)


def install_defaults(catalog: Catalog, progress: Callable[[str], None]) -> None:
    """Record default packs that are already on disk. VOXPrint MDL fetches weights."""
    del progress
    adopt_required(models_dir(), catalog)
    for spec in default_specs(catalog):
        install_spec(spec, lambda _message: None, confirm=None)


def install_spec(
    spec: ModelSpec,
    progress: Callable[[str], None],
    confirm: Callable[[str], bool] | None,
) -> None:
    """Record one pack when it is already on disk. Canvas does not download it."""
    del progress, confirm
    if withheld_from_setup(spec, catalog=None):
        return
    root = models_dir()
    try:
        folder = snapshot_ready(spec, root)
    except SnapshotError:
        return
    if folder is None or is_view(root, folder):
        # A view's blobs are already refcounted by VOXPrint MDL.
        return
    remember(spec)


def ensure_plan(
    catalog: Catalog,
    plan: CallPlan,
    progress: Callable[[str], None],
    confirm: Callable[[str], bool] | None,
) -> None:
    """Record packs that are already on disk. VOXPrint MDL fetches anything missing."""
    for spec in plan_specs(catalog, plan):
        install_spec(spec, progress, confirm)


def repair(catalog: Catalog, progress: Callable[[str], None]) -> list[str]:
    """Replace missing or corrupt shared files and return the component ids fixed.

    A file whose recorded size and time still match is not hashed again. Nothing
    is downloaded when every recorded file is present and unchanged.
    """
    shared = shared_dir()
    try:
        components = load_manifest(shared)
    except ManifestError as exc:
        if exc.args and exc.args[0] == "manifest.lock timed out":
            progress(text("startup.lock"))
        else:
            progress(str(exc))
        return []
    cache = VerifiedCache.open(False)
    fixed: list[str] = []
    try:
        for component_id, component in components.items():
            if not _needs_repair(shared, component_id, component, catalog, cache):
                continue
            progress(text("splash.repairing", name=component_id))
            _repair_one(catalog, component_id, component, progress)
            fixed.append(component_id)
    finally:
        cache.flush()
    return fixed


def remember(spec: ModelSpec, app: str = APP_ID) -> None:
    """Record a pack that is already on disk."""
    folder = weight_dir(models_dir(), snapshot_repo(spec))
    try:
        relative = _relative_pack(folder)
    except SnapshotError:
        return
    digest = lock_sha256(dict(spec.files))
    register_component(
        shared_dir(),
        f"model-{digest[:12]}",
        kind="model",
        version=spec.id,
        path=relative,
        sha256=digest,
        app=app,
    )


def record_runtime(app: str = APP_ID) -> None:
    """The shared runtime root is not registered. Refs are per model only."""
    del app


def record_wheels(app: str = APP_ID) -> None:
    """Wheel and runtime roots are not registered. Refs are per model only."""
    del app


def record_tensorrt(app: str = APP_ID) -> None:
    """The TensorRT directory is not registered. Refs are per model only."""
    del app


def _repair_one(
    catalog: Catalog, component_id: str, component: Component, progress: Callable[[str], None],
) -> None:
    kind = component.kind
    if kind == "model":
        progress(text("splash.leaving", name=component_id))
        return
    if kind == "wheels":
        refill_wheels(progress)
        record_wheels()
        return
    if kind == "runtime" and (component_id == TRT_ID or component.path == TRT_PATH):
        restore_tensorrt(progress)
        record_tensorrt()
        return
    if kind == "runtime":
        if any(ref != APP_ID for ref in component.refs):
            progress(text("splash.leaving", name=component_id))
            return
        restore_runtime(progress)
        record_runtime()
        return
    if kind == "tool":
        return
    progress(text("splash.leaving", name=component_id))


def _needs_repair(
    shared: Path, component_id: str, component: Component, catalog: Catalog, cache: VerifiedCache,
) -> bool:
    if APP_ID not in component.refs:
        return False
    if component.kind == "runtime" and (component_id == TRT_ID or component.path == TRT_PATH):
        if not python_supports_tensorrt():
            return False
    if component.kind == "model":
        return False
    if component.kind in ("wheels", "runtime"):
        return not (shared / component.path).exists()
    return False


def _spec_for(catalog: Catalog, component_id: str, component: Component) -> ModelSpec | None:
    for key in (component_id, component.version):
        try:
            return catalog.by_id(key)
        except RegistryError:
            continue
    for spec in catalog.models:
        try:
            if _relative_pack(weight_dir(models_dir(), snapshot_repo(spec))) == component.path:
                return spec
        except SnapshotError:
            continue
    return None


def _relative_pack(folder: Path) -> str:
    """The manifest path: shared-relative, or absolute inside ``models_dir()``."""
    shared = shared_dir().resolve()
    resolved = folder.resolve()
    try:
        return resolved.relative_to(shared).as_posix()
    except ValueError:
        pass
    models = models_dir().resolve()
    if resolved != models and models in resolved.parents:
        return resolved.as_posix()
    raise SnapshotError("the model folder is outside the models directory")
