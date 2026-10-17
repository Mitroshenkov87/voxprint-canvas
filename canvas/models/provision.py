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
    ORT_ID,
    ORT_VERSION,
    PYTHON_ID,
    PYTHON_VERSION,
    RUNTIME_REL,
    TORCH_ID,
    TORCH_VERSION,
    TRT_ID,
    TRT_PATH,
    TRT_VERSION,
    file_sha256,
    hash_tree,
    interpreter_marker,
    python_supports_tensorrt,
    refill_wheels,
    restore_runtime,
    restore_tensorrt,
    wheel_dir,
)
from canvas.models.features import withheld_from_setup
from canvas.models.loader import plan_specs
from canvas.models.plan import CallPlan
from canvas.models.registry import Catalog, ModelSpec, RegistryError
from canvas.models.snapshots import (
    SnapshotError,
    download_snapshot,
    snapshot_ready,
    snapshot_repo,
    snapshot_status,
    weight_dir,
)
from canvas.models.verified import VerifiedCache

_RATE = 8 * 1024 * 1024
_SAFETY_ROLES = ("prompt-guard", "zero-shot", "face-detector", "age", "nsfw")
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
    """Download the models that ship with the default pack, other than the generator."""
    for spec in default_specs(catalog):
        if spec.role == "generator":
            continue
        install_spec(spec, progress, confirm=None)


def install_defaults(catalog: Catalog, progress: Callable[[str], None]) -> None:
    """Download the default packs during setup."""
    for spec in default_specs(catalog):
        progress(text("splash.installing", name=pack_title(spec)))
        install_spec(spec, progress, confirm=None)


def install_spec(
    spec: ModelSpec,
    progress: Callable[[str], None],
    confirm: Callable[[str], bool] | None,
) -> None:
    """Download one pack when it is missing, then record it."""
    if withheld_from_setup(spec, catalog=None):
        return
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
    """Replace missing or corrupt shared files and return the component ids fixed.

    A file whose recorded size and time still match is not hashed again. Nothing
    is downloaded when every recorded file is present and unchanged.
    """
    shared = shared_dir()
    try:
        components = load_manifest(shared)
    except ManifestError as exc:
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
    """Record the suite interpreter when the setup installed one."""
    marker = interpreter_marker()
    if marker is None:
        return
    register_component(
        shared_dir(),
        PYTHON_ID,
        kind="runtime",
        version=PYTHON_VERSION,
        path=RUNTIME_REL,
        sha256=file_sha256(marker),
        app=app,
    )


def record_wheels(app: str = APP_ID) -> None:
    """Record torch, the GPU runtime, and TensorRT when those files are present."""
    files = hash_tree(wheel_dir())
    torch_files = {
        name: digest for name, digest in files.items() if _wheel_name(name).startswith(("torch-", "torchvision-"))
    }
    if torch_files:
        register_component(
            shared_dir(),
            TORCH_ID,
            kind="wheels",
            version=TORCH_VERSION,
            path=RUNTIME_REL if interpreter_marker() is not None else "wheels",
            sha256=lock_sha256(torch_files),
            app=app,
        )
    ort_files = {name: digest for name, digest in files.items() if "onnxruntime_gpu" in _wheel_name(name)}
    if ort_files:
        register_component(
            shared_dir(),
            ORT_ID,
            kind="wheels",
            version=ORT_VERSION,
            path=RUNTIME_REL if interpreter_marker() is not None else "wheels",
            sha256=lock_sha256(ort_files),
            app=app,
        )
    record_tensorrt(app)


def record_tensorrt(app: str = APP_ID) -> None:
    """Record TensorRT 10 when its component directory has files."""
    folder = shared_dir() / TRT_PATH
    files = hash_tree(folder)
    if not files:
        return
    register_component(
        shared_dir(),
        TRT_ID,
        kind="runtime",
        version=TRT_VERSION,
        path=TRT_PATH,
        sha256=lock_sha256(files),
        app=app,
    )


def _repair_one(
    catalog: Catalog, component_id: str, component: Component, progress: Callable[[str], None],
) -> None:
    kind = component.kind
    if kind == "model":
        spec = _spec_for(catalog, component_id, component)
        if spec is None:
            progress(text("splash.leaving", name=component_id))
            return
        download_snapshot(spec, models_dir(), progress=progress)
        remember(spec)
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
        spec = _spec_for(catalog, component_id, component)
        if spec is None or withheld_from_setup(spec, catalog):
            return False
        return snapshot_status(spec, models_dir(), cache) != "ready"
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


def _wheel_name(name: str) -> str:
    return Path(name).name


def _relative_pack(folder: Path) -> str:
    shared = shared_dir().resolve()
    try:
        return folder.resolve().relative_to(shared).as_posix()
    except ValueError as exc:
        raise SnapshotError("the model folder is outside the shared directory") from exc
