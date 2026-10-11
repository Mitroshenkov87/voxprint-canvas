"""Install, offer, and repair the shared model packs.

The default pack is Z-Image-Turbo plus the models that ship with it. Optional
packs stay out of the setup and are offered, with a size and an estimate,
when a picture needs them.
"""
from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from pathlib import Path

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
from canvas.models.edit_route import FP8_EDIT_ID, set_edit_route_override
from canvas.models.loader import plan_specs
from canvas.models.nunchaku import (
    NUNCHAKU_COMPONENT_ID,
    NunchakuUnavailable,
    NunchakuWheel,
    install_runtime_files,
    runtime_home,
    runtime_is_installed,
    select_nunchaku_wheel,
)
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
    "qwen-image-edit-2511-fp8",
    "qwen-image-edit-2511-int4-r256",
    "qwen-image-edit-2511-int4-r128",
    "qwen-image-edit-2511-int4-r256-mirror",
)


class DownloadDeclined(Exception):
    """The user left an optional pack uninstalled."""

    def __init__(self) -> None:
        super().__init__("The download was cancelled.")


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
        return "ControlNet lite"
    if spec.role == "control":
        return "ControlNet"
    if spec.id == "z-image-creative-bf16":
        return "Creative Z-Image"
    if spec.id == "qwen-image-edit-2511-fp8":
        return "Qwen-Image-Edit-2511"
    if spec.quant == "int4" and spec.role == "edit":
        return "Qwen-Image-Edit-2511 INT4"
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


def nunchaku_runtime_notice() -> str:
    """Size and estimated time for the private INT4 runtime on this PC."""
    plan = select_nunchaku_wheel(sys.platform, platform.machine())
    total = plan.nbytes
    seconds = 1 if total <= 0 else max(1, (total + _RATE - 1) // _RATE)
    return f"Nunchaku INT4 runtime is {format_bytes(total)}. Estimated time {format_duration(seconds)}."


def ensure_nunchaku_runtime(
    progress: Callable[[str], None],
    confirm: Callable[[str], bool] | None,
) -> None:
    """Download the INT4 runtime when it is missing. The notice is the size and the estimate."""
    plan = select_nunchaku_wheel(sys.platform, platform.machine())
    if runtime_is_installed(plan):
        return
    notice = nunchaku_runtime_notice()
    progress(notice)
    if confirm is not None and not confirm(notice):
        raise DownloadDeclined()
    install_runtime_files(plan, runtime_home(), progress)
    _record_nunchaku(plan)


def ensure_plan(
    catalog: Catalog,
    plan: CallPlan,
    progress: Callable[[str], None],
    confirm: Callable[[str], bool] | None,
) -> None:
    """Download the packs one plan needs. Optional packs ask first."""
    if plan.mode == "instruction" and plan.quant == "int4":
        _ensure_int4_edit(catalog, plan, progress, confirm)
        return
    for spec in plan_specs(catalog, plan):
        install_spec(spec, progress, confirm)


def _ensure_int4_edit(
    catalog: Catalog,
    plan: CallPlan,
    progress: Callable[[str], None],
    confirm: Callable[[str], bool] | None,
) -> None:
    """Offer the INT4 runtime and weights. A declined download installs the FP8 pack."""
    try:
        ensure_nunchaku_runtime(progress, confirm)
        install_spec(catalog.by_id(plan.model_id), progress, confirm)
    except DownloadDeclined:
        set_edit_route_override("fp8")
        install_spec(catalog.by_id(FP8_EDIT_ID), progress, confirm)


def _record_nunchaku(plan: NunchakuWheel) -> None:
    register_component(
        shared_dir(),
        NUNCHAKU_COMPONENT_ID,
        kind="runtime",
        version=plan.file("nunchaku").filename,
        path="runtime/nunchaku-side",
        sha256=plan.file("nunchaku").sha256,
        app=APP_ID,
    )


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
            progress(f"Repairing {component_id}")
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
            progress(f"Leaving {component_id} unchanged")
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
    if kind == "runtime" and component_id == NUNCHAKU_COMPONENT_ID:
        try:
            wheel = select_nunchaku_wheel(sys.platform, platform.machine())
        except NunchakuUnavailable:
            progress(f"Leaving {component_id} unchanged")
            return
        install_runtime_files(wheel, runtime_home(), progress)
        _record_nunchaku(wheel)
        return
    if kind == "runtime":
        if any(ref != APP_ID for ref in component.refs):
            progress(f"Leaving {component_id} unchanged")
            return
        restore_runtime(progress)
        record_runtime()
        return
    if kind == "tool":
        return
    progress(f"Leaving {component_id} unchanged")


def _needs_repair(
    shared: Path, component_id: str, component: Component, catalog: Catalog, cache: VerifiedCache,
) -> bool:
    if APP_ID not in component.refs:
        return False
    if component.kind == "runtime" and (component_id == TRT_ID or component.path == TRT_PATH):
        if not python_supports_tensorrt():
            return False
    if component.kind == "runtime" and component_id == NUNCHAKU_COMPONENT_ID:
        try:
            wheel = select_nunchaku_wheel(sys.platform, platform.machine())
        except NunchakuUnavailable:
            return False
        return not runtime_is_installed(wheel)
    if component.kind == "model":
        spec = _spec_for(catalog, component_id, component)
        if spec is None:
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
