"""Open a generator and the check stack from the registry.

A missing pin, a missing file or a bad hash refuses the stack. Generation runs
when the pinned snapshot is already on disk. Downloading is a separate call.
"""
from pathlib import Path
from typing import assert_never

from canvas.app.i18n import text
from canvas.infra.paths import models_dir
from canvas.models.modes import Mode, Tier
from canvas.models.pin import PinError, require_pin, verify_file
from canvas.models.plan import CallPlan, build_plan
from canvas.models.registry import Catalog, ModelSpec, RegistryError, load_catalog
from canvas.models.runtime import WeightsLoadError, render
from canvas.models.snapshots import SnapshotError, fallback_ready, snapshot_ready, weight_dir
from canvas.pipeline.bf16_schedule import (
    EmbeddingCache,
    GpuLedger,
    group_offload_cost,
    run_bf16_16gb,
    uses_bf16_16gb,
)
from canvas.pipeline.schedule import Placement, choose_schedule
from canvas.pipeline.vram_probe import PeakVramProbe
from canvas.safety.adapters import FairFaceAge, NsfwViT, Qwen3GuardGen, SiglipZeroShot, YuNetFaces
from canvas.safety.gate import SafetyStack, SafetyUnavailable
from canvas.safety.image import RawImage, ScreenedInput

_SAFETY_ROLES = ("prompt-guard", "zero-shot", "face-detector", "age", "nsfw")


class ModelNotReady(Exception):
    """The selected pipeline is known, and its weights are not runnable yet."""

    def __init__(self, plan: CallPlan) -> None:
        super().__init__(text("gate.unavailable"))
        self.plan = plan

    def __str__(self) -> str:
        return text("gate.unavailable")


def _weight_path(root: Path, spec: ModelSpec, name: str) -> Path:
    return root / spec.repo.replace("/", "--") / name


def ensure_weights(spec: ModelSpec, root: Path, folder: Path | None = None) -> None:
    """Refuse when any file is unpinned, missing or hashed wrong.

    ``folder`` is the model folder to check (a view over the MDL blobs or the
    legacy repo folder); by default the legacy folder under ``root``.
    """
    if not spec.files:
        raise SafetyUnavailable("missing_model")
    for name, digest in spec.files.items():
        try:
            require_pin(digest)
        except PinError as exc:
            raise SafetyUnavailable("missing_model") from exc
        path = (folder / name) if folder is not None else _weight_path(root, spec, name)
        if not path.is_file():
            raise SafetyUnavailable("missing_model")
        try:
            verify_file(path, digest)
        except PinError as exc:
            raise SafetyUnavailable("hash_mismatch") from exc


def load_safety_stack(catalog: Catalog, root: Path) -> SafetyStack:
    """Open the check stack after every weight file is verified."""
    folders: dict[str, Path] = {}
    for role in _SAFETY_ROLES:
        found = catalog.by_role(role)
        if len(found) != 1:
            raise SafetyUnavailable("missing_model")
        folder = _safety_folder(found[0], root)
        ensure_weights(found[0], root, folder)
        folders[role] = folder
    return SafetyStack(
        Qwen3GuardGen(folders["prompt-guard"]),
        SiglipZeroShot(folders["zero-shot"]),
        YuNetFaces(folders["face-detector"]),
        FairFaceAge(folders["age"]),
        NsfwViT(folders["nsfw"]),
    )


def _safety_folder(spec: ModelSpec, root: Path) -> Path:
    """The view over the MDL blobs when it is complete, else the legacy repo folder.

    Either way ``ensure_weights`` checks every file against its pin before use, and
    a view that cannot be built refuses the stack (fail closed).
    """
    try:
        folder = snapshot_ready(spec, root)
    except SnapshotError as exc:
        raise SafetyUnavailable("missing_model") from exc
    return folder if folder is not None else weight_dir(root, spec.repo)


def load_catalog_stack(root: Path, catalog: Catalog | None = None) -> SafetyStack:
    """Open the check stack from the registry after every weight file is verified."""
    return load_safety_stack(catalog or load_catalog(), root)


def request_fp32(spec: ModelSpec) -> bool:
    """Queue the FP32 ``<id>-upstream`` entry with VOXPrint MDL. False when it cannot be queued."""
    if not spec.fallback_files:
        return False
    # Resolved at call time: a missing MDL client returns False, and tests patch enqueue.
    try:
        from voxprint_common.mdl import enqueue
        from voxprint_common.mdl.catalog import CatalogError

        from canvas.models.mdl_registry import manifest_path
    except ImportError:
        return False
    try:
        enqueue("canvas", [f"{spec.id}-upstream"], manifests=[manifest_path()])
    except (CatalogError, OSError, ValueError):
        return False
    return True


class ZImageGenerator:
    """Diffusers generator. ``generate`` accepts only a ``ScreenedInput``."""

    def __init__(self, catalog: Catalog, tier: Tier, ledger: GpuLedger | None = None) -> None:
        self.catalog = catalog
        self.tier = tier
        self.last_plan: CallPlan | None = None
        self.last_size: tuple[int, int] | None = None
        self.unloaded = False
        self.ledger = ledger if ledger is not None else GpuLedger(16.0)
        self.embeddings = EmbeddingCache()
        self.probe: PeakVramProbe | None = None
        self.seed = 0
        self.img2img_strength: float | None = None
        self.models: Path | None = None

    def attach_probe(self, probe: PeakVramProbe | None) -> None:
        """Stage marks land on this probe. The pipeline calls it before ``generate``."""
        self.probe = probe

    def unload(self) -> None:
        """Drop GPU weights. Encoder embeddings stay, so a later seed can skip the encoder."""
        self.unloaded = True
        self.ledger.drop_all()

    def generate(self, screened: ScreenedInput, width: int | None = None, height: int | None = None) -> RawImage:
        """Run the plan when the pinned snapshot is on disk. A 16 GB BF16 job schedules memory first."""
        if type(screened) is not ScreenedInput:
            raise TypeError("the generator only accepts a ScreenedInput")
        if width is not None and height is not None:
            self.last_size = (width, height)
        plan = build_plan(self.catalog, screened, self.tier)
        self.last_plan = plan
        mode = Mode(plan.mode)
        self._schedule(plan, mode, 1024 if width is None else width, 1024 if height is None else height, screened.prompt)
        root = self.models if self.models is not None else models_dir()
        spec = _spec_for(self.catalog, plan)
        folder = snapshot_ready(spec, root)
        if folder is None or width is None or height is None:
            raise ModelNotReady(plan)
        control_file = _control_file(self.catalog, plan, root)
        try:
            try:
                return render(
                    plan, folder, screened, width, height, self.tier,
                    seed=self.seed, control_file=control_file,
                    strength=self.img2img_strength,
                )
            except WeightsLoadError:
                # The BF16 weights did not open: use the author's FP32 -upstream entry.
                fp32 = fallback_ready(spec, root)
                if fp32 is None or fp32 == folder:
                    request_fp32(spec)
                    raise ModelNotReady(plan) from None
                return render(
                    plan, fp32, screened, width, height, self.tier,
                    seed=self.seed, control_file=control_file,
                    strength=self.img2img_strength,
                )
        except ImportError as exc:
            raise ModelNotReady(plan) from exc

    def _schedule(self, plan: CallPlan, mode: Mode, width: int, height: int, prompt: str) -> None:
        """Plan memory before weights move. 24 GB stays resident and does not mark stages."""
        if self.tier is Tier.GB24:
            return
        choice = choose_schedule(16.0, mode, 0.0)
        if choice.placement is Placement.RESIDENT:
            return
        if choice.placement is Placement.GROUP_OFFLOAD:
            if uses_bf16_16gb(self.tier.value, plan):
                run_bf16_16gb(
                    prompt=prompt,
                    mode=mode,
                    width=width,
                    height=height,
                    ledger=self.ledger,
                    cache=self.embeddings,
                    probe=self.probe,
                    cost=group_offload_cost(),
                )
            return
        assert_never(choice.placement)


def plan_specs(catalog: Catalog, plan: CallPlan) -> tuple[ModelSpec, ...]:
    """The generator snapshot for one plan, plus a control file when the plan needs one."""
    specs = [_spec_for(catalog, plan)]
    control = _control_spec(catalog, plan)
    if control is not None:
        specs.append(control)
    return tuple(specs)


def _spec_for(catalog: Catalog, plan: CallPlan) -> ModelSpec:
    if plan.model_id:
        try:
            return catalog.by_id(plan.model_id)
        except RegistryError:
            pass
    repo = plan.repos[0]
    ranked = [
        spec for spec in catalog.models
        if spec.repo == repo and plan.pipeline_class in spec.pipeline_classes
    ]
    for spec in ranked:
        if spec.tier == plan.tier:
            return spec
    if ranked:
        return ranked[0]
    for spec in catalog.models:
        if spec.repo == repo:
            return spec
    raise ModelNotReady(plan)


def _control_spec(catalog: Catalog, plan: CallPlan) -> ModelSpec | None:
    if len(plan.repos) < 2:
        return None
    repo = plan.repos[1]
    matches = [spec for spec in catalog.models if spec.repo == repo and spec.role == "control"]
    for spec in matches:
        if spec.tier == plan.tier:
            return spec
    if matches:
        return matches[0]
    return None


def _control_file(catalog: Catalog, plan: CallPlan, root: Path) -> Path | None:
    spec = _control_spec(catalog, plan)
    if spec is None:
        return None
    folder = snapshot_ready(spec, root)
    if folder is None:
        return None
    for name in spec.files:
        if name.endswith(".safetensors"):
            path = folder / name
            if path.is_file():
                return path
    return None
