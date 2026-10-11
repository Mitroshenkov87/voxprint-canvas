"""Open a generator and the check stack from the registry.

A missing pin, a missing file or a bad hash refuses the stack. Generation runs
when the pinned snapshot is already on disk. Downloading is a separate call.
"""
from pathlib import Path
from typing import assert_never

from canvas.infra.paths import models_dir
from canvas.models.modes import Mode, Tier
from canvas.models.nunchaku import NUNCHAKU_ID, load_nunchaku_generator, nunchaku_ready
from canvas.models.pin import PinError, require_pin, verify_file
from canvas.models.plan import CallPlan, build_plan
from canvas.models.registry import Catalog, ModelSpec, load_catalog
from canvas.models.runtime import render
from canvas.models.snapshots import download_snapshot, snapshot_ready, weight_dir
from canvas.pipeline.bf16_schedule import (
    EmbeddingCache,
    GpuLedger,
    group_offload_cost,
    run_bf16_16gb,
    uses_bf16_16gb,
)
from canvas.pipeline.schedule import (
    Placement,
    choose_schedule,
    run_edit_16gb,
    system_ram_gb,
)
from canvas.pipeline.vram_probe import PeakVramProbe
from canvas.safety.adapters import FairFaceAge, NsfwViT, Qwen3GuardGen, SiglipZeroShot, YuNetFaces
from canvas.safety.gate import SafetyStack, SafetyUnavailable
from canvas.safety.image import RawImage, ScreenedInput

_SAFETY_ROLES = ("prompt-guard", "zero-shot", "face-detector", "age", "nsfw")


class ModelNotReady(Exception):
    """The selected pipeline is known, and its weights are not runnable yet."""

    def __init__(self, plan: CallPlan) -> None:
        super().__init__("Image generation is unavailable.")
        self.plan = plan


def _weight_path(root: Path, spec: ModelSpec, name: str) -> Path:
    return root / spec.repo.replace("/", "--") / name


def ensure_weights(spec: ModelSpec, root: Path) -> None:
    """Refuse when any file is unpinned, missing or hashed wrong."""
    if not spec.files:
        raise SafetyUnavailable("missing_model")
    for name, digest in spec.files.items():
        try:
            require_pin(digest)
        except PinError as exc:
            raise SafetyUnavailable("missing_model") from exc
        path = _weight_path(root, spec, name)
        if not path.is_file():
            raise SafetyUnavailable("missing_model")
        try:
            verify_file(path, digest)
        except PinError as exc:
            raise SafetyUnavailable("hash_mismatch") from exc


def load_safety_stack(catalog: Catalog, root: Path) -> SafetyStack:
    """Open the check stack after every weight file is verified."""
    bound: dict[str, ModelSpec] = {}
    for role in _SAFETY_ROLES:
        found = catalog.by_role(role)
        if len(found) != 1:
            raise SafetyUnavailable("missing_model")
        ensure_weights(found[0], root)
        bound[role] = found[0]
    return SafetyStack(
        Qwen3GuardGen(weight_dir(root, bound["prompt-guard"].repo)),
        SiglipZeroShot(weight_dir(root, bound["zero-shot"].repo)),
        YuNetFaces(weight_dir(root, bound["face-detector"].repo)),
        FairFaceAge(weight_dir(root, bound["age"].repo)),
        NsfwViT(weight_dir(root, bound["nsfw"].repo)),
    )


def load_catalog_stack(root: Path, catalog: Catalog | None = None) -> SafetyStack:
    """Open the check stack from the registry after every weight file is verified."""
    return load_safety_stack(catalog or load_catalog(), root)


class ZImageGenerator:
    """Diffusers generator. ``generate`` accepts only a ``ScreenedInput``."""

    def __init__(self, catalog: Catalog, tier: Tier, ledger: GpuLedger | None = None,
                 ram_gb: float | None = None) -> None:
        self.catalog = catalog
        self.tier = tier
        self.last_plan: CallPlan | None = None
        self.last_size: tuple[int, int] | None = None
        self.unloaded = False
        self.ledger = ledger if ledger is not None else GpuLedger(16.0)
        self.embeddings = EmbeddingCache()
        self.probe: PeakVramProbe | None = None
        self.seed = 0
        self.models: Path | None = None
        self.ram_gb = ram_gb

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
            return render(
                plan, folder, screened, width, height, self.tier, seed=self.seed, control_file=control_file,
            )
        except ImportError as exc:
            raise ModelNotReady(plan) from exc

    def _schedule(self, plan: CallPlan, mode: Mode, width: int, height: int, prompt: str) -> None:
        """Plan memory before weights move. 24 GB text stays resident and does not mark stages."""
        if self.tier is Tier.GB24 and mode is not Mode.INSTRUCTION:
            return
        budget = 16.0 if self.tier is Tier.GB16 else 24.0
        ready = False
        if mode is not Mode.INSTRUCTION:
            ready = nunchaku_ready(self.catalog.by_id(NUNCHAKU_ID))
        ram = 0.0 if mode is not Mode.INSTRUCTION else self._ram()
        choice = choose_schedule(budget, mode, ram, nunchaku_ready=ready)
        if choice.placement is Placement.NUNCHAKU_INT4:
            load_nunchaku_generator(self.catalog.by_id(NUNCHAKU_ID))
            raise ModelNotReady(plan)
        if choice.placement is Placement.RESIDENT:
            return
        if choice.placement is Placement.GROUP_OFFLOAD:
            if mode is Mode.INSTRUCTION:
                run_edit_16gb(width=width, height=height, ledger=self.ledger, probe=self.probe)
                return
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

    def _ram(self) -> float:
        if self.ram_gb is not None:
            return float(self.ram_gb)
        return system_ram_gb()


def download_safety(catalog: Catalog, root: Path) -> None:
    """Download every content-check snapshot. A bad hash raises ``SnapshotError``."""
    for role in _SAFETY_ROLES:
        found = catalog.by_role(role)
        if len(found) != 1:
            raise SafetyUnavailable("missing_model")
        download_snapshot(found[0], root)


def download_generator(catalog: Catalog, plan: CallPlan, root: Path) -> None:
    """Download the generator snapshot for one plan, including a control file when the plan needs one."""
    for spec in plan_specs(catalog, plan):
        download_snapshot(spec, root)


def plan_specs(catalog: Catalog, plan: CallPlan) -> tuple[ModelSpec, ...]:
    """The generator snapshot for one plan, plus a control file when the plan needs one."""
    specs = [_spec_for(catalog, plan)]
    control = _control_spec(catalog, plan)
    if control is not None:
        specs.append(control)
    return tuple(specs)


def _spec_for(catalog: Catalog, plan: CallPlan) -> ModelSpec:
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
