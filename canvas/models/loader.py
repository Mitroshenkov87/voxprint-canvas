"""Open a generator and the check stack from the registry.

A missing pin, a missing file or a bad hash refuses the stack. Generation runs
when the pinned snapshot is already on disk. Downloading is a separate call.
"""
from pathlib import Path
from typing import NoReturn, assert_never

from canvas.app.i18n import text
from canvas.app.preflight import PreflightFailed, Reason, preflight
from canvas.infra.paths import models_dir
from canvas.models.modes import Mode, Tier
from canvas.models.pin import PinError, require_pin, verify_file
from canvas.models.plan import CallPlan, build_plan
from canvas.models.registry import Catalog, ModelSpec, RegistryError, load_catalog
from canvas.models.runtime import WeightsLoadError, render
from canvas.models.user_lora import UserLora, resolve_selected
from canvas.models.snapshots import SnapshotError, fallback_ready, request_download, snapshot_ready
from canvas.models.verified import VerifiedCache, flush_process_cache, process_cache
from canvas.pipeline.bf16_schedule import (
    EmbeddingCache,
    GpuLedger,
    group_offload_cost,
    run_bf16_16gb,
    uses_bf16_16gb,
)
from canvas.pipeline.schedule import Placement, choose_schedule
from canvas.pipeline.vram_probe import PeakVramProbe
from canvas.safety.adapters import FaceAgeRegressor, FairFaceAge, NsfwViT, Qwen3GuardGen, SiglipZeroShot, YuNetFaces
from canvas.safety.gate import SafetyStack, SafetyUnavailable
from canvas.safety.image import RawImage, ScreenedInput

_SAFETY_ROLES = ("prompt-guard", "zero-shot", "face-detector", "age", "nsfw", "age-fine")


class ModelNotReady(Exception):
    """The selected pipeline is known, and its weights are not runnable yet."""

    def __init__(self, plan: CallPlan, *, missing: bool = False) -> None:
        self.plan = plan
        self.missing = missing
        super().__init__(text("model.missing") if missing else text("gate.unavailable"))

    def __str__(self) -> str:
        if self.missing:
            return text("model.missing")
        return text("gate.unavailable")


class ModelMissing(SafetyUnavailable):
    """The MDL view is incomplete. A legacy copy is not opened."""

    message_key = "model.missing"


def ensure_weights(spec: ModelSpec, root: Path, folder: Path | None = None) -> None:
    """Refuse when any file in ``folder`` is unpinned, missing or hashed wrong.

    ``folder`` is the MDL view. There is no legacy-folder default.
    """
    del root
    if not spec.files or folder is None:
        raise ModelMissing("missing_model")
    for name, digest in spec.files.items():
        try:
            require_pin(digest)
        except PinError as exc:
            raise ModelMissing("missing_model") from exc
        path = folder / name
        if not path.is_file():
            raise ModelMissing("missing_model")
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
        FaceAgeRegressor(folders["age-fine"]),
    )


def _safety_folder(spec: ModelSpec, root: Path) -> Path:
    """The view over the MDL blobs. An incomplete view is not a legacy folder.

    ``ensure_weights`` checks every file against its pin before use. A view that
    cannot be built refuses the stack. The ``<org>--<name>`` tree is left as it is.
    """
    try:
        folder = snapshot_ready(spec, root)
    except SnapshotError as exc:
        raise ModelMissing("missing_model") from exc
    if folder is None:
        request_download(spec.id)
        raise ModelMissing("missing_model")
    return folder


def load_catalog_stack(root: Path, catalog: Catalog | None = None) -> SafetyStack:
    """Open the check stack after preflight. A failed preflight does not load a model."""
    chosen = catalog or load_catalog()
    result = preflight(root, chosen)
    if not result.ok:
        _refuse_preflight(result.reason)
    return load_safety_stack(chosen, root)


def _refuse_preflight(reason: Reason) -> NoReturn:
    """Stop before the stack is constructed. A missing view uses the MDL sentence."""
    match reason:
        case "view" | "gate":
            raise ModelMissing("missing_model")
        case "disk" | "ram" | "budget":
            raise PreflightFailed(reason)
        case "ok":
            raise RuntimeError("preflight ok")
        case _ as unexpected:
            assert_never(unexpected)


def request_fp32(spec: ModelSpec) -> bool:
    """Queue the FP32 ``<id>-upstream`` entry with VOXPrint MDL. False when it cannot be queued."""
    if not spec.fallback_files:
        return False
    return request_download(f"{spec.id}-upstream")


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
        self.negative_prompt = ""
        self.models: Path | None = None
        self.lora_path = ""
        self.lora_strength = 0.7
        self.bound_lora: UserLora | None = None

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
        cache = process_cache()
        try:
            folder = snapshot_ready(spec, root, cache)
            if folder is None:
                request_download(spec.id)
                raise ModelNotReady(plan, missing=True)
            if width is None or height is None:
                raise ModelNotReady(plan)
            control_file = _control_file(self.catalog, plan, root, cache)
            lora, lora_strength = self._selected_lora()
            try:
                return render(
                    plan, folder, screened, width, height, self.tier,
                    seed=self.seed, control_file=control_file,
                    strength=self.img2img_strength,
                    lora=lora, lora_strength=lora_strength,
                    negative=self.negative_prompt,
                )
            except WeightsLoadError:
                # The BF16 weights did not open: use the author's FP32 -upstream entry.
                fp32 = fallback_ready(spec, root, cache)
                if fp32 is None or fp32 == folder:
                    request_fp32(spec)
                    raise ModelNotReady(plan, missing=True) from None
                return render(
                    plan, fp32, screened, width, height, self.tier,
                    seed=self.seed, control_file=control_file,
                    strength=self.img2img_strength,
                    lora=lora, lora_strength=lora_strength,
                    negative=self.negative_prompt,
                )
            except ImportError as exc:
                raise ModelNotReady(plan) from exc
        finally:
            flush_process_cache()

    def _selected_lora(self) -> tuple[UserLora | None, float | None]:
        """The listed LoRA for the expert field, and the strength to apply."""
        if self.bound_lora is not None:
            return self.bound_lora, self.lora_strength
        if not self.lora_path:
            return None, None
        selected = resolve_selected(self.lora_path, self.models)
        if selected is None:
            return None, None
        return selected, self.lora_strength

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


def _control_file(
    catalog: Catalog, plan: CallPlan, root: Path, cache: VerifiedCache | None = None,
) -> Path | None:
    spec = _control_spec(catalog, plan)
    if spec is None:
        return None
    folder = snapshot_ready(spec, root, cache)
    if folder is None:
        return None
    for name in spec.files:
        if name.endswith(".safetensors"):
            path = folder / name
            if path.is_file():
                return path
    return None
