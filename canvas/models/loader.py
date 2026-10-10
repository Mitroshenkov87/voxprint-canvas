"""Open a generator and the classifier stack from the registry.

Weights are not downloaded here. A missing pin, a missing file or a bad hash
refuses the stack. The generator stub records the diffusers plan and refuses
until inference is wired.
"""
from pathlib import Path

from canvas.models.modes import Tier
from canvas.models.pin import PinError, require_pin, verify_bytes
from canvas.models.plan import CallPlan, build_plan
from canvas.models.registry import Catalog, ModelSpec, load_catalog
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
            verify_bytes(path.read_bytes(), digest)
        except PinError as exc:
            raise SafetyUnavailable("hash_mismatch") from exc


def load_safety_stack(catalog: Catalog, root: Path) -> SafetyStack:
    """Build the five adapters after every safety weight has been checked."""
    for role in _SAFETY_ROLES:
        found = catalog.by_role(role)
        if len(found) != 1:
            raise SafetyUnavailable("missing_model")
        ensure_weights(found[0], root)
    return SafetyStack(Qwen3GuardGen(), SiglipZeroShot(), YuNetFaces(), FairFaceAge(), NsfwViT())


def load_catalog_stack(root: Path, catalog: Catalog | None = None) -> SafetyStack:
    return load_safety_stack(catalog or load_catalog(), root)


class ZImageGenerator:
    """Diffusers generator stub. ``generate`` accepts only a ``ScreenedInput``."""

    def __init__(self, catalog: Catalog, tier: Tier) -> None:
        self.catalog = catalog
        self.tier = tier
        self.last_plan: CallPlan | None = None

    def generate(self, screened: ScreenedInput) -> RawImage:
        if type(screened) is not ScreenedInput:
            raise TypeError("the generator only accepts a ScreenedInput")
        plan = build_plan(self.catalog, screened, self.tier)
        self.last_plan = plan
        raise ModelNotReady(plan)
