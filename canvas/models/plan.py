"""Which registry entries and diffusers classes a mode uses. No weights are loaded."""
from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from canvas.app.i18n import text
from canvas.models.modes import Mode, Tier
from canvas.models.pins import (
    DIFFUSERS_PIN,
    PIPELINE_CONTROL,
    PIPELINE_CONTROL_INPAINT,
    PIPELINE_IMG2IMG,
    PIPELINE_INPAINT,
    PIPELINE_TEXT,
)
from canvas.models.default_base import selected_text_base
from canvas.models.registry import Catalog, ModelSpec, RegistryError
from canvas.safety.image import ScreenedInput
from canvas.safety.inputs import ImageRole

PIPELINE_MOSS = "MOSSImagePipeline"


class ModeUnavailable(Exception):
    """This GPU tier does not offer the mode. The UI hides it."""

    def __init__(self) -> None:
        super().__init__(text("error.mode"))

    def __str__(self) -> str:
        return text("error.mode")


class BaseNotSelected(Exception):
    """Text to image has no default until a comparison picks one."""

    def __init__(self) -> None:
        super().__init__(text("error.base"))

    def __str__(self) -> str:
        return text("error.base")


class OptionalModelUnavailable(Exception):
    """The optional model is not in the catalog or is not available to generate."""

    def __init__(self) -> None:
        super().__init__(text("error.optional_model"))

    def __str__(self) -> str:
        return text("error.optional_model")


@dataclass(frozen=True)
class CallPlan:
    """The diffusers call selected for one request. No weights are loaded."""

    diffusers: str
    pipeline_class: str
    repos: tuple[str, ...]
    quant: str
    steps: int | None
    guidance_scale: float | None
    text_encoder_offload: str
    tier: str
    mode: str
    guidance_kind: str = ""
    text_encoder: str = ""
    model_id: str = ""


def _one(catalog: Catalog, model_id: str) -> ModelSpec:
    return catalog.by_id(model_id)


def _lookup(catalog: Catalog, model_id: str) -> ModelSpec | None:
    """Return the row, or None when the optional id is not in the catalog."""
    try:
        return catalog.by_id(model_id)
    except RegistryError:
        return None


def build_plan(
    catalog: Catalog,
    screened: ScreenedInput,
    tier: Tier,
    *,
    base: str | None = None,
    model_id: str | None = None,
) -> CallPlan:
    """Return the diffusers call for one screened request. Hidden modes raise.

    ``base`` selects the text-to-image family. Other modes ignore it and stay on Z-Image.
    ``model_id`` is a specific installed generate row (the Canvas model pick). It
    wins for words-to-image when the catalog lists the row; an explicit ``base``
    (a bound LoRA's trained family) still wins over a pick of another family, and
    an uncatalogued pick leaves the family mapping in charge.
    """
    if type(screened) is not ScreenedInput:
        raise TypeError("a generation plan requires a screened request")
    mode = Mode(screened.mode)
    roles = set(screened.roles())
    has_mask = ImageRole.MASK.value in roles
    if tier is Tier.GB16:
        turbo_id = "z-image-turbo-bf16-16gb"
        control_id = "z-image-turbo-fun-controlnet-union-bf16-lite"
    elif tier is Tier.GB24:
        turbo_id = "z-image-turbo-bf16"
        control_id = "z-image-turbo-fun-controlnet-union-bf16"
    else:
        assert_never(tier)
    turbo = _one(catalog, turbo_id)
    control = _one(catalog, control_id)

    if mode is Mode.TEXT_CREATIVE:
        if tier is not Tier.GB24:
            raise ModeUnavailable()
    if mode is Mode.TEXT:
        spec, pipeline = _text_model(catalog, tier, base, model_id)
        return _plan(mode, tier, pipeline, (spec,), spec.steps, spec.guidance_scale, spec.text_encoder_offload)
    if mode is Mode.TEXT_CREATIVE:
        creative = _one(catalog, "z-image-creative-bf16")
        return _plan(mode, tier, PIPELINE_TEXT, (creative,), creative.steps, creative.guidance_scale, "")
    if mode is Mode.RESTYLE:
        return _plan(
            mode, tier, PIPELINE_IMG2IMG, (turbo,), turbo.steps, turbo.guidance_scale, turbo.text_encoder_offload,
        )
    if mode is Mode.INPAINT or mode is Mode.OUTPAINT:
        return _plan(
            mode, tier, PIPELINE_INPAINT, (turbo, control), turbo.steps, turbo.guidance_scale, turbo.text_encoder_offload,
        )
    if mode is Mode.POSE or mode is Mode.DEPTH or mode is Mode.EDGE:
        pipeline = PIPELINE_CONTROL_INPAINT if has_mask else PIPELINE_CONTROL
        return _plan(
            mode, tier, pipeline, (turbo, control), turbo.steps, turbo.guidance_scale, turbo.text_encoder_offload,
        )
    assert_never(mode)


def _text_model(
    catalog: Catalog,
    tier: Tier,
    base: str | None = None,
    model_id: str | None = None,
) -> tuple[ModelSpec, str]:
    """The text-to-image row for ``base``, or the shared ``model.default`` when ``base`` is omitted.

    A specific ``model_id`` (the Canvas model pick) wins when the catalog lists
    the row as a generator. An explicit ``base`` — a bound LoRA's trained
    family — still wins over a pick of a different family, and a pick the
    catalog does not know (for example a compact 4-bit build pinned nowhere
    yet) leaves the family mapping in charge, so catalogued models keep
    planning.
    """
    family = base or selected_text_base()
    if not family:
        raise BaseNotSelected()
    if model_id:
        picked = _picked_text_row(catalog, model_id, explicit_base=base)
        if picked is not None:
            return picked
    if family == "z-image-turbo":
        if tier is Tier.GB16:
            return _one(catalog, "z-image-turbo-bf16-16gb"), PIPELINE_TEXT
        if tier is Tier.GB24:
            return _one(catalog, "z-image-turbo-bf16"), PIPELINE_TEXT
        assert_never(tier)
    if family == "moss-image-sft":
        if tier is Tier.GB16:
            moss = _lookup(catalog, "voxprint-moss-image-sft-compact") or _lookup(
                catalog, "moss-image-sft-bf16-16gb"
            )
        elif tier is Tier.GB24:
            moss = _lookup(catalog, "moss-image-sft-bf16")
        else:
            assert_never(tier)
        if moss is None:
            raise OptionalModelUnavailable()
        return moss, PIPELINE_MOSS
    if family == "z-image":
        # Optional full Z-Image. There is no Canvas pin yet; stay absent-safe.
        wanted = "16gb" if tier is Tier.GB16 else "24gb" if tier is Tier.GB24 else None
        if wanted is None:
            assert_never(tier)
        full = next((spec for spec in catalog.models if spec.text_base == "z-image" and spec.tier == wanted), None)
        if full is None:
            raise OptionalModelUnavailable()
        return full, PIPELINE_TEXT
    if family not in catalog.text_bases:
        raise BaseNotSelected()
    raise BaseNotSelected()


def _picked_text_row(catalog: Catalog, model_id: str, *, explicit_base: str | None) -> tuple[ModelSpec, str] | None:
    """The picked generate row and its pipeline, or None when the pick cannot run.

    The catalog must list the id with the ``generator`` role. An explicit
    ``base`` (a bound LoRA's trained family) keeps a pick of another family
    from loading that LoRA onto the wrong weights; a row without a text base
    (creative) runs under any family. Uncatalogued picks return None so the
    family mapping plans instead.
    """
    try:
        spec = catalog.by_id(model_id)
    except RegistryError:
        return None
    if spec.role != "generator":
        return None
    if explicit_base and spec.text_base and spec.text_base != explicit_base:
        return None
    pipeline = PIPELINE_MOSS if PIPELINE_MOSS in spec.pipeline_classes else PIPELINE_TEXT
    return spec, pipeline


def _plan(mode: Mode, tier: Tier, pipeline: str, specs: tuple[ModelSpec, ...], steps: int | None,
          guidance: float | None, offload: str) -> CallPlan:
    spec = specs[0]
    return CallPlan(
        diffusers=DIFFUSERS_PIN,
        pipeline_class=pipeline,
        repos=tuple(item.repo for item in specs),
        quant=spec.quant,
        steps=steps,
        guidance_scale=guidance,
        text_encoder_offload=offload,
        tier=tier.value,
        mode=mode.value,
        guidance_kind=spec.guidance_kind,
        text_encoder=spec.text_encoder,
        model_id=spec.id,
    )
