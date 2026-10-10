"""Which registry entries and diffusers classes a mode uses. No weights are loaded."""
from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from canvas.models.modes import Mode, Tier
from canvas.models.pins import (
    DIFFUSERS_PIN,
    PIPELINE_CONTROL,
    PIPELINE_CONTROL_INPAINT,
    PIPELINE_EDIT,
    PIPELINE_IMG2IMG,
    PIPELINE_INPAINT,
    PIPELINE_TEXT,
)
from canvas.models.registry import Catalog, ModelSpec
from canvas.safety.image import ScreenedInput
from canvas.safety.inputs import ImageRole


class ModeUnavailable(Exception):
    """This GPU tier does not offer the mode. The UI hides it."""

    def __init__(self) -> None:
        super().__init__("This mode is not available on this GPU.")


@dataclass(frozen=True)
class CallPlan:
    diffusers: str
    pipeline_class: str
    repos: tuple[str, ...]
    quant: str
    steps: int | None
    guidance_scale: float | None
    text_encoder_offload: str
    tier: str
    mode: str


def _one(catalog: Catalog, model_id: str) -> ModelSpec:
    return catalog.by_id(model_id)


def build_plan(catalog: Catalog, screened: ScreenedInput, tier: Tier) -> CallPlan:
    """Return the diffusers call for one screened request. Hidden modes raise."""
    if type(screened) is not ScreenedInput:
        raise TypeError("a generation plan requires a screened request")
    mode = Mode(screened.mode)
    roles = set(screened.roles())
    has_mask = ImageRole.MASK.value in roles
    if tier is Tier.GB16:
        turbo_id = "z-image-turbo-fp8"
        control_id = "z-image-turbo-fun-controlnet-union-fp8"
    elif tier is Tier.GB24:
        turbo_id = "z-image-turbo-bf16"
        control_id = "z-image-turbo-fun-controlnet-union-bf16"
    else:
        assert_never(tier)
    turbo = _one(catalog, turbo_id)
    control = _one(catalog, control_id)

    if mode is Mode.TEXT_CREATIVE or mode is Mode.INSTRUCTION:
        if tier is not Tier.GB24:
            raise ModeUnavailable()
    if mode is Mode.TEXT:
        return _plan(mode, tier, PIPELINE_TEXT, (turbo,), turbo.steps, turbo.guidance_scale, turbo.text_encoder_offload)
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
    if mode is Mode.INSTRUCTION:
        edit = _one(catalog, "qwen-image-edit-2511-fp8")
        return _plan(mode, tier, PIPELINE_EDIT, (edit,), edit.steps, edit.guidance_scale, "")
    assert_never(mode)


def _plan(mode: Mode, tier: Tier, pipeline: str, specs: tuple[ModelSpec, ...], steps: int | None,
          guidance: float | None, offload: str) -> CallPlan:
    return CallPlan(
        diffusers=DIFFUSERS_PIN,
        pipeline_class=pipeline,
        repos=tuple(spec.repo for spec in specs),
        quant=specs[0].quant,
        steps=steps,
        guidance_scale=guidance,
        text_encoder_offload=offload,
        tier=tier.value,
        mode=mode.value,
    )
