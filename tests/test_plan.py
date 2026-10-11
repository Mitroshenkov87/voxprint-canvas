"""Diffusers 0.41.0 plans for the two tiers."""
from dataclasses import replace

import pytest

from canvas.models.modes import Mode, Tier, visible_modes
from canvas.models.pins import (
    DIFFUSERS_PIN,
    ENCODER_VL,
    PIPELINE_CONTROL,
    PIPELINE_CONTROL_INPAINT,
    PIPELINE_EDIT,
    PIPELINE_IMG2IMG,
    PIPELINE_INPAINT,
    PIPELINE_TEXT,
    REPO_CONTROL,
    REPO_EDIT,
    REPO_TURBO,
)
from canvas.models.plan import BaseNotSelected, ModeUnavailable, build_plan
from canvas.models.registry import RegistryError, load_catalog
from canvas.safety.inputs import ImageRole, PendingInput
from tests.fakes import World, rgb


def _screened(mode: Mode, *roles: ImageRole):
    world = World()
    inputs = [PendingInput(role, rgb()) for role in roles]
    return world.gate().release_request("a red boat", mode.value, inputs)


def test_pin_and_visibility():
    catalog = load_catalog()
    assert catalog.diffusers == "0.41.0"
    assert catalog.default_text_base == "z-image-turbo"
    assert catalog.text_bases == ("z-image-turbo",)
    assert DIFFUSERS_PIN == "0.41.0"
    assert Mode.INSTRUCTION not in visible_modes(Tier.GB16)
    assert Mode.TEXT_CREATIVE not in visible_modes(Tier.GB16)
    assert Mode.INSTRUCTION in visible_modes(Tier.GB24)
    assert Mode.TEXT_CREATIVE in visible_modes(Tier.GB24)
    lite = catalog.by_id("z-image-turbo-fun-controlnet-union-bf16-lite")
    full = catalog.by_id("z-image-turbo-fun-controlnet-union-bf16")
    assert lite.repo == REPO_CONTROL and lite.variant == "lite" and lite.quant == "bf16"
    assert lite.tier == "16gb" and lite.quality_gate == "passed"
    assert full.variant == "full" and full.quant == "bf16" and full.quality_gate == "passed"
    edit = catalog.by_id("qwen-image-edit-2511-fp8")
    assert edit.repo == REPO_EDIT and edit.tier == "24gb"
    assert edit.text_encoder == ENCODER_VL and edit.quality_gate == "passed"
    assert edit.mirror_status == "published"
    assert edit.text_encoder_offload == "model-cpu-offload"
    ids = {spec.id for spec in catalog.models}
    assert "qwen-image-2512-int4" not in ids
    assert "qwen-image-2512-fp8" not in ids
    creative = catalog.by_id("z-image-creative-bf16")
    assert creative.repo == "Tongyi-MAI/Z-Image" and creative.tier == "24gb" and creative.steps == 50
    assert "diffusers==0.41.0" in open("pyproject.toml", encoding="utf-8").read()


def test_text_plan_waits_until_a_base_is_chosen():
    catalog = replace(load_catalog(), default_text_base=None)
    with pytest.raises(BaseNotSelected):
        build_plan(catalog, _screened(Mode.TEXT), Tier.GB16)


def test_text_and_restyle_plans():
    catalog = load_catalog()
    text = build_plan(catalog, _screened(Mode.TEXT), Tier.GB16)
    assert text.diffusers == "0.41.0"
    assert text.pipeline_class == PIPELINE_TEXT
    assert text.repos == (REPO_TURBO,)
    assert text.quant == "bf16"
    assert text.steps == 8 and text.guidance_scale == 0.0
    assert text.text_encoder == "Qwen/Qwen3-4B"
    assert text.text_encoder_offload == "free-after-encode"
    larger = build_plan(catalog, _screened(Mode.TEXT), Tier.GB24)
    assert larger.quant == "bf16" and larger.text_encoder_offload == ""
    restyle = build_plan(catalog, _screened(Mode.RESTYLE, ImageRole.INIT), Tier.GB16)
    assert restyle.pipeline_class == PIPELINE_IMG2IMG
    assert restyle.repos == (REPO_TURBO,)
    qwen = replace(
        catalog,
        default_text_base="qwen-image-2512",
        text_bases=("z-image-turbo", "qwen-image-2512"),
    )
    with pytest.raises(RegistryError, match="has no plan"):
        build_plan(qwen, _screened(Mode.TEXT), Tier.GB16)
    with pytest.raises(RegistryError, match="has no plan"):
        build_plan(qwen, _screened(Mode.TEXT), Tier.GB24)
    restyle_on_qwen = build_plan(qwen, _screened(Mode.RESTYLE, ImageRole.INIT), Tier.GB16)
    assert restyle_on_qwen.repos == (REPO_TURBO,) and restyle_on_qwen.pipeline_class == PIPELINE_IMG2IMG


def test_control_and_edit_plans():
    catalog = load_catalog()
    inpaint = build_plan(catalog, _screened(Mode.INPAINT, ImageRole.INIT, ImageRole.MASK), Tier.GB16)
    assert inpaint.pipeline_class == PIPELINE_INPAINT
    assert inpaint.repos == (REPO_TURBO, REPO_CONTROL)
    outpaint = build_plan(catalog, _screened(Mode.OUTPAINT, ImageRole.INIT, ImageRole.MASK), Tier.GB24)
    assert outpaint.pipeline_class == PIPELINE_INPAINT and outpaint.quant == "bf16"
    pose = build_plan(catalog, _screened(Mode.POSE, ImageRole.POSE), Tier.GB16)
    assert pose.pipeline_class == PIPELINE_CONTROL
    depth = build_plan(catalog, _screened(Mode.DEPTH, ImageRole.DEPTH, ImageRole.MASK), Tier.GB24)
    assert depth.pipeline_class == PIPELINE_CONTROL_INPAINT
    edge = build_plan(catalog, _screened(Mode.EDGE, ImageRole.EDGE), Tier.GB16)
    assert edge.pipeline_class == PIPELINE_CONTROL
    edit = build_plan(catalog, _screened(Mode.INSTRUCTION, ImageRole.INIT), Tier.GB24)
    assert edit.pipeline_class == PIPELINE_EDIT
    assert edit.repos == (REPO_EDIT,)
    assert edit.steps == 40 and edit.guidance_scale == 4.0
    assert edit.guidance_kind == "true_cfg_scale"
    assert edit.text_encoder_offload == "model-cpu-offload"
    creative = build_plan(catalog, _screened(Mode.TEXT_CREATIVE), Tier.GB24)
    assert creative.pipeline_class == PIPELINE_TEXT
    assert creative.steps == 50 and creative.guidance_scale is None
    assert creative.repos == ("Tongyi-MAI/Z-Image",)
    with pytest.raises(ModeUnavailable):
        build_plan(catalog, _screened(Mode.INSTRUCTION, ImageRole.INIT), Tier.GB16)
    with pytest.raises(ModeUnavailable):
        build_plan(catalog, _screened(Mode.TEXT_CREATIVE), Tier.GB16)
