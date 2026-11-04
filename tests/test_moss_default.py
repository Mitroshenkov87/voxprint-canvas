"""MOSS-Image is the default text generator. Z-Image stays available by name."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canvas.models.loader import Generator, ZImageGenerator
from canvas.models.mdl_manifest import render
from canvas.models.modes import Mode, Tier
from canvas.models.moss import render_moss
from canvas.models.pins import PIPELINE_TEXT, REPO_TURBO, TEXT_ENCODER
from canvas.models.plan import PIPELINE_IMG2IMG, PIPELINE_MOSS, CallPlan, build_plan
from canvas.models.registry import load_catalog
from canvas.models.required import required_specs
from canvas.models.runtime import InferenceUnavailable
from canvas.models.user_lora import base_matches, running_base_id
from canvas.safety.image import RawImage
from canvas.safety.inputs import ImageRole
from tests.fakes import World, rgb


def _screened(mode: Mode = Mode.TEXT, *roles: ImageRole):
    from canvas.safety.inputs import PendingInput

    world = World()
    frames = [PendingInput(role, rgb()) for role in roles]
    return world.gate().release_request("a red teapot on a table", mode.value, frames)


def test_catalog_loads_with_moss_as_the_default_text_base():
    catalog = load_catalog()
    assert catalog.default_text_base == "moss-image-sft"
    assert catalog.text_bases[0] == "moss-image-sft"
    assert "z-image-turbo" in catalog.text_bases
    assert "moss-image-sft" in catalog.sizes
    compact = catalog.by_id("moss-image-sft-compact-16gb")
    assert compact.pinned() is False
    assert compact.optional is False
    assert compact.quant == "compact-tbd"
    assert compact.vram_hint_gb == 13
    raw = json.loads(Path("canvas/models/catalog.json").read_text(encoding="utf-8"))
    compact_row = next(row for row in raw["models"] if row["id"] == compact.id)
    assert compact_row["credit"] == "OpenMOSS-Team/MOSS-Image-SFT, Qwen/Qwen3.5-9B, Apache-2.0"
    assert compact_row["peak_vram_gb"] == 14
    assert compact_row["repo"] == "voxprint/moss-image-sft-compact"
    assert compact.text_base == "moss-image-sft"
    assert compact.pipeline_classes == ("MOSSImagePipeline",)
    assert compact.pin_note.startswith("PENDING:")
    for model_id, offload, hint, minimum in (
        ("moss-image-sft-bf16-16gb", "release-after-encode", 11, 16),
        ("moss-image-sft-bf16", "none", 27, 24),
    ):
        spec = catalog.by_id(model_id)
        assert spec.optional is True
        assert spec.pinned() is False
        assert spec.quant == "bf16"
        assert spec.steps == 50 and spec.guidance_scale == 3.0
        assert spec.text_encoder == "Qwen/Qwen3.5-9B"
        assert spec.text_encoder_offload == offload
        assert spec.vram_hint_gb == hint and spec.min_vram_gb == minimum
        assert spec.pin_note.startswith("TBD:")
    required = {spec.id for spec in required_specs(catalog)}
    assert "moss-image-sft-compact-16gb" in required
    for model_id in (
        "moss-image-sft-bf16-16gb",
        "moss-image-sft-bf16",
        "z-image-turbo-bf16-16gb",
        "z-image-turbo-bf16",
        "z-image-creative-bf16",
        "z-image-turbo-fun-controlnet-union-bf16-lite",
        "z-image-turbo-fun-controlnet-union-bf16",
    ):
        assert catalog.by_id(model_id).optional is True
        assert model_id not in required
    assert Generator is ZImageGenerator


def test_text_plans_pick_moss_unless_the_caller_asks_for_z_image():
    catalog = load_catalog()
    small = build_plan(catalog, _screened(), Tier.GB16)
    assert small.model_id == "moss-image-sft-compact-16gb"
    assert small.pipeline_class == PIPELINE_MOSS
    assert small.repos == ("voxprint/moss-image-sft-compact",)
    assert small.steps == 50 and small.guidance_scale == 3.0
    large = build_plan(catalog, _screened(), Tier.GB24)
    assert large.model_id == "moss-image-sft-bf16"
    assert large.pipeline_class == PIPELINE_MOSS
    assert large.text_encoder_offload == "none"
    assert large.repos == ("OpenMOSS-Team/MOSS-Image-SFT",)
    z16 = build_plan(catalog, _screened(), Tier.GB16, base="z-image-turbo")
    assert z16.model_id == "z-image-turbo-bf16-16gb"
    assert z16.pipeline_class == PIPELINE_TEXT
    z24 = build_plan(catalog, _screened(), Tier.GB24, base="z-image-turbo")
    assert z24.model_id == "z-image-turbo-bf16"
    restyle = build_plan(catalog, _screened(Mode.RESTYLE, ImageRole.INIT), Tier.GB16)
    assert restyle.pipeline_class == PIPELINE_IMG2IMG
    assert restyle.repos == (REPO_TURBO,)
    creative = build_plan(catalog, _screened(Mode.TEXT_CREATIVE), Tier.GB24)
    assert creative.model_id == "z-image-creative-bf16"


def test_unpinned_rows_are_pending_and_z_image_rows_are_optional():
    document = json.loads(render())
    listed = {row["id"] for row in document["models"]}
    pending = {item["id"]: item["note"] for item in document["pending"]}
    for model_id in (
        "moss-image-sft-compact-16gb",
        "moss-image-sft-bf16-16gb",
        "moss-image-sft-bf16",
    ):
        assert model_id not in listed
        assert model_id in pending
        assert pending[model_id]
    rows = {row["id"]: row for row in document["models"]}
    for model_id in (
        "z-image-turbo-bf16-16gb",
        "z-image-turbo-bf16",
        "z-image-creative-bf16",
        "z-image-turbo-fun-controlnet-union-bf16",
        "z-image-turbo-fun-controlnet-union-bf16-lite",
    ):
        assert rows[model_id]["optional"] is True


def test_running_base_id_follows_the_catalog_and_moss_aliases_match():
    assert running_base_id() == "moss-image-sft"
    assert base_matches("moss-image-sft")
    assert base_matches("moss-image")
    assert base_matches("moss-image-base")
    assert base_matches("z-image", "z-image-turbo")
    assert base_matches("z-image-turbo", "z-image-turbo")
    assert base_matches("z-image") is False
    assert base_matches("other", "moss-image-sft") is False


class _Encoder:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode(self, prompt: str, negative: str) -> tuple[str, str]:
        self.prompts.append(prompt)
        return prompt, negative


def test_the_encoder_is_released_before_the_pipe_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    order: list[str] = []
    encoder = _Encoder()

    def load_encoder(folder: Path) -> _Encoder:
        order.append("load_encoder")
        assert folder == tmp_path
        return encoder

    def release(held: object) -> None:
        order.append("release")
        assert held is encoder

    def load_pipe(folder: Path) -> object:
        order.append("load_pipe")
        assert folder == tmp_path
        return object()

    monkeypatch.setattr("canvas.models.moss._load_encoder", load_encoder)
    monkeypatch.setattr("canvas.models.moss._release", release)
    monkeypatch.setattr("canvas.models.moss._load_pipe", load_pipe)
    monkeypatch.setattr(
        "canvas.models.moss._run",
        lambda *args, **kwargs: RawImage(1, 1, b"\x00\x00\x00"),
    )
    image = render_moss(
        build_plan(load_catalog(), _screened(), Tier.GB16),
        tmp_path,
        _screened(),
        64,
        64,
        Tier.GB16,
    )
    assert order == ["load_encoder", "release", "load_pipe"]
    assert order.index("release") < order.index("load_pipe")
    assert encoder.prompts == ["a red teapot on a table"]
    assert image.width == 1


def test_a_missing_vendored_pipeline_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("canvas.models.moss._load_encoder", lambda folder: _Encoder())
    monkeypatch.setattr("canvas.models.moss._release", lambda encoder: None)
    with pytest.raises(InferenceUnavailable, match="pipeline_moss_image"):
        render_moss(
            build_plan(load_catalog(), _screened(), Tier.GB16),
            tmp_path,
            _screened(),
            32,
            32,
            Tier.GB16,
        )


def test_moss_generate_skips_the_z_image_schedule(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[str] = []
    monkeypatch.setattr("canvas.models.loader.snapshot_ready", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(
        "canvas.models.loader.render",
        lambda *args, **kwargs: calls.append("render") or RawImage(1, 1, b"\x00\x00\x00"),
    )
    monkeypatch.setattr(
        "canvas.models.moss.render_moss",
        lambda *args, **kwargs: calls.append("moss") or RawImage(1, 1, b"\x00\x00\x00"),
    )
    monkeypatch.setattr("canvas.models.loader.run_bf16_16gb", lambda **kwargs: calls.append("schedule"))
    monkeypatch.setattr("canvas.models.loader.sample_latent", lambda *args, **kwargs: calls.append("sample"))
    generator = ZImageGenerator(load_catalog(), Tier.GB16)
    screened = _screened()
    generator.sample(screened, 32, 32)
    image = generator.generate(screened, 32, 32)
    assert calls == ["moss"]
    assert image.width == 1
    plan = CallPlan(
        "0.41.0",
        PIPELINE_MOSS,
        ("voxprint/moss-image-sft-compact",),
        "bf16",
        50,
        3.0,
        "release-after-encode",
        "16gb",
        "text",
        text_encoder=TEXT_ENCODER,
        model_id="moss-image-sft-compact-16gb",
    )
    generator._schedule(plan, Mode.TEXT, 64, 64, "a red teapot on a table")
    assert "schedule" not in calls
