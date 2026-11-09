"""Z-Image-Turbo is the default text generator. MOSS-Image stays an optional download."""
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


def test_catalog_loads_with_z_image_as_the_default_text_base():
    catalog = load_catalog()
    assert catalog.default_text_base == "z-image-turbo"
    assert catalog.text_bases[0] == "z-image-turbo"
    assert "moss-image-sft" in catalog.text_bases
    assert "z-image-turbo" in catalog.sizes
    compact = catalog.by_id("voxprint-moss-image-sft-compact")
    assert compact.pinned() is False
    assert compact.optional is True
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
    assert compact_row["notes"] == "(fp8/int8/nf4, chosen by moss-prequant)"
    assert "(fp8/int8/nf4, chosen by moss-prequant)" in compact.pin_note
    assert "moss-image-compressed" in compact.pin_note
    assert "no longer lists moss-image-compressed" in compact.pin_note
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
    assert "z-image-turbo-bf16-16gb" in required
    assert "voxprint-moss-image-sft-compact" not in required
    assert "moss-image-sft-compact-16gb" not in required
    assert catalog.by_id("z-image-turbo-bf16-16gb").optional is False
    for model_id in (
        "voxprint-moss-image-sft-compact",
        "moss-image-sft-bf16-16gb",
        "moss-image-sft-bf16",
        "z-image-turbo-bf16",
        "z-image-creative-bf16",
        "z-image-turbo-fun-controlnet-union-bf16-lite",
        "z-image-turbo-fun-controlnet-union-bf16",
    ):
        assert catalog.by_id(model_id).optional is True
        assert model_id not in required
    assert Generator is ZImageGenerator



def test_the_common_compressed_placeholder_is_not_a_canvas_pin():
    """voxprint-common 0.7.0 no longer lists moss-image-compressed.

    Looking the id up must be absent-safe (None, not an error). Canvas must
    not invent a hash or treat a removed placeholder as a pin. Z-Image is
    the shared default; MOSS stays optional.
    """
    from voxprint_common.mdl.index import load_index
    from voxprint_common.models import default_model
    from voxprint_common.suite_settings import model_default

    index = load_index(remote=False)
    assert index.get("moss-image-compressed") is None
    z_image = index.get("z-image")
    assert z_image is not None
    assert z_image.default is True
    assert z_image.optional is False
    assert default_model() == "z-image"
    assert model_default() == "z-image"
    catalog = load_catalog()
    compact = catalog.by_id("voxprint-moss-image-sft-compact")
    assert compact.pinned() is False
    assert compact.optional is True
    raw = json.loads(Path("canvas/models/catalog.json").read_text(encoding="utf-8"))
    compact_row = next(row for row in raw["models"] if row["id"] == compact.id)
    assert compact_row.get("revision") in ("", None)
    ids = {row["id"] for row in raw["models"]}
    assert "moss-image-compressed" not in ids



def test_text_plans_pick_z_image_unless_the_caller_asks_for_moss():
    catalog = load_catalog()
    small = build_plan(catalog, _screened(), Tier.GB16)
    assert small.model_id == "z-image-turbo-bf16-16gb"
    assert small.pipeline_class == PIPELINE_TEXT
    assert small.steps == 8 and small.guidance_scale == 0.0
    large = build_plan(catalog, _screened(), Tier.GB24)
    assert large.model_id == "z-image-turbo-bf16"
    assert large.pipeline_class == PIPELINE_TEXT
    moss16 = build_plan(catalog, _screened(), Tier.GB16, base="moss-image-sft")
    assert moss16.model_id == "voxprint-moss-image-sft-compact"
    assert moss16.pipeline_class == PIPELINE_MOSS
    assert moss16.repos == ("voxprint/moss-image-sft-compact",)
    assert moss16.steps == 50 and moss16.guidance_scale == 3.0
    moss24 = build_plan(catalog, _screened(), Tier.GB24, base="moss-image-sft")
    assert moss24.model_id == "moss-image-sft-bf16"
    assert moss24.pipeline_class == PIPELINE_MOSS
    assert moss24.text_encoder_offload == "none"
    assert moss24.repos == ("OpenMOSS-Team/MOSS-Image-SFT",)
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
        "voxprint-moss-image-sft-compact",
        "moss-image-sft-bf16-16gb",
        "moss-image-sft-bf16",
    ):
        assert model_id not in listed
        assert model_id in pending
        assert pending[model_id]
    assert "moss-image-sft-compact-16gb" not in pending
    assert "(fp8/int8/nf4, chosen by moss-prequant)" in pending["voxprint-moss-image-sft-compact"]
    rows = {row["id"]: row for row in document["models"]}
    assert rows["z-image-turbo-bf16-16gb"]["optional"] is False
    for model_id in (
        "z-image-turbo-bf16",
        "z-image-creative-bf16",
        "z-image-turbo-fun-controlnet-union-bf16",
        "z-image-turbo-fun-controlnet-union-bf16-lite",
    ):
        assert rows[model_id]["optional"] is True


def test_running_base_id_follows_the_catalog_and_aliases_match():
    assert running_base_id() == "z-image-turbo"
    assert base_matches("z-image")
    assert base_matches("z-image-turbo")
    assert base_matches("z-image", "z-image-turbo")
    assert base_matches("z-image-turbo", "z-image-turbo")
    assert base_matches("moss") is False
    assert base_matches("moss-image-sft") is False
    assert base_matches("moss", "moss5b")
    assert base_matches("moss-image", "moss5b")
    assert base_matches("moss5b", "moss-image-sft")
    assert base_matches("moss-image-base", "moss-image-sft")
    assert base_matches("other", "z-image-turbo") is False


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
        build_plan(load_catalog(), _screened(), Tier.GB16, base="moss-image-sft"),
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
            build_plan(load_catalog(), _screened(), Tier.GB16, base="moss-image-sft"),
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
    generator.base = "moss-image-sft"
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
        model_id="voxprint-moss-image-sft-compact",
    )
    generator._schedule(plan, Mode.TEXT, 64, 64, "a red teapot on a table")
    assert "schedule" not in calls


def test_a_text_job_does_not_require_moss_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A default generate queues Z-Image, never MOSS, when weights are absent."""
    from canvas.models.loader import ModelNotReady

    queued: list[str] = []
    monkeypatch.setattr("canvas.models.loader.snapshot_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "canvas.models.loader.request_download",
        lambda model_id: queued.append(model_id) or True,
    )
    generator = ZImageGenerator(load_catalog(), Tier.GB16)
    generator.models = tmp_path
    with pytest.raises(ModelNotReady) as caught:
        generator.generate(_screened(), 32, 32)
    assert caught.value.missing is True
    assert "voxprint-moss-image-sft-compact" not in queued
    assert "z-image-turbo-bf16-16gb" in queued
    required = {spec.id for spec in required_specs(load_catalog())}
    assert "voxprint-moss-image-sft-compact" not in required


def test_users_without_an_explicit_model_pick_follow_the_catalog_default():
    """No stored base means the catalog default, which is now Z-Image-Turbo."""
    catalog = load_catalog()
    plan = build_plan(catalog, _screened(), Tier.GB16, base=None)
    assert plan.model_id == "z-image-turbo-bf16-16gb"
    moss = build_plan(catalog, _screened(), Tier.GB16, base="moss-image-sft")
    assert moss.model_id == "voxprint-moss-image-sft-compact"


def test_a_stored_moss_choice_selects_the_optional_moss_plan(tmp_path, monkeypatch):
    """model.default=moss is an explicit pick; implicit plans then use the compact row."""
    from voxprint_common.suite_settings import model_default, set_model_default

    from canvas.models.default_base import selected_text_base

    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    assert model_default() == "z-image"
    assert selected_text_base() == "z-image-turbo"
    set_model_default("moss")
    assert model_default() == "moss"
    assert selected_text_base() == "moss-image-sft"
    plan = build_plan(load_catalog(), _screened(), Tier.GB16)
    assert plan.model_id == "voxprint-moss-image-sft-compact"


def test_a_catalog_without_the_compact_moss_row_still_starts_and_plans(tmp_path):
    """Missing voxprint-moss-image-sft-compact must not raise RegistryError at load or generate."""
    from canvas.models.plan import OptionalModelUnavailable
    from canvas.models.registry import RegistryError

    raw = json.loads(Path("canvas/models/catalog.json").read_text(encoding="utf-8"))
    raw["models"] = [row for row in raw["models"] if row["id"] != "voxprint-moss-image-sft-compact"]
    dest = tmp_path / "catalog.json"
    dest.write_text(json.dumps(raw), encoding="utf-8")
    try:
        catalog = load_catalog(dest)
    except RegistryError as exc:
        raise AssertionError(f"startup RegistryError without the compact row: {exc}") from exc
    assert "voxprint-moss-image-sft-compact" not in {spec.id for spec in catalog.models}
    implicit = build_plan(catalog, _screened(), Tier.GB16)
    assert implicit.model_id == "z-image-turbo-bf16-16gb"
    moss = build_plan(catalog, _screened(), Tier.GB16, base="moss-image-sft")
    assert moss.model_id == "moss-image-sft-bf16-16gb"

    raw2 = json.loads(Path("canvas/models/catalog.json").read_text(encoding="utf-8"))
    raw2["models"] = [row for row in raw2["models"] if row.get("text_base") != "moss-image-sft"]
    dest.write_text(json.dumps(raw2), encoding="utf-8")
    catalog = load_catalog(dest)
    assert "moss-image-sft" not in catalog.text_bases
    implicit = build_plan(catalog, _screened(), Tier.GB16)
    assert implicit.model_id == "z-image-turbo-bf16-16gb"
    with pytest.raises(OptionalModelUnavailable):
        build_plan(catalog, _screened(), Tier.GB16, base="moss-image-sft")
