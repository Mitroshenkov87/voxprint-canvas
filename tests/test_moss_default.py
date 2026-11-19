"""Z-Image-Turbo is the default text generator. MOSS is no longer a Canvas option."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canvas.models.generate_picker import retired_generate_id, stored_generate_model
from canvas.models.loader import Generator, ZImageGenerator
from canvas.models.mdl_manifest import render
from canvas.models.modes import Mode, Tier
from canvas.models.pins import PIPELINE_TEXT, REPO_TURBO
from canvas.models.plan import PIPELINE_IMG2IMG, OptionalModelUnavailable, build_plan
from canvas.models.registry import load_catalog
from canvas.models.required import required_specs
from canvas.models.user_lora import base_matches, running_base_id
from canvas.safety.inputs import ImageRole
from tests.fakes import World, rgb


_MOSS_GENERATOR_IDS = (
    "voxprint-moss-image-sft-compact",
    "moss-image-sft-bf16-16gb",
    "moss-image-sft-bf16",
)


def _store_model_default(value: str) -> None:
    """Write model.default to suite.json, including leftover ids common 0.7.4 rejects."""
    from voxprint_common import suite_settings

    path = suite_settings.file()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, ValueError):
            loaded = {}
        if isinstance(loaded, dict):
            payload = loaded
    payload["model.default"] = value
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    suite_settings._good_token = None
    suite_settings._good_path = None


def _screened(mode: Mode = Mode.TEXT, *roles: ImageRole):
    from canvas.safety.inputs import PendingInput

    world = World()
    frames = [PendingInput(role, rgb()) for role in roles]
    return world.gate().release_request("a red teapot on a table", mode.value, frames)


def test_catalog_loads_with_z_image_turbo_as_the_only_text_base():
    catalog = load_catalog()
    assert catalog.default_text_base == "z-image-turbo"
    assert catalog.text_bases == ("z-image-turbo",)
    assert "z-image-turbo" in catalog.sizes
    ids = {spec.id for spec in catalog.models}
    for model_id in _MOSS_GENERATOR_IDS:
        assert model_id not in ids
    required = {spec.id for spec in required_specs(catalog)}
    assert "z-image-turbo-bf16-16gb" in required
    assert catalog.by_id("z-image-turbo-bf16-16gb").optional is False
    for model_id in (
        "z-image-turbo-fun-controlnet-union-bf16-lite",
        "z-image-turbo-fun-controlnet-union-bf16",
    ):
        assert catalog.by_id(model_id).optional is True
        assert model_id not in required
    assert Generator is ZImageGenerator


def test_the_common_compressed_placeholder_is_not_a_canvas_pin():
    """voxprint-common 0.7.0 no longer lists moss-image-compressed.

    Looking the id up must be absent-safe (None, not an error). Canvas must
    not invent a hash or treat a removed placeholder as a pin.
    """
    from voxprint_common.mdl.index import load_index
    from voxprint_common.models import default_model
    from voxprint_common.suite_settings import model_default

    index = load_index(remote=False)
    assert index.get("moss-image-compressed") is None
    turbo = index.get("z-image-turbo")
    assert turbo is not None
    assert turbo.default is True
    assert turbo.optional is False
    z_image = index.get("z-image")
    assert z_image is not None
    assert z_image.default is False
    assert default_model() == "z-image-turbo"
    assert model_default() == "z-image-turbo"
    raw = json.loads(Path("canvas/models/catalog.json").read_text(encoding="utf-8"))
    ids = {row["id"] for row in raw["models"]}
    assert "moss-image-compressed" not in ids
    for model_id in _MOSS_GENERATOR_IDS:
        assert model_id not in ids


def test_text_plans_use_z_image_turbo_and_refuse_a_moss_family():
    catalog = load_catalog()
    small = build_plan(catalog, _screened(), Tier.GB16)
    assert small.model_id == "z-image-turbo-bf16-16gb"
    assert small.pipeline_class == PIPELINE_TEXT
    assert small.steps == 8 and small.guidance_scale == 0.0
    large = build_plan(catalog, _screened(), Tier.GB24)
    assert large.model_id == "z-image-turbo-bf16-16gb"  # the FP8 16 GB pack is the only Turbo build
    assert large.pipeline_class == PIPELINE_TEXT
    for family in ("moss-image-sft", "moss", "moss5b"):
        with pytest.raises(OptionalModelUnavailable):
            build_plan(catalog, _screened(), Tier.GB16, base=family)
    restyle = build_plan(catalog, _screened(Mode.RESTYLE, ImageRole.INIT), Tier.GB16)
    assert restyle.pipeline_class == PIPELINE_IMG2IMG
    assert restyle.repos == (REPO_TURBO,)


def test_unpinned_rows_do_not_list_moss_and_z_image_rows_are_optional():
    document = json.loads(render())
    listed = {row["id"] for row in document["models"]}
    pending = {item["id"]: item["note"] for item in document.get("pending") or []}
    for model_id in _MOSS_GENERATOR_IDS:
        assert model_id not in listed
        assert model_id not in pending
    assert "moss-image-sft-compact-16gb" not in pending
    rows = {row["id"]: row for row in document["models"]}
    assert rows["z-image-turbo-bf16-16gb"]["optional"] is False  # the FP8 pack is the required Turbo row
    assert "z-image-turbo-bf16-16gb-upstream" not in rows  # the unlisted fallback is not emitted
    for model_id in (
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


def test_a_text_job_does_not_require_moss_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A default generate queues Z-Image-Turbo, never MOSS, when weights are absent."""
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
    for model_id in _MOSS_GENERATOR_IDS:
        assert model_id not in queued
    assert "z-image-turbo-bf16-16gb" in queued
    required = {spec.id for spec in required_specs(load_catalog())}
    for model_id in _MOSS_GENERATOR_IDS:
        assert model_id not in required


def test_users_without_an_explicit_model_pick_follow_the_catalog_default():
    catalog = load_catalog()
    plan = build_plan(catalog, _screened(), Tier.GB16, base=None)
    assert plan.model_id == "z-image-turbo-bf16-16gb"


def test_a_leftover_moss_suite_choice_remaps_to_turbo(tmp_path, monkeypatch):
    """A leftover model.default=moss is read as Turbo and plans the compact Turbo row."""
    from voxprint_common.suite_settings import model_default

    from canvas.models.default_base import selected_text_base

    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    assert model_default() == "z-image-turbo"
    assert selected_text_base() == "z-image-turbo"
    _store_model_default("moss")
    assert model_default() == "z-image-turbo"
    assert selected_text_base() == "z-image-turbo"
    plan = build_plan(load_catalog(), _screened(), Tier.GB16)
    assert plan.model_id == "z-image-turbo-bf16-16gb"


def test_a_saved_moss_generate_model_is_retired():
    assert retired_generate_id("moss-image-sft-bf16")
    assert retired_generate_id("moss-image-sft-bf16-16gb")
    assert retired_generate_id("voxprint-moss-image-sft-compact")
    assert stored_generate_model("moss-image-sft-bf16") == ""
    assert stored_generate_model("z-image-turbo-bf16-16gb") == "z-image-turbo-bf16-16gb"


def test_full_z_image_is_an_optional_download(tmp_path, monkeypatch):
    """model.default=z-image is the optional full model, not an alias of Turbo."""
    from voxprint_common.suite_settings import model_default, set_model_default

    from canvas.models.default_base import selected_text_base
    from canvas.models.registry import RegistryError

    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    set_model_default("z-image")
    assert model_default() == "z-image"
    assert selected_text_base() == "z-image"
    catalog = load_catalog()
    assert "z-image" not in catalog.text_bases
    # The catalog has no full z-image text row yet: planning must not raise RegistryError.
    try:
        build_plan(catalog, _screened(), Tier.GB16, base="z-image")
    except OptionalModelUnavailable:
        pass
    except RegistryError as exc:
        raise AssertionError(f"full z-image raised RegistryError: {exc}") from exc
    else:
        raise AssertionError("expected OptionalModelUnavailable when the full z-image row is absent")
    # Default (omitted base) still follows the stored z-image pick, which is optional/unavailable.
    try:
        build_plan(catalog, _screened(), Tier.GB16)
    except OptionalModelUnavailable:
        pass
    except RegistryError as exc:
        raise AssertionError(f"implicit full z-image raised RegistryError: {exc}") from exc
    else:
        raise AssertionError("expected OptionalModelUnavailable for a stored full z-image pick")
