"""CPU-only stubs for the headless render_batch entry point. No weights are loaded."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from canvas.core.formats import OutputFormat
from canvas.safety.image import SafeImage
from canvas.tools.render_batch import (
    Hooks,
    compose_prompt,
    load_prompts,
    main,
    photo_id,
    preset_values,
)
from tests.fakes import World, rgb


def _picture() -> SafeImage:
    return World().gate().review_image(rgb(32, 32, (12, 24, 48)), "an adult woman, 28 years old")


def _prompts(path: Path, rows: list[object]) -> Path:
    path.write_text(json.dumps({"prompts": rows}), encoding="utf-8")
    return path


def _persona(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "name": "Mira Ellison",
                "age": 28,
                "fictional": True,
                "trigger": "<SYNTH_TRIGGER>",
                "description": "A 28-year-old adult woman named Mira Ellison, with a clearly adult face and body.",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_presets_come_from_common() -> None:
    steps, guidance, sampler = preset_values("balanced")
    assert steps == 8 and guidance == 0.0 and sampler == "euler"
    assert preset_values("fast")[0] == 4
    assert preset_values("best")[0] == 16


def test_a_missing_lora_exits_nonzero(tmp_path: Path) -> None:
    prompts = _prompts(tmp_path / "p.json", [{"id": "one", "prompt": "a 28-year-old adult woman in a studio"}])
    log = io.StringIO()
    code = main(
        [
            "--prompts",
            str(prompts),
            "--lora",
            str(tmp_path / "missing.safetensors"),
            "--preset",
            "balanced",
            "--adult-content",
            "on",
            "--out",
            str(tmp_path / "out"),
            "--seed",
            "1001",
        ],
        hooks=Hooks(weights_ready=lambda: True, log=log),
    )
    assert code != 0
    assert "missing" in log.getvalue().lower()


def test_missing_weights_exit_with_a_clear_message(tmp_path: Path) -> None:
    prompts = _prompts(tmp_path / "p.json", [{"prompt": "a 28-year-old adult woman standing at a window"}])
    log = io.StringIO()
    code = main(
        [
            "--prompts",
            str(prompts),
            "--preset",
            "balanced",
            "--adult-content",
            "on",
            "--out",
            str(tmp_path / "out"),
            "--seed",
            "1001",
        ],
        hooks=Hooks(weights_ready=lambda: False, log=log),
    )
    assert code == 2
    assert "Z-Image-Turbo weights are missing" in log.getvalue()


def test_render_goes_through_the_gate_and_writes_lossy_webp(tmp_path: Path, monkeypatch) -> None:
    prompts = _prompts(
        tmp_path / "p.json",
        [{"id": "studio", "prompt": "a 28-year-old adult woman, face clearly visible, studio light"}],
    )
    seen: list[tuple[str, int, bool]] = []
    saved: list[tuple[object, object]] = []

    def generate(prompt: str, seed: int, *, explicit: bool) -> SafeImage:
        seen.append((prompt, seed, explicit))
        return _picture()

    def capture(image, path, fmt, *, metadata=None, purpose=None, **kwargs):
        saved.append((fmt, purpose, metadata, path))
        path.write_bytes(b"RIFF....WEBP")

    monkeypatch.setattr("canvas.tools.render_batch.save_image", capture)
    log = io.StringIO()
    code = main(
        [
            "--prompts",
            str(prompts),
            "--preset",
            "balanced",
            "--adult-content",
            "on",
            "--out",
            str(tmp_path / "out"),
            "--seed",
            "1001",
        ],
        hooks=Hooks(weights_ready=lambda: True, generate=generate, log=log),
    )
    assert code == 0
    assert seen == [("a 28-year-old adult woman, face clearly visible, studio light", 1001, True)]
    assert saved[0][0] is OutputFormat.WEBP
    assert saved[0][1] == "picture"
    # Metadata is off, but the seed is always written.
    assert saved[0][2] == {"seed": 1001}
    record = json.loads((tmp_path / "out" / "studio.json").read_text(encoding="utf-8"))
    assert record["seed"] == 1001
    assert record["steps"] == 8
    assert record["guidance"] == 0.0
    assert record["gate"] == "passed"
    assert "seconds" in record
    assert record["base"] == "z-image-turbo"


def test_dataset_writes_the_forge_photos_layout(tmp_path: Path, monkeypatch) -> None:
    persona = _persona(tmp_path / "persona.json")
    prompts = _prompts(
        tmp_path / "shots.json",
        [
            {"id": "close", "prompt": "Close-up portrait, face clearly visible, window light.", "short": "close-up, window light"},
            {"id": "full", "prompt": "Full body, standing, golden hour.", "caption": "full body, golden hour"},
        ],
    )
    seen_seeds: list[int] = []

    def generate(prompt: str, seed: int, *, explicit: bool) -> SafeImage:
        del explicit
        seen_seeds.append(seed)
        assert "28-year-old" in prompt
        return _picture()

    def capture(image, path, fmt, *, metadata=None, purpose=None, **kwargs):
        del image, fmt, metadata
        # Distinct bytes so each photo_id differs.
        path.write_bytes(b"WEBP" + path.name.encode() + b"\x00" * 16)
        assert purpose == "master"

    monkeypatch.setattr("canvas.tools.render_batch.save_image", capture)
    log = io.StringIO()
    out = tmp_path / "set"
    code = main(
        [
            "--mode",
            "dataset",
            "--persona",
            str(persona),
            "--prompts",
            str(prompts),
            "--preset",
            "balanced",
            "--adult-content",
            "on",
            "--out",
            str(out),
            "--seed",
            "2000",
        ],
        hooks=Hooks(weights_ready=lambda: True, generate=generate, log=log),
    )
    assert code == 0
    assert seen_seeds == [2000, 2001]
    photos = sorted((out / "photos").glob("*.webp"))
    assert [p.name for p in photos] == ["01.webp", "02.webp"]
    # Only image files in the photos folder (Forge collect() filters by suffix).
    assert {p.suffix for p in (out / "photos").iterdir()} == {".webp"}
    lines = (out / "captions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["id"] == photo_id(photos[0].read_bytes())
    assert "long" in first and "short" in first
    assert first["id"] != json.loads(lines[1])["id"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "dataset"
    assert manifest["base"] == "z-image-turbo"
    assert manifest["photos_dir"] == "photos"
    assert manifest["captions"] == "captions.jsonl"
    assert manifest["lora"] is None
    assert manifest["items"][0]["seed"] == 2000
    assert manifest["items"][0]["id"] == first["id"]
    assert manifest["items"][0]["gate"] == "passed"


def test_dataset_rejects_a_lora_flag(tmp_path: Path) -> None:
    log = io.StringIO()
    code = main(
        [
            "--mode",
            "dataset",
            "--persona",
            str(_persona(tmp_path / "persona.json")),
            "--prompts",
            str(_prompts(tmp_path / "p.json", [{"prompt": "a 28-year-old adult woman"}])),
            "--lora",
            str(tmp_path / "any.safetensors"),
            "--adult-content",
            "on",
            "--out",
            str(tmp_path / "out"),
        ],
        hooks=Hooks(weights_ready=lambda: True, log=log),
    )
    assert code == 2
    assert "does not take a LoRA" in log.getvalue()


def test_compose_prompt_prefixes_the_persona_once() -> None:
    from canvas.tools.render_batch import PromptItem

    persona = {"description": "A 28-year-old adult woman named Mira Ellison."}
    item = PromptItem(prompt="Close-up, window light.")
    text = compose_prompt(item, persona)
    assert text.startswith("A 28-year-old adult woman named Mira Ellison.")
    assert compose_prompt(PromptItem(prompt=text), persona) == text


def test_load_prompts_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no prompts"):
        load_prompts(path)


def test_preflight_names_the_models_dir_and_missing_ids(tmp_path: Path, monkeypatch) -> None:
    from canvas.models.modes import Tier
    from canvas.tools.render_batch import format_preflight_error, missing_model_ids

    monkeypatch.setattr("canvas.models.snapshots.snapshot_ready", lambda spec, root, cache=None: None)
    missing = missing_model_ids(tmp_path / "models", Tier.GB16)
    assert "z-image-turbo-bf16-16gb" in missing
    for model_id in (
        "qwen3guard-gen-0.6b",
        "siglip2-so400m-patch14-384",
        "yunet-face",
        "fairface-age",
        "mivolo-v2-age",
        "nsfw-image-detection",
    ):
        assert model_id in missing
    message = format_preflight_error(tmp_path / "models", "VOXPRINT_MODELS_DIR", missing)
    assert str(tmp_path / "models") in message
    assert "VOXPRINT_MODELS_DIR" in message
    assert "qwen3guard-gen-0.6b" in message
    assert "voxprint-mdl download" in message


def test_render_batch_prints_models_dir_and_exits_on_a_live_preflight_miss(
    tmp_path: Path, monkeypatch,
) -> None:
    from canvas.app.platform import GpuCheck
    from canvas.models.modes import Tier

    prompts = _prompts(tmp_path / "p.json", [{"id": "one", "prompt": "a 32-year-old adult woman in a studio"}])
    models = tmp_path / "mdl-root"
    models.mkdir()
    monkeypatch.setattr("canvas.app.platform.check_gpu", lambda info: GpuCheck(ok=True, tier=Tier.GB16, message=""))
    monkeypatch.setattr("canvas.infra.gpu.probe_gpu", lambda: object())
    monkeypatch.setattr("canvas.models.snapshots.snapshot_ready", lambda spec, root, cache=None: None)
    monkeypatch.setattr("canvas.tools.render_batch.missing_gate_imports", lambda: [])
    log = io.StringIO()
    # No weights_ready hook: this is the live preflight path.
    code = main(
        [
            "--prompts",
            str(prompts),
            "--preset",
            "balanced",
            "--adult-content",
            "on",
            "--out",
            str(tmp_path / "out"),
            "--seed",
            "1001",
            "--models-dir",
            str(models),
        ],
        hooks=Hooks(log=log),
    )
    text = log.getvalue()
    assert code == 2
    assert f"Models folder: {models.resolve()} (from VOXPRINT_MODELS_DIR)" in text
    assert "qwen3guard-gen-0.6b" in text
    assert "z-image-turbo-bf16-16gb" in text
    assert "voxprint-mdl download" in text


def test_missing_gate_imports_names_a_missing_package(monkeypatch) -> None:
    import importlib

    from canvas.tools.render_batch import format_deps_error, missing_gate_imports

    real = importlib.import_module

    def fake(name: str, package: str | None = None):
        if name in {"transformers", "timm"}:
            raise ImportError(name)
        return real(name, package)

    monkeypatch.setattr("canvas.tools.render_batch.importlib.import_module", fake)
    missing = missing_gate_imports()
    assert "transformers" in missing
    assert "timm" in missing
    message = format_deps_error(missing)
    assert "transformers" in message
    assert "timm" in message
    assert "pip install .[gen]" in message


def test_missing_gate_imports_rejects_opencv_without_utils_logging(monkeypatch) -> None:
    import importlib
    import types

    from canvas.tools.render_batch import missing_gate_imports

    real = importlib.import_module
    fake_cv2 = types.SimpleNamespace(__version__="4.10.0", utils=types.SimpleNamespace())

    def fake(name: str, package: str | None = None):
        if name == "cv2":
            return fake_cv2
        return real(name, package)

    monkeypatch.setattr("canvas.tools.render_batch.importlib.import_module", fake)
    missing = missing_gate_imports()
    assert any("opencv-python-headless" in item and "4.10.0" in item and "5.x" in item for item in missing)


def test_render_batch_exits_on_missing_gate_imports(tmp_path: Path, monkeypatch) -> None:
    from canvas.app.platform import GpuCheck
    from canvas.models.modes import Tier

    prompts = _prompts(tmp_path / "p.json", [{"id": "one", "prompt": "a 32-year-old adult woman in a studio"}])
    models = tmp_path / "mdl-root"
    models.mkdir()
    monkeypatch.setattr("canvas.tools.render_batch.missing_gate_imports", lambda: ["transformers", "timm"])
    monkeypatch.setattr("canvas.app.platform.check_gpu", lambda info: GpuCheck(ok=True, tier=Tier.GB16, message=""))
    monkeypatch.setattr("canvas.infra.gpu.probe_gpu", lambda: object())
    monkeypatch.setattr("canvas.models.snapshots.snapshot_ready", lambda spec, root, cache=None: None)
    log = io.StringIO()
    code = main(
        [
            "--prompts",
            str(prompts),
            "--preset",
            "balanced",
            "--adult-content",
            "on",
            "--out",
            str(tmp_path / "out"),
            "--seed",
            "1001",
            "--models-dir",
            str(models),
        ],
        hooks=Hooks(log=log),
    )
    text = log.getvalue()
    assert code == 2
    assert "transformers" in text
    assert "timm" in text
    assert "pip install .[gen]" in text
    assert "qwen3guard-gen-0.6b" not in text


def test_safety_error_fields_record_type_and_one_line_cause() -> None:
    from canvas.safety.gate import SafetyUnavailable
    from canvas.tools.render_batch import safety_error_fields

    bare = SafetyUnavailable("classifier_error")
    assert safety_error_fields(bare) == {"error_type": "SafetyUnavailable"}
    assert safety_error_fields(None) == {}
    assert safety_error_fields(RuntimeError("no")) == {}

    try:
        raise SafetyUnavailable("classifier_error") from AttributeError(
            "module 'cv2.utils' has no attribute 'logging'\nsecond line"
        )
    except SafetyUnavailable as exc:
        fields = safety_error_fields(exc)
    assert fields["error_type"] == "SafetyUnavailable"
    assert fields["error_cause"] == (
        "AttributeError: module 'cv2.utils' has no attribute 'logging' second line"
    )
    assert "\n" not in fields["error_cause"]


def test_render_records_a_safety_error_cause_in_the_image_json(tmp_path: Path) -> None:
    from canvas.safety.gate import SafetyUnavailable

    def generate(prompt: str, seed: int, *, explicit: bool) -> SafeImage:
        del prompt, seed, explicit
        raise SafetyUnavailable("classifier_error") from AttributeError(
            "module 'cv2.utils' has no attribute 'logging'"
        )

    prompts = _prompts(
        tmp_path / "p.json",
        [{"id": "blocked", "prompt": "a 32-year-old adult woman in a studio"}],
    )
    log = io.StringIO()
    code = main(
        [
            "--prompts",
            str(prompts),
            "--preset",
            "balanced",
            "--adult-content",
            "on",
            "--out",
            str(tmp_path / "out"),
            "--seed",
            "1001",
        ],
        hooks=Hooks(weights_ready=lambda: True, generate=generate, log=log),
    )
    record = json.loads((tmp_path / "out" / "blocked.json").read_text(encoding="utf-8"))
    assert code == 1
    assert not (tmp_path / "out" / "blocked.webp").exists()
    assert record["gate"] == "classifier_error"
    assert record["error_type"] == "SafetyUnavailable"
    assert record["error_cause"] == "AttributeError: module 'cv2.utils' has no attribute 'logging'"
    assert "prompt" not in record["error_cause"]
    assert "\n" not in record["error_cause"]
    assert "error" not in record
