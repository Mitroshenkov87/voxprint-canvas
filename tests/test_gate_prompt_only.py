"""Prompt-check-only generation (owner rule 2026-11-16).

Canvas screens the request, not the result:

- a generated picture leaves through ``release_generated``: no NSFW score, no
  zero-shot, no face detection and no age estimate on generated pixels, with
  no LoRA, an unverified LoRA or a Forge-verified LoRA;
- generation warms only the prompt guard; the image check models load lazily,
  only when an uploaded input image is screened;
- input-image checks still run and still refuse;
- the prompt check stays fail-closed: a guard that cannot load, errors or
  answers junk refuses the request and the generator is never called.

The prompt refusal rules themselves are covered, unchanged, by the existing
gate tests (``tests/test_gate.py`` and friends).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from canvas.lora_verify import ALGORITHM, SCHEMA_ID, file_sha256, record_path_for, sign_record
from canvas.models.modes import Mode, Tier
from canvas.pipeline.pipeline import Pipeline
from canvas.safety.classifiers import Face, GuardVerdict
from canvas.safety.gate import (
    AgeGate,
    GenerationBlocked,
    SafetyStack,
    SafetyUnavailable,
    release_generated,
)
from canvas.safety.image import SafeImage
from canvas.safety.inputs import ImageRole, PendingInput
from tests.fakes import World, rgb

IMAGE_CALLS = {"nsfw", "image-score", "faces", "ages", "fine"}
IMAGE_MODELS = ("nsfw", "zeroshot", "faces", "ages", "fine")


@pytest.fixture(autouse=True)
def _suite_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "suite"))


def _flagged_world() -> World:
    """Image checks that would refuse any picture they saw (young face, high NSFW score)."""
    world = World()
    world.nsfw = 0.95
    world.faces = [Face(0, 0, 1, 1)]
    world.ages = [(1.0, 0.0)]
    return world


def _with_loaders(world: World) -> tuple[AgeGate, list[str]]:
    """The world's gate, with a ``load`` on every component that records its name."""
    gate = world.gate()
    loaded: list[str] = []
    for name in ("guard", *IMAGE_MODELS):
        component = getattr(gate._stack, name)
        component.load = (lambda name=name: loaded.append(name))  # type: ignore[attr-defined]
    return gate, loaded


class _Generator:
    def __init__(self) -> None:
        self.calls = 0
        self.bound_lora = None

    def generate(self, screened, width=None, height=None):
        del screened, width, height
        self.calls += 1
        return rgb(4, 4, (7, 8, 9))


def _signed(tmp_path: Path) -> Path:
    weights = tmp_path / "step-001600.safetensors"
    weights.write_bytes(b"lora-weights")
    record = {
        "schema": SCHEMA_ID, "record_id": "forge-1", "lora_file": weights.name,
        "lora_sha256": file_sha256(weights), "project_id": "p", "dataset_image_count": 3,
        "age_gate_version": "5", "verified": True, "result": "pass",
        "verified_at": "2026-11-15T00:30:00+03:00", "algorithm": ALGORITHM,
    }
    record["signature"] = sign_record(record)
    record_path_for(weights).write_text(json.dumps(record), encoding="utf-8")
    return weights


# release_generated


def test_release_generated_mints_a_safe_image_and_wipes_the_raw_buffer() -> None:
    raw = rgb(2, 2, (1, 2, 3))
    safe = release_generated(raw)
    assert type(safe) is SafeImage
    assert (safe.width, safe.height, safe.data) == (2, 2, bytes([1, 2, 3]) * 4)
    assert raw.to_bytes() == b"" and raw.width == 0


def test_release_generated_wipes_the_raw_buffer_when_minting_fails(monkeypatch) -> None:
    def broken(_raw):
        raise RuntimeError("mint failed")

    monkeypatch.setattr("canvas.safety.gate.issue", broken)
    raw = rgb()
    with pytest.raises(RuntimeError):
        release_generated(raw)
    assert raw.to_bytes() == b""


# warm_prompt


def test_warm_prompt_loads_only_the_guard_once() -> None:
    gate, loaded = _with_loaders(World())
    gate.warm_prompt()
    gate.warm_prompt()
    gate.review_prompt("a red boat")
    assert loaded == ["guard"]
    assert gate._loaded == {"guard"}


def test_an_input_image_loads_the_age_models_lazily_for_an_18_plus_edit_only() -> None:
    gate, loaded = _with_loaders(World())
    gate.warm_prompt()
    gate.release_request("a red boat", "restyle", [PendingInput(ImageRole.INIT, rgb())])
    assert loaded == ["guard"]
    gate.release_request("nude, adult woman", "restyle", [PendingInput(ImageRole.INIT, rgb())])
    assert loaded == ["guard", *(name for name in IMAGE_MODELS if name != "nsfw")]


def test_warm_prompt_is_fail_closed_when_the_guard_cannot_load() -> None:
    gate = World().gate()

    def broken() -> None:
        raise SafetyUnavailable("classifier_error")

    gate._stack.guard.load = broken  # type: ignore[attr-defined]
    with pytest.raises(SafetyUnavailable):
        gate.warm_prompt()
    # The failure sticks: the next prompt check refuses too.
    with pytest.raises(SafetyUnavailable):
        gate.review_prompt("a red boat")


# Generation through the pipeline


@pytest.mark.parametrize("lora", ["none", "unverified", "verified"])
def test_generation_runs_no_output_check_and_loads_no_image_model(tmp_path, lora) -> None:
    world = _flagged_world()
    gate, loaded = _with_loaders(world)
    generator = _Generator()
    if lora != "none":
        weights = _signed(tmp_path) if lora == "verified" else tmp_path / "plain.safetensors"
        if lora == "unverified":
            weights.write_bytes(b"x")
        generator.bound_lora = SimpleNamespace(weights=weights)
    gate.warm_prompt()
    image = Pipeline(gate, generator, Tier.GB24).generate("a red boat", Mode.TEXT)
    assert type(image) is SafeImage
    assert image.data == bytes([7, 8, 9]) * 16
    assert generator.calls == 1
    assert world.calls == ["guard"]
    assert set(loaded) == {"guard"}


def test_generation_never_calls_review_image(monkeypatch) -> None:
    def no_check(*_args, **_kwargs):
        raise AssertionError("no generate path may call review_image")

    monkeypatch.setattr(AgeGate, "review_image", no_check)
    image = Pipeline(_flagged_world().gate(), _Generator(), Tier.GB24).generate("a red boat", Mode.TEXT)
    assert type(image) is SafeImage


def test_an_input_image_of_a_minor_in_an_18_plus_edit_is_still_checked_and_refused() -> None:
    world = _flagged_world()
    generator = _Generator()
    source = rgb()
    with pytest.raises(GenerationBlocked) as caught:
        Pipeline(world.gate(), generator, Tier.GB24).generate(
            "nude", Mode.RESTYLE, [PendingInput(ImageRole.INIT, source)],
        )
    assert caught.value.code == "input"
    assert generator.calls == 0
    assert "ages" in world.calls and "nsfw" not in world.calls
    assert source.to_bytes() == b""


def test_a_clean_input_image_is_checked_and_the_output_is_not() -> None:
    world = World()
    generator = _Generator()
    Pipeline(world.gate(), generator, Tier.GB24).generate(
        "a red boat", Mode.RESTYLE, [PendingInput(ImageRole.INIT, rgb())],
    )
    assert generator.calls == 1
    # The edit is not 18+: no image model runs on the input, nothing after the generator.
    assert world.calls == ["guard"]


def test_a_minors_prompt_is_refused_before_the_generator() -> None:
    world = World()
    world.safety = "unsafe"
    world.categories = ("Minors", "Sexual Content or Sexual Acts")
    generator = _Generator()
    with pytest.raises(GenerationBlocked):
        Pipeline(world.gate(), generator, Tier.GB24).generate("a red boat", Mode.TEXT)
    assert generator.calls == 0


def test_a_guard_that_errors_refuses_and_nothing_is_generated() -> None:
    world = World()
    world.error = "guard"
    generator = _Generator()
    with pytest.raises(SafetyUnavailable):
        Pipeline(world.gate(), generator, Tier.GB24).generate("a red boat", Mode.TEXT)
    assert generator.calls == 0


def test_a_guard_that_cannot_load_refuses_and_nothing_is_written(tmp_path) -> None:
    gate = World().gate()

    def broken() -> None:
        raise SafetyUnavailable("classifier_error")

    gate._stack.guard.load = broken  # type: ignore[attr-defined]
    generator = _Generator()
    path = tmp_path / "out.webp"
    with pytest.raises(SafetyUnavailable):
        Pipeline(gate, generator, Tier.GB24).export("a red boat", Mode.TEXT, (), path)
    assert generator.calls == 0
    assert not path.exists()


def test_a_guard_that_answers_junk_refuses() -> None:
    class Junk:
        def assess(self, text: str) -> GuardVerdict:
            del text
            return GuardVerdict("maybe", ())

    stack = World().stack()
    stack = SafetyStack(Junk(), stack.zeroshot, stack.faces, stack.ages, stack.nsfw, stack.fine)  # type: ignore[arg-type]
    generator = _Generator()
    with pytest.raises(SafetyUnavailable):
        Pipeline(AgeGate(stack), generator, Tier.GB24).generate("a red boat", Mode.TEXT)
    assert generator.calls == 0


def test_no_generate_path_calls_the_output_check() -> None:
    """Source guard: generate paths use release_generated and warm_prompt, never review_image or warm."""
    root = Path(__file__).resolve().parents[1] / "canvas"
    paths = (
        "pipeline/pipeline.py", "pipeline/worker.py", "pipeline/series.py", "pipeline/batch.py",
        "app/shell_gate.py", "app/main.py", "tools/render_batch.py", "worker_jobs.py",
    )
    for relative in paths:
        source = (root / relative).read_text(encoding="utf-8")
        assert "review_image(" not in source, relative
        assert "review_or_skip(" not in source, relative
        assert "gate.warm()" not in source, relative
