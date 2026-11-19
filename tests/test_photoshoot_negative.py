"""photoshoot --negative: the negative prompt is checked like the prompt."""
from __future__ import annotations

import json
from pathlib import Path

from canvas.app.main import _parser, main
from canvas.app.photoshoot_cli import model_generate, run_command
from canvas.models.modes import Tier
from canvas.safety.gate import GenerationBlocked
from canvas.safety.image import RawImage
from tests.fakes import World, rgb
from tests.test_photoshoot import IDEA, _payload
from tests.test_photoshoot_cli import _ready

NEGATIVE = "blurry, washed out colours"


def _rewrite(prompt: str) -> str:
    body = json.loads(prompt)
    return _payload(str(body["idea"]), int(body["count"]), start=len(body.get("previous", [])))


def _stub_gate(refuse: str = ""):
    """The fake gate with a stubbed prompt check that records every text it sees."""
    gate = World().gate()
    real = gate.review_prompt
    seen: list[str] = []

    def review_prompt(prompt: str) -> None:
        seen.append(prompt)
        if refuse and prompt == refuse:
            raise GenerationBlocked("prompt")
        real(prompt)

    gate.review_prompt = review_prompt
    return gate, seen


def _run(tmp_path: Path, gate, *, idea: str = IDEA, negative: str = ""):
    rendered: list[str] = []
    written: list[str] = []

    def rewrite(prompt: str) -> str:
        written.append(prompt)
        return _rewrite(prompt)

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> RawImage:
        rendered.append(line)
        return rgb()

    payload, code = run_command(
        idea=idea, folder=tmp_path / "series", frames=2, seed=1, style="photo", places="same",
        adult=False, age_gate=gate, rewrite=rewrite, generate=generate, negative=negative,
    )
    return payload, code, rendered, written


def test_the_negative_is_passed_to_the_prompt_check(tmp_path: Path) -> None:
    gate, seen = _stub_gate()
    payload, code, rendered, _written = _run(tmp_path, gate, negative=NEGATIVE)
    assert NEGATIVE in seen
    assert code == 0 and payload["ok"] is True
    assert rendered


def test_a_refused_negative_generates_nothing(tmp_path: Path) -> None:
    gate, seen = _stub_gate(refuse=NEGATIVE)
    payload, code, rendered, written = _run(tmp_path, gate, negative=NEGATIVE)
    assert NEGATIVE in seen
    assert code == 1
    assert payload["ok"] is False
    assert payload["made"] == 0
    assert payload["frames"] == []
    assert rendered == []
    assert written == []
    assert not (tmp_path / "series" / "scenario.txt").exists()


def test_a_refused_negative_stops_like_a_refused_prompt(tmp_path: Path) -> None:
    gate, _seen = _stub_gate(refuse=NEGATIVE)
    by_negative, negative_code, _, _ = _run(tmp_path / "negative", gate, negative=NEGATIVE)
    gate, _seen = _stub_gate(refuse=IDEA)
    by_prompt, prompt_code, _, _ = _run(tmp_path / "prompt", gate)
    keys = ("ok", "error", "message", "code", "made", "frames", "shots")
    assert {key: by_negative.get(key) for key in keys} == {key: by_prompt.get(key) for key in keys}
    assert negative_code == prompt_code


def test_an_empty_negative_adds_no_extra_check(tmp_path: Path) -> None:
    gate, seen = _stub_gate()
    _run(tmp_path, gate)
    assert "" not in seen


def test_the_cli_option_reaches_the_check_and_the_generator(monkeypatch, tmp_path: Path) -> None:
    _ready(monkeypatch)
    calls: dict[str, object] = {}

    def fake_generate(*args: object) -> object:
        calls["generate_args"] = args
        return lambda *_frame: None

    def fake_run(**kwargs: object) -> tuple[dict[str, object], int]:
        calls["run"] = kwargs
        return {"ok": True, "frames": [], "scenario": None, "shots": []}, 0

    monkeypatch.setattr("canvas.app.main.model_rewrite", lambda *_args: _rewrite)
    monkeypatch.setattr("canvas.app.main.model_generate", fake_generate)
    monkeypatch.setattr("canvas.app.main.run_command", fake_run)
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    code = main([
        "photoshoot", "--prompt", IDEA, "--negative", NEGATIVE, "--out", str(tmp_path / "out"), "--json",
    ])
    assert code == 0
    run = calls["run"]
    assert isinstance(run, dict) and run["negative"] == NEGATIVE
    generate_args = calls["generate_args"]
    assert isinstance(generate_args, tuple) and generate_args[-1] == NEGATIVE


def test_the_parser_default_is_empty(tmp_path: Path) -> None:
    args = _parser().parse_args(["photoshoot", "--prompt", IDEA, "--out", str(tmp_path)])
    assert args.negative == ""


def test_the_frame_negative_carries_the_user_text(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    negatives: list[str] = []

    class _Gen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.seed = 0
            self.lora_path = ""
            self.lora_strength = 0.7
            self.img2img_strength = None
            self.models = None
            self.negative_prompt = ""

        def generate(self, screened: object, width: int | None = None, height: int | None = None) -> RawImage:
            del screened, width, height
            negatives.append(self.negative_prompt)
            return rgb()

    monkeypatch.setattr("canvas.app.shell_run.ZImageGenerator", _Gen)
    gate, seen = _stub_gate()
    payload, code = run_command(
        idea=IDEA, folder=tmp_path / "series", frames=2, seed=1, style="photo", places="same",
        adult=False, age_gate=gate, rewrite=_rewrite,
        generate=model_generate(gate, Tier.GB16, "photo", False, negative=NEGATIVE),
        negative=NEGATIVE,
    )
    assert code == 0, payload
    assert NEGATIVE in seen
    assert negatives and all(NEGATIVE in text for text in negatives)
