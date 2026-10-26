"""The Photoshoot bench times six frames and does not invent a duration."""
from __future__ import annotations

import json
from pathlib import Path

from canvas.app.main import _parser, main
from canvas.app.photoshoot_bench import SERIES_FRAMES, live_generate, live_rewrite, measure_series
from canvas.pipeline.photoshoot import TEST_SERIES_CHARACTER, scenario_request, scenario_token_budget
from canvas.pipeline.series import REFERENCE_STRENGTH
from canvas.safety.image import RawImage
from canvas.app.platform import GpuCheck, OsCheck
from canvas.infra.gpu import GpuInfo
from canvas.models.modes import Tier
from tests.fakes import World
from tests.test_photoshoot import IDEA, _frame, _payload
from tests.test_photoshoot_cli import _arm_runner, _rewrite


def test_measure_series_times_six_frames_from_text(tmp_path: Path) -> None:
    heroes: list[object | None] = []
    strengths: list[float | None] = []

    def generate(line: str, seed: int, hero: object | None, strength: float) -> object:
        del line, seed
        heroes.append(hero)
        strengths.append(strength)
        return _frame()

    report = measure_series(
        idea=IDEA,
        folder=tmp_path,
        seed=3,
        rewrite=lambda _prompt: _payload(IDEA),
        age_gate=World().gate(),
        generate=generate,
    )
    assert report["frames"] == SERIES_FRAMES == 6
    assert report["kept"] == 6
    assert report["stopped"] == ""
    assert report["ok"] is True
    assert isinstance(report["seconds"], float)
    assert report["seconds"] >= 0
    assert strengths == [None] * 6
    assert all(hero is None for hero in heroes)


def test_live_generate_copies_the_reused_hero(monkeypatch, tmp_path: Path) -> None:
    _arm_runner(monkeypatch, tmp_path)
    gate = World().gate()
    heroes: list[object | None] = []
    copies: list[tuple[int, int]] = []
    real_image = RawImage

    def tracking(width: int, height: int, data: bytes, mode: str = "RGB") -> RawImage:
        image = real_image(width, height, data, mode)
        copies.append((id(image), width))
        return image

    monkeypatch.setattr("canvas.app.photoshoot_bench.RawImage", tracking)
    inner = live_generate(gate, Tier.GB16)

    def generate(line: str, seed: int, hero: object | None, strength: float) -> object | None:
        heroes.append(hero)
        return inner(line, seed, hero, strength)

    report = measure_series(
        idea=IDEA,
        folder=tmp_path / "series",
        seed=1,
        rewrite=_rewrite,
        age_gate=gate,
        generate=generate,
        strength=REFERENCE_STRENGTH,
    )
    assert report["ok"] is True
    assert report["kept"] == SERIES_FRAMES
    assert heroes[0] is None
    assert len(copies) == SERIES_FRAMES - 1
    assert copies[1][1] > 0
    assert copies[1][0] != id(heroes[2])
    assert getattr(heroes[2], "width") > 0


def test_live_rewrite_caps_tokens_at_the_request_count(monkeypatch) -> None:
    seen: dict[str, int] = {}

    class _Rewriter:
        def complete(self, prompt: str, *, system: str) -> str:
            del prompt, system
            return "{}"

        def close(self) -> None:
            return None

    def open_rewriter(*_args: object, **kwargs: object) -> _Rewriter:
        seen["tokens"] = int(kwargs["max_new_tokens"])  # type: ignore[arg-type]
        return _Rewriter()

    monkeypatch.setattr("canvas.app.photoshoot_bench.open_rewriter", open_rewriter)
    request = scenario_request(IDEA, SERIES_FRAMES, "photo", "same")
    assert request is not None
    live_rewrite(Tier.GB16)(request)
    assert seen["tokens"] == scenario_token_budget(SERIES_FRAMES)


def test_photoshoot_bench_command_asks_for_six_frames(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr(
        "canvas.app.main.probe_gpu",
        lambda: GpuInfo("RTX 4090", 16.0, 20, 8, 9, "fake"),
    )
    monkeypatch.setattr("canvas.app.main.check_gpu", lambda _info: GpuCheck(True, "", Tier.GB16))
    monkeypatch.setattr("canvas.app.main.install_supporting", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("canvas.app.main.load_catalog_stack", lambda *_args, **_kwargs: object())
    seen: dict[str, object] = {}

    def run_bench(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"ok": True, "frames": 6, "kept": 6, "stopped": "", "seconds": 1.25, "folder": str(kwargs["folder"])}

    monkeypatch.setattr("canvas.app.main.run_bench", run_bench)
    class _Gate:
        def warm(self) -> None:
            return None

    monkeypatch.setattr("canvas.app.main.AgeGate", lambda _stack: _Gate())
    code = main(["photoshoot-bench", "--prompt", IDEA, "--out", str(tmp_path), "--json", "--seed", "7"])
    body = json.loads(capsys.readouterr().out)
    assert code == 0
    assert seen["prompt"] == IDEA
    assert seen["seed"] == 7
    assert body["frames"] == 6
    assert body["seconds"] == 1.25


def test_photoshoot_bench_without_a_gpu_prints_no_duration(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr(
        "canvas.app.main.probe_gpu",
        lambda: GpuInfo("none", 0.0, 0.0, 0, 0, "fake"),
    )
    monkeypatch.setattr("canvas.app.main.check_gpu", lambda _info: GpuCheck(False, "no gpu", None))
    code = main(["photoshoot-bench", "--prompt", IDEA, "--out", str(tmp_path), "--json"])
    body = json.loads(capsys.readouterr().out)
    assert code == 2
    assert body["ok"] is False
    assert "seconds" not in body


def test_the_bench_prompt_defaults_to_the_fixed_character(tmp_path: Path) -> None:
    args = _parser().parse_args(["photoshoot-bench", "--out", str(tmp_path)])
    assert args.prompt == TEST_SERIES_CHARACTER


def test_parser_accepts_the_bench_command(tmp_path: Path) -> None:
    args = _parser().parse_args(["photoshoot-bench", "--prompt", IDEA, "--out", str(tmp_path)])
    assert args.command == "photoshoot-bench"
    assert args.prompt == IDEA
