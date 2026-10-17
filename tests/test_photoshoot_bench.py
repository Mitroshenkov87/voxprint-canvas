"""The Photoshoot bench times six frames and does not invent a duration."""
from __future__ import annotations

import json
from pathlib import Path

from canvas.app.main import _parser, main
from canvas.app.photoshoot_bench import HERO_STRENGTH, SERIES_FRAMES, measure_series
from canvas.app.platform import GpuCheck, OsCheck
from canvas.infra.gpu import GpuInfo
from canvas.models.modes import Tier
from tests.fakes import World
from tests.test_photoshoot import IDEA, _frame, _payload


def test_measure_series_times_six_frames_with_the_hero_strength(tmp_path: Path) -> None:
    heroes: list[object | None] = []
    strengths: list[float] = []

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
    assert strengths == [HERO_STRENGTH] * 6
    assert heroes[0] is None
    assert heroes[1] is not None
    assert all(hero is heroes[1] for hero in heroes[1:])


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
    monkeypatch.setattr("canvas.app.main.AgeGate", lambda _stack: object())
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


def test_parser_accepts_the_bench_command(tmp_path: Path) -> None:
    args = _parser().parse_args(["photoshoot-bench", "--prompt", IDEA, "--out", str(tmp_path)])
    assert args.command == "photoshoot-bench"
    assert args.prompt == IDEA
