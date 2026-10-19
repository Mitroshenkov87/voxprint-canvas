"""The photoshoot command writes a chosen number of frames."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canvas.app.main import _parser, main
from canvas.app.photoshoot_cli import run_command
from canvas.app.platform import GpuCheck, OsCheck
from canvas.infra.gpu import GpuInfo
from canvas.models.modes import Tier
from canvas.pipeline.adult_direction import ADULT_ANCHOR
from canvas.pipeline.photoshoot import SYSTEM_PROMPT
from canvas.pipeline.series import SeriesResult
from tests.fakes import World, avif_file
from tests.test_photoshoot import IDEA, _frame, _payload


def _ready(monkeypatch) -> None:
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr(
        "canvas.app.main.probe_gpu",
        lambda: GpuInfo("RTX 4090", 16.0, 20, 8, 9, "fake"),
    )
    monkeypatch.setattr("canvas.app.main.check_gpu", lambda _info: GpuCheck(True, "", Tier.GB16))
    monkeypatch.setattr("canvas.app.main.install_supporting", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("canvas.app.main.load_catalog_stack", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("canvas.app.main.AgeGate", lambda _stack: World().gate())


def _writers(monkeypatch, generate) -> None:
    def rewrite(tier: Tier, adult: bool):
        del tier, adult

        def _write(prompt: str) -> str:
            body = json.loads(prompt)
            return _payload(str(body["idea"]), int(body["count"]), start=len(body.get("previous", [])))

        return _write

    monkeypatch.setattr("canvas.app.main.model_rewrite", rewrite)
    monkeypatch.setattr("canvas.app.main.model_generate", lambda *_args: generate)


def test_parser_accepts_one_and_fifty_frames(tmp_path: Path) -> None:
    one = _parser().parse_args(["photoshoot", "--prompt", IDEA, "--frames", "1", "--out", str(tmp_path)])
    fifty = _parser().parse_args(["photoshoot", "--prompt", IDEA, "--frames", "50", "--out", str(tmp_path)])
    default = _parser().parse_args(["photoshoot", "--prompt", IDEA, "--out", str(tmp_path)])
    assert one.frames == 1
    assert fifty.frames == 50
    assert default.frames == 6
    assert default.style == "photo"
    assert default.places == "same"
    assert default.seed == 1
    assert default.adult is False
    assert default.command == "photoshoot"


@pytest.mark.parametrize("frames", ["0", "51"])
def test_frames_outside_one_to_fifty_are_rejected(capsys, tmp_path: Path, frames: str) -> None:
    code = main(["photoshoot", "--prompt", IDEA, "--frames", frames, "--out", str(tmp_path), "--json"])
    body = json.loads(capsys.readouterr().out)
    assert code == 2
    assert body["ok"] is False
    assert body["error"] == "bad_frames"
    assert body["requested"] == int(frames)
    assert body["made"] == 0
    assert body["frames"] == []


def test_command_writes_scenario_and_frame_files(monkeypatch, capsys, tmp_path: Path) -> None:
    _ready(monkeypatch)

    def generate(line: str, seed: int, hero: object | None, strength: float) -> object:
        del line, seed, hero
        assert strength == 0.45
        return _frame()

    _writers(monkeypatch, generate)
    code = main([
        "photoshoot", "--prompt", IDEA, "--frames", "4", "--out", str(tmp_path),
        "--json", "--seed", "3", "--style", "photo", "--places", "same",
    ])
    body = json.loads(capsys.readouterr().out)
    assert code == 0
    assert body["ok"] is True
    assert body["requested"] == 4
    assert body["made"] == 4
    assert len(body["frames"]) == 4
    assert [shot["code"] for shot in body["shots"]] == ["ok"] * 4
    for path in body["frames"]:
        file = Path(path)
        assert file.is_file()
        assert file.suffix == ".avif"
        assert avif_file(file.read_bytes())
    scenario = Path(body["scenario"])
    assert scenario == tmp_path / "scenario.txt"
    lines = scenario.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert all(line.startswith(IDEA) for line in lines)


def test_a_stopped_frame_is_not_produced(monkeypatch, capsys, tmp_path: Path) -> None:
    _ready(monkeypatch)

    def generate(line: str, seed: int, hero: object | None, strength: float) -> object | None:
        del seed, hero, strength
        if "three-quarter" in line:
            return None
        return _frame()

    _writers(monkeypatch, generate)
    code = main(["photoshoot", "--prompt", IDEA, "--frames", "6", "--out", str(tmp_path), "--json"])
    body = json.loads(capsys.readouterr().out)
    assert code == 0
    assert body["requested"] == 6
    assert body["made"] == 5
    assert len(body["frames"]) == 5
    blocked = [shot for shot in body["shots"] if shot["code"] == "not_produced"]
    assert len(blocked) == 1
    assert blocked[0]["message"] == "Shot blocked"
    assert "error" not in body
    lines = Path(body["scenario"]).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6


@pytest.mark.parametrize("adult", [False, True])
def test_adult_flag_reaches_run_series_and_scenario_system(monkeypatch, tmp_path: Path, adult: bool) -> None:
    _ready(monkeypatch)
    seen: dict[str, object] = {}

    def run_series(**kwargs: object) -> SeriesResult:
        seen["adult"] = kwargs["adult"]
        seen["rewrite"] = kwargs["rewrite"]
        return SeriesResult("scenario", "", (), None)

    class _Rewriter:
        def complete(self, prompt: str, *, system: str) -> str:
            del prompt
            seen["system"] = system
            return "{}"

        def close(self) -> None:
            seen["closed"] = True

    monkeypatch.setattr("canvas.app.photoshoot_cli.run_series", run_series)
    monkeypatch.setattr("canvas.app.photoshoot_cli.open_rewriter", lambda *_args, **_kwargs: _Rewriter())
    argv = ["photoshoot", "--prompt", IDEA, "--out", str(tmp_path), "--json", "--frames", "2"]
    if adult:
        argv.append("--adult")
    code = main(argv)
    assert code == 1
    assert seen["adult"] is adult
    rewrite = seen["rewrite"]
    assert callable(rewrite)
    rewrite("ping")
    assert seen["closed"] is True
    system = str(seen["system"])
    assert system.startswith(SYSTEM_PROMPT)
    assert (ADULT_ANCHOR in system) is adult


def test_json_reports_the_reason_code(tmp_path: Path) -> None:
    payload, code = run_command(
        idea=IDEA,
        folder=tmp_path,
        frames=1,
        seed=1,
        style="photo",
        places="same",
        adult=False,
        age_gate=World().gate(),
        rewrite=lambda _prompt: "Sorry, I cannot write that.",
        generate=lambda *_args: _frame(),
    )
    assert code == 1
    assert payload["error"] == "scenario"
    assert payload["code"] == "scenario.declined"
    assert payload["message"] == "The scenario was not written."


def test_run_command_uses_the_injected_writer(tmp_path: Path) -> None:
    calls: list[bool] = []

    def rewrite(prompt: str) -> str:
        del prompt
        return _payload(IDEA, 2)

    def generate(line: str, seed: int, hero: object | None, strength: float) -> object:
        del line, seed, hero, strength
        calls.append(True)
        return _frame()

    payload, code = run_command(
        idea=IDEA,
        folder=tmp_path,
        frames=2,
        seed=1,
        style="cinematic",
        places="same",
        adult=False,
        age_gate=World().gate(),
        rewrite=rewrite,
        generate=generate,
    )
    assert code == 0
    assert payload["made"] == 2
    assert len(calls) == 2
    assert (tmp_path / "scenario.txt").is_file()
