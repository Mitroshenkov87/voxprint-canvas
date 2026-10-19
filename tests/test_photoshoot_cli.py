"""The photoshoot command writes a chosen number of frames."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canvas.app.main import _parser, main
from canvas.app.photoshoot_cli import model_generate, reject_frames, run_command
from canvas.app.platform import GpuCheck, OsCheck
from canvas.app.shell_run import RunStopped, ShellRunner
from canvas.infra.gpu import GpuInfo
from canvas.models.modes import Mode, Tier
from canvas.pipeline.adult_direction import ADULT_ANCHOR
from canvas.pipeline.photoshoot import SYSTEM_PROMPT
from canvas.pipeline.series import SeriesResult
from canvas.safety.constants import IMAGE_MINOR_SEXUAL
from canvas.safety.gate import StylisedBlocked
from canvas.safety.image import RawImage
from canvas.ui.shell_layout import PictureJob
from tests.fakes import World, avif_file, rgb
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


def _arm_runner(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.delenv("VOXPRINT_DEBUG", raising=False)

    class _Gen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.seed = 0
            self.lora_path = ""
            self.lora_strength = 0.7
            self.img2img_strength = None
            self.models = None

        def generate(self, screened: object, width: int | None = None, height: int | None = None) -> RawImage:
            del screened, width, height
            return rgb()

    monkeypatch.setattr("canvas.app.shell_run.ZImageGenerator", _Gen)


def _rewrite(prompt: str) -> str:
    body = json.loads(prompt)
    return _payload(str(body["idea"]), int(body["count"]), start=len(body.get("previous", [])))


def test_series_copies_the_hero_before_the_next_shot(monkeypatch, tmp_path: Path) -> None:
    _arm_runner(monkeypatch, tmp_path)
    gate = World().gate()
    heroes: list[object | None] = []
    screened: list[tuple[int, int, int]] = []
    original = gate.release_request

    def release_request(prompt: str, mode: str, inputs: tuple[object, ...] = (), *, explicit: bool = False):
        hero = heroes[-1] if heroes else None
        for item in inputs:
            image = item.image
            screened.append((id(image), image.width, id(hero) if hero is not None else 0))
        return original(prompt, mode, inputs, explicit=explicit)

    gate.release_request = release_request
    inner = model_generate(gate, Tier.GB16, "photo", False)

    def generate(line: str, seed: int, hero: object | None, strength: float) -> object | None:
        heroes.append(hero)
        return inner(line, seed, hero, strength)

    payload, code = run_command(
        idea=IDEA,
        folder=tmp_path / "series",
        frames=5,
        seed=1,
        style="photo",
        places="same",
        adult=False,
        age_gate=gate,
        rewrite=_rewrite,
        generate=generate,
    )
    assert code == 0
    assert payload["ok"] is True
    assert payload["made"] == 5
    assert payload["scenario_source"] == "writer"
    assert payload["code"] == ""
    assert heroes[0] is None
    assert all(getattr(hero, "width", 0) > 0 for hero in heroes[1:])
    assert len(screened) == 4
    third = screened[1]
    assert third[1] > 0
    assert third[0] != third[2]
    assert getattr(heroes[2], "width") > 0


def test_a_check_failure_stops_with_partial_frames(monkeypatch, tmp_path: Path) -> None:
    _arm_runner(monkeypatch, tmp_path)
    gate = World().gate()
    seen = {"frames": 0}
    original = gate.release_request
    score = gate._stack.nsfw.sexual_score
    score_image = gate._stack.zeroshot.score_image

    def release_request(prompt: str, mode: str, inputs: tuple[object, ...] = (), *, explicit: bool = False):
        seen["frames"] += 1
        return original(prompt, mode, inputs, explicit=explicit)

    def sexual_score(image: RawImage) -> float:
        if seen["frames"] >= 3:
            return 1.0
        return score(image)

    def score_image_later(image: RawImage, labels: object) -> dict[str, float]:
        if seen["frames"] >= 3:
            raise RuntimeError(f"image scores failed {IDEA}")
        return score_image(image, labels)

    gate.release_request = release_request
    gate._stack.nsfw.sexual_score = sexual_score
    gate._stack.zeroshot.score_image = score_image_later
    payload, code = run_command(
        idea=IDEA,
        folder=tmp_path / "series",
        frames=5,
        seed=1,
        style="photo",
        places="same",
        adult=False,
        age_gate=gate,
        rewrite=_rewrite,
        generate=model_generate(gate, Tier.GB16, "photo", False),
    )
    assert code == 1
    assert payload["ok"] is False
    assert payload["made"] == 2
    assert len(payload["frames"]) == 2
    for path in payload["frames"]:
        assert Path(str(path)).is_file()
    assert [shot["code"] for shot in payload["shots"][:2]] == ["ok", "ok"]
    assert [shot["code"] for shot in payload["shots"][2:]] == ["not_produced", "not_produced", "not_produced"]
    assert payload["code"] == "check.classifier_error"
    assert payload["scenario_source"] == "writer"
    scenario = Path(str(payload["scenario"]))
    assert scenario.is_file()
    log = (tmp_path / "home" / "logs" / "voxprint.log").read_text(encoding="utf-8")
    crash = [line for line in log.splitlines() if line.startswith("crash ")]
    assert len(crash) == 1
    assert "RuntimeError" in crash[0]
    assert "Traceback" in crash[0]
    assert IDEA not in log
    assert "kitchen table" not in log
    assert "close view" not in log
    assert "years" not in log


def test_a_frame_refusal_continues_without_an_internal_code(monkeypatch, tmp_path: Path) -> None:
    _arm_runner(monkeypatch, tmp_path)
    gate = World().gate()
    seen = {"frames": 0}
    original = gate.release_request

    def release_request(prompt: str, mode: str, inputs: tuple[object, ...] = (), *, explicit: bool = False):
        seen["frames"] += 1
        if seen["frames"] == 3:
            raise StylisedBlocked("prompt")
        return original(prompt, mode, inputs, explicit=explicit)

    gate.release_request = release_request
    payload, code = run_command(
        idea=IDEA,
        folder=tmp_path / "series",
        frames=5,
        seed=1,
        style="photo",
        places="same",
        adult=False,
        age_gate=gate,
        rewrite=_rewrite,
        generate=model_generate(gate, Tier.GB16, "photo", False),
    )
    assert code == 0
    assert payload["ok"] is True
    assert payload["made"] == 4
    assert payload["code"] == ""
    assert payload["scenario_source"] == "writer"
    assert [shot["code"] for shot in payload["shots"]] == ["ok", "ok", "not_produced", "ok", "ok"]
    log = tmp_path / "home" / "logs" / "voxprint.log"
    assert not log.exists() or "crash " not in log.read_text(encoding="utf-8")
    assert "internal" not in str(payload["code"])


def _picture_verdict(world: World) -> None:
    world.nsfw = 0.9
    world.image_scores[IMAGE_MINOR_SEXUAL] = 0.9


def _prompt_verdict(world: World) -> None:
    world.safety = "unsafe"
    world.categories = ("Minors",)


@pytest.mark.parametrize(
    ("stage", "arm"),
    [("output", _picture_verdict), ("input", _picture_verdict), ("prompt", _prompt_verdict)],
)
def test_a_real_frame_verdict_stops_the_series(monkeypatch, tmp_path: Path, stage: str, arm) -> None:
    """The real gate verdict on frame 3 ends the series with scenario.minor. Frames 1 and 2 stay."""
    _arm_runner(monkeypatch, tmp_path)
    world = World()
    gate = world.gate()
    inner = model_generate(gate, Tier.GB16, "photo", False)
    calls = {"n": 0}

    def generate(line: str, seed: int, hero: object | None, strength: float) -> object | None:
        calls["n"] += 1
        if calls["n"] == 3 and stage != "output":
            arm(world)
        image = inner(line, seed, hero, strength)
        if calls["n"] == 3 and stage == "output":
            arm(world)
        return image

    folder = tmp_path / "series"
    payload, code = run_command(
        idea=IDEA,
        folder=folder,
        frames=5,
        seed=1,
        style="photo",
        places="same",
        adult=False,
        age_gate=gate,
        rewrite=_rewrite,
        generate=generate,
    )
    assert code == 1
    assert payload["ok"] is False
    assert payload["code"] == "scenario.minor"
    assert payload["made"] == 2
    assert calls["n"] == 3
    assert [shot["code"] for shot in payload["shots"][:2]] == ["ok", "ok"]
    for path in payload["frames"]:
        assert Path(str(path)).is_file()
    numbered = sorted(path.name for path in next(folder.rglob("scenario.txt")).parent.iterdir() if path.stem.isdigit())
    assert len(numbered) == 2
    log = tmp_path / "home" / "logs" / "voxprint.log"
    assert not log.exists() or "crash " not in log.read_text(encoding="utf-8")


def test_an_unexpected_runner_error_is_internal(monkeypatch, tmp_path: Path) -> None:
    _arm_runner(monkeypatch, tmp_path)

    class _Boom:
        calls = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.seed = 0
            self.lora_path = ""
            self.lora_strength = 0.7
            self.img2img_strength = None
            self.models = None

        def generate(self, screened: object, width: int | None = None, height: int | None = None) -> RawImage:
            del screened, width, height
            type(self).calls += 1
            if type(self).calls == 3:
                raise RuntimeError(f"boom {IDEA}")
            return rgb()

    monkeypatch.setattr("canvas.app.shell_run.ZImageGenerator", _Boom)
    gate = World().gate()
    payload, code = run_command(
        idea=IDEA,
        folder=tmp_path / "series",
        frames=5,
        seed=1,
        style="photo",
        places="same",
        adult=False,
        age_gate=gate,
        rewrite=_rewrite,
        generate=model_generate(gate, Tier.GB16, "photo", False),
    )
    assert code == 1
    assert payload["ok"] is False
    assert payload["made"] == 2
    assert payload["code"] == "internal"
    assert payload["scenario_source"] == "writer"
    assert payload["message"] == "Something went wrong. The series stopped."
    log = (tmp_path / "home" / "logs" / "voxprint.log").read_text(encoding="utf-8")
    crash = [line for line in log.splitlines() if line.startswith("crash ")]
    assert len(crash) == 1
    assert "RuntimeError" in crash[0]
    assert "Traceback" in crash[0]
    assert IDEA not in log


def test_runner_codes_stay_stable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))

    class _Gen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.seed = 0
            self.lora_path = ""
            self.lora_strength = 0.7

    class _Pipe:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def generate(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr("canvas.app.shell_run.load_catalog", lambda: object())
    monkeypatch.setattr("canvas.app.shell_run.ZImageGenerator", _Gen)
    monkeypatch.setattr("canvas.app.shell_run.Pipeline", _Pipe)
    runner = ShellRunner(gate=World().gate(), tier=Tier.GB16, enlarge=object())
    job = PictureJob("a red boat", "", Mode.TEXT, "1:1", "native", "avif", 1, 4, 1.0, "euler", 1)
    with pytest.raises(RunStopped) as caught:
        runner(job, ())
    assert caught.value.unavailable is True
    assert caught.value.code == "internal"


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
    assert code == 0
    assert payload["scenario_source"] == "planned"
    assert payload["code"] == ""
    assert payload["made"] == 1
    stopped, stop_code = run_command(
        idea="   ",
        folder=tmp_path / "empty",
        frames=1,
        seed=1,
        style="photo",
        places="same",
        adult=False,
        age_gate=World().gate(),
        rewrite=lambda _prompt: "",
        generate=lambda *_args: _frame(),
    )
    assert stop_code == 1
    assert stopped["error"] == "scenario"
    assert stopped["code"] == "scenario.format"
    assert stopped["scenario_source"] == ""
    assert stopped["message"] == "The scenario cannot be used."
    rejected = reject_frames(0)
    assert rejected["error"] == "bad_frames"
    assert rejected["code"] == ""
    assert rejected["scenario_source"] == ""


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
