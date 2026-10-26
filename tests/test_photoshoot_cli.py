"""The photoshoot command writes a chosen number of frames."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canvas.app.main import _parser, main
from canvas.app.photoshoot_cli import model_generate, model_rewrite, reject_frames, run_command
from canvas.app.platform import GpuCheck, OsCheck
from canvas.app.shell_run import RunStopped, ShellRunner
from canvas.infra.gpu import GpuInfo
from canvas.models.modes import Mode, Tier
from canvas.pipeline.adult_direction import ADULT_ANCHOR
from canvas.pipeline.photoshoot import SYSTEM_PROMPT, scenario_request, scenario_token_budget
from canvas.pipeline.series import REFERENCE_STRENGTH, SeriesResult
from canvas.safety.constants import IMAGE_MINOR_SEXUAL
from canvas.safety.gate import GenerationBlocked, StylisedBlocked
from canvas.safety.image import RawImage
from canvas.safety.inputs import ImageRole
from canvas.ui.shell_layout import PictureJob
from tests.fakes import World, avif_file, rgb
from tests.test_photoshoot import IDEA, PERSON, _frame, _payload, fixed_appearance


def _series_line(prompt: str) -> str:
    prefix = "Suitable for a general audience. "
    suffix = " photorealistic photograph"
    assert prompt.startswith(prefix)
    assert prompt.endswith(suffix)
    return prompt[len(prefix):-len(suffix)]


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


def test_parser_accepts_one_and_ninety_nine_frames(tmp_path: Path) -> None:
    one = _parser().parse_args(["photoshoot", "--prompt", IDEA, "--frames", "1", "--out", str(tmp_path)])
    full = _parser().parse_args(["photoshoot", "--prompt", IDEA, "--frames", "99", "--out", str(tmp_path)])
    default = _parser().parse_args(["photoshoot", "--prompt", IDEA, "--out", str(tmp_path)])
    assert one.frames == 1
    assert full.frames == 99
    assert default.frames == 6
    assert default.style == "photo"
    assert default.places == "same"
    assert default.seed == 1
    assert default.adult is False
    assert default.export_format == "png"
    assert default.command == "photoshoot"
    assert default.vram_trace is False
    assert default.vram_trace is False


@pytest.mark.parametrize("frames", ["0", "100"])
def test_frames_outside_one_to_ninety_nine_are_rejected(capsys, tmp_path: Path, frames: str) -> None:
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
        del line, seed
        assert strength is None
        assert hero is None
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
        assert file.suffix == ".png"
        assert file.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    scenario = Path(body["scenario"])
    assert scenario == tmp_path / "scenario.txt"
    lines = scenario.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert all(line.startswith(f"A single photograph of {IDEA}, still, 50 mm, ") for line in lines)


def test_export_format_avif_is_still_a_frame_file(monkeypatch, capsys, tmp_path: Path) -> None:
    _ready(monkeypatch)

    def generate(line: str, seed: int, hero: object | None, strength: float) -> object:
        del line, seed, hero, strength
        return _frame()

    _writers(monkeypatch, generate)
    code = main([
        "photoshoot", "--prompt", IDEA, "--frames", "1", "--out", str(tmp_path),
        "--json", "--export-format", "avif",
    ])
    body = json.loads(capsys.readouterr().out)
    assert code == 0
    file = Path(body["frames"][0])
    assert file.suffix == ".avif"
    assert avif_file(file.read_bytes())


def test_ten_frames_vary_pose_camera_and_seed(monkeypatch, tmp_path: Path) -> None:
    """Default frames are text. High consistency restyles at the shared strength."""
    calls: list[tuple[str, Mode, int, float | None, tuple[ImageRole, ...]]] = []

    class _Runner:
        def __init__(self, *, gate: object, tier: object, enlarge: object) -> None:
            del gate, tier, enlarge
            self.img2img_strength: float | None = None

        def __call__(self, job: PictureJob, inputs: tuple[object, ...]) -> object:
            roles = tuple(item.role for item in inputs if isinstance(item.role, ImageRole))
            calls.append((job.prompt, job.mode, job.seed, self.img2img_strength, roles))
            return _frame()

    monkeypatch.setattr("canvas.app.photoshoot_cli.ShellRunner", _Runner)

    def rewrite(prompt: str) -> str:
        body = json.loads(prompt)
        count = int(body["count"])
        start = len(body.get("previous", []))
        shots = []
        for offset in range(count):
            number = start + offset
            shots.append({
                "scene": f"scene {number}",
                "pose": f"pose {number}",
                "camera": f"{30 + number} mm lens",
                "place": "kitchen table",
            })
        return json.dumps({
            "subject_block": str(body["idea"]),
            "descriptors": ["teapot"],
            "place": "kitchen table",
            "shots": shots,
        })

    gate = World().gate()
    payload, code = run_command(
        idea=IDEA,
        folder=tmp_path / "text",
        frames=10,
        seed=1,
        style="photo",
        places="same",
        adult=False,
        age_gate=gate,
        rewrite=rewrite,
        generate=model_generate(gate, Tier.GB16, "photo", False),
    )
    assert code == 0 and payload["made"] == 10
    lines = [_series_line(prompt) for prompt, _mode, _seed, _strength, _roles in calls]
    assert len(lines) == 10
    assert len(set(lines)) == 10
    assert [seed for _prompt, _mode, seed, _strength, _roles in calls] == list(range(1, 11))
    for index, (line, (_prompt, mode, _seed, strength, roles)) in enumerate(zip(lines, calls, strict=True)):
        assert line.startswith(f"A single photograph of {IDEA}, pose {index}, {30 + index} mm lens, ")
        assert line.endswith(", kitchen table")
        assert IDEA in line
        assert mode is Mode.TEXT
        assert strength is None
        assert roles == ()

    calls.clear()
    high, high_code = run_command(
        idea=IDEA,
        folder=tmp_path / "high",
        frames=10,
        seed=1,
        style="photo",
        places="same",
        adult=False,
        age_gate=gate,
        rewrite=rewrite,
        generate=model_generate(gate, Tier.GB16, "photo", False),
        strength=REFERENCE_STRENGTH,
    )
    assert high_code == 0 and high["made"] == 10
    assert REFERENCE_STRENGTH >= 0.75
    assert calls[0][1] is Mode.TEXT and calls[0][3] is None and calls[0][4] == ()
    for prompt, mode, _seed, strength, roles in calls[1:]:
        assert _series_line(prompt).startswith("A single photograph of ")
        assert "pose " in _series_line(prompt)
        assert mode is Mode.RESTYLE
        assert strength == REFERENCE_STRENGTH
        assert roles == (ImageRole.INIT,)


def test_a_person_series_anchors_by_default(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Mode, int, float | None, tuple[ImageRole, ...]]] = []

    class _Runner:
        def __init__(self, *, gate: object, tier: object, enlarge: object) -> None:
            del gate, tier, enlarge
            self.img2img_strength: float | None = None

        def __call__(self, job: PictureJob, inputs: tuple[object, ...]) -> object:
            roles = tuple(item.role for item in inputs if isinstance(item.role, ImageRole))
            calls.append((job.prompt, job.mode, job.seed, self.img2img_strength, roles))
            return _frame()

    monkeypatch.setattr("canvas.app.photoshoot_cli.ShellRunner", _Runner)
    gate = World().gate()
    payload, code = run_command(
        idea=PERSON,
        folder=tmp_path,
        frames=5,
        seed=3,
        style="photo",
        places="same",
        adult=False,
        age_gate=gate,
        rewrite=_rewrite,
        generate=model_generate(gate, Tier.GB16, "photo", False),
    )
    look = fixed_appearance(PERSON, 3)
    assert code == 0 and payload["made"] == 5
    assert look
    lines = Path(str(payload["scenario"])).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    assert all(line.count(look) == 1 and PERSON in line for line in lines)
    assert calls[0][1] is Mode.TEXT and calls[0][3] is None and calls[0][4] == ()
    assert look in calls[0][0]
    for prompt, mode, _seed, strength, roles in calls[1:]:
        assert look in prompt
        assert mode is Mode.RESTYLE
        assert strength == REFERENCE_STRENGTH
        assert roles == (ImageRole.INIT,)


def test_model_rewrite_caps_tokens_at_the_requested_frame_count(monkeypatch) -> None:
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

    monkeypatch.setattr("canvas.app.photoshoot_cli.open_rewriter", open_rewriter)
    request = scenario_request(PERSON, 5, "photo", "same")
    assert request is not None
    model_rewrite(Tier.GB16, False)(request)
    assert seen["tokens"] == scenario_token_budget(5)


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
    copies: list[tuple[int, int]] = []
    real_image = RawImage

    def tracking(width: int, height: int, data: bytes, mode: str = "RGB") -> RawImage:
        image = real_image(width, height, data, mode)
        copies.append((id(image), width))
        return image

    monkeypatch.setattr("canvas.app.photoshoot_cli.RawImage", tracking)
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
        strength=REFERENCE_STRENGTH,
    )
    assert code == 0
    assert payload["ok"] is True
    assert payload["made"] == 5
    assert payload["scenario_source"] == "writer"
    assert payload["code"] == ""
    assert heroes[0] is None
    assert all(getattr(hero, "width", 0) > 0 for hero in heroes[1:])
    assert len(copies) == 4
    third = copies[1]
    assert third[1] > 0
    assert third[0] != id(heroes[2])
    assert getattr(heroes[2], "width") > 0


def test_a_check_failure_stops_with_partial_frames(monkeypatch, tmp_path: Path) -> None:
    _arm_runner(monkeypatch, tmp_path)
    gate = World().gate()
    seen = {"frames": 0}
    original = gate.review_image
    score = gate._stack.nsfw.sexual_score
    score_image = gate._stack.zeroshot.score_image

    def review_image(raw: object, prompt: str = "") -> object:
        seen["frames"] += 1
        return original(raw, prompt)

    def sexual_score(image: RawImage) -> float:
        if seen["frames"] >= 3:
            return 1.0
        return score(image)

    def score_image_later(image: RawImage, labels: object) -> dict[str, float]:
        if seen["frames"] >= 3:
            raise RuntimeError(f"image scores failed {IDEA}")
        return score_image(image, labels)

    gate.review_image = review_image
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
    original = gate.review_prompt

    def review_prompt(prompt: str) -> None:
        if "from above" in prompt:
            raise StylisedBlocked("prompt")
        original(prompt)

    gate.review_prompt = review_prompt
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


@pytest.mark.parametrize("stage", ["output", "prompt"])
def test_a_real_frame_verdict_stops_the_series(monkeypatch, tmp_path: Path, stage: str) -> None:
    """Frame 3's verdict ends the saves with scenario.minor. Frames 1 and 2 stay."""
    _arm_runner(monkeypatch, tmp_path)
    world = World()
    gate = world.gate()
    if stage == "prompt":
        real_prompt = gate.review_prompt

        def review_prompt(prompt: str) -> None:
            if "from above" in prompt:
                raise GenerationBlocked("prompt")
            real_prompt(prompt)

        gate.review_prompt = review_prompt
    inner = model_generate(gate, Tier.GB16, "photo", False)
    calls = {"n": 0}
    reviewed = {"n": 0}
    if stage == "output":
        real_image = gate.review_image

        def review_image(raw: object, prompt: str = "") -> object:
            reviewed["n"] += 1
            if reviewed["n"] == 3:
                _picture_verdict(world)
            return real_image(raw, prompt)

        gate.review_image = review_image

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object | None:
        calls["n"] += 1
        return inner(line, seed, hero, strength)

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
    assert payload["made"] == (2 if stage == "output" else 0)
    assert calls["n"] == (5 if stage == "output" else 0)
    if stage == "output":
        assert [shot["code"] for shot in payload["shots"][:2]] == ["ok", "ok"]
        for path in payload["frames"]:
            assert Path(str(path)).is_file()
        numbered = sorted(
            path.name for path in next(folder.rglob("scenario.txt")).parent.iterdir() if path.stem.isdigit()
        )
        assert len(numbered) == 2
    else:
        assert payload["frames"] == []
        assert payload["shots"] == []
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
