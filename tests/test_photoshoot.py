"""Photoshoot writes a scenario from the prompt and generates the series."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from canvas.infra.stage_timer import StageTimes
from canvas.pipeline.enhance import stated_ages
from canvas.pipeline.photoshoot import (
    FORMAT_REMINDER,
    SCENARIO_TOKENS,
    SYSTEM_PROMPT,
    TOKENS_PER_FRAME,
    Draft,
    Shot,
    _BUILD,
    _ETHNICITY,
    _EYES,
    _FACE,
    _HAIR,
    _object,
    accept_shot,
    assemble_line,
    fixed_appearance,
    planned_shots,
    scenario_request,
    scenario_token_budget,
    tokens_for_request,
    truncate_for_encoder,
    validate_scenario,
)
from canvas.pipeline.series import REFERENCE_STRENGTH, ages_allowed, run_series
from canvas.safety.denylist import scan
from canvas.safety.gate import AgeGate, GenerationBlocked, SafetyUnavailable, StylisedBlocked
from tests.fakes import World


IDEA = "Yellow ceramic teapot with a blue handle"
AGED = "a woman, 34 years old, in a red coat"
PERSON = "invented adult woman, 25 years old"
_BANNED_LOOK = re.compile(
    r"\b(?:young|youthful|youth|teen|teenage|child|girl|boy|kid|kids|petite|baby|infant|"
    r"smooth|round|mature|older|elderly|aged|aging|ageing|wrinkle|wrinkles|grey|gray|"
    r"year|years|old|toddler|minor|school|student)\b",
    re.IGNORECASE,
)


def _shots(count: int, place: str, *, scene_age: str = "", start: int = 0) -> list[dict[str, str]]:
    scenes = [
        "close view",
        "three-quarter view",
        "from above",
        "from the side",
        "by the window",
        "among cups",
        "on a tray",
        "beside a spoon",
        "under a lamp",
        "near the sink",
        "against the tiles",
        "with a cup",
    ]
    found = []
    for offset in range(count):
        number = start + offset
        scene = scenes[number] if number < len(scenes) else f"angle {number}"
        if offset == 1 and scene_age:
            scene = f"{scene}, {scene_age}"
        found.append({"scene": scene, "pose": "still", "camera": "50 mm", "place": place})
    return found


def _payload(
    idea: str,
    count: int = 6,
    *,
    place_mode: str = "same",
    subject: str | None = None,
    scene_age: str = "",
    start: int = 0,
) -> str:
    place = "kitchen table" if place_mode == "same" else ""
    places = ["kitchen table"] * count if place_mode == "same" else [f"room {start + index}" for index in range(count)]
    shots = _shots(count, places[0], scene_age=scene_age, start=start)
    for shot, name in zip(shots, places, strict=True):
        shot["place"] = name
    body = {
        "subject_block": idea if subject is None else subject,
        "descriptors": ["Yellow ceramic teapot", "blue handle"] if "teapot" in idea else ["red coat"],
        "place": place,
        "shots": shots,
    }
    return json.dumps(body)


def test_a_shot_that_describes_several_views_is_rejected() -> None:
    scenes = (
        "a collage of the teapot",
        "a grid of views",
        "three panels",
        "a sequence of pours",
        "a series of angles",
        "before/after the pour",
        "before and after",
        "multiple views of the handle",
        "a multi-panel layout",
        "split screen",
    )
    for scene in scenes:
        assert accept_shot(IDEA, IDEA, "same", "kitchen table", (), Draft(scene, "still", "50 mm", "")) is None
    kept = accept_shot(IDEA, IDEA, "same", "kitchen table", (), Draft("close view", "still", "50 mm", ""))
    assert isinstance(kept, Shot)


def test_appearance_stays_inside_the_encoder_head() -> None:
    look = "Southeast Asian, straight blonde hair, blue eyes, oval face, slender"
    subject = f"{look}, a woman in a red coat"
    shot = Shot(" ".join(["wide room"] * 400), "still", "50 mm", "kitchen table")
    line = assemble_line(subject, shot, include_camera=True)
    assert line.startswith(f"A single photograph of {look},")
    assert len(line.split()) <= 512 - 48
    assert truncate_for_encoder(line, limit=12).startswith("A single photograph of Southeast")


def test_assemble_line_leads_with_the_subject() -> None:
    shot = Shot("close view", "still", "50 mm", "kitchen table")
    plain = assemble_line(IDEA, shot, include_camera=False)
    detailed = assemble_line(IDEA, shot, include_camera=True)
    assert plain == f"A single photograph of {IDEA}, still, close view, kitchen table"
    assert "50 mm" not in plain
    assert detailed == f"A single photograph of {IDEA}, still, 50 mm, close view, kitchen table"
    assert IDEA in detailed
    assert detailed.index(IDEA) < detailed.index("50 mm")


def test_changed_age_discards_the_scenario() -> None:
    raw = _payload(AGED, subject="a woman in a red coat")
    assert validate_scenario(AGED, raw, count=6, place_mode="same") is None


def test_a_shot_that_changes_the_age_is_dropped() -> None:
    raw = _payload(AGED, scene_age="now 12 years old")
    scenario = validate_scenario(AGED, raw, count=6, place_mode="same")
    assert scenario is not None
    assert len(scenario.shots) == 5
    assert all("12 years" not in shot.scene for shot in scenario.shots)


def test_place_mode_accepts_one_place_or_distinct_places() -> None:
    same = validate_scenario(IDEA, _payload(IDEA), count=6, place_mode="same")
    assert same is not None and {shot.place for shot in same.shots} == {"kitchen table"}
    distinct = validate_scenario(IDEA, _payload(IDEA, place_mode="different"), count=6, place_mode="different")
    assert distinct is not None and len({shot.place for shot in distinct.shots}) == 6
    parsed = json.loads(_payload(IDEA, place_mode="different"))
    parsed["shots"][1]["place"] = parsed["shots"][0]["place"]
    assert validate_scenario(IDEA, json.dumps(parsed), count=6, place_mode="different") is None


def test_frame_count_bounds_are_one_to_ninety_nine() -> None:
    assert scenario_request(IDEA, 0, "photo", "same") is None
    assert scenario_request(IDEA, 100, "photo", "same") is None
    assert scenario_request(IDEA, 1, "photo", "same") is not None
    assert scenario_request(IDEA, 99, "photo", "same") is not None
    one = validate_scenario(IDEA, _payload(IDEA, 1), count=1, place_mode="same")
    full = validate_scenario(IDEA, _payload(IDEA, 99), count=99, place_mode="same")
    assert one is not None and len(one.shots) == 1
    assert full is not None and len(full.shots) == 99
    assert validate_scenario(IDEA, _payload(IDEA, 1), count=0, place_mode="same") is None
    assert validate_scenario(IDEA, _payload(IDEA, 1), count=100, place_mode="same") is None


def test_one_frame_series_is_kept(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        count=1,
        rewrite=lambda prompt: _payload(IDEA, int(json.loads(prompt)["count"])),
    )
    assert result.stopped == ""
    assert len(result.kept) == 1


def test_a_long_scenario_is_requested_in_chunks_until_the_count(tmp_path: Path) -> None:
    calls: list[str] = []

    def rewrite(prompt: str) -> str:
        calls.append(prompt)
        assert "write one different shot" not in prompt
        body = json.loads(prompt)
        count = int(body["count"])
        assert count <= 10
        previous = body.get("previous", [])
        assert isinstance(previous, list)
        return _payload(IDEA, count, start=len(previous))

    result = _run(tmp_path, count=23, rewrite=rewrite)
    assert result.stopped == ""
    assert len(result.kept) == 23
    counts = [int(json.loads(prompt)["count"]) for prompt in calls]
    assert counts == [10, 10, 3]
    assert "previous" not in json.loads(calls[0])
    assert len(json.loads(calls[1])["previous"]) == 10
    assert len(json.loads(calls[2])["previous"]) == 20
    lines = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))["lines"]
    assert len(lines) == 23
    assert len({line for line in lines}) == 23


def test_blocked_idea_stops_before_the_model(tmp_path: Path) -> None:
    calls: list[str] = []
    result = _run(tmp_path, idea="loli", rewrite=lambda prompt: calls.append(prompt) or _payload(IDEA))
    assert result.stopped == "idea"
    assert calls == []
    assert result.kept == ()


def test_a_gate_failure_stops_the_idea_without_the_model(tmp_path: Path) -> None:
    world = World()
    world.error = "guard"
    calls: list[str] = []
    result = _run(
        tmp_path,
        age_gate=world.gate(),
        rewrite=lambda prompt: calls.append(prompt) or _payload(IDEA),
    )
    assert result.stopped == "idea"
    assert calls == []
    assert "guard failed" not in result.reason


def test_blocked_scenario_stops_and_is_not_rewritten(tmp_path: Path) -> None:
    calls: list[str] = []

    def rewrite(prompt: str) -> str:
        calls.append(prompt)
        parsed = json.loads(_payload(IDEA))
        parsed["shots"][0]["scene"] = "loli"
        return json.dumps(parsed)

    result = _run(tmp_path, rewrite=rewrite)
    assert result.stopped == "scenario"
    assert len(calls) == 1
    assert result.kept == ()


def test_blocked_shot_line_is_dropped_without_a_rewrite(tmp_path: Path) -> None:
    calls: list[str] = []

    def line_for(index: int) -> str:
        if index == 1:
            return f"{IDEA}, kitchen table, three-quarter view, still hentai"
        return f"{IDEA}, kitchen table, view {index}, still"

    result = _run(tmp_path, rewrite=lambda prompt: calls.append(prompt) or _payload(IDEA), line_for=line_for)
    document = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert result.stopped == ""
    assert len(calls) == 1
    assert len(result.kept) == 5
    assert document["blocked"] == [1]
    assert all("hentai" not in line for line in document["lines"])


def test_a_scenario_minor_frame_stops_the_series(tmp_path: Path) -> None:
    """A frame refused with code scenario.minor stops the series. Later frames are not saved."""
    drawn: list[int] = []

    def line_for(index: int) -> str:
        if index == 1:
            return f"{IDEA}, kitchen table, three-quarter view, still loli"
        return f"{IDEA}, kitchen table, view {index}, still"

    def generate(line: str, seed: int, hero: object | None, strength: float) -> object:
        del line, hero, strength
        drawn.append(seed)
        return _frame()

    result = _run(tmp_path, line_for=line_for, generate=generate)
    document = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert result.stopped == "scenario"
    assert result.code == "scenario.minor"
    assert result.folder == tmp_path
    assert len(result.kept) == 1
    assert len(drawn) == 1
    assert (tmp_path / "01.png").is_file()
    assert not (tmp_path / "02.png").exists()
    assert document["blocked"] == [1]
    assert all("loli" not in line for line in document["lines"])


def test_blocked_frame_is_dropped_without_a_rewrite(tmp_path: Path) -> None:
    calls: list[str] = []
    heroes: list[object | None] = []
    gate = World().gate()
    real = gate.review_image

    def review_image(raw: object, prompt: str = "") -> object:
        if "three-quarter" in prompt:
            raise SafetyUnavailable("classifier_error")
        return real(raw, prompt)

    gate.review_image = review_image

    def generate(line: str, seed: int, hero: object | None, strength: float) -> object:
        del seed
        heroes.append(hero)
        assert strength is None
        assert line.startswith(f"A single photograph of {IDEA}, still, 50 mm, ")
        assert IDEA in line
        return _frame(b"\x10\x20\x30")

    result = _run(tmp_path, age_gate=gate, rewrite=lambda prompt: calls.append(prompt) or _payload(IDEA), generate=generate)
    assert result.stopped == "error"
    assert result.code == "check.classifier_error"
    assert len(calls) == 1
    assert len(heroes) == 6
    assert len(result.kept) == 1
    assert all(hero is None for hero in heroes)
    assert all(path.suffix == ".png" for path in result.kept)
    assert (tmp_path / "series.json").is_file()
    assert (tmp_path / "overview.png").is_file()
    assert (tmp_path / "series.zip").is_file()


def test_an_image_gate_failure_drops_every_frame(tmp_path: Path) -> None:
    world = World()
    world.nsfw = 0.9
    world.error = "faces"
    calls: list[str] = []
    result = _run(tmp_path, age_gate=world.gate(), rewrite=lambda prompt: calls.append(prompt) or _payload(IDEA))
    assert result.stopped == "error"
    assert result.code == "check.classifier_error"
    assert result.kept == ()
    assert len(calls) == 1


def test_the_gate_is_called_by_name_on_every_text_and_frame(tmp_path: Path, monkeypatch) -> None:
    prompts: list[str] = []
    frames: list[str] = []
    real_prompt = AgeGate.review_prompt
    real_image = AgeGate.review_image

    def spy_prompt(self: AgeGate, prompt: str) -> None:
        prompts.append(prompt)
        real_prompt(self, prompt)

    def spy_image(self: AgeGate, raw: object, prompt: str = "") -> object:
        frames.append(prompt)
        return real_image(self, raw, prompt)

    monkeypatch.setattr(AgeGate, "review_prompt", spy_prompt)
    monkeypatch.setattr(AgeGate, "review_image", spy_image)
    result = _run(tmp_path)
    assert result.stopped == ""
    assert prompts[0] == IDEA
    assert any("\n" in item for item in prompts)
    assert len(prompts) == 7
    assert all(item.startswith(IDEA + "\n") for item in prompts[1:])
    assert frames == [item.split("\n", 1)[1] for item in prompts[1:]]
    assert len(frames) == 6


def test_a_shot_that_drops_the_age_is_replaced_by_a_planned_shot(tmp_path: Path) -> None:
    calls: list[str] = []

    def rewrite(prompt: str) -> str:
        calls.append(prompt)
        return _payload(AGED, scene_age="now 12 years old")

    result = _run(tmp_path, idea=AGED, rewrite=rewrite)
    document = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert result.stopped == ""
    assert result.source == "mixed"
    assert len(result.kept) == 6
    assert len(calls) == 2
    assert all("12 years" not in line for line in document["lines"])
    assert all(AGED in line and not line.startswith(AGED) for line in document["lines"])


def test_cancel_keeps_finished_frames(tmp_path: Path) -> None:
    produced = {"n": 0}

    def generate(line: str, seed: int, hero: object | None, strength: float) -> object:
        del line, seed, hero, strength
        produced["n"] += 1
        return _frame()

    result = _run(tmp_path, generate=generate, cancelled=lambda: produced["n"] >= 1)
    assert result.stopped == "cancelled"
    assert len(result.kept) == 1
    assert produced["n"] == 1
    assert (tmp_path / "01.png").is_file()
    assert not (tmp_path / "02.png").exists()


def test_object_reads_prose_around_json_and_rejects_a_refusal() -> None:
    body = _payload(IDEA, 1)
    before = _object("The shoot is below.\n" + body)
    after = _object(body + "\nThat is the shoot.")
    assert before is not None and before["subject_block"] == IDEA
    assert after is not None and after["place"] == "kitchen table"
    braced = (
        'prefix {"scene": "a } brace", "pose": "still", "camera": "50 mm", "place": "kitchen {table}"} tail'
    )
    shot = _object(braced)
    assert shot is not None
    assert shot["scene"] == "a } brace"
    assert shot["place"] == "kitchen {table}"
    assert _object("Sorry, I cannot write that.") is None
    parsed = validate_scenario(IDEA, "Notes.\n" + _payload(IDEA) + "\nEnd.", count=6, place_mode="same")
    assert parsed is not None and len(parsed.shots) == 6


def test_retry_sends_a_different_request(tmp_path: Path) -> None:
    calls: list[str] = []

    def rewrite(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "Sorry, I cannot write that."
        return _payload(IDEA, 1)

    result = _run(tmp_path, count=1, rewrite=rewrite)
    assert result.stopped == ""
    assert result.code == ""
    assert calls[1] == f"{calls[0]}\n{FORMAT_REMINDER}"
    assert calls[0] != calls[1]


def test_each_reason_code(tmp_path: Path, monkeypatch) -> None:
    refused = _run(tmp_path / "policy", age_gate=_guard_error(), rewrite=lambda _prompt: _payload(IDEA))
    assert refused.stopped == "idea" and refused.code == "idea.policy"

    monkeypatch.setattr("canvas.pipeline.series.ages_allowed", lambda _text: False)
    minor_idea = _run(tmp_path / "idea-minor", adult=True)
    assert minor_idea.stopped == "idea" and minor_idea.code == "idea.minor"
    monkeypatch.setattr("canvas.pipeline.series.ages_allowed", ages_allowed)

    bad_place = _run(tmp_path / "place", place_mode="beside")
    assert bad_place.stopped == "scenario" and bad_place.code == "scenario.format"
    empty_idea = _run(tmp_path / "empty-idea", idea="   ")
    assert empty_idea.stopped == "scenario" and empty_idea.code == "scenario.format"
    prose = _run(tmp_path / "prose", rewrite=lambda _prompt: "The table is empty. " * 30)
    assert prose.stopped == "" and prose.code == "" and prose.source == "planned"
    assert len(prose.kept) == 6

    declined = _run(tmp_path / "declined", rewrite=lambda _prompt: "Sorry, I cannot write that.")
    assert declined.stopped == "" and declined.source == "planned"

    def short(prompt: str) -> str:
        if FORMAT_REMINDER in prompt:
            return "No object here."
        raw = json.loads(_payload(IDEA, 6))
        raw["shots"][1]["scene"] = raw["shots"][0]["scene"]
        return json.dumps(raw)

    counted = _run(tmp_path / "count", rewrite=short)
    assert counted.stopped == "" and counted.source == "mixed" and len(counted.kept) == 6

    blocked = _run(tmp_path / "lines", age_gate=_refuse_lines())
    assert blocked.code == "scenario.policy"

    base = scan(IDEA)

    def fake_scan(text: str):
        if "close view" in text:
            return replace(base, minor=True)
        return base

    monkeypatch.setattr("canvas.pipeline.series.scan", fake_scan)
    hard = _run(tmp_path / "hard", age_gate=_refuse_close())
    assert hard.code == "scenario.minor"
    monkeypatch.setattr("canvas.pipeline.series.scan", scan)

    def joined(text: str) -> bool:
        return "\n" not in text

    monkeypatch.setattr("canvas.pipeline.series.ages_allowed", joined)
    joined_stop = _run(tmp_path / "joined", adult=True)
    assert joined_stop.code == "scenario.minor"
    monkeypatch.setattr("canvas.pipeline.series.ages_allowed", ages_allowed)

    empty = _run(tmp_path / "empty", generate=lambda *_args: None)
    assert empty.stopped == "" and empty.code == "not_produced" and empty.kept == ()


def test_a_declined_scenario_logs_only_the_code_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("VOXPRINT_DEBUG", raising=False)
    result = _run(tmp_path / "series", idea=AGED, rewrite=lambda _prompt: "Sorry, I cannot write 34 lines.")
    assert result.stopped == "" and result.source == "planned"
    log = (tmp_path / "home" / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "scenario.declined" in log
    assert "scenario.raw" not in log and "cannot write" not in log
    assert "34" not in _without_metric_lines(log)
    assert "red coat" not in log


def test_the_debug_switch_adds_the_masked_reply(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VOXPRINT_DEBUG", "1")
    _run(tmp_path / "series", idea=AGED, rewrite=lambda _prompt: "Sorry, I cannot write 34 lines.")
    log = (tmp_path / "home" / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "debug scenario.raw" in log and "cannot write" in log
    raw = [line for line in log.splitlines() if "scenario.raw" in line]
    assert raw and all("34" not in line for line in raw)
    assert "34" not in _without_metric_lines(log)
    assert "red coat" not in log


def _without_metric_lines(log: str) -> str:
    """Drop timing, stage, and VRAM lines. Their numbers are not the reply."""
    kept = [
        line for line in log.splitlines()
        if not line.startswith(("timing ", "stage ", "vram "))
    ]
    return "\n".join(kept)


def _guard_error():
    world = World()
    world.error = "guard"
    return world.gate()


def _refuse_lines():
    gate = World().gate()
    real = gate.review_prompt

    def review(prompt: str) -> None:
        if "\n" in prompt:
            raise StylisedBlocked("prompt")
        real(prompt)

    gate.review_prompt = review
    return gate


def _refuse_close():
    gate = World().gate()
    real = gate.review_prompt

    def review(prompt: str) -> None:
        if "close view" in prompt:
            raise GenerationBlocked("prompt")
        real(prompt)

    gate.review_prompt = review
    return gate


class _Frame:
    def __init__(self, data: bytes) -> None:
        self.width = 2
        self.height = 2
        self.data = data * 4

    def to_bytes(self) -> bytes:
        return self.data


def _frame(data: bytes = b"\x01\x02\x03") -> _Frame:
    return _Frame(data)


def _run(folder: Path, **overrides: object):
    idea = str(overrides.pop("idea", IDEA))
    age_gate = overrides.pop("age_gate", None)
    rewrite = overrides.pop("rewrite", None)
    generate = overrides.pop("generate", None)
    cancelled = overrides.pop("cancelled", None)
    count = overrides.pop("count", 6)
    adult = overrides.pop("adult", False)
    place_mode = overrides.pop("place_mode", "same")
    line_for = overrides.pop("line_for", None)
    timings = overrides.pop("timings", None)
    return run_series(
        idea=idea,
        count=int(count) if isinstance(count, int) else 6,
        style="photo",
        place_mode=str(place_mode),
        adult=bool(adult),
        seed=3,
        strength=None,
        include_camera=lambda: True,
        rewrite=rewrite if rewrite is not None else (lambda _prompt: _payload(idea)),
        age_gate=age_gate if isinstance(age_gate, AgeGate) else World().gate(),
        draw_frame=generate if generate is not None else (lambda line, seed, hero, strength: _frame()),
        cancelled=cancelled if cancelled is not None else (lambda: False),
        folder=folder,
        line_for=line_for if callable(line_for) else None,
        timings=timings if isinstance(timings, StageTimes) else None,
    )


def _lines(count: int, *, start: int = 0, place: str = "kitchen table", different: bool = False) -> str:
    rows = []
    for offset in range(count):
        number = start + offset
        where = f"room {number}" if different else place
        rows.append(f"{offset + 1}. view {number}, soft light | still | 50 mm, eye level | {where}")
    return "Here is the plan.\n" + "\n".join(rows) + "\nDone."


def test_numbered_lines_are_read_as_shots(tmp_path: Path) -> None:
    result = _run(tmp_path, rewrite=lambda prompt: _lines(int(json.loads(prompt)["count"])))
    document = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert result.stopped == "" and result.source == "writer"
    assert len(result.kept) == 6
    assert document["appearance"] == ""
    assert document["age"] is None
    assert document["subject_block"] == IDEA
    assert document["lines"][0] == (
        f"A single photograph of {IDEA}, still, 50 mm, eye level, view 0, soft light, kitchen table"
    )


def test_lines_without_a_place_use_the_series_place(tmp_path: Path) -> None:
    def rewrite(prompt: str) -> str:
        count = int(json.loads(prompt)["count"])
        rows = [f"{index + 1}. view {index} | still | 50 mm" for index in range(count)]
        rows[0] += " | kitchen table"
        return "\n".join(rows)

    result = _run(tmp_path, rewrite=rewrite)
    document = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert result.source == "writer"
    assert all(line.endswith(", kitchen table") for line in document["lines"])


def test_json_shots_without_a_place_use_the_shared_place(tmp_path: Path) -> None:
    def rewrite(prompt: str) -> str:
        raw = json.loads(_payload(IDEA, int(json.loads(prompt)["count"])))
        for shot in raw["shots"]:
            del shot["place"]
        return json.dumps(raw)

    result = _run(tmp_path, rewrite=rewrite)
    assert result.stopped == "" and result.source == "writer" and len(result.kept) == 6


def test_planned_fallback_prompts_differ(tmp_path: Path) -> None:
    prompts: list[str] = []

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        del seed, hero, strength
        prompts.append(line)
        return _frame()

    result = _run(tmp_path, rewrite=lambda _prompt: "no lines here", generate=generate)
    assert result.source == "planned"
    assert len(prompts) == 6
    assert len(set(prompts)) == 6
    assert all(IDEA in line and not line.startswith(IDEA) for line in prompts)


def test_different_places_stay_distinct_with_planned_shots(tmp_path: Path) -> None:
    def rewrite(prompt: str) -> str:
        body = json.loads(prompt)
        return _lines(2, start=len(body.get("previous", [])), different=True)

    result = _run(tmp_path, count=14, place_mode="different", rewrite=rewrite)
    document = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert result.stopped == "" and len(result.kept) == 14
    assert len(set(document["lines"])) == 14


def test_planned_shots_cover_ninety_nine_frames_in_either_place_mode() -> None:
    for mode in ("same", "different"):
        for adult in (False, True):
            shots = planned_shots(AGED, 99, mode, place="", taken=(), adult=adult)
            assert len(shots) == 99
            assert len({shot.scene for shot in shots}) == 99
            if mode == "different":
                assert len({shot.place for shot in shots}) == 99
            for shot in shots:
                line = assemble_line(AGED, shot, include_camera=True)
                assert line.startswith(f"A single photograph of {AGED}, {shot.pose}")
                assert AGED in line
                assert sorted(stated_ages(line)) == [34]
    assert planned_shots(IDEA, 6, "same", place="", taken=()) == planned_shots(IDEA, 6, "same", place="", taken=())


def test_planned_lines_go_through_the_same_prompt_review(tmp_path: Path) -> None:
    prompts: list[str] = []
    gate = World().gate()
    real = gate.review_prompt

    def review(prompt: str) -> None:
        prompts.append(prompt)
        real(prompt)

    gate.review_prompt = review
    result = _run(tmp_path, age_gate=gate, rewrite=lambda _prompt: "no lines here")
    assert result.source == "planned"
    assert len(prompts) == 7
    assert all(item.startswith(IDEA + "\n") for item in prompts[1:])


def test_a_refused_line_still_stops_the_series_and_is_not_replaced(tmp_path: Path) -> None:
    calls: list[str] = []

    def rewrite(prompt: str) -> str:
        calls.append(prompt)
        rows = _lines(int(json.loads(prompt)["count"])).splitlines()
        rows[1] = rows[1].replace("view 0", "loli")
        return "\n".join(rows)

    result = _run(tmp_path, rewrite=rewrite)
    assert result.stopped == "scenario"
    assert len(calls) == 1
    assert result.kept == ()


def test_the_retry_is_sent_only_when_no_line_can_be_read(tmp_path: Path) -> None:
    calls: list[str] = []

    def rewrite(prompt: str) -> str:
        calls.append(prompt)
        return "1. only | one line"

    result = _run(tmp_path, count=2, rewrite=rewrite)
    assert result.source == "planned"
    assert len(calls) == 2
    assert calls[1] == f"{calls[0]}\n{FORMAT_REMINDER}"


def test_a_three_part_line_is_read_by_its_last_part(tmp_path: Path) -> None:
    def rewrite(prompt: str) -> str:
        count = int(json.loads(prompt)["count"])
        start = len(json.loads(prompt).get("previous", []))
        return "\n".join(f"{index + 1}. view {start + index}, seated | 50 mm lens, eye level | room {start + index}"
                         for index in range(count))

    result = _run(tmp_path, place_mode="different", rewrite=rewrite)
    document = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert result.source == "writer"
    places = {line.rsplit(", ", 1)[-1] for line in document["lines"]}
    assert len(places) == 6
    still = _run(tmp_path / "same", rewrite=lambda _prompt: "\n".join(
        f"{index + 1}. view {index} | still | 85 mm lens" for index in range(6)))
    assert still.source == "writer"


def test_fixed_appearance_is_stable_and_names_a_concrete_look() -> None:
    assert fixed_appearance(IDEA, 3) == ""
    assert fixed_appearance("a manual camera", 1) == ""
    assert fixed_appearance("the manager of the shop", 1) == ""
    assert fixed_appearance("personal computer", 1) == ""
    assert fixed_appearance("humanoid robot", 1) == ""
    assert fixed_appearance(PERSON, 3) == fixed_appearance(f"  {PERSON}  ", 3)
    assert fixed_appearance("a man in a wool coat", 1)
    assert fixed_appearance("женщина в красном пальто", 2)
    for table in (_ETHNICITY, _HAIR, _EYES, _FACE, _BUILD):
        for item in table:
            assert _BANNED_LOOK.search(item) is None
            assert stated_ages(item) == ()
    for seed in range(400):
        look = fixed_appearance(PERSON, seed)
        assert look == fixed_appearance(PERSON, seed)
        assert _BANNED_LOOK.search(look) is None
        assert stated_ages(look) == ()
        assert len([part for part in look.split(",") if part.strip()]) == 5


def test_a_person_series_repeats_one_appearance_and_anchors_the_hero(tmp_path: Path) -> None:
    seen: list[tuple[str, object | None, float | None]] = []

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        del seed
        seen.append((line, hero, strength))
        return _frame()

    result = _run(tmp_path, idea=PERSON, generate=generate)
    look = fixed_appearance(PERSON, 3)
    document = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert result.stopped == ""
    assert len(result.kept) == 6
    assert look
    assert document["appearance"] == look
    assert document["subject_block"] == f"{look}, {PERSON}"
    assert all(line.count(look) == 1 and PERSON in line for line in document["lines"])
    assert all(sorted(stated_ages(line)) == [25] for line in document["lines"])
    assert seen[0][1] is None
    assert all(item[1] is seen[1][1] for item in seen[1:])
    assert all(item[0].startswith(f"A single photograph of {look},") for item in seen)
    assert [item[2] for item in seen] == [REFERENCE_STRENGTH] * 6
    assert document["age"] == 25


def test_a_blocked_frame_does_not_replace_the_hero(tmp_path: Path) -> None:
    seen: list[object | None] = []

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object | None:
        del line, seed, strength
        seen.append(hero)
        if len(seen) == 2:
            return None
        return _frame()

    result = _run(tmp_path, idea=PERSON, count=4, generate=generate)
    assert result.stopped == ""
    assert len(result.kept) == 3
    assert seen[0] is None
    assert seen[1] is not None
    assert seen[2] is seen[1]
    assert seen[3] is seen[1]


def test_a_series_records_peak_vram(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("canvas.pipeline.series.peak_allocated_bytes", lambda: 1500 * 1024 * 1024)
    clock = StageTimes()
    result = _run(tmp_path, idea=PERSON, count=2, timings=clock)
    assert result.stopped == ""
    assert "peak_vram=mib1500" in clock.log_line()


def test_a_planned_person_series_shares_the_same_appearance(tmp_path: Path) -> None:
    prompts: list[str] = []
    strengths: list[float | None] = []

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        del seed, hero
        prompts.append(line)
        strengths.append(strength)
        return _frame()

    result = _run(tmp_path, idea=PERSON, rewrite=lambda _prompt: "no lines here", generate=generate)
    look = fixed_appearance(PERSON, 3)
    assert result.source == "planned"
    assert len(prompts) == 6
    assert all(line.count(look) == 1 for line in prompts)
    assert strengths == [REFERENCE_STRENGTH] * 6


def test_an_explicit_strength_is_kept_for_a_person(tmp_path: Path) -> None:
    seen: list[float | None] = []

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        del line, seed, hero
        seen.append(strength)
        return _frame()

    def once(folder: Path, strength: float) -> None:
        seen.clear()
        run_series(
            idea=PERSON, count=2, style="photo", place_mode="same", seed=3, strength=strength,
            include_camera=lambda: True, rewrite=lambda _prompt: _payload(PERSON, 2),
            age_gate=World().gate(), draw_frame=generate, cancelled=lambda: False, folder=folder,
        )

    once(tmp_path / "kept", 0.9)
    assert seen == [0.9, 0.9]
    once(tmp_path / "text", 0.5)
    assert seen == [None, None]


def test_a_frame_prompt_has_no_series_words(tmp_path: Path) -> None:
    idea = "same woman in every frame, varied shots and angles, a woman in a red coat"
    prompts: list[str] = []
    prepared: list[list[str]] = []

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        del seed, hero, strength
        prompts.append(line)
        return _frame()

    def prepare(lines: list[str]) -> None:
        prepared.append(list(lines))

    generate.prepare = prepare  # type: ignore[attr-defined]
    result = _run(tmp_path, idea=idea, generate=generate, count=3)
    look = fixed_appearance(idea, 3)
    assert result.stopped == ""
    assert prompts
    banned = re.compile(
        r"\b(?:every\s+frames?|series|shots|angles|panels|grids?|collages?|sheets?|sequences?|multiple|sets?)\b",
        re.IGNORECASE,
    )
    joined = "\n".join(prompts)
    for line in prompts:
        assert banned.search(line) is None
        assert line.startswith(f"A single photograph of {look},")
        assert "red coat" in line
        assert "woman" in line
        assert line.count("A single photograph of") == 1
    assert prepared and prepared[0] == prompts
    assert joined not in prepared[0]


def test_scenario_token_budget_follows_the_frame_count() -> None:
    assert scenario_token_budget(1) == TOKENS_PER_FRAME
    assert scenario_token_budget(5) == 5 * TOKENS_PER_FRAME
    assert scenario_token_budget(5) < SCENARIO_TOKENS
    assert scenario_token_budget(99) == min(SCENARIO_TOKENS, 99 * TOKENS_PER_FRAME)
    assert "under 18 words" in SYSTEM_PROMPT
    assert "appearance first" in SYSTEM_PROMPT
    assert scenario_token_budget(True) == TOKENS_PER_FRAME
    request = scenario_request(PERSON, 5, "photo", "same")
    assert request is not None
    assert tokens_for_request(request) == scenario_token_budget(5)
    assert tokens_for_request(f"{request}\n{FORMAT_REMINDER}") == scenario_token_budget(5)
    assert tokens_for_request("ping") == scenario_token_budget(1)
