"""Photoshoot writes a scenario from the prompt and generates the series."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from canvas.pipeline.enhance import stated_ages
from canvas.pipeline.photoshoot import (
    FORMAT_REMINDER,
    Shot,
    _object,
    assemble_line,
    planned_shots,
    scenario_request,
    validate_scenario,
)
from canvas.pipeline.series import ages_allowed, run_series
from canvas.safety.denylist import scan
from canvas.safety.gate import AgeGate, GenerationBlocked
from tests.fakes import World


IDEA = "Yellow ceramic teapot with a blue handle"
AGED = "a woman, 34 years old, in a red coat"


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


def test_simple_line_omits_the_camera() -> None:
    shot = Shot("close view", "still", "50 mm", "kitchen table")
    plain = assemble_line(IDEA, shot, include_camera=False)
    detailed = assemble_line(IDEA, shot, include_camera=True)
    assert "50 mm" not in plain
    assert plain == f"{IDEA}, kitchen table, close view, still"
    assert detailed.endswith("50 mm")


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


def test_frame_count_bounds_are_one_to_fifty() -> None:
    assert scenario_request(IDEA, 0, "photo", "same") is None
    assert scenario_request(IDEA, 51, "photo", "same") is None
    assert scenario_request(IDEA, 1, "photo", "same") is not None
    assert scenario_request(IDEA, 50, "photo", "same") is not None
    one = validate_scenario(IDEA, _payload(IDEA, 1), count=1, place_mode="same")
    fifty = validate_scenario(IDEA, _payload(IDEA, 50), count=50, place_mode="same")
    assert one is not None and len(one.shots) == 1
    assert fifty is not None and len(fifty.shots) == 50
    assert validate_scenario(IDEA, _payload(IDEA, 1), count=0, place_mode="same") is None
    assert validate_scenario(IDEA, _payload(IDEA, 1), count=51, place_mode="same") is None


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
            return f"{IDEA}, kitchen table, three-quarter view, still loli"
        return f"{IDEA}, kitchen table, view {index}, still"

    result = _run(tmp_path, rewrite=lambda prompt: calls.append(prompt) or _payload(IDEA), line_for=line_for)
    document = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert result.stopped == ""
    assert len(calls) == 1
    assert len(result.kept) == 5
    assert document["blocked"] == [1]
    assert all("loli" not in line for line in document["lines"])


def test_blocked_frame_is_dropped_without_a_rewrite(tmp_path: Path) -> None:
    calls: list[str] = []
    heroes: list[object | None] = []
    gate = World().gate()
    real = gate.review_image

    def review_image(raw: object, prompt: str = "") -> object:
        if "three-quarter" in prompt:
            raise GenerationBlocked("image")
        return real(raw, prompt)

    gate.review_image = review_image

    def generate(line: str, seed: int, hero: object | None, strength: float) -> object:
        del seed
        heroes.append(hero)
        assert strength == 0.45
        assert "50 mm" not in line
        return _frame(b"\x10\x20\x30")

    result = _run(tmp_path, age_gate=gate, rewrite=lambda prompt: calls.append(prompt) or _payload(IDEA), generate=generate)
    assert result.stopped == ""
    assert len(calls) == 1
    assert len(result.kept) == 5
    assert heroes[0] is None
    first = heroes[1]
    assert first is not None
    assert all(hero is first for hero in heroes[2:])
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
    assert result.stopped == ""
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
    assert len(prompts) == 13
    assert all(item.startswith(IDEA + "\n") for item in prompts[1:])
    assert frames == [item.split("\n", 1)[1] for item in prompts[7:]]
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
    assert all(line.startswith(AGED) for line in document["lines"])


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


def test_a_declined_scenario_logs_the_code_and_the_masked_reply(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    result = _run(tmp_path / "series", idea=AGED, rewrite=lambda _prompt: "Sorry, I cannot write 34 lines.")
    assert result.stopped == "" and result.source == "planned"
    log = (tmp_path / "home" / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "scenario.declined" in log
    assert "debug scenario.raw" in log and "cannot write" in log
    assert "34" not in log
    assert "red coat" not in log


def _guard_error():
    world = World()
    world.error = "guard"
    return world.gate()


def _refuse_lines():
    gate = World().gate()
    real = gate.review_prompt

    def review(prompt: str) -> None:
        if "\n" in prompt:
            raise GenerationBlocked("prompt")
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
    return run_series(
        idea=idea,
        count=int(count) if isinstance(count, int) else 6,
        style="photo",
        place_mode=str(place_mode),
        adult=bool(adult),
        seed=3,
        strength=0.45,
        include_camera=lambda: False,
        rewrite=rewrite if rewrite is not None else (lambda _prompt: _payload(idea)),
        age_gate=age_gate if isinstance(age_gate, AgeGate) else World().gate(),
        draw_frame=generate if generate is not None else (lambda line, seed, hero, strength: _frame()),
        cancelled=cancelled if cancelled is not None else (lambda: False),
        folder=folder,
        line_for=overrides.pop("line_for", None),
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
    assert document["lines"][0] == f"{IDEA}, kitchen table, view 0, soft light, still"


def test_lines_without_a_place_use_the_series_place(tmp_path: Path) -> None:
    def rewrite(prompt: str) -> str:
        count = int(json.loads(prompt)["count"])
        rows = [f"{index + 1}. view {index} | still | 50 mm" for index in range(count)]
        rows[0] += " | kitchen table"
        return "\n".join(rows)

    result = _run(tmp_path, rewrite=rewrite)
    document = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert result.source == "writer"
    assert all(", kitchen table, " in line for line in document["lines"])


def test_json_shots_without_a_place_use_the_shared_place(tmp_path: Path) -> None:
    def rewrite(prompt: str) -> str:
        raw = json.loads(_payload(IDEA, int(json.loads(prompt)["count"])))
        for shot in raw["shots"]:
            del shot["place"]
        return json.dumps(raw)

    result = _run(tmp_path, rewrite=rewrite)
    assert result.stopped == "" and result.source == "writer" and len(result.kept) == 6


def test_different_places_stay_distinct_with_planned_shots(tmp_path: Path) -> None:
    def rewrite(prompt: str) -> str:
        body = json.loads(prompt)
        return _lines(2, start=len(body.get("previous", [])), different=True)

    result = _run(tmp_path, count=14, place_mode="different", rewrite=rewrite)
    document = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert result.stopped == "" and len(result.kept) == 14
    assert len(set(document["lines"])) == 14


def test_planned_shots_cover_fifty_frames_in_either_place_mode() -> None:
    for mode in ("same", "different"):
        for adult in (False, True):
            shots = planned_shots(AGED, 50, mode, place="", taken=(), adult=adult)
            assert len(shots) == 50
            assert len({shot.scene for shot in shots}) == 50
            if mode == "different":
                assert len({shot.place for shot in shots}) == 50
            for shot in shots:
                line = assemble_line(AGED, shot, include_camera=True)
                assert line.startswith(AGED)
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
    assert len(prompts) == 13
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
