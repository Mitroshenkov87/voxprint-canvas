"""Photoshoot writes a scenario from the prompt and generates the series."""
from __future__ import annotations

import json
from pathlib import Path

from canvas.pipeline.photoshoot import (
    Shot,
    assemble_line,
    scenario_request,
    validate_scenario,
)
from canvas.pipeline.series import run_series


IDEA = "Yellow ceramic teapot with a blue handle"
AGED = "a woman, 34 years old, in a red coat"


def _shots(count: int, place: str, *, scene_age: str = "") -> list[dict[str, str]]:
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
    for index in range(count):
        scene = scenes[index % len(scenes)]
        if index == 1 and scene_age:
            scene = f"{scene}, {scene_age}"
        found.append({"scene": scene, "pose": "still", "camera": "50 mm", "place": place})
    return found


def _payload(idea: str, count: int = 6, *, place_mode: str = "same", subject: str | None = None, scene_age: str = "") -> str:
    place = "kitchen table" if place_mode == "same" else ""
    places = ["kitchen table"] * count if place_mode == "same" else [f"room {index}" for index in range(count)]
    shots = _shots(count, places[0], scene_age=scene_age)
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


def test_count_outside_the_series_is_refused() -> None:
    message = scenario_request(IDEA, 4, "photo", "same")
    assert message is None


def test_blocked_idea_stops_before_the_model() -> None:
    calls: list[str] = []

    def rewrite(prompt: str) -> str:
        calls.append(prompt)
        return _payload(IDEA)

    result = run_series(
        idea="BLOCK this idea",
        count=6,
        style="photo",
        place_mode="same",
        seed=3,
        strength=0.45,
        include_camera=lambda: False,
        rewrite=rewrite,
        review_prompt=lambda text: "blocked" if "BLOCK" in text else None,
        generate=lambda line, seed, hero, strength: _frame(),
        review_image=lambda image, line: None,
        cancelled=lambda: False,
        folder=Path("/tmp/unused-photoshoot"),
    )
    assert result.stopped == "idea"
    assert calls == []
    assert result.kept == ()


def test_blocked_scenario_stops_and_is_not_rewritten(tmp_path: Path) -> None:
    calls: list[str] = []

    def rewrite(prompt: str) -> str:
        calls.append(prompt)
        return _payload(IDEA)

    result = run_series(
        idea=IDEA,
        count=6,
        style="photo",
        place_mode="same",
        seed=3,
        strength=0.45,
        include_camera=lambda: False,
        rewrite=rewrite,
        review_prompt=lambda text: "blocked" if "close view" in text else None,
        generate=lambda line, seed, hero, strength: _frame(),
        review_image=lambda image, line: None,
        cancelled=lambda: False,
        folder=tmp_path,
    )
    assert result.stopped == "scenario"
    assert len(calls) == 1
    assert result.kept == ()


def test_blocked_frame_is_dropped_without_a_rewrite(tmp_path: Path) -> None:
    calls: list[str] = []
    heroes: list[object | None] = []

    def rewrite(prompt: str) -> str:
        calls.append(prompt)
        return _payload(IDEA)

    def generate(line: str, seed: int, hero: object | None, strength: float) -> object:
        heroes.append(hero)
        assert strength == 0.45
        assert "50 mm" not in line
        return _frame(b"\x10\x20\x30")

    result = run_series(
        idea=IDEA,
        count=6,
        style="photo",
        place_mode="same",
        seed=3,
        strength=0.45,
        include_camera=lambda: False,
        rewrite=rewrite,
        review_prompt=lambda text: None,
        generate=generate,
        review_image=lambda image, line: "blocked" if "three-quarter" in line else None,
        cancelled=lambda: False,
        folder=tmp_path,
    )
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


class _Frame:
    def __init__(self, data: bytes) -> None:
        self.width = 2
        self.height = 2
        self.data = data * 4

    def to_bytes(self) -> bytes:
        return self.data


def _frame(data: bytes = b"\x01\x02\x03") -> _Frame:
    return _Frame(data)
