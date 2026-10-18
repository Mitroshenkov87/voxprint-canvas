"""Adult photo direction: lint, adult anchor, variety, and refusal without rewriting. No generation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canvas.pipeline import adult_direction as adult
from canvas.pipeline.enhance import SYSTEM_PROMPT as ENHANCE_SYSTEM
from canvas.pipeline.photoshoot import SYSTEM_PROMPT as SCENARIO_SYSTEM
from canvas.pipeline.photoshoot import scenario_request
from canvas.pipeline.series import run_series
from canvas.safety.gate import GenerationBlocked
from tests.fakes import World

IDEA = "boudoir shoot in a boutique hotel suite"


class _Frame:
    width = 2
    height = 2

    def to_bytes(self) -> bytes:
        return b"\x05\x06\x07" * 4


def _adult_world_gate():
    world = World()
    world.safety = "unsafe"
    world.categories = ("Sexual Content or Sexual Acts",)
    return world.gate()


def _director(pose_override: dict[int, str] | None = None):
    """Stub language model: follows the request's direction list."""
    calls: list[str] = []

    def rewrite(request: str) -> str:
        calls.append(request)
        body = json.loads(request)
        shots = []
        for index, step in enumerate(body["direction"]):
            pose = (pose_override or {}).get(index, step["pose_family"])
            shots.append({
                "scene": f"{step['shot_type']}, {step['light']}, silk slip dress",
                "pose": pose,
                "camera": "85 mm, eye level",
                "place": "boutique hotel suite",
            })
        return json.dumps({
            "subject_block": IDEA,
            "descriptors": ["boudoir shoot"],
            "place": "boutique hotel suite",
            "shots": shots,
        })

    return rewrite, calls


def _series(tmp_path: Path, rewrite, lines: list[str], statuses: list[tuple[int, str]], gate=None):
    def draw(line, seed, hero, strength):
        del seed, hero, strength
        lines.append(line)
        return _Frame()

    return run_series(
        idea=IDEA, count=6, style="photo", place_mode="same", seed=1, strength=0.45,
        include_camera=lambda: True, rewrite=rewrite, age_gate=gate or _adult_world_gate(),
        draw_frame=draw, cancelled=lambda: False, folder=tmp_path / "series",
        on_status=lambda index, status: statuses.append((index, status)), adult=True,
    )


def test_module_text_and_examples_have_no_youth_cues() -> None:
    hits = {text[:60]: adult.youth_cues(text) for text in adult.prompt_texts() if adult.youth_cues(text)}
    assert hits == {}
    assert adult.youth_cues("a petite schoolgirl in anime style") == ("petite", "schoolgirl", "anime")


def test_adult_anchor_only_when_the_policy_is_on() -> None:
    assert adult.ADULT_ANCHOR in adult.enhancer_system(ENHANCE_SYSTEM, True)
    assert adult.ADULT_ANCHOR in adult.scenario_system(SCENARIO_SYSTEM, True)
    assert adult.enhancer_system(ENHANCE_SYSTEM, False) == ENHANCE_SYSTEM
    assert adult.scenario_system(SCENARIO_SYSTEM, False) == SCENARIO_SYSTEM
    assert "direction" not in json.loads(scenario_request(IDEA, 6, "photo", "same") or "{}")
    assert len(json.loads(scenario_request(IDEA, 6, "photo", "same", adult=True) or "{}")["direction"]) == 6
    assert all(prompt.startswith(adult.ADULT_ANCHOR) or "year-old" in prompt for _idea, prompt in adult.EXAMPLES)
    assert adult.with_adult_anchor("person on a velvet sofa").endswith(adult.ADULT_SUFFIX)
    assert adult.with_adult_anchor("34-year-old man on a sofa") == "34-year-old man on a sofa"
    assert not adult.ages_allowed("a 17-year-old")


def test_six_shots_never_repeat_a_pose_or_a_light(tmp_path: Path) -> None:
    rewrite, _calls = _director()
    lines: list[str] = []
    result = _series(tmp_path, rewrite, lines, [])
    assert result.stopped == ""
    assert len(lines) == 6
    poses = [next(p for p in adult.POSE_FAMILIES if p in line) for line in lines]
    lights = [next(light for light in adult.LIGHT_SETUPS if light in line) for line in lines]
    assert len(set(poses)) == 6
    assert len(set(lights)) == 6
    assert all(adult.ADULT_SUFFIX in line for line in lines)
    assert all(not adult.youth_cues(line) for line in lines)
    plan = adult.shot_plan(20)
    for start in range(15):
        window = plan[start:start + 6]
        assert len({step["pose_family"] for step in window}) == 6
        assert len({step["light"] for step in window}) == 6


def test_a_shot_with_a_youth_cue_is_refused_and_not_rewritten(tmp_path: Path) -> None:
    rewrite, calls = _director({2: "petite pose by the window"})
    lines: list[str] = []
    statuses: list[tuple[int, str]] = []
    result = _series(tmp_path, rewrite, lines, statuses)
    assert result.stopped == ""
    assert (2, "blocked") in statuses
    assert len(lines) == 5
    assert all("petite" not in line for line in lines)
    assert len(calls) == 1


def test_a_scenario_the_gate_refuses_stops_without_a_rewrite(tmp_path: Path) -> None:
    rewrite, calls = _director({2: "schoolgirl pose by the window"})
    lines: list[str] = []
    result = _series(tmp_path, rewrite, lines, [])
    assert result.stopped == "scenario"
    assert result.reason
    assert lines == []
    assert len(calls) == 1


def test_the_gate_filter_refuses_a_youth_line(tmp_path: Path) -> None:
    gate = _adult_world_gate()
    gate.review_prompt(f"{adult.ADULT_ANCHOR}, boutique hotel suite, black lace lingerie")
    with pytest.raises(GenerationBlocked):
        gate.review_prompt("teen in lingerie, boudoir")


class _Rewriter:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[str] = []
        self.closed = False

    def complete(self, prompt: str, *, system: str) -> str:
        self.calls.append(system)
        return self.reply

    def close(self) -> None:
        self.closed = True


def test_improve_refuses_a_youth_reply_without_rewriting() -> None:
    rewriter = _Rewriter("petite model on a bed, soft window light")
    with pytest.raises(GenerationBlocked):
        adult.improve_adult(rewriter, IDEA, ENHANCE_SYSTEM)
    assert len(rewriter.calls) == 1 and rewriter.closed
    assert adult.ADULT_ANCHOR in rewriter.calls[0]


def test_improve_keeps_an_adult_reply_and_refuses_an_under_18_age() -> None:
    reply = f"{adult.ADULT_ANCHOR}, boutique hotel suite, satin robe, golden hour sun through sheer curtains"
    assert adult.improve_adult(_Rewriter(reply), IDEA, ENHANCE_SYSTEM) == reply
    untouched = _Rewriter(reply)
    with pytest.raises(GenerationBlocked):
        adult.improve_adult(untouched, "a 16-year-old in a robe", ENHANCE_SYSTEM)
    assert untouched.calls == []
