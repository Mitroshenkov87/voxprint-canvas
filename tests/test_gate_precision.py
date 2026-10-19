"""gate: precision cases."""
import json
from pathlib import Path

import pytest

from canvas.pipeline.series import run_series
from canvas.safety.classifiers import Face
from canvas.safety.constants import (
    CLEAR_CHILD_AT,
    IMAGE_ADULT_SEXUAL,
    IMAGE_ANIME,
    IMAGE_CHIBI,
    IMAGE_CLASSROOM,
    IMAGE_LABELS,
    IMAGE_MINOR,
    IMAGE_MINOR_SEXUAL,
    IMAGE_SCHOOL_UNIFORM,
    ZERO_SHOT_BLOCK_AT,
)
from canvas.safety.gate import GenerationBlocked, image_block_reason
from tests.fakes import World, rgb

PASS_A = [
    "hey baby, nude",
    "hey baby nude photo",
    "oh baby, nude",
    "yes baby, take it off, nude",
    "my baby, nude on the bed",
    "baby, nude",
    "\"Come here, baby,\" she said, nude",
    "adult woman, 30 years old, nude, hey, baby",
]
BLOCK_A = [
    "a baby, nude",
    "the baby, nude",
    "nude baby",
    "naked baby",
    "baby nude",
    "my baby nude",
    "nude, baby",
    "nude babies",
    "nude infant",
    "nude newborn",
    "nude toddler",
]


@pytest.mark.parametrize("prompt", PASS_A)
def test_a_pass(prompt):
    World().gate().review_prompt(prompt)


@pytest.mark.parametrize("prompt", BLOCK_A)
def test_a_block(prompt):
    with pytest.raises(GenerationBlocked):
        World().gate().review_prompt(prompt)


class _Frame:
    width = 2
    height = 2

    def to_bytes(self) -> bytes:
        return rgb().to_bytes()


def _series(tmp_path: Path, idea: str, scenes: list[str]):
    shots = [{"scene": scene, "pose": "standing", "camera": "50 mm", "place": "studio"} for scene in scenes]
    payload = json.dumps({"subject_block": SUBJECT, "descriptors": ["red hair"], "place": "studio", "shots": shots})
    return run_series(
        idea=idea,
        count=6,
        style="photo",
        place_mode="same",
        seed=3,
        strength=None,
        include_camera=lambda: True,
        rewrite=lambda _prompt: payload,
        age_gate=World().gate(),
        draw_frame=lambda line, seed, hero, strength: _Frame(),
        cancelled=lambda: False,
        folder=tmp_path / "series",
    )


SUBJECT = "a woman, 30 years old, red hair"
SCENES = ["in a classroom, fully clothed", "nude on a bed", "nude by a window", "in a red dress", "nude in the shower", "on a sofa"]


def test_b_shots_are_screened_one_by_one(tmp_path):
    result = _series(tmp_path, SUBJECT, SCENES)
    assert result.stopped == ""
    assert len(result.kept) == 6


def test_b_explicit_idea_drops_the_school_shot_only(tmp_path):
    result = _series(tmp_path, f"{SUBJECT}, nude series", SCENES)
    assert result.stopped == ""
    assert len(result.kept) == 5


def test_b_idea_with_both_stops_the_series(tmp_path):
    result = _series(tmp_path, "nude schoolgirl series", SCENES)
    assert result.stopped == "idea"
    assert result.kept == ()


def test_b_hard_term_in_a_shot_stops_the_series(tmp_path):
    result = _series(tmp_path, SUBJECT, ["loli", *SCENES[1:]])
    assert result.stopped == "scenario"
    assert result.kept == ()


def test_b_every_shot_refused_stops_the_series(tmp_path):
    result = _series(tmp_path, f"{SUBJECT}, nude", [f"in a classroom {n}" for n in range(6)])
    assert result.stopped == "scenario"


def _scores(**values: float) -> dict[str, float]:
    found = {label: 0.0 for label in IMAGE_LABELS}
    found.update(values)
    return found


ADULT_FACE = [(0.05, 0.01)]


def test_c_classroom_needs_a_second_signal():
    assert image_block_reason(0.95, ADULT_FACE, _scores(**{IMAGE_CLASSROOM: 0.9})) == "allow"
    assert image_block_reason(0.95, [], _scores(**{IMAGE_CLASSROOM: 0.9})) == "allow"
    scores = _scores(**{IMAGE_CLASSROOM: 0.9, IMAGE_MINOR: ZERO_SHOT_BLOCK_AT})
    assert image_block_reason(0.95, ADULT_FACE, scores) == "school-context"
    assert image_block_reason(0.10, ADULT_FACE, scores) == "allow"


@pytest.mark.parametrize("label", [IMAGE_SCHOOL_UNIFORM, IMAGE_ANIME, IMAGE_CHIBI])
def test_c_single_labels(label):
    assert image_block_reason(0.95, ADULT_FACE, _scores(**{label: ZERO_SHOT_BLOCK_AT})) != "allow"
    assert image_block_reason(0.10, ADULT_FACE, _scores(**{label: 0.9})) == "allow"


def _faceless(**scores: float) -> World:
    world = World()
    world.nsfw = 0.9
    world.image_scores.update(scores)
    return world


def test_e_faceless_minor_sexual():
    world = _faceless(**{IMAGE_MINOR_SEXUAL: 0.9})
    with pytest.raises(GenerationBlocked):
        world.gate().review_image(rgb())
    assert "image-score" in world.calls and "faces" in world.calls and "ages" not in world.calls


def test_e_faceless_minor_body():
    world = _faceless(**{IMAGE_MINOR: 0.9})
    with pytest.raises(GenerationBlocked):
        world.gate().review_image(rgb())
    assert "image-score" in world.calls and "ages" not in world.calls


def test_e_faceless_adult_passes():
    world = _faceless(**{IMAGE_ADULT_SEXUAL: 0.9})
    world.gate().review_image(rgb())
    assert "image-score" in world.calls


def _probe(bands: list[tuple[float, float]], prompt: str = "", **scores: float) -> bool:
    world = World()
    world.nsfw = 0.9
    world.image_scores.update(scores)
    world.faces = [Face(0, 0, 2, 2)] * len(bands)
    world.ages = bands
    try:
        world.gate().review_image(rgb(), prompt)
    except GenerationBlocked:
        return False
    return True


def test_f_must_pass():
    assert _probe([(0.30, 0.04)], **{IMAGE_ADULT_SEXUAL: 0.9})
    assert _probe([(0.30, 0.04)], "a woman, 22 years old", **{IMAGE_ADULT_SEXUAL: 0.9})


def test_f_must_fail():
    assert not _probe([(0.51, 0.17)], **{IMAGE_MINOR: 0.9})
    assert not _probe([(CLEAR_CHILD_AT, 0.0)])
    assert not _probe(ADULT_FACE, **{IMAGE_MINOR_SEXUAL: 0.9})
    assert not _probe(ADULT_FACE, **{IMAGE_SCHOOL_UNIFORM: 0.9})
    assert not _probe(ADULT_FACE, **{IMAGE_ANIME: 0.9})
