"""Fail-closed checks, issued images, and wiped buffers."""
import logging
import time

import pytest

from canvas.safety.classifiers import Face
from canvas.safety.adapters import age_bands, apparent_age_from_bucket
from canvas.safety.constants import (
    ADOLESCENT_16_17_AT,
    ADOLESCENT_16_17_AT_AGE_18,
    APPARENT_AGE_UNCERTAIN_UNDER,
    BLOCKED_MESSAGE,
    CLASSIFIER_TIMEOUT_S,
    CLEAR_CHILD_AT,
    IMAGE_ADULT_SEXUAL,
    IMAGE_LABELS,
    IMAGE_MINOR,
    IMAGE_MINOR_SEXUAL,
    IMAGE_TIMEOUT_S,
    SEXUAL_SCORE_BLOCK_AT,
    UNAVAILABLE_MESSAGE,
    ZERO_SHOT_BLOCK_AT,
)
from canvas.safety.denylist import explain
from canvas.safety.gate import (
    GenerationBlocked,
    SafetyUnavailable,
    _wait,
    adult_age_support,
    image_block_reason,
    refusal_stage,
)
from canvas.safety.image import SafeImage, ScreenedInput, _INPUT_TOKEN, _OUTPUT_TOKEN
from canvas.safety.inputs import ImageRole, PendingInput
from tests.fakes import World, rgb

SHOWER = (
    "Эротическая художественная фотография: обнажённая женщина 30 лет с длинными тёмными волосами "
    "стоит под струями воды в душе из тёмного камня, капли на коже, пар, мягкий контровой свет, "
    "уверенный взгляд взрослой женщины, реалистичная текстура кожи, кинематографичная цветокоррекция, "
    "среднеформатная камера, 85 мм"
)


def test_thresholds_are_compile_time_constants():
    assert APPARENT_AGE_UNCERTAIN_UNDER == 25
    assert CLEAR_CHILD_AT == 0.70
    assert ADOLESCENT_16_17_AT == 0.16
    assert ADOLESCENT_16_17_AT_AGE_18 == 0.18
    assert SEXUAL_SCORE_BLOCK_AT == 0.50
    assert ZERO_SHOT_BLOCK_AT == 0.35
    assert CLASSIFIER_TIMEOUT_S == 20.0
    assert IMAGE_TIMEOUT_S == 30.0
    source = open("canvas/safety/gate.py", encoding="utf-8").read()
    assert "timeout=CLASSIFIER_TIMEOUT_S" in source
    assert "timeout=IMAGE_TIMEOUT_S" in source
    assert "os.environ" not in source


def test_wait_uses_the_supplied_timeout_instead_of_sleeping_the_full_budget():
    def slow() -> None:
        time.sleep(0.4)

    started = time.monotonic()
    with pytest.raises(SafetyUnavailable) as caught:
        _wait(slow, timeout=0.05)
    assert caught.value.code == "timeout"
    assert time.monotonic() - started < 1.0
    assert str(caught.value) == UNAVAILABLE_MESSAGE


def test_classifier_errors_hide_the_cause():
    def boom() -> None:
        raise RuntimeError("secret-boom")

    with pytest.raises(SafetyUnavailable) as caught:
        _wait(boom, timeout=1)
    assert caught.value.code == "classifier_error"
    assert "secret-boom" not in str(caught.value)


def test_adult_sexual_prompt_is_allowed_and_a_minor_conjunction_is_not():
    world = World()
    world.safety = "unsafe"
    world.categories = ("Sexual Content or Sexual Acts",)
    world.gate().review_prompt("nude")
    with pytest.raises(GenerationBlocked) as caught:
        world.gate().review_prompt("nude child")
    assert str(caught.value) == BLOCKED_MESSAGE
    assert caught.value.code == "prompt"


def test_adult_russian_shower_prompt_stays_open_when_the_guard_says_sexual():
    detail = explain(SHOWER)
    assert detail.age is False and detail.minor is False and detail.sexual is True
    world = World()
    world.safety = "unsafe"
    world.categories = ("Sexual Content or Sexual Acts",)
    assert refusal_stage(SHOWER, world.safety, world.categories) == "allow"
    world.gate().review_prompt(SHOWER)
    world.safety = "controversial"
    assert refusal_stage(SHOWER, world.safety, world.categories) == "allow"
    world.gate().review_prompt(SHOWER)
    world.categories = ("Non-consensual sexual acts",)
    assert refusal_stage("a red boat", world.safety, world.categories) == "guard-restricted"
    with pytest.raises(GenerationBlocked):
        world.gate().review_prompt("a red boat")
    assert refusal_stage("loli", "safe", ()) == "lexicon-minor-sexual"
    assert refusal_stage("nude child", "safe", ()) == "lexicon-minor"


def test_denylist_is_not_cleared_by_a_safe_verdict(caplog):
    world = World()
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(GenerationBlocked):
            world.gate().review_prompt("nude child secret-phrase")
    assert "secret-phrase" not in caplog.text
    # The product keeps no record of a content decision.
    assert not [record for record in caplog.records if record.name.startswith("voxprint")]


def test_prompt_screening_ignores_text_zero_shot_scores():
    world = World()
    for label in world.text_scores:
        world.text_scores[label] = 1.0
    world.gate().review_prompt("a red boat")
    assert "text" not in world.calls


def test_explicit_minor_sexual_terms_block_without_a_second_word():
    world = World()
    with pytest.raises(GenerationBlocked):
        world.gate().review_prompt("loli")
    with pytest.raises(GenerationBlocked):
        world.gate().review_prompt("лоликон")
    world.gate().review_prompt("a child flying a kite in a park")


def test_prompt_gate_fails_closed_on_the_guard():
    world = World()
    world.error = "guard"
    with pytest.raises(SafetyUnavailable) as caught:
        world.gate().review_prompt("a red boat")
    assert caught.value.code == "classifier_error"
    world.error = "zero-text"
    world.gate().review_prompt("a red boat")


def _adult_distribution(bucket: str) -> dict[str, float]:
    labels = ("0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+")
    return {label: (1.0 if label == bucket else 0.0) for label in labels}


def _image_scores(**overrides: float) -> dict[str, float]:
    scores = {label: 0.0 for label in IMAGE_LABELS}
    scores.update(overrides)
    return scores


def test_shower_prompt_allows_and_a_20_29_face_does_not_block_an_adult_picture():
    assert refusal_stage(SHOWER, "unsafe", ("Sexual Content or Sexual Acts",)) == "allow"
    assert adult_age_support(SHOWER) == "strong"
    assert adult_age_support("a 22-year-old woman") == "strong"
    assert adult_age_support("an 18-year-old woman") == "adult"
    assert adult_age_support("a 17-year-old") == "none"
    assert adult_age_support("a 25-year-old and a 10-year-old") == "none"
    assert adult_age_support("a red boat") == "none"
    assert age_bands(_adult_distribution("20-29")) == (0.0, 0.0)
    assert age_bands(_adult_distribution("30-39")) == (0.0, 0.0)
    assert age_bands(_adult_distribution("10-19")) == pytest.approx((0.6, 0.2))
    assert age_bands(_adult_distribution("3-9")) == (1.0, 0.0)
    assert apparent_age_from_bucket("20-29") == 20.0
    scores = _image_scores(**{IMAGE_ADULT_SEXUAL: 0.9, IMAGE_MINOR: 0.9})
    # Highest bands measured on public-domain faces aged 20-25 stay open.
    assert image_block_reason(0.95, [(0.67, 0.09)], scores) == "allow"
    assert image_block_reason(0.95, [(0.50, 0.14)], scores) == "allow"
    assert image_block_reason(0.95, [(0.69, 0.0)], scores) == "allow"
    assert image_block_reason(0.95, [(0.70, 0.0)], scores, support="strong") == "clear-child"
    assert image_block_reason(0.95, [(0.0, 0.16)], scores) == "adolescent"
    assert image_block_reason(0.95, [(0.0, 0.16)], scores, support="strong") == "allow"
    assert image_block_reason(0.95, [(0.0, 0.17)], scores, support="adult") == "allow"
    assert image_block_reason(0.95, [(0.0, 0.18)], scores, support="adult") == "adolescent"
    assert image_block_reason(0.95, [(0.0, 0.20)], scores, support="strong") == "allow"
    assert image_block_reason(0.1, [], _image_scores(**{IMAGE_MINOR_SEXUAL: 0.35})) == "minor-sexual"
    with pytest.raises(SafetyUnavailable):
        image_block_reason(0.0, [], {IMAGE_MINOR_SEXUAL: 0.0, IMAGE_ADULT_SEXUAL: 0.0})


def test_image_probability_boundary_and_adult_sexual_image():
    world = World()
    world.nsfw = 0.90
    world.gate().review_image(rgb())
    world.faces = [Face(0, 0, 2, 2)]
    world.ages = [(0.0, 0.0)]
    released = world.gate().review_image(rgb(), "a 22-year-old woman")
    assert type(released) is SafeImage
    world.ages = [(0.69, 0.15)]
    world.gate().review_image(rgb())
    world.ages = [(0.70, 0.0)]
    raw = rgb(color=(9, 8, 7))
    with pytest.raises(GenerationBlocked):
        world.gate().review_image(raw, "a 25-year-old woman")
    assert raw.to_bytes() == b""
    assert raw.width == 0


def test_image_label_and_failed_age_are_fail_closed():
    world = World()
    world.image_scores[IMAGE_MINOR_SEXUAL] = 0.35
    # Two-stage gate: a clothed frame (NSFW under the line) stops after the NSFW pass.
    assert type(world.gate().review_image(rgb())) is SafeImage
    world.nsfw = 0.9
    with pytest.raises(GenerationBlocked):
        world.gate().review_image(rgb())
    world.image_scores[IMAGE_MINOR_SEXUAL] = 0.0
    world.faces = [Face(0, 0, 1, 1)]
    world.ages = []
    with pytest.raises(SafetyUnavailable) as caught:
        world.gate().review_image(rgb())
    assert caught.value.code == "classifier_error"
    world.ages = [(float("nan"), 0.0)]
    with pytest.raises(SafetyUnavailable):
        world.gate().review_image(rgb())
    world.ages = [(0.0, 0.0)]
    world.nsfw = True  # type: ignore[assignment]
    # An unusable NSFW score counts as sexual, so a child face then blocks.
    world.ages = [(1.0, 0.0)]
    with pytest.raises(GenerationBlocked):
        world.gate().review_image(rgb())
    world.nsfw = 0.9
    world.ages = [(30.0, 0.0)]
    with pytest.raises(SafetyUnavailable):
        world.gate().review_image(rgb())
    world.ages = [(0.0, 0.0)]
    world.image_scores[IMAGE_MINOR] = 0.90
    world.image_scores[IMAGE_ADULT_SEXUAL] = 0.90
    released = world.gate().review_image(rgb())
    assert type(released) is SafeImage


def test_every_input_is_screened_and_sources_are_wiped():
    world = World()
    first = rgb(color=(1, 2, 3))
    second = rgb(color=(4, 5, 6))
    third = rgb(color=(7, 8, 9))
    screened = world.gate().release_request(
        "a red boat",
        "restyle",
        [
            PendingInput(ImageRole.INIT, first),
            PendingInput(ImageRole.REFERENCE, second),
            PendingInput(ImageRole.STYLE, third),
        ],
    )
    assert type(screened) is ScreenedInput
    assert screened.roles() == ("init", "reference", "style")
    assert world.calls.count("nsfw") == 3  # two-stage gate: clothed inputs get the NSFW pass only
    assert first.to_bytes() == b"" and second.to_bytes() == b"" and third.to_bytes() == b""
    assert screened.frames[0].to_bytes() == bytes([1, 2, 3]) * 4


def test_a_blocked_input_wipes_every_staged_image():
    world = World()
    world.nsfw = 0.9
    world.faces = [Face(0, 0, 1, 1)]
    world.ages = [(1.0, 0.0)]
    images = [rgb(), rgb()]
    with pytest.raises(GenerationBlocked) as caught:
        world.gate().release_request(
            "a red boat",
            "inpaint",
            [PendingInput(ImageRole.INIT, images[0]), PendingInput(ImageRole.MASK, images[1])],
        )
    assert caught.value.code == "input"
    assert images[0].to_bytes() == b"" and images[1].to_bytes() == b""


def test_issued_types_cannot_be_built_outside_the_gate():
    with pytest.raises(RuntimeError):
        ScreenedInput(_INPUT_TOKEN, "a red boat", "text", ())
    with pytest.raises(RuntimeError):
        SafeImage(_OUTPUT_TOKEN, 1, 1, "RGB", b"rgb")


def test_environment_cannot_turn_the_checks_off(monkeypatch):
    monkeypatch.setenv("CANVAS_DISABLE_SAFETY", "1")
    monkeypatch.setenv("APPARENT_AGE_UNCERTAIN_UNDER", "0")
    monkeypatch.setenv("CLEAR_CHILD_AT", "1")
    monkeypatch.setenv("ADOLESCENT_16_17_AT", "1")
    world = World()
    with pytest.raises(GenerationBlocked):
        world.gate().review_prompt("nude child")
    world.nsfw = 0.9
    world.faces = [Face(0, 0, 1, 1)]
    world.ages = [(1.0, 0.0)]
    with pytest.raises(GenerationBlocked):
        world.gate().review_image(rgb(), "a 25-year-old woman")
