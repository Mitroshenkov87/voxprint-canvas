"""Fail-closed checks, issued images, and wiped buffers."""
import logging
import time

import pytest

from canvas.safety.classifiers import Face
from canvas.safety.constants import (
    APPARENT_AGE_UNCERTAIN_UNDER,
    BLOCKED_MESSAGE,
    CLASSIFIER_TIMEOUT_S,
    IMAGE_MINOR,
    IMAGE_MINOR_SEXUAL,
    IMAGE_TIMEOUT_S,
    SEXUAL_SCORE_BLOCK_AT,
    UNAVAILABLE_MESSAGE,
    ZERO_SHOT_BLOCK_AT,
)
from canvas.safety.denylist import explain
from canvas.safety.gate import GenerationBlocked, SafetyUnavailable, _wait, refusal_stage
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
    with caplog.at_level(logging.INFO):
        with pytest.raises(GenerationBlocked):
            world.gate().review_prompt("nude child secret-phrase")
    assert "secret-phrase" not in caplog.text
    assert "content check refused a prompt" in caplog.text


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


def test_image_age_boundary_and_adult_sexual_image():
    world = World()
    world.nsfw = 0.90
    world.gate().review_image(rgb())
    world.faces = [Face(0, 0, 2, 2)]
    world.ages = [25.0]
    released = world.gate().review_image(rgb())
    assert type(released) is SafeImage
    world.ages = [24.9]
    raw = rgb(color=(9, 8, 7))
    with pytest.raises(GenerationBlocked):
        world.gate().review_image(raw)
    assert raw.to_bytes() == b""
    assert raw.width == 0


def test_image_label_and_failed_age_are_fail_closed():
    world = World()
    world.image_scores[IMAGE_MINOR_SEXUAL] = 0.35
    with pytest.raises(GenerationBlocked):
        world.gate().review_image(rgb())
    world.image_scores[IMAGE_MINOR_SEXUAL] = 0.0
    world.faces = [Face(0, 0, 1, 1)]
    world.ages = []
    with pytest.raises(SafetyUnavailable) as caught:
        world.gate().review_image(rgb())
    assert caught.value.code == "classifier_error"
    world.ages = [float("nan")]
    with pytest.raises(SafetyUnavailable):
        world.gate().review_image(rgb())
    world.ages = [30.0]
    world.nsfw = True  # type: ignore[assignment]
    with pytest.raises(SafetyUnavailable):
        world.gate().review_image(rgb())
    world.nsfw = 0.9
    world.image_scores[IMAGE_MINOR] = 0.35
    with pytest.raises(GenerationBlocked):
        world.gate().review_image(rgb())


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
    assert world.calls.count("faces") == 3
    assert first.to_bytes() == b"" and second.to_bytes() == b"" and third.to_bytes() == b""
    assert screened.frames[0].to_bytes() == bytes([1, 2, 3]) * 4


def test_a_blocked_input_wipes_every_staged_image():
    world = World()
    world.nsfw = 0.9
    world.faces = [Face(0, 0, 1, 1)]
    world.ages = [10.0]
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
    world = World()
    with pytest.raises(GenerationBlocked):
        world.gate().review_prompt("nude child")
