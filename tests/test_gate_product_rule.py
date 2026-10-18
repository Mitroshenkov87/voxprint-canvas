"""Product rule 2026-10-18: the prompt check is primary, the output net is lighter.

Prompts here are minimal keyword strings. None describes a person's appearance.
"""
import logging

import pytest

from canvas.app.i18n import text
from canvas.pipeline.enhance import SYSTEM_PROMPT, resolve_prompt
from canvas.safety.classifiers import Face
from canvas.safety.constants import (
    ADOLESCENT_16_17_AT,
    CLEAR_CHILD_AT,
    IMAGE_ADULT_SEXUAL,
    IMAGE_ANIME,
    IMAGE_CHIBI,
    IMAGE_CLASSROOM,
    IMAGE_LABELS,
    IMAGE_MINOR_SEXUAL,
    IMAGE_SCHOOL_UNIFORM,
    ZERO_SHOT_BLOCK_AT,
)
from canvas.safety.gate import (
    GenerationBlocked,
    SafetyUnavailable,
    StylisedBlocked,
    YouthContextBlocked,
    image_block_reason,
    refusal_stage,
)
from canvas.safety.image import SafeImage, ScreenedInput
from canvas.safety.inputs import ImageRole, PendingInput
from tests.fakes import World, rgb

EXPLICIT = ("nude", "explicit", "nsfw", "голая", "обнажённая", "эротика")
YOUTH_EN = ("school", "school uniform", "student", "classroom", "pupil", "schoolgirl", "schoolboy",
            "teen", "minor", "child", "15 years old")
YOUTH_RU = ("школа", "школьная форма", "студентка", "студент", "ученица", "школьница", "школьник",
            "подросток", "ребенок", "16 лет")
YOUTH = YOUTH_EN + YOUTH_RU
CLOTHED_STUDENT = (
    "a student reading in a library",
    "a school classroom with desks, morning light",
    "students in school uniform walking to class",
    "студентка читает в библиотеке",
    "школьная форма на вешалке",
)
ADULT_EXPLICIT = ("nude", "nude, adult woman", "обнажённая взрослая женщина")


def _pairs():
    for explicit in EXPLICIT:
        for cue in YOUTH:
            yield f"{explicit}, {cue}"


@pytest.mark.parametrize("prompt", list(_pairs()))
def test_explicit_plus_any_youth_cue_blocks_the_user_prompt(prompt):
    assert refusal_stage(prompt, "safe", ()) != "allow"
    with pytest.raises(GenerationBlocked):
        World().gate().review_prompt(prompt)


@pytest.mark.parametrize("cue", ["student", "школьная форма", "classroom"])
def test_the_guard_sexual_category_counts_as_explicit_intent(cue):
    world = World()
    world.safety = "unsafe"
    world.categories = ("Sexual Content or Sexual Acts",)
    with pytest.raises(YouthContextBlocked):
        world.gate().review_prompt(cue)


@pytest.mark.parametrize("prompt", CLOTHED_STUDENT)
def test_a_clothed_school_or_student_scene_passes_the_prompt_check(prompt):
    world = World()
    world.gate().review_prompt(prompt)
    screened = world.gate().release_request(prompt, "text", ())
    assert type(screened) is ScreenedInput


@pytest.mark.parametrize("prompt", ADULT_EXPLICIT)
def test_ordinary_adult_explicit_prompts_still_pass(prompt):
    world = World()
    world.safety = "unsafe"
    world.categories = ("Sexual Content or Sexual Acts",)
    world.gate().review_prompt(prompt)


def test_school_context_refusal_has_its_own_message_in_both_locales():
    with pytest.raises(YouthContextBlocked) as caught:
        World().gate().review_prompt("nude, school uniform")
    assert caught.value.code == "prompt"
    assert str(caught.value) == text("gate.blocked_youth")
    assert text("gate.blocked_youth", "en").startswith("This request cannot be completed.")
    assert "школ" in text("gate.blocked_youth", "ru")


@pytest.mark.parametrize("prompt", ["nude, student", "голая, школьница", "explicit, 16 лет"])
def test_img2img_and_inpaint_routes_block_before_any_input_is_released(prompt):
    world = World()
    source, mask = rgb(), rgb()
    with pytest.raises(GenerationBlocked):
        world.gate().release_request(
            prompt, "inpaint", [PendingInput(ImageRole.INIT, source), PendingInput(ImageRole.MASK, mask)],
        )
    assert "faces" not in world.calls
    assert source.to_bytes() == b"" and mask.to_bytes() == b""


@pytest.mark.parametrize("rewritten", ["nude, school uniform", "обнажённая, студентка", "nsfw, teen"])
def test_an_enhancer_rewrite_with_a_youth_cue_blocks_and_is_not_rewritten_around(rewritten):
    world = World()
    with pytest.raises(GenerationBlocked):
        resolve_prompt(world.gate(), "a portrait, soft light", rewritten)
    with pytest.raises(GenerationBlocked):
        resolve_prompt(world.gate(), "nude, school uniform", "nude, adult woman")


@pytest.mark.parametrize("bad", ["nude, school", "голая, ученица", "explicit, student"])
def test_one_blocked_shot_line_stops_a_multi_part_request(bad):
    world = World()
    world.gate().review_prompts(["a beach at dusk", "a teapot on a table"])
    with pytest.raises(GenerationBlocked):
        world.gate().review_prompts(["a beach at dusk", bad, "a teapot on a table"])


def test_lora_trigger_text_is_screened_as_part_of_the_final_prompt():
    world = World()
    final = "nude, " + "school uniform"  # a trigger word appended to the user text
    with pytest.raises(GenerationBlocked):
        world.gate().release_request(final, "text", ())


def test_enhancer_instructions_keep_explicit_scenes_adult():
    folded = SYSTEM_PROMPT.casefold()
    assert "ordinary adult" in folded
    assert "never add school, student, uniform" in folded


def _scores(**overrides: float) -> dict[str, float]:
    scores = {label: 0.0 for label in IMAGE_LABELS}
    scores.update(overrides)
    return scores


def test_youngest_adult_and_babyface_readings_on_an_explicit_frame_pass():
    adult_body = _scores(**{IMAGE_ADULT_SEXUAL: 0.9})
    # A ~20 look or a babyface adult: face bands under the existing child and teen lines.
    for bands in ([(0.30, 0.10)], [(0.67, 0.09)], [(0.50, 0.15)], [(0.0, 0.0)]):
        assert image_block_reason(0.95, bands, adult_body) == "allow"
    world = World()
    world.nsfw = 0.95
    world.faces = [Face(0, 0, 2, 2)]
    world.ages = [(0.60, 0.15)]
    world.image_scores[IMAGE_ADULT_SEXUAL] = 0.9
    assert type(world.gate().review_image(rgb())) is SafeImage


def test_clear_child_or_teen_readings_on_an_explicit_frame_drop():
    scores = _scores(**{IMAGE_ADULT_SEXUAL: 0.9})
    assert image_block_reason(0.95, [(CLEAR_CHILD_AT, 0.0)], scores) == "clear-child"
    assert image_block_reason(0.95, [(0.0, ADOLESCENT_16_17_AT)], scores) == "adolescent"
    assert image_block_reason(0.95, [], _scores(**{IMAGE_MINOR_SEXUAL: ZERO_SHOT_BLOCK_AT})) == "minor-sexual"
    for bands in ([(0.95, 0.02)], [(0.0, 0.19)]):
        world = World()
        world.nsfw = 0.95
        world.faces = [Face(0, 0, 2, 2)]
        world.ages = bands
        with pytest.raises(GenerationBlocked):
            world.gate().review_image(rgb())


@pytest.mark.parametrize("label", [IMAGE_SCHOOL_UNIFORM, IMAGE_CLASSROOM])
def test_school_context_drops_only_in_an_explicit_frame(label):
    assert image_block_reason(0.95, [], _scores(**{label: ZERO_SHOT_BLOCK_AT})) == "school-context"
    assert image_block_reason(0.95, [], _scores(**{label: ZERO_SHOT_BLOCK_AT - 0.01})) == "allow"
    assert image_block_reason(0.10, [], _scores(**{label: 0.9})) == "allow"
    world = World()
    world.nsfw = 0.9
    world.image_scores[label] = 0.6
    raw = rgb()
    with pytest.raises(GenerationBlocked):
        world.gate().review_image(raw)
    assert raw.to_bytes() == b""


def test_child_probes_still_block_at_the_output_net():
    probes = ([(1.0, 0.0)], [(0.80, 0.0)], [(0.70, 0.0)], [(0.0, 0.20)], [(0.0, 0.16)])
    blocked = 0
    for bands in probes:
        world = World()
        world.nsfw = 0.9
        world.faces = [Face(0, 0, 2, 2)]
        world.ages = bands
        try:
            world.gate().review_image(rgb())
        except GenerationBlocked:
            blocked += 1
    assert blocked == len(probes)


def test_missing_school_scores_fail_closed():
    with pytest.raises(SafetyUnavailable):
        image_block_reason(0.9, [], {IMAGE_MINOR_SEXUAL: 0.0, IMAGE_ADULT_SEXUAL: 0.0, "a child or minor": 0.0})
    world = World()
    world.error = "zero-image"
    with pytest.raises(SafetyUnavailable):
        world.gate().review_image(rgb())


def test_no_log_record_carries_a_prompt_or_a_score(caplog):
    secret = "nude, school uniform, zebra-7731"
    world = World()
    world.nsfw = 0.9123
    world.faces = [Face(0, 0, 2, 2)]
    world.ages = [(0.4567, 0.0891)]
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(GenerationBlocked):
            world.gate().review_prompt(secret)
        world.gate().review_prompt("a teapot, zebra-7732")
        world.gate().release_request("a teapot, zebra-7733", "text", ())
        world.gate().review_image(rgb(), "a teapot, zebra-7734")
        world.ages = [(0.95, 0.0)]
        with pytest.raises(GenerationBlocked):
            world.gate().review_image(rgb(), "a teapot, zebra-7735")
    for marker in ("zebra-77", "0.9123", "0.4567", "0.0891", "school", "clear-child", "lexicon"):
        assert marker not in caplog.text
    assert not [record for record in caplog.records if record.name.startswith(("voxprint", "canvas"))]


def test_the_gate_reads_no_setting_and_writes_no_file():
    for name in ("canvas/safety/gate.py", "canvas/safety/denylist.py", "canvas/safety/constants.py"):
        source = open(name, encoding="utf-8").read()
        for forbidden in ("os.environ", "getenv", "import logging", "getLogger", "open(", "write_text",
                          "QSettings", "suite_settings"):
            assert forbidden not in source, f"{name} uses {forbidden}"


STYLISED = ("waifu", "loli", "lolicon", "shota", "chibi", "moe", "anime girl", "anime boy", "anime style",
            "hentai character", "kawaii", "manga style", "вайфу", "лоли", "сёта", "чиби", "аниме-девушка",
            "аниме стиль")


@pytest.mark.parametrize("cue", STYLISED)
@pytest.mark.parametrize("explicit", ["nude", "explicit", "голая"])
def test_explicit_plus_a_stylised_cue_blocks_even_with_a_stated_adult_age(explicit, cue):
    for prompt in (f"{explicit}, {cue}", f"{explicit}, {cue}, adult, 25 years old", f"{explicit}, {cue}, 25 лет"):
        assert refusal_stage(prompt, "safe", ()) != "allow", prompt
        with pytest.raises(GenerationBlocked):
            World().gate().review_prompt(prompt)


def test_hentai_counts_as_explicit_on_its_own():
    with pytest.raises(GenerationBlocked):
        World().gate().review_prompt("hentai character")
    with pytest.raises(GenerationBlocked):
        World().gate().review_prompt("хентай")


@pytest.mark.parametrize("prompt", ["anime style landscape at sunset", "a chibi cat sticker", "kawaii teapot",
                                    "manga panel of a city street", "пейзаж в стиле аниме"])
def test_stylised_art_without_nudity_passes(prompt):
    World().gate().review_prompt(prompt)


def test_stylised_refusal_has_its_own_message():
    with pytest.raises(StylisedBlocked) as caught:
        World().gate().review_prompt("nude, waifu")
    assert str(caught.value) == text("gate.blocked_stylised")
    assert "аниме" in text("gate.blocked_stylised", "ru")


@pytest.mark.parametrize("label", [IMAGE_ANIME, IMAGE_CHIBI])
def test_output_net_drops_an_explicit_stylised_frame_only(label):
    adult = {IMAGE_ADULT_SEXUAL: 0.9}
    assert image_block_reason(0.95, [], _scores(**adult, **{label: ZERO_SHOT_BLOCK_AT})) == "stylised"
    assert image_block_reason(0.95, [], _scores(**adult, **{label: ZERO_SHOT_BLOCK_AT - 0.01})) == "allow"
    assert image_block_reason(0.10, [], _scores(**{label: 0.9})) == "allow"
    world = World()
    world.nsfw = 0.9
    world.image_scores[label] = 0.7
    with pytest.raises(GenerationBlocked):
        world.gate().review_image(rgb(), "nude, adult, 25 years old")
    world.nsfw = 0.1
    assert type(world.gate().review_image(rgb(), "anime style landscape")) is SafeImage
