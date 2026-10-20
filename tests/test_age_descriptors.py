"""Enhancer and scenario writer keep the stated age and add no aging or youthful descriptors."""
from __future__ import annotations

import pytest

from canvas.app.i18n import text
from canvas.pipeline import adult_direction, photoshoot
from canvas.pipeline.enhance import (
    AGING_DESCRIPTOR_MIN_AGE,
    AGING_TERMS,
    SYSTEM_PROMPT,
    added_aging_terms,
    aging_terms,
    fallback_message,
    resolve_prompt,
    stated_ages,
)
from canvas.pipeline.photoshoot import Draft, Shot, accept_shot
from canvas.ui.styles import LOOKS

YOUTHFUL_TERMS = ("young-looking", "youthful", "petite", "smooth skin", "round face", "baby face")


class _Gate:
    def review_prompt(self, prompt: str) -> None:
        return None


def test_rewrite_that_adds_aging_words_falls_back() -> None:
    original = "invented adult woman, 32 years old, red coat"
    result = resolve_prompt(_Gate(), original, "a mature woman, 32 years old, fine wrinkles, red coat, 85 mm")
    assert result.prompt == original
    assert result.fallback == "age_descriptor"
    assert fallback_message(result)


def test_users_own_aging_words_are_kept() -> None:
    original = "invented adult woman, 32 years old, mature adult face with fine smile lines"
    rewritten = "invented adult woman, 32 years old, mature adult face with fine smile lines, soft window light, 85 mm"
    result = resolve_prompt(_Gate(), original, rewritten)
    assert result.prompt == rewritten
    assert result.fallback == ""


@pytest.mark.parametrize("age", (18, 25, 30, 32))
def test_stated_age_passes_through_as_is(age: int) -> None:
    original = f"adult woman, {age} years old, studio portrait"
    rewritten = f"adult woman, {age} years old, studio portrait, softbox key light, 85 mm lens"
    result = resolve_prompt(_Gate(), original, rewritten)
    assert result.prompt == rewritten
    assert f"{age} years old" in result.prompt
    assert stated_ages(result.prompt) == (age,)


def test_terms_match_whole_words_only() -> None:
    assert aging_terms("Crow's feet and SMILE LINES") == ("crow's feet", "smile lines")
    assert aging_terms("teenaged") == ()
    assert added_aging_terms("mature face", "mature face, wrinkles") == ("wrinkles",)


def test_catalog_has_the_new_key_in_both_languages() -> None:
    assert text("enhance.age_descriptor", "en")
    assert text("enhance.age_descriptor", "ru")


def test_system_prompts_carry_the_rule() -> None:
    for prompt in (SYSTEM_PROMPT, photoshoot.SYSTEM_PROMPT):
        folded = prompt.casefold()
        assert "smile lines" in folded and "mature" in folded and "young-looking" in folded


def _rule_free(prompt: str) -> str:
    return "\n".join(
        line for line in prompt.splitlines()
        if not any(rule in line.casefold() for rule in ("do not add", "never add", "or older"))
    )


def test_prompt_texts_add_no_aging_or_youthful_words() -> None:
    data = [
        *adult_direction.prompt_texts()[3:],
        *(look.suffix for look in LOOKS),
        *photoshoot.PLAN_FRAMINGS, *photoshoot.PLAN_LIGHTS, *photoshoot.PLAN_POSES,
        *photoshoot.PLAN_CAMERAS, *photoshoot.PLAN_PLACES,
        _rule_free(SYSTEM_PROMPT), _rule_free(photoshoot.SYSTEM_PROMPT),
        *(_rule_free(item) for item in adult_direction.prompt_texts()[:3]),
    ]
    for item in data:
        assert aging_terms(item) == (), item
        assert not any(term in item.casefold() for term in YOUTHFUL_TERMS), item
    assert "older" in AGING_TERMS


def test_scenario_shot_that_adds_aging_words_is_rejected() -> None:
    idea = "invented adult woman, 32 years old, red coat"
    bad = Draft("waist-up shot, middle-aged look, soft light", "seated", "85 mm lens, eye level", "")
    good = Draft("waist-up shot, soft light", "seated", "85 mm lens, eye level", "")
    assert accept_shot(idea, idea, "same", "loft", (), bad) is None
    shot = accept_shot(idea, idea, "same", "loft", (), good)
    assert isinstance(shot, Shot)
    line = photoshoot.assemble_line(idea, shot, include_camera=True)
    assert "32 years old" in line and stated_ages(line) == (32,)


def test_threshold_is_45() -> None:
    assert AGING_DESCRIPTOR_MIN_AGE == 45


@pytest.mark.parametrize(("age", "allowed"), ((44, False), (45, True)))
def test_rewrite_aging_words_44_vs_45(age: int, allowed: bool) -> None:
    original = f"adult woman, {age} years old, red coat"
    rewritten = f"adult woman, {age} years old, fine smile lines, red coat, 85 mm"
    result = resolve_prompt(_Gate(), original, rewritten)
    assert (result.fallback == "") is allowed
    assert result.prompt == (rewritten if allowed else original)


def test_no_stated_age_never_allows_aging_words() -> None:
    result = resolve_prompt(_Gate(), "adult woman, red coat", "adult woman, red coat, wrinkles")
    assert result.fallback == "age_descriptor"


@pytest.mark.parametrize(("age", "allowed"), ((44, False), (45, True)))
def test_scenario_aging_words_44_vs_45(age: int, allowed: bool) -> None:
    idea = f"invented adult woman, {age} years old, red coat"
    draft = Draft("waist-up shot, weathered face, soft light", "seated", "85 mm lens, eye level", "")
    assert (accept_shot(idea, idea, "same", "loft", (), draft) is not None) is allowed
