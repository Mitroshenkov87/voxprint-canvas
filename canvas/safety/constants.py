"""Constants."""
from __future__ import annotations

from canvas.app.i18n import text

APPARENT_AGE_UNCERTAIN_UNDER = 25

CLEAR_CHILD_AT = 0.70

ADOLESCENT_16_17_AT = 0.16

ADOLESCENT_16_17_AT_AGE_18 = 0.18

ESTIMATE_CONFIRM_BELOW = 18.0

ESTIMATE_DROP_BELOW = 13.0

SEXUAL_SCORE_BLOCK_AT = 0.50

ZERO_SHOT_BLOCK_AT = 0.35

CLASSIFIER_TIMEOUT_S = 20.0
IMAGE_TIMEOUT_S = 30.0

BLOCKED_MESSAGE = text("gate.blocked", "en")

UNAVAILABLE_MESSAGE = text("gate.unavailable", "en")

PROMPT_MINOR_SEXUAL = "sexual or suggestive content involving a child or minor"
PROMPT_SEXUAL = "sexual or suggestive content"
PROMPT_MINOR = "a child or minor"

IMAGE_MINOR_SEXUAL = "a sexual or nude image of a child or minor"
IMAGE_ADULT_SEXUAL = "a sexual or nude image of an adult"
IMAGE_MINOR = "a child or minor"
IMAGE_SCHOOL_UNIFORM = "a person wearing a school uniform"
IMAGE_CLASSROOM = "a school classroom"
IMAGE_ANIME = "an anime, manga or cartoon style character"
IMAGE_CHIBI = "a chibi character with a large head and a small childlike body"

PROMPT_LABELS = (PROMPT_MINOR_SEXUAL, PROMPT_SEXUAL, PROMPT_MINOR)
IMAGE_LABELS = (
    IMAGE_MINOR_SEXUAL, IMAGE_ADULT_SEXUAL, IMAGE_MINOR, IMAGE_SCHOOL_UNIFORM, IMAGE_CLASSROOM, IMAGE_ANIME, IMAGE_CHIBI,
)

GUARD_SEXUAL_CATEGORY = "sexual content or sexual acts"

FORBIDDEN_MODEL_MARKERS = ("nudenet",)
