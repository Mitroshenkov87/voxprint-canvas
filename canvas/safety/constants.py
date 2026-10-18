"""Compile-time safety thresholds.

These names are constants. The gate does not read environment variables, suite.json,
CLI flags or plugins to change them. Tests pin the values.
"""
from __future__ import annotations

from canvas.app.i18n import text

# Pinned historical cutoff. The image gate does not compare an apparent age with this
# value. A face in the 20-29 bucket used to be read as 20 and then stopped.
APPARENT_AGE_UNCERTAIN_UNDER = 25

# P(age <= 15) at or above this, together with sexual content, blocks the image.
# Buckets 0-2 and 3-9 count in full. Six of the ten years in 10-19 count, so a
# pure 10-19 distribution is 0.60 and stays under this line. A stated age cannot
# move it. Face crops of clothed portraits aged 20-25 stayed at or below 0.67.
CLEAR_CHILD_AT = 0.70

# P(age in 16 or 17). Only two of the ten years in the 10-19 bucket count, so
# this line is that bucket at 0.80 or more. It is the higher-confidence band.
# A stated age of 20 or more can support a pass here. It cannot move CLEAR_CHILD_AT.
ADOLESCENT_16_17_AT = 0.16

# A stated age of 18 or 19 raises the 16-17 line to a 10-19 bucket of 0.90.
# A stated age of 20 or more removes the 16-17 line.
ADOLESCENT_16_17_AT_AGE_18 = 0.18

# Falconsai/nsfw_image_detection score at or above this counts as sexual content.
SEXUAL_SCORE_BLOCK_AT = 0.50

# SigLIP2 zero-shot cosine score at or above this counts as a match for that label.
ZERO_SHOT_BLOCK_AT = 0.35

# A classifier that does not return in time is a failure, and a failure refuses generation.
CLASSIFIER_TIMEOUT_S = 20.0
IMAGE_TIMEOUT_S = 30.0

# Shown for a blocked prompt or a blocked image. No reason, no prompt echo.
BLOCKED_MESSAGE = text("gate.blocked", "en")

# Shown when a classifier is missing, mismatched, or fails. Same silence about why.
UNAVAILABLE_MESSAGE = text("gate.unavailable", "en")

# Prompt-side SigLIP2 labels. The wording is part of the gate, not a user setting.
PROMPT_MINOR_SEXUAL = "sexual or suggestive content involving a child or minor"
PROMPT_SEXUAL = "sexual or suggestive content"
PROMPT_MINOR = "a child or minor"

# Image-side SigLIP2 labels.
IMAGE_MINOR_SEXUAL = "a sexual or nude image of a child or minor"
IMAGE_ADULT_SEXUAL = "a sexual or nude image of an adult"
IMAGE_MINOR = "a child or minor"
# School context. Either label at ZERO_SHOT_BLOCK_AT in a sexual or nude frame drops it.
IMAGE_SCHOOL_UNIFORM = "a person wearing a school uniform"
IMAGE_CLASSROOM = "a school classroom"
# Stylised characters. Either label at ZERO_SHOT_BLOCK_AT in a sexual or nude frame drops it.
IMAGE_ANIME = "an anime, manga or cartoon style character"
IMAGE_CHIBI = "a chibi character with a large head and a small childlike body"

PROMPT_LABELS = (PROMPT_MINOR_SEXUAL, PROMPT_SEXUAL, PROMPT_MINOR)
IMAGE_LABELS = (
    IMAGE_MINOR_SEXUAL, IMAGE_ADULT_SEXUAL, IMAGE_MINOR, IMAGE_SCHOOL_UNIFORM, IMAGE_CLASSROOM, IMAGE_ANIME, IMAGE_CHIBI,
)

# Qwen3Guard category that counts as a sexual signal. Controversial is treated as unsafe (strict).
GUARD_SEXUAL_CATEGORY = "sexual content or sexual acts"

# Repos and import names that must never be wired in. NudeNet is AGPL.
FORBIDDEN_MODEL_MARKERS = ("nudenet",)
