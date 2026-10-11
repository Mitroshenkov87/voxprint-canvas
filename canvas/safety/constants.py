"""Compile-time safety thresholds.

These names are constants. The gate does not read environment variables, suite.json,
CLI flags or plugins to change them. Tests pin the values.
"""
from __future__ import annotations

# Pinned historical cutoff. The image gate does not compare an apparent age with this
# value. A face in the 20-29 bucket used to be read as 20 and then stopped.
APPARENT_AGE_UNCERTAIN_UNDER = 25

# P(age < 18) at or above this, together with sexual content, blocks the image.
# The 20-29 bucket contributes nothing to that probability.
MINOR_FACE_PROBABILITY_AT = 0.50

# Falconsai/nsfw_image_detection score at or above this counts as sexual content.
SEXUAL_SCORE_BLOCK_AT = 0.50

# SigLIP2 zero-shot cosine score at or above this counts as a match for that label.
ZERO_SHOT_BLOCK_AT = 0.35

# A classifier that does not return in time is a failure, and a failure refuses generation.
CLASSIFIER_TIMEOUT_S = 20.0
IMAGE_TIMEOUT_S = 30.0

# Shown for a blocked prompt or a blocked image. No reason, no prompt echo.
BLOCKED_MESSAGE = "This request cannot be completed."

# Shown when a classifier is missing, mismatched, or fails. Same silence about why.
UNAVAILABLE_MESSAGE = "Image generation is unavailable."

# Prompt-side SigLIP2 labels. The wording is part of the gate, not a user setting.
PROMPT_MINOR_SEXUAL = "sexual or suggestive content involving a child or minor"
PROMPT_SEXUAL = "sexual or suggestive content"
PROMPT_MINOR = "a child or minor"

# Image-side SigLIP2 labels.
IMAGE_MINOR_SEXUAL = "a sexual or nude image of a child or minor"
IMAGE_ADULT_SEXUAL = "a sexual or nude image of an adult"
IMAGE_MINOR = "a child or minor"

PROMPT_LABELS = (PROMPT_MINOR_SEXUAL, PROMPT_SEXUAL, PROMPT_MINOR)
IMAGE_LABELS = (IMAGE_MINOR_SEXUAL, IMAGE_ADULT_SEXUAL, IMAGE_MINOR)

# Qwen3Guard category that counts as a sexual signal. Controversial is treated as unsafe (strict).
GUARD_SEXUAL_CATEGORY = "sexual content or sexual acts"

# Repos and import names that must never be wired in. NudeNet is AGPL.
FORBIDDEN_MODEL_MARKERS = ("nudenet",)
