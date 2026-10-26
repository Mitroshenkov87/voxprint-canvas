"""One frame's picture prompt.

The diffusion prompt is the appearance and that frame's own shot. Series
wording (a grid, a sheet, "every frame") is removed before the text encoder
sees it. The log records the length and a hash, not the words.
"""
from __future__ import annotations

import hashlib
import re

from canvas.infra.stage_timer import log_detail

# Phrases that make a 1024 frame come back as a contact sheet.
_SERIES_WORDS = re.compile(
    r"\b(?:every\s+frames?|series|shots|angles|panels|grids?|collages?|sheets?|sequences?|multiple|sets?)\b",
    re.IGNORECASE,
)
_SPACES = re.compile(r"[ \t]{2,}")
_COMMA_GAP = re.compile(r"\s+,")
_COMMA_RUN = re.compile(r"(?:,\s*){2,}")
_LEAD = "A single photograph of"
_NUMBERED = re.compile(r"^\s*\d{1,2}\s*[.)]\s*")


def strip_series_words(text: str) -> str:
    """Drop series and plural wording. The subject and the shot stay."""
    cleaned = _SERIES_WORDS.sub(" ", text)
    cleaned = _COMMA_GAP.sub(",", cleaned)
    cleaned = _COMMA_RUN.sub(", ", cleaned)
    cleaned = _SPACES.sub(" ", cleaned)
    return cleaned.strip(" ,")


def split_frame_prompts(prompts: tuple[str, ...]) -> tuple[str, ...]:
    """One entry per frame. A joined scenario is not one embedding."""
    frames: list[str] = []
    for prompt in prompts:
        frames.extend(_split_one(prompt))
    return tuple(frames)


def log_frame_prompt(prompt: str) -> None:
    """Record the length and hash of one final frame prompt. The words stay out of the log."""
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    log_detail("frame", "prompt", "cpu", f"chars={len(prompt)} sha={digest}")


def _split_one(prompt: str) -> tuple[str, ...]:
    text = prompt.strip()
    if not text:
        return ()
    marker = _LEAD.lower()
    if text.lower().count(marker) >= 2:
        parts = re.split(rf"(?i)(?={re.escape(_LEAD)})", text)
        return tuple(part.strip() for part in parts if part.strip())
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    numbered = [line for line in lines if _NUMBERED.match(line)]
    if len(lines) >= 2 and len(numbered) == len(lines):
        frames: list[str] = []
        for line in numbered:
            body = strip_series_words(_NUMBERED.sub("", line).strip())
            if not body:
                continue
            if body.lower().startswith(marker):
                frames.append(body)
            else:
                frames.append(f"{_LEAD} {body}")
        return tuple(frames)
    return (text,)
