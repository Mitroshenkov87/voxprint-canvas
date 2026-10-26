"""One photograph's picture prompt.

The diffusion prompt is the appearance and that shot's own pose, camera,
scene, and place. Series wording (a grid, a sheet, "every shot") is removed
before the text encoder sees it. A parenthesized list contributes one item.
The log records the length and a hash, not the words.
"""
from __future__ import annotations

import hashlib
import re

from canvas.infra.stage_timer import log_detail

# ``frame`` is replaced with ``shot`` before this runs, so ``frames`` becomes
# ``shots`` and is removed, while a singular ``shot`` stays. ``photograph``
# has no word boundary around ``frame`` and is left as the lead.
_FRAME_WORD = re.compile(r"\bframes?\b", re.IGNORECASE)
_SERIES_WORDS = re.compile(
    r"(?:\b(?:in|of|with|on|for|at|to|from|into|onto)\s+)?"
    r"\b(?:every\s+(?:shots?|frames?)|series|shots|angles|panels?|grids?|collages?|sheets?|sequences?"
    r"|multiple|sets?|borders?|photoshoots?|photo[-\s]+shoots?)\b",
    re.IGNORECASE,
)
_DANGLING = re.compile(
    r"\b(?:in|of|with|on|for|at|to|from|into|onto|and|or)\b(?=\s*(?:,|$))",
    re.IGNORECASE,
)
_DIFFERENT = re.compile(r"\b(?:different|various)\s+[A-Za-z][\w'-]*\b", re.IGNORECASE)
_CHOICE = re.compile(r"\(([^()]*)\)")
_CHOICE_SPLIT = re.compile(r"\s*(?:,|\bor\b)\s*", re.IGNORECASE)
_SPACES = re.compile(r"[ \t]{2,}")
_COMMA_GAP = re.compile(r"\s+,")
_COMMA_RUN = re.compile(r"(?:,\s*){2,}")
_LEAD = "A single photograph of"
_NUMBERED = re.compile(r"^\s*\d{1,2}\s*[.)]\s*")


def strip_series_words(text: str) -> str:
    """Drop series wording. ``frame`` becomes ``shot``. The subject stays.

    ``A single photograph of`` is unchanged: ``frame`` is a whole word, so it
    does not match inside ``photograph``.
    """
    cleaned = _FRAME_WORD.sub(_as_shot, text)
    cleaned = _DIFFERENT.sub(" ", cleaned)
    cleaned = _SERIES_WORDS.sub(" ", cleaned)
    return _undangle(cleaned)


def choice_lists(text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Parentheses that hold a comma or ``or``, and their items.

    A bare comma list, such as an appearance, is not a choice. ``and`` does
    not split an item.
    """
    found: list[tuple[str, tuple[str, ...]]] = []
    for match in _CHOICE.finditer(text):
        items = tuple(
            part for part in (piece.strip(" ,") for piece in _CHOICE_SPLIT.split(match.group(1))) if part
        )
        if len(items) >= 2:
            found.append((match.group(0), items))
    return tuple(found)


def chosen_items(text: str, index: int, seed: int) -> tuple[str, ...]:
    """The one item each choice list contributes for this shot."""
    groups = choice_lists(text)
    offset = index if isinstance(index, int) and not isinstance(index, bool) else 0
    base = seed if isinstance(seed, int) and not isinstance(seed, bool) else 0
    return tuple(items[(base + offset) % len(items)] for _raw, items in groups)


def with_choices(text: str, index: int, seed: int) -> str:
    """Remove each choice list and append one item, ``(seed + index)`` along the list.

    The same seed and index always pick the same item. Consecutive indexes
    cover the list. Appearance commas outside parentheses stay.
    """
    picks = list(chosen_items(text, index, seed))
    for raw, _items in choice_lists(text):
        text = text.replace(raw, " ", 1)
    text = _tidy(text)
    if not picks:
        return text
    extra = ", ".join(picks)
    return f"{text}, {extra}" if text else extra


def frame_identity(text: str) -> str:
    """Subject text with choice lists and series words removed.

    The chosen item is appended after this text, so the result stays inside
    the assembled line.
    """
    for raw, _items in choice_lists(text):
        text = text.replace(raw, " ", 1)
    return strip_series_words(text)


def _as_shot(match: re.Match[str]) -> str:
    return "shots" if match.group(0).casefold().endswith("s") else "shot"


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


def _tidy(text: str) -> str:
    cleaned = _COMMA_GAP.sub(",", text)
    cleaned = _COMMA_RUN.sub(", ", cleaned)
    cleaned = _SPACES.sub(" ", cleaned)
    return cleaned.strip(" ,")


def _undangle(text: str) -> str:
    """Drop a preposition or conjunction left behind when a series phrase goes.

    ``same woman in every frame`` becomes ``same woman``, not ``same woman in,``.
    ``jeans and sweater`` stays: ``and`` is not sitting against a comma or the end.
    """
    cleaned = _tidy(text)
    for _pass in range(4):
        nxt = _tidy(_DANGLING.sub(" ", cleaned))
        if nxt == cleaned:
            return cleaned
        cleaned = nxt
    return cleaned


def _without_partial(head: str, phrase: str) -> str:
    """Remove a cut-off prefix of ``phrase`` from the end of ``head``."""
    words = head.split()
    phrase_words = phrase.split()
    for count in range(len(phrase_words) - 1, 0, -1):
        if words[-count:] == phrase_words[:count]:
            return " ".join(words[:-count])
    return head


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
