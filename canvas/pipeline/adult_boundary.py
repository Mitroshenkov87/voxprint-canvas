"""The Adult content (18+) setting and its OFF boundary. Prompt policy only.

``canvas.adult_content`` is the one stored value. The shared layout still reads
``image.adult_content`` for its own prompt policy, so Canvas writes both keys
(see :class:`canvas.ui.shell_layout.CanvasLayout`) and reads only its own.

OFF allows swapping one outfit for another, in generation and in every edit.
OFF refuses taking all clothing off and any nude or unclothed picture. ON
allows both. This module is not the minors gate: that gate runs in both
states and nothing here changes it. It is a fixed list of the policy's own
words, with no scoring and no model.
"""
from __future__ import annotations

import re
from typing import Protocol

from voxprint_common.ui.safety import AdultPromptPolicy, adult_prompt_policy

from canvas.app.i18n import text

KEY = "canvas.adult_content"
COMMON_KEY = "image.adult_content"

# Added to the shared OFF negatives so a model that is not told to refuse also leans away.
_OFF_NEGATIVES = "nude, naked, nudity, topless, unclothed"

_NUDE = (
    r"nude|nudes|nudity|naked|unclothed|undressed|topless|bottomless|birthday suit|in the buff|"
    r"undress\w*|disrobe\w*|"
    r"голая|голый|голые|голое|обнажённ\w*|обнажен\w*|раздет\w*|раздень\w*|раздева\w*|нюд\w*|"
    r"nackt\w*|entkleid\w*"
)
_CLOTHES = r"clothes|clothing|garments|outfit|everything|одежд\w*|вещи|kleidung|kleider"
_REMOVE = r"remove|removing|take off|taking off|strip|stripping|get rid of|lose|losing|сними\w*|снять|убери\w*|убрать"
_REFUSED = (
    re.compile(rf"(?<!\w)(?:{_NUDE})(?!\w)", re.IGNORECASE),
    re.compile(rf"(?<!\w)(?:{_REMOVE})\s+(?:\w+\s+){{0,2}}?(?:{_CLOTHES})(?!\w)", re.IGNORECASE),
    re.compile(rf"(?<!\w)(?:without|no|not wearing|wearing no|без)\s+(?:any\s+|a\s+stitch\s+of\s+)?(?:{_CLOTHES})(?!\w)",
               re.IGNORECASE),
    re.compile(rf"(?<!\w)(?:{_CLOTHES})\s+(?:off|removed|gone)(?!\w)", re.IGNORECASE),
    re.compile(r"(?<!\w)wearing nothing(?!\w)", re.IGNORECASE),
)
_NEGATION = {"no", "not", "non", "never", "avoid", "без", "не", "kein", "keine"}
_WORD = re.compile(r"\w+")


class _HasState(Protocol):
    def get(self, key: str) -> object: ...


def enabled(layout: _HasState) -> bool:
    """The stored setting. A layout that does not hold it reads as OFF."""
    try:
        return layout.get(KEY) is True
    except (KeyError, ValueError):
        return False


def policy(allowed: bool) -> AdultPromptPolicy:
    """The shared policy, with the OFF negatives tightened. ON adds nothing."""
    base = adult_prompt_policy(allowed=allowed)
    if allowed:
        return base
    return AdultPromptPolicy(base.template, ", ".join(part for part in (base.negatives, _OFF_NEGATIVES) if part))


def crosses_off_boundary(prompt: str) -> bool:
    """True when the words ask to remove all clothing or for a nude or unclothed picture."""
    for pattern in _REFUSED:
        for found in pattern.finditer(prompt):
            before = _WORD.findall(prompt[: found.start()].lower())[-1:]
            if before and before[0] in _NEGATION:
                continue
            return True
    return False


def refusal(prompt: str, *, allowed: bool) -> str | None:
    """The plain refusal when the setting is OFF and ``prompt`` crosses the boundary, else None.

    Outfit swaps ("change the red dress to a blue coat") pass in both states.
    """
    if allowed or not crosses_off_boundary(prompt):
        return None
    return text("adult_off.refused")
