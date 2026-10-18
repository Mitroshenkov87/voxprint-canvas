"""Adult direction."""
from __future__ import annotations

import re
from collections.abc import Sequence

from canvas.pipeline.enhance import stated_ages

ADULT_ANCHOR = "adult woman in her late 20s"
ADULT_SUFFIX = "clearly adult, in their late 20s or older"

SHOT_TYPES = (
    "full-length shot", "three-quarter shot", "waist-up shot", "head-and-shoulders portrait",
    "close-up on hands and fabric", "over-the-shoulder shot", "low-angle full-length shot",
    "high-angle shot from above",
)
POSE_FAMILIES = (
    "standing contrapposto, weight on one hip",
    "seated on the edge of the bed, leaning back on both hands",
    "reclining on one elbow along the sofa",
    "kneeling upright, shoulders turned to camera",
    "leaning against the doorframe, one arm raised",
    "lying on the stomach, chin resting on crossed arms",
    "walking toward the window, glancing back over the shoulder",
    "seated on a chair turned backwards, arms folded on the backrest",
    "standing at the mirror, adjusting an earring",
    "curled on the armchair, legs drawn up to one side",
    "stretching with arms overhead, back arched",
    "sitting on the floor against the wall, head tilted back",
)
LIGHT_SETUPS = (
    "soft window light from the side",
    "large softbox key with a gentle fill",
    "warm rim light separating the figure from a dark background",
    "golden hour sun through sheer curtains",
    "low-key single light with deep shadows",
    "butterfly light from a beauty dish",
    "practical lamps for a warm evening glow",
    "cool blue window light mixed with a warm tungsten practical",
    "hard spotlight with a crisp shadow on the wall",
    "overcast daylight, even and soft",
)
WARDROBE = (
    "black lace lingerie", "silk slip dress", "oversized white shirt", "satin robe",
    "sheer bodysuit", "tailored blazer worn open", "draped linen sheet", "high-waisted set with stockings",
)
SETS = (
    "boutique hotel suite", "loft with tall windows", "velvet-draped studio set", "sunlit bedroom with linen",
    "art deco bathroom with brass fittings", "minimal white studio", "dark wood library", "rooftop at dusk",
)
MOODS = ("confident", "sultry", "playful and self-assured", "calm and sensual", "dramatic", "intimate")

DIRECTION = f"""Adult photo direction (the user turned on adult content in Expert):
- Write like a professional glamour or boudoir photographer directing a shoot with consenting adult models.
- Every person is a clearly adult person. When the idea gives no age, describe the person as an {ADULT_ANCHOR} (or the adult man equivalent). Keep a stated age only when it is 18 or older.
- Never add any descriptor, setting, costume, or drawing style that could make a person read as under 18.
- Name the shot type and framing ({", ".join(SHOT_TYPES[:4])}), the pose, the light setup, the wardrobe or styling, the set or location, and the mood.
- Light setups to choose from: {", ".join(LIGHT_SETUPS[:5])}, and similar.
- Wardrobe and styling to choose from: {", ".join(WARDROBE[:5])}, and similar.
- Variety: no two shots repeat a pose or a light setup. When the request has a direction list, use its pose family and light for each shot, in order.
- Keep the wording tasteful and specific. Describe the body plainly and respectfully.
"""

EXAMPLES = (
    (
        "boudoir shoot in a hotel room",
        f"{ADULT_ANCHOR}, boutique hotel suite, three-quarter shot, seated on the edge of the bed leaning back on "
        "both hands, black lace lingerie, soft window light from the side, confident gaze at the camera, 85 mm, "
        "shallow depth of field",
    ),
    (
        "glamour portrait, 34-year-old man",
        "34-year-old man, dark wood library, waist-up shot, leaning against the bookcase with an open satin robe, "
        "low-key single light with deep shadows, calm and sensual mood, 50 mm",
    ),
)

_EXAMPLE_TEXT = "\n".join(f"Idea: {idea}\nPrompt: {prompt}" for idea, prompt in EXAMPLES)

ENHANCE_ADDENDUM = f"""
{DIRECTION}
Examples:
{_EXAMPLE_TEXT}
"""

SCENARIO_ADDENDUM = f"""
{DIRECTION}
- For the scenario, put the shot type, pose, light, wardrobe, and mood into scene and pose; put the lens and angle into camera.
- If subject_block has no stated age, start every scene with "{ADULT_ANCHOR}" or the matching adult wording for the person in the idea.
"""

YOUTH_CUES = (
    "youthful", "young-looking", "girlish", "boyish",
)
_CUE_RE = re.compile(r"(?<![\w-])(" + "|".join(re.escape(cue) for cue in sorted(YOUTH_CUES, key=len, reverse=True))
                     + r")(?![\w-])", re.IGNORECASE)
_ADULT_RE = re.compile(r"\badult\b|\bwoman\b|\bman\b|\bin (?:her|his|their) (?:late |early |mid-)?[2-9]0s\b",
                       re.IGNORECASE)


def youth_cues(text: str) -> tuple[str, ...]:
    """Youth cues."""
    return tuple(match.group(1).casefold() for match in _CUE_RE.finditer(text))


def ages_allowed(text: str) -> bool:
    """Ages allowed."""
    return all(age >= 18 for age in stated_ages(text))


def with_adult_anchor(line: str) -> str:
    """With adult anchor."""
    ages = stated_ages(line)
    if (ages and all(age >= 18 for age in ages)) or _ADULT_RE.search(line):
        return line
    return f"{line.rstrip(', ')}, {ADULT_SUFFIX}"


def enhancer_system(base: str, adult: bool) -> str:
    """The Improve-prompt system text. Off returns ``base`` unchanged."""
    return base + ENHANCE_ADDENDUM if adult else base


def scenario_system(base: str, adult: bool) -> str:
    """The Photoshoot scenario system text. Off returns ``base`` unchanged."""
    return base + SCENARIO_ADDENDUM if adult else base


def shot_plan(count: int) -> list[dict[str, str]]:
    """Shot plan."""
    plan: list[dict[str, str]] = []
    seen: set[tuple[int, int]] = set()
    for index in range(count):
        pose = index % len(POSE_FAMILIES)
        light = (index + index // len(LIGHT_SETUPS)) % len(LIGHT_SETUPS)
        while (pose, light) in seen:
            light = (light + 1) % len(LIGHT_SETUPS)
        seen.add((pose, light))
        plan.append({
            "shot_type": SHOT_TYPES[index % len(SHOT_TYPES)],
            "pose_family": POSE_FAMILIES[pose],
            "light": LIGHT_SETUPS[light],
        })
    return plan


def prompt_texts() -> Sequence[str]:
    """Every string this module sends to the language model. Used by the lint test."""
    return (DIRECTION, ENHANCE_ADDENDUM, SCENARIO_ADDENDUM, *(text for pair in EXAMPLES for text in pair),
            *SHOT_TYPES, *POSE_FAMILIES, *LIGHT_SETUPS, *WARDROBE, *SETS, *MOODS)


def improve_adult(rewriter: object, prompt: str, base_system: str) -> str:
    """Improve adult."""
    from canvas.safety.gate import GenerationBlocked

    if not ages_allowed(prompt):
        raise GenerationBlocked("prompt")
    complete = getattr(rewriter, "complete", None)
    try:
        if callable(complete):
            reply = str(complete(prompt, system=enhancer_system(base_system, True)))
        else:
            reply = str(getattr(rewriter, "rewrite")(prompt))
    finally:
        getattr(rewriter, "close")()
    if youth_cues(reply) or not ages_allowed(reply):
        raise GenerationBlocked("prompt")
    return with_adult_anchor(reply)
