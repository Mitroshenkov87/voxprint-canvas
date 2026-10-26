"""Scenario text for a Photoshoot series.

The language model returns numbered lines, one per shot. This module reads
those lines (or a JSON object from an older reply), builds a planned shot for
any frame the reply did not cover, and assembles the line for each shot. It
does not load a model and it does not call the gate.
"""
from __future__ import annotations

import json
import re
import unicodedata
import zlib
from dataclasses import dataclass
from typing import Literal, assert_never

from canvas.pipeline.adult_direction import LIGHT_SETUPS, POSE_FAMILIES, shot_plan
from canvas.pipeline.enhance import AGING_DESCRIPTOR_MIN_AGE, added_aging_terms, quoted_spans, stated_ages
from canvas.pipeline.frame_prompt import (
    _undangle,
    _without_partial,
    chosen_items,
    frame_identity,
    strip_series_words,
    with_choices,
)

PlaceMode = Literal["same", "different"]
MAX_FRAMES = 99
DEFAULT_FRAMES = 6
SCENARIO_CHUNK = 10
SCENARIO_TOKENS = 8192

SYSTEM_PROMPT = f"""You plan the shots of a photo series for a text-to-image model.
The user message is JSON with idea, count, style, and place_mode. It can also have direction and previous.

Write exactly count lines and nothing else. Number the lines 1., 2., 3. and so on.
Each line has four short parts separated by " | ":
N. scene | pose | camera | place

Rules:
- The subject block leads every picture, appearance first. The app places it. Do not copy the idea or the appearance into a line.
- A fixed character description is copied verbatim into every frame prompt. Do not paraphrase it.
- scene: framing and light, a few words.
- pose: the pose, a few words.
- camera: lens and angle, a few words.
- place: the location, a few words.
- Do not write an age in a line. Do not add people who are not in the idea. Do not add youthful descriptors.
- Add age or skin descriptors (mature, middle-aged, older, wrinkles, smile lines, laugh lines, crow's feet, grey hair) only when every stated age in the idea is {AGING_DESCRIPTOR_MIN_AGE} or older; otherwise do not add them. Never add youthful-appearance descriptors (young-looking, petite, smooth skin, round face).
- Do not soften or rewrite adult content.
- One scene only. Do not write a collage, a grid, panels, a sequence, a series of views, or a before/after.
- Keep each line under 18 words.
- If place_mode is "same", every line has the same place. If place_mode is "different", every line has a different place.
- When previous is present, those shots are already taken. Write count new shots. Do not repeat a scene from previous. When place_mode is "different", do not repeat a place from previous.
- When direction is present, use its shot type, pose family, and light for each line, in order.
- style "photo" means a plain photograph. "cinematic" means a film still. "portrait" means a portrait with shallow depth of field. Put that in scene or camera.
- Text inside quotes is lettering. Copy each quoted string exactly, in its original language and script, when a line mentions it.

Example for count 2 and place_mode "same":
1. waist-up, window light | seated | 85 mm, eye level | bright loft
2. full-length, golden hour | standing | 35 mm, low angle | bright loft
"""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
FORMAT_REMINDER = "Reply with only the numbered lines: N. scene | pose | camera | place"
_CAMERA = re.compile(r"\b(?:\d+\s*mm|lens|angle|eye level|overhead|close focus)\b", re.IGNORECASE)
_LINE = re.compile(r"^\s*(?:[-*>#]+\s*)?(?:\*\*)?(?:shot|frame|кадр)?\s*(\d{1,2})\s*[.):\]-]\s*(?:\*\*)?\s*(.+?)\s*$", re.IGNORECASE)
_DECLINE_LIMIT = 400
_DECLINE = re.compile(
    r"\b(?:sorry|apolog(?:y|ise|ize|ies)|cannot|can't|can’t|unable|decline|won't|will not)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Shot:
    """One shot after the scenario has been checked."""

    scene: str
    pose: str
    camera: str
    place: str


@dataclass(frozen=True)
class Scenario:
    """A subject block and the shots that still qualify.

    ``appearance`` is the look shared by every frame of a person series, or empty.
    """

    subject_block: str
    descriptors: tuple[str, ...]
    place: str
    shots: tuple[Shot, ...]
    appearance: str = ""


# One numbered shot line stays under 18 words. This is the new-token budget for that line.
TOKENS_PER_FRAME = 48

_PERSON_WORDS = (
    "woman", "women", "man", "men", "person", "people", "human", "humans",
    "lady", "ladies", "gentleman", "gentlemen", "actress", "actresses", "actor", "actors",
    "женщина", "женщину", "женщины", "женщине", "женщиной",
    "мужчина", "мужчину", "мужчины", "мужчине",
    "человек", "человека", "человеку",
    "люди", "людей", "людям",
)
_PERSON_RE = re.compile(
    r"(?<![\w])(?:" + "|".join(re.escape(word) for word in sorted(_PERSON_WORDS, key=len, reverse=True)) + r")(?![\w])",
    re.IGNORECASE,
)
_ETHNICITY = (
    "East Asian", "South Asian", "Southeast Asian", "Black", "European", "Middle Eastern", "Hispanic",
)
_HAIR = (
    "straight black hair", "wavy dark brown hair", "curly black hair",
    "straight chestnut hair", "wavy auburn hair", "straight blonde hair",
)
_EYES = ("dark brown eyes", "brown eyes", "hazel eyes", "green eyes", "amber eyes", "blue eyes")
_FACE = ("oval face", "square face", "oblong face", "angular face", "diamond face", "heart-shaped face")
_BUILD = ("athletic build", "average build", "broad-shouldered build", "lean build", "sturdy build", "solid build")
TEST_SERIES_CHARACTER = "an Asian woman, about 30, blonde, athletic with defined muscles"


def fixed_appearance(idea: str, seed: int) -> str:
    """One concrete look for a person series, or empty when the idea has no person.

    The same idea and seed always return the same text. The text names ethnicity,
    hair, eyes, face shape, and build.

    Args:
        idea: The prompt-field text.
        seed: The series seed.
    """
    if idea and TEST_SERIES_CHARACTER in idea:
        return TEST_SERIES_CHARACTER
    if not idea or _PERSON_RE.search(idea) is None:
        return ""
    if isinstance(seed, bool) or not isinstance(seed, int):
        seed = 0
    mix = zlib.crc32(f"{seed}\n{idea.strip()}".encode("utf-8"))
    tables = (_ETHNICITY, _HAIR, _EYES, _FACE, _BUILD)
    parts = [table[(mix >> (shift * 5)) % len(table)] for shift, table in enumerate(tables)]
    return ", ".join(parts)


def scenario_token_budget(count: int) -> int:
    """New-token cap for a scenario reply of ``count`` shots.

    The cap grows with the frame count and never exceeds ``SCENARIO_TOKENS``.

    Args:
        count: How many shots this reply should contain.
    """
    frames = count if isinstance(count, int) and not isinstance(count, bool) and count >= 1 else 1
    return min(SCENARIO_TOKENS, frames * TOKENS_PER_FRAME)


def tokens_for_request(prompt: str) -> int:
    """New-token cap for one scenario request, from the JSON ``count`` in ``prompt``.

    Args:
        prompt: The user message, a JSON object, optionally followed by a reminder.
    """
    count = 1
    start = prompt.find("{")
    end = prompt.rfind("}")
    if start >= 0 and end > start:
        try:
            body = json.loads(prompt[start:end + 1])
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict):
            raw = body.get("count")
            if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1:
                count = raw
    return scenario_token_budget(count)


def scenario_request(
    idea: str,
    count: int,
    style: str,
    place_mode: str,
    *,
    adult: bool = False,
    previous: tuple[str, ...] = (),
) -> str | None:
    """JSON user message for one scenario call. None when the chips are not a series.

    Args:
        idea: The prompt-field text.
        count: How many shots this call should return, from 1 to ``MAX_FRAMES``.
        style: ``photo``, ``cinematic``, or ``portrait``.
        place_mode: ``same`` or ``different``.
        adult: Expert adult content is on. Adds a per-shot pose and light plan.
        previous: Lines already accepted, so this call continues the series.
    """
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or count > MAX_FRAMES
        or style not in ("photo", "cinematic", "portrait")
        or place_mode not in ("same", "different")
    ):
        return None
    if not idea.strip():
        return None
    payload: dict[str, object] = {"idea": idea, "count": count, "style": style, "place_mode": place_mode}
    if TEST_SERIES_CHARACTER in idea:
        payload["character"] = TEST_SERIES_CHARACTER
    if previous:
        payload["previous"] = list(previous)
    if adult:
        start = len(previous)
        payload["direction"] = shot_plan(start + count)[start:]
    return json.dumps(payload, ensure_ascii=False)


SINGLE_FRAME = "A single photograph of"
COLLAGE_NEGATIVE = "collage, grid, panels, split screen"
# The text encoder keeps this many tokens, including its chat wrapper.
ENCODER_TOKEN_LIMIT = 512
_ENCODER_WRAPPER_TOKENS = 48
_MULTI_SCENE = re.compile(
    r"\b(?:collage|grids?|panels?|split[- ]screen|before\s*/\s*after|before and after"
    r"|series of|sequences?|multiple views|multi[- ]panel)\b",
    re.IGNORECASE,
)


def single_frame(line: str) -> str:
    """Lead with one photograph. The picture model does not read a negative."""
    text = line.strip()
    if text.lower().startswith(SINGLE_FRAME.lower()):
        return text
    if not text:
        return SINGLE_FRAME
    return f"{SINGLE_FRAME} {text}"


def one_scene(text: str) -> bool:
    """False when a shot line describes more than one view."""
    return _MULTI_SCENE.search(text) is None


def truncate_for_encoder(
    prompt: str, limit: int | None = None, keep: tuple[str, ...] = (),
) -> str:
    """Keep the head of ``prompt`` inside the text encoder's token cap.

    A token is at most one word, so the first ``limit`` words fit. The lead and
    the appearance sit in that head, and the tail is what the cap drops. Each
    phrase in ``keep`` (the per-frame outfit, light, and place) is put back
    when the cut would drop it. The default leaves room for the encoder's chat
    wrapper.
    """
    if limit is None:
        limit = ENCODER_TOKEN_LIMIT - _ENCODER_WRAPPER_TOKENS
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        limit = ENCODER_TOKEN_LIMIT - _ENCODER_WRAPPER_TOKENS
    words = prompt.split()
    kept = tuple(phrase.strip() for phrase in keep if phrase and phrase.strip())
    if len(words) <= limit and all(phrase in prompt for phrase in kept):
        return prompt
    reserve = 0
    head_budget = " ".join(words[:limit])
    for phrase in kept:
        if phrase not in head_budget:
            reserve += len(phrase.split())
    room = max(0, limit - reserve)
    head = _undangle(" ".join(words[:room]))
    for phrase in kept:
        head = _without_partial(head, phrase)
        if phrase not in head:
            head = f"{head}, {phrase}" if head else phrase
    return head


def frame_negative(negative: str) -> str:
    """The caller's negative plus the collage wording."""
    if COLLAGE_NEGATIVE in negative:
        return negative
    return ", ".join(part for part in (negative.strip(), COLLAGE_NEGATIVE) if part)


def _with_test_character(text: str, subject: str) -> str:
    """Keep the test-series character verbatim at the head of a frame prompt."""
    character = TEST_SERIES_CHARACTER
    if character not in subject or character in text:
        return text
    return f"{character}, {text}" if text else character


def assemble_line(
    subject: str, shot: Shot, *, include_camera: bool, index: int = 0, seed: int = 0,
) -> str:
    """The line sent for one shot.

    Appearance leads, then this shot's pose, camera, scene, and place. Each
    parenthesized list in the subject contributes one item, chosen from the
    seed. Series wording is removed. The line starts with one photograph and
    stays inside the text encoder's token cap.

    Args:
        subject: Appearance, then the idea. Lists and series instructions are stripped.
        shot: Scene, pose, camera, and place for this shot only.
        include_camera: When true, the lens and angle stay on the line.
        index: Which shot this is. Alternative lists walk forward from here.
        seed: Series seed. The same seed and index always pick the same item.
    """
    sources = [subject, shot.pose]
    if include_camera:
        sources.append(shot.camera)
    sources.extend((shot.scene, shot.place))
    kept = tuple(
        item for source in sources for item in chosen_items(source, index, seed)
    )
    parts = [strip_series_words(with_choices(source, index, seed)) for source in sources]
    body = ", ".join(part for part in parts if part)
    body = _with_test_character(body, subject)
    return truncate_for_encoder(single_frame(body), keep=kept)


def validate_scenario(idea: str, raw: str, *, count: int, place_mode: PlaceMode) -> Scenario | None:
    """Return the checked scenario, or None when the reply cannot be used.

    A shot that changes a stated age or repeats a scene is left out. A bad
    subject block or a broken place mode rejects the whole reply.

    Args:
        idea: The prompt-field text.
        raw: The model reply.
        count: How many shots were requested, from 1 to ``MAX_FRAMES``.
        place_mode: ``same`` or ``different``.
    """
    payload = _object(raw)
    if payload is None or isinstance(count, bool) or not isinstance(count, int) or count < 1 or count > MAX_FRAMES:
        return None
    subject = payload.get("subject_block")
    descriptors = payload.get("descriptors")
    place = payload.get("place")
    shots = payload.get("shots")
    if not isinstance(subject, str) or not isinstance(place, str) or not isinstance(shots, list):
        return None
    if not isinstance(descriptors, list) or not _descriptors(subject, descriptors):
        return None
    folded_idea = _fold(idea)
    folded_subject = _fold(subject)
    if not folded_subject or folded_subject not in folded_idea or subject not in idea:
        return None
    if sorted(stated_ages(subject)) != sorted(stated_ages(idea)):
        return None
    if len(shots) != count:
        return None
    fill = place if place_mode == "same" else ""
    parsed = tuple(_shot(item, fill) for item in shots)
    if any(item is None for item in parsed):
        return None
    ready = tuple(item for item in parsed if item is not None)
    if not _places(place_mode, place, ready):
        return None
    kept = _keep_shots(idea, subject, ready)
    return Scenario(subject, tuple(str(item) for item in descriptors), place, kept)


def _keep_shots(idea: str, subject: str, shots: tuple[Shot, ...]) -> tuple[Shot, ...]:
    kept: list[Shot] = []
    scenes: set[str] = set()
    ages = sorted(stated_ages(idea))
    required = [piece for piece in quoted_spans(idea) if piece in subject]
    for shot in shots:
        line = assemble_line(subject, shot, include_camera=True)
        identity = frame_identity(subject)
        if identity and identity not in line:
            continue
        if sorted(stated_ages(line)) != ages:
            continue
        if any(piece not in line for piece in required):
            continue
        if shot.scene in scenes:
            continue
        scenes.add(shot.scene)
        kept.append(shot)
    return tuple(kept)


def _places(place_mode: PlaceMode, place: str, shots: tuple[Shot, ...]) -> bool:
    if place_mode == "same":
        return bool(place.strip()) and all(shot.place == place for shot in shots)
    if place_mode == "different":
        names = [shot.place.strip() for shot in shots]
        return place == "" and all(names) and len(set(names)) == len(names)
    assert_never(place_mode)


def _descriptors(subject: str, descriptors: list[object]) -> bool:
    if not descriptors:
        return False
    for item in descriptors:
        if not isinstance(item, str) or not item or item not in subject:
            return False
    return True


def _shot(item: object, place_default: str = "") -> Shot | None:
    if not isinstance(item, dict):
        return None
    scene = item.get("scene")
    pose = item.get("pose")
    camera = item.get("camera")
    place = item.get("place", place_default) or place_default
    if not all(isinstance(value, str) and value.strip() for value in (scene, pose, camera, place)):
        return None
    return Shot(str(scene).strip(), str(pose).strip(), str(camera).strip(), str(place).strip())


def reply_declined(raw: str) -> bool:
    """True when a reply has no JSON object and is a short decline."""
    text = _FENCE.sub("", raw.strip())
    if "{" in text or len(text) > _DECLINE_LIMIT:
        return False
    return _DECLINE.search(text) is not None


def _object(raw: str) -> dict[str, object] | None:
    text = _FENCE.sub("", raw.strip())
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        span = _balanced_object(text)
        if span is None:
            return None
        try:
            payload = json.loads(span)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    return payload


def _balanced_object(text: str) -> str | None:
    """The first top-level ``{...}`` span, ignoring braces inside JSON strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _fold(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


@dataclass(frozen=True)
class Draft:
    """One shot read from a reply. ``place`` is empty when the line left it out."""

    scene: str
    pose: str
    camera: str
    place: str


def read_drafts(raw: str) -> tuple[Draft, ...]:
    """Shots from numbered lines, or from a JSON reply when there are no lines.

    Args:
        raw: The model reply.
    """
    drafts = _line_drafts(raw)
    if drafts:
        return drafts
    payload = _object(raw)
    if payload is None:
        return ()
    shots = payload.get("shots")
    if not isinstance(shots, list):
        return ()
    shared = payload.get("place")
    fill = shared.strip() if isinstance(shared, str) else ""
    found: list[Draft] = []
    for item in shots:
        shot = _shot(item, fill) if fill else _draft_from_object(item)
        if isinstance(shot, Shot):
            found.append(Draft(shot.scene, shot.pose, shot.camera, shot.place))
        elif isinstance(shot, Draft):
            found.append(shot)
    return tuple(found)


def accept_shot(
    idea: str,
    subject: str,
    mode: PlaceMode,
    place: str,
    taken: tuple[Shot, ...],
    draft: Draft,
) -> Shot | None:
    """The shot for one draft, or None when it does not fit the series.

    Args:
        idea: The prompt-field text.
        subject: The verbatim subject block.
        mode: ``same`` or ``different``.
        place: The series place in ``same`` mode.
        taken: Shots already accepted.
        draft: The shot to check.
    """
    if mode == "same":
        where = place
    elif mode == "different":
        where = draft.place.strip()
        if not where or _key(where) in {_key(shot.place) for shot in taken}:
            return None
    else:
        assert_never(mode)
    shot = Shot(draft.scene.strip(), draft.pose.strip(), draft.camera.strip(), where)
    if not shot.scene:
        return None
    if not one_scene(" ".join((shot.scene, shot.pose, shot.camera, shot.place))):
        return None
    if _key(shot.scene) in {_key(item.scene) for item in taken}:
        return None
    if added_aging_terms(idea, " ".join((shot.scene, shot.pose, shot.camera, shot.place))):
        return None
    if not _keep_shots(idea, subject, (shot,)):
        return None
    return shot


def planned_shots(
    idea: str,
    count: int,
    mode: PlaceMode,
    *,
    place: str,
    taken: tuple[Shot, ...],
    adult: bool = False,
) -> tuple[Shot, ...]:
    """Shots built from fixed lists, for frames the reply did not cover.

    Each shot varies framing, light, pose, and camera. In ``different`` mode
    each shot also gets its own place. The same order comes back for the same
    idea. Every shot still passes ``accept_shot``.

    Args:
        idea: The prompt-field text.
        count: How many shots to add.
        mode: ``same`` or ``different``.
        place: The series place in ``same`` mode. It can be empty.
        taken: Shots already accepted.
        adult: Expert adult content is on. Uses the same pose and light plan as the request.
    """
    subject = idea.strip()
    shots = list(taken)
    added: list[Shot] = []
    offset = zlib.crc32(subject.encode("utf-8")) % len(PLAN_FRAMINGS)
    pairs = len(POSE_FAMILIES) * len(LIGHT_SETUPS)
    plan = shot_plan(min(len(taken) + count * 2, pairs)) if adult else []
    step = 0
    limit = len(PLAN_FRAMINGS) * len(PLAN_LIGHTS) * len(PLAN_POSES)
    while len(added) < count and step < limit:
        index = len(taken) + step
        framing = PLAN_FRAMINGS[(offset + index) % len(PLAN_FRAMINGS)]
        light = PLAN_LIGHTS[(index // len(PLAN_FRAMINGS)) % len(PLAN_LIGHTS)]
        pose = PLAN_POSES[(offset + index) % len(PLAN_POSES)]
        if adult and index < len(plan):
            framing = plan[index]["shot_type"]
            light = plan[index]["light"]
            pose = plan[index]["pose_family"]
        turn = index // (len(PLAN_FRAMINGS) * len(PLAN_LIGHTS))
        if turn:
            light = f"{light}, variation {turn + 1}"
        camera = PLAN_CAMERAS[index % len(PLAN_CAMERAS)]
        where = place if mode == "same" else _planned_place(index)
        draft = Draft(f"{framing}, {light}", pose, camera, where)
        step += 1
        shot = accept_shot(idea, subject, mode, place, tuple(shots), draft)
        if shot is None:
            continue
        shots.append(shot)
        added.append(shot)
    return tuple(added)


PLAN_FRAMINGS = (
    "wide shot", "medium shot", "close-up", "detail shot", "full-length shot", "three-quarter view",
    "overhead view", "profile view", "waist-up shot", "low-angle shot", "high-angle shot",
    "centred symmetrical composition",
)
PLAN_LIGHTS = (
    "soft window light from the side", "golden hour sunlight", "overcast daylight, even and soft",
    "large softbox key with a gentle fill", "warm rim light against a dark background", "cool blue hour light",
    "hard spotlight with a crisp shadow", "warm light from practical lamps",
)
PLAN_POSES = (
    "subject centred in the shot", "subject on the left third of the shot", "subject on the right third of the shot",
    "subject turned three-quarters to the camera", "subject seen through a soft foreground",
    "subject against a plain background", "subject filling the shot", "subject seen at a slight angle",
)
PLAN_CAMERAS = (
    "35 mm lens, eye level", "50 mm lens, slightly above eye level", "85 mm lens, eye level",
    "24 mm lens, low angle", "100 mm lens, close focus", "35 mm lens, high angle",
    "50 mm lens, three-quarter angle", "70 mm lens, side angle",
)
PLAN_PLACES = (
    "minimal white studio", "loft with tall windows", "outdoors in open shade", "rooftop at dusk",
    "sunlit room with linen curtains", "against a dark textured wall", "quiet courtyard", "beside a large window",
    "wooden terrace", "softly lit hallway", "garden with greenery", "against a concrete wall",
)
_PLACE_TIMES = (
    "",
    "in the morning",
    "in the evening",
    "at night",
    "in warm afternoon light",
    "at dawn",
    "at noon",
    "in late afternoon",
    "under moonlight",
)


def _planned_place(index: int) -> str:
    base = PLAN_PLACES[index % len(PLAN_PLACES)]
    when = _PLACE_TIMES[(index // len(PLAN_PLACES)) % len(_PLACE_TIMES)]
    return f"{base}, {when}" if when else base


def _line_drafts(raw: str) -> tuple[Draft, ...]:
    found: list[Draft] = []
    seen: set[int] = set()
    for row in _FENCE.sub("", raw.strip()).splitlines():
        match = _LINE.match(row)
        if match is None:
            continue
        number = int(match.group(1))
        parts = [part.strip(" *\t") for part in match.group(2).split("|")]
        parts = [part for part in parts if part]
        if number in seen or len(parts) < 3:
            continue
        seen.add(number)
        if len(parts) == 3 and _CAMERA.search(parts[2]):
            found.append(Draft(parts[0], parts[1], parts[2], ""))
        elif len(parts) == 3:
            found.append(Draft(parts[0], "", parts[1], parts[2]))
        else:
            found.append(Draft(parts[0], parts[1], ", ".join(parts[2:-1]), parts[-1]))
    return tuple(found)


def _draft_from_object(item: object) -> Draft | None:
    if not isinstance(item, dict):
        return None
    values = [item.get(key) for key in ("scene", "pose", "camera")]
    if not all(isinstance(value, str) and value.strip() for value in values):
        return None
    where = item.get("place")
    return Draft(str(values[0]), str(values[1]), str(values[2]), where if isinstance(where, str) else "")


def _key(value: str) -> str:
    return " ".join(_fold(value).casefold().split())
