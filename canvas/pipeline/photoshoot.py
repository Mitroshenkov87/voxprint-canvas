"""Scenario text for a Photoshoot series.

The language model returns JSON. This module checks that JSON and builds the
line for each shot. It does not load a model and it does not call the gate.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, assert_never

from canvas.pipeline.adult_direction import shot_plan
from canvas.pipeline.enhance import quoted_spans, stated_ages

PlaceMode = Literal["same", "different"]
MAX_FRAMES = 50
DEFAULT_FRAMES = 6
SCENARIO_CHUNK = 10

SYSTEM_PROMPT = """You write a photo-shoot scenario as JSON for a text-to-image model.
The user message is JSON with idea, count, style, and place_mode.

Return one JSON object and nothing else.
{
  "subject_block": string,
  "descriptors": [string, ...],
  "place": string,
  "shots": [{"scene": string, "pose": string, "camera": string, "place": string}, ...]
}

Rules:
- subject_block is copied character for character from idea. Do not translate it. Do not shorten it by dropping a stated age or a subject descriptor.
- descriptors are short phrases copied character for character from subject_block. One phrase is not enough if the idea names more than one identifying fact. Include every stated age phrase.
- shots has exactly count objects.
- When previous is present, those lines are shots already accepted. Return count new shots that continue the series. Do not repeat a scene from previous. When place_mode is "different", do not repeat a place from previous. Keep subject_block the same text.
- scene, pose, and camera are natural English.
- Do not add people who are not in the idea. Do not add youthful descriptors.
- Do not invent, drop, round, or change a stated age. Do not soften or rewrite adult content.
- Text inside quotes is lettering. Copy each quoted string exactly, in its original language and script.
- If place_mode is "same", place is one location, and every shot.place equals that same string.
- If place_mode is "different", every shot.place is a different location, and place is an empty string.
- style "photo" means a plain photograph. "cinematic" means a film still. "portrait" means a portrait with shallow depth of field. Put that only in the wording of scene or camera, not in subject_block.
"""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
JSON_REMINDER = "Reply with only the JSON object."
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
    """A subject block and the shots that still qualify."""

    subject_block: str
    descriptors: tuple[str, ...]
    place: str
    shots: tuple[Shot, ...]


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
    if previous:
        payload["previous"] = list(previous)
    if adult:
        start = len(previous)
        payload["direction"] = shot_plan(start + count)[start:]
    return json.dumps(payload, ensure_ascii=False)


def assemble_line(subject: str, shot: Shot, *, include_camera: bool) -> str:
    """The line sent for one shot. Simple mode leaves the camera clause out.

    Args:
        subject: The verbatim subject block.
        shot: Scene, pose, camera, and place.
        include_camera: True in Expert, where the lens and angle stay on the line.
    """
    parts = [subject.strip(), shot.place.strip(), shot.scene.strip(), shot.pose.strip()]
    if include_camera and shot.camera.strip():
        parts.append(shot.camera.strip())
    return ", ".join(part for part in parts if part)


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
    parsed = tuple(_shot(item) for item in shots)
    if any(item is None for item in parsed):
        return None
    ready = tuple(item for item in parsed if item is not None)
    if not _places(place_mode, place, ready):
        return None
    kept = _keep_shots(idea, subject, ready)
    return Scenario(subject, tuple(str(item) for item in descriptors), place, kept)


def one_shot_request(subject: str, place_mode: PlaceMode) -> str:
    """User message for a replacement shot. It does not mention the gate.

    Args:
        subject: The verbatim subject block.
        place_mode: ``same`` or ``different``.
    """
    return json.dumps(
        {"subject_block": subject, "place_mode": place_mode, "ask": "write one different shot"},
        ensure_ascii=False,
    )


def parse_shot(raw: str) -> Shot | None:
    """One replacement shot, or None when the reply is not that object.

    Args:
        raw: The model reply.
    """
    payload = _object(raw)
    if payload is None:
        return None
    return _shot(payload)


def _keep_shots(idea: str, subject: str, shots: tuple[Shot, ...]) -> tuple[Shot, ...]:
    kept: list[Shot] = []
    scenes: set[str] = set()
    ages = sorted(stated_ages(idea))
    required = [piece for piece in quoted_spans(idea) if piece in subject]
    for shot in shots:
        line = assemble_line(subject, shot, include_camera=True)
        if subject not in line:
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


def _shot(item: object) -> Shot | None:
    if not isinstance(item, dict):
        return None
    scene = item.get("scene")
    pose = item.get("pose")
    camera = item.get("camera")
    place = item.get("place")
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
