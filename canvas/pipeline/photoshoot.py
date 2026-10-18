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

from canvas.pipeline.enhance import quoted_spans, stated_ages

PlaceMode = Literal["same", "different"]
COUNTS = (6, 12, 20)

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
- scene, pose, and camera are natural English.
- Do not add people who are not in the idea. Do not add youthful descriptors.
- Do not invent, drop, round, or change a stated age. Do not soften or rewrite adult content.
- Text inside quotes is lettering. Copy each quoted string exactly, in its original language and script.
- If place_mode is "same", place is one location, and every shot.place equals that same string.
- If place_mode is "different", every shot.place is a different location, and place is an empty string.
- style "photo" means a plain photograph. "cinematic" means a film still. "portrait" means a portrait with shallow depth of field. Put that only in the wording of scene or camera, not in subject_block.
"""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


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


def scenario_request(idea: str, count: int, style: str, place_mode: str) -> str | None:
    """JSON user message for one scenario call. None when the chips are not a series.

    Args:
        idea: The prompt-field text.
        count: 6, 12, or 20.
        style: ``photo``, ``cinematic``, or ``portrait``.
        place_mode: ``same`` or ``different``.
    """
    if count not in COUNTS or style not in ("photo", "cinematic", "portrait") or place_mode not in ("same", "different"):
        return None
    if not idea.strip():
        return None
    return json.dumps(
        {"idea": idea, "count": count, "style": style, "place_mode": place_mode},
        ensure_ascii=False,
    )


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
        count: How many shots were requested.
        place_mode: ``same`` or ``different``.
    """
    payload = _object(raw)
    if payload is None or count not in COUNTS:
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


def _object(raw: str) -> dict[str, object] | None:
    text = _FENCE.sub("", raw.strip())
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _fold(value: str) -> str:
    return unicodedata.normalize("NFKC", value)
