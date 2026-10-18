"""Run one Photoshoot series: scenario, gate, then one frame at a time."""
from __future__ import annotations

import json
import struct
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, assert_never

from canvas.app.i18n import text
from canvas.pipeline.enhance import stated_ages
from canvas.pipeline.adult_direction import ages_allowed, with_adult_anchor, youth_cues
from canvas.pipeline.photoshoot import (
    PlaceMode,
    Scenario,
    Shot,
    assemble_line,
    one_shot_request,
    parse_shot,
    scenario_request,
    validate_scenario,
)
from canvas.safety.gate import AgeGate, SafetyError
from canvas.safety.image import RawImage, SafeImage

StopCode = Literal["", "idea", "scenario", "cancelled"]


class SeriesFrame(Protocol):
    """Pixels a frame callback can return. The gate reviews a copy."""

    width: int
    height: int

    def to_bytes(self) -> bytes:
        """RGB bytes, three per pixel."""


@dataclass(frozen=True)
class SeriesResult:
    """What the series kept, and why it stopped when it did."""

    stopped: StopCode
    reason: str
    kept: tuple[Path, ...]
    folder: Path | None


def run_series(
    *,
    idea: str,
    count: int,
    style: str,
    place_mode: str,
    seed: int,
    strength: float,
    include_camera: Callable[[], bool],
    rewrite: Callable[[str], str],
    age_gate: AgeGate,
    draw_frame: Callable[[str, int, SeriesFrame | None, float], SeriesFrame | None],
    cancelled: Callable[[], bool],
    folder: Path,
    on_scenario: Callable[[Scenario], None] | None = None,
    line_for: Callable[[int], str] | None = None,
    on_status: Callable[[int, str], None] | None = None,
    adult: bool = False,
) -> SeriesResult:
    """Write the scenario, check it, and save the frames the gate allows.

    A blocked idea or scenario stops the series. The rewrite is not asked to
    soften a blocked text. A blocked frame is left out and is not the hero.

    Args:
        idea: Prompt-field text.
        count: 6, 12, or 20.
        style: Everyday look id.
        place_mode: ``same`` or ``different``.
        seed: Hero seed. Shot ``i`` uses ``seed + i``.
        strength: Img2img strength for shots after the hero.
        include_camera: True when Expert keeps the camera clause on the line.
        rewrite: One language-model call. The argument is the user message.
        age_gate: The minors gate. ``review_prompt`` and ``review_image`` are called by name.
        draw_frame: Returns pixels, or None when that frame is blocked.
        cancelled: True when the user has cancelled.
        folder: Series directory. Created only after the scenario is accepted.
        on_scenario: Shows the accepted shots before the first frame.
        line_for: The line to send for that index, after an Expert edit.
        on_status: ``drawing``, ``kept``, or ``blocked`` for one row.
        adult: Expert adult content is on. Lines get the adult wording, a line with
            a youth cue is refused (not rephrased), and a stated age under 18 stops
            the series. The gate still reviews every text and frame.
    """
    if place_mode == "same":
        mode: PlaceMode = "same"
    elif place_mode == "different":
        mode = "different"
    else:
        return SeriesResult("scenario", "", (), None)
    request = scenario_request(idea, count, style, mode, adult=adult)
    if request is None:
        return SeriesResult("scenario", "", (), None)
    reason = _refuse_prompt(age_gate, idea)
    if reason:
        return SeriesResult("idea", reason, (), None)
    if adult and not ages_allowed(idea):
        return SeriesResult("idea", "", (), None)
    scenario = _scenario(idea, count, mode, request, rewrite)
    if scenario is None or len(scenario.shots) < 6:
        return SeriesResult("scenario", "", (), None)
    lines = [assemble_line(scenario.subject_block, shot, include_camera=True) for shot in scenario.shots]
    reason = _refuse_prompt(age_gate, "\n".join(lines))
    if reason:
        return SeriesResult("scenario", reason, (), None)
    if adult and not ages_allowed("\n".join(lines)):
        return SeriesResult("scenario", "", (), None)
    if on_scenario is not None:
        on_scenario(scenario)
    folder.mkdir(parents=True, exist_ok=True)
    hero: SeriesFrame | None = None
    kept: list[Path] = []
    blocked: list[int] = []
    saved_lines: list[str] = []
    seeds: list[int] = []
    for index, shot in enumerate(scenario.shots):
        if cancelled():
            _write_meta(folder, idea, scenario, style, mode, saved_lines, seeds, blocked, kept)
            return SeriesResult("cancelled", "", tuple(kept), folder)
        line = (
            assemble_line(scenario.subject_block, shot, include_camera=include_camera())
            if line_for is None
            else line_for(index)
        )
        if scenario.subject_block not in line:
            blocked.append(index)
            _status(on_status, index, "blocked")
            continue
        if adult:
            if youth_cues(line):
                blocked.append(index)
                _status(on_status, index, "blocked")
                continue
            line = with_adult_anchor(line)
        shot_reason = _refuse_prompt(age_gate, line)
        if shot_reason:
            blocked.append(index)
            _status(on_status, index, "blocked")
            continue
        _status(on_status, index, "drawing")
        if cancelled():
            _write_meta(folder, idea, scenario, style, mode, saved_lines, seeds, blocked, kept)
            return SeriesResult("cancelled", "", tuple(kept), folder)
        image = draw_frame(line, seed + index, hero, strength)
        width = 0 if image is None else image.width
        height = 0 if image is None else image.height
        pixels = None if image is None else _released_frame(age_gate, image, line)
        if pixels is None:
            blocked.append(index)
            _status(on_status, index, "blocked")
            continue
        path = folder / f"{index + 1:02}.png"
        _write_png(path, width, height, pixels)
        kept.append(path)
        saved_lines.append(line)
        seeds.append(seed + index)
        if hero is None:
            hero = image
        _status(on_status, index, "kept")
        _write_meta(folder, idea, scenario, style, mode, saved_lines, seeds, blocked, kept)
    _write_meta(folder, idea, scenario, style, mode, saved_lines, seeds, blocked, kept)
    return SeriesResult("", "", tuple(kept), folder)


def _refuse_prompt(age_gate: AgeGate, prompt: str) -> str | None:
    """The gate's own sentence, or None when ``prompt`` may continue.

    A failure stops this text. The words are not sent back to the rewriter.
    """
    try:
        age_gate.review_prompt(prompt)
    except SafetyError as exc:
        return exc.user_message
    except Exception:
        return text("gate.unavailable")
    return None


def _released_frame(age_gate: AgeGate, image: SeriesFrame, line: str) -> bytes | None:
    """Pixels ``review_image`` released, or None when the frame is dropped."""
    raw = RawImage(image.width, image.height, image.to_bytes())
    try:
        released = age_gate.review_image(raw, line)
    except SafetyError:
        return None
    except Exception:
        return None
    if isinstance(released, SafeImage):
        return released.data
    return image.to_bytes()


def _status(callback: Callable[[int, str], None] | None, index: int, status: str) -> None:
    if callback is not None:
        callback(index, status)


def _scenario(
    idea: str,
    count: int,
    mode: PlaceMode,
    request: str,
    rewrite: Callable[[str], str],
) -> Scenario | None:
    scenario = validate_scenario(idea, rewrite(request), count=count, place_mode=mode)
    if scenario is None:
        scenario = validate_scenario(idea, rewrite(request), count=count, place_mode=mode)
    if scenario is None:
        return None
    shots = list(scenario.shots)
    attempts = 0
    while len(shots) < count and attempts < 2:
        attempts += 1
        extra = parse_shot(rewrite(one_shot_request(scenario.subject_block, mode)))
        if extra is None:
            continue
        candidate = _accept_extra(idea, scenario.subject_block, mode, scenario.place, shots, extra)
        if candidate is not None:
            shots.append(candidate)
    if len(shots) < 6:
        return None
    return Scenario(scenario.subject_block, scenario.descriptors, scenario.place, tuple(shots))


def _accept_extra(
    idea: str,
    subject: str,
    mode: PlaceMode,
    place: str,
    shots: list[Shot],
    extra: Shot,
) -> Shot | None:
    line = assemble_line(subject, extra, include_camera=True)
    if subject not in line or sorted(stated_ages(line)) != sorted(stated_ages(idea)):
        return None
    if extra.scene in {shot.scene for shot in shots}:
        return None
    if mode == "same":
        return extra if extra.place == place else None
    if mode == "different":
        if not extra.place.strip() or extra.place in {shot.place for shot in shots}:
            return None
        return extra
    assert_never(mode)


def _write_meta(
    folder: Path,
    idea: str,
    scenario: Scenario,
    style: str,
    mode: PlaceMode,
    lines: list[str],
    seeds: list[int],
    blocked: list[int],
    kept: list[Path],
) -> None:
    document = {
        "idea": idea,
        "subject_block": scenario.subject_block,
        "place_mode": mode,
        "style": style,
        "lines": lines,
        "seeds": seeds,
        "blocked": blocked,
    }
    (folder / "series.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    if not kept:
        return
    _write_sheet(folder / "overview.png", kept)
    with zipfile.ZipFile(folder / "series.zip", "w") as archive:
        for path in kept:
            archive.write(path, path.name)
        archive.write(folder / "overview.png", "overview.png")


def _write_sheet(path: Path, frames: list[Path]) -> None:
    images = [_read_png(frame) for frame in frames]
    cell = 64
    gap = 8
    width = len(images) * cell + (len(images) + 1) * gap
    height = cell + 2 * gap
    canvas = bytearray(b"\x20\x20\x29" * (width * height))
    for index, (src_w, src_h, src) in enumerate(images):
        left = gap + index * (cell + gap)
        for y in range(cell):
            for x in range(cell):
                sx = min(src_w - 1, x * src_w // cell)
                sy = min(src_h - 1, y * src_h // cell)
                pixel = src[(sy * src_w + sx) * 3:(sy * src_w + sx) * 3 + 3]
                dest = ((gap + y) * width + left + x) * 3
                canvas[dest:dest + 3] = pixel
    _write_png(path, width, height, bytes(canvas))


def _read_png(path: Path) -> tuple[int, int, bytes]:
    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("overview sheet needs a png frame")
    pos = 8
    width = height = 0
    compressed = b""
    while pos + 8 <= len(raw):
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        tag = raw[pos + 4:pos + 8]
        data = raw[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
        elif tag == b"IDAT":
            compressed += data
        elif tag == b"IEND":
            break
    inflated = zlib.decompress(compressed)
    rows = []
    stride = width * 3
    cursor = 0
    for _ in range(height):
        cursor += 1
        rows.append(inflated[cursor:cursor + stride])
        cursor += stride
    return width, height, b"".join(rows)


def _write_png(path: Path, width: int, height: int, rgb: bytes) -> None:
    if len(rgb) != width * height * 3:
        raise ValueError("png bytes do not match the frame size")

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + rgb[y * width * 3:(y + 1) * width * 3] for y in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
