"""Run one Photoshoot series: scenario, gate, then one frame at a time."""
from __future__ import annotations

import json
import struct
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from canvas.app.i18n import text
from canvas.app.shell_run import RunStopped
from canvas.core.formats import OutputFormat, extension, save_image
from canvas.infra.errors import note_debug, note_error
from canvas.pipeline.enhance import mask_numbers
from canvas.pipeline.adult_direction import ages_allowed, with_adult_anchor, youth_cues
from canvas.pipeline.photoshoot import (
    FORMAT_REMINDER,
    SCENARIO_CHUNK,
    PlaceMode,
    Scenario,
    Shot,
    accept_shot,
    assemble_line,
    planned_shots,
    read_drafts,
    reply_declined,
    scenario_request,
)
from canvas.safety.denylist import scan
from canvas.safety.gate import AgeGate, SafetyError
from canvas.safety.image import RawImage, SafeImage

StopCode = Literal["", "idea", "scenario", "cancelled", "error"]
Source = Literal["", "writer", "mixed", "planned"]
_RAW_LIMIT = 6000


class SeriesFrame(Protocol):
    """Pixels a frame callback can return. The gate reviews a copy."""

    width: int
    height: int

    def to_bytes(self) -> bytes:
        """RGB bytes, three per pixel."""


@dataclass(frozen=True)
class SeriesResult:
    """SeriesResult."""

    stopped: StopCode
    reason: str
    kept: tuple[Path, ...]
    folder: Path | None
    code: str = ""
    source: Source = ""


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
    container: OutputFormat | None = None,
) -> SeriesResult:
    """Run series."""
    if place_mode == "same":
        mode: PlaceMode = "same"
    elif place_mode == "different":
        mode = "different"
    else:
        return _stop("scenario", "", "scenario.format")
    request = scenario_request(idea, count, style, mode, adult=adult)
    if request is None:
        return _stop("scenario", "", "scenario.format")
    reason = _refuse_prompt(age_gate, idea)
    if reason:
        return _stop("idea", reason, "idea.policy")
    if adult and not ages_allowed(idea):
        return _stop("idea", "", "idea.minor")
    scenario, fail_code, source = _scenario(idea, count, style, mode, rewrite, adult=adult)
    if scenario is None or len(scenario.shots) < count:
        return _stop("scenario", "", fail_code if scenario is None else "scenario.count")
    lines = [assemble_line(scenario.subject_block, shot, include_camera=True) for shot in scenario.shots]
    refused = [_refuse_prompt(age_gate, _with_idea(idea, line)) for line in lines]
    hard = [reason for line, reason in zip(lines, refused, strict=True) if reason and scan(line).minor]
    if hard or all(refused):
        code = "scenario.minor" if hard else "scenario.policy"
        return _stop("scenario", (hard or refused)[0] or "", code, source)
    if adult and not ages_allowed("\n".join(lines)):
        return _stop("scenario", "", "scenario.minor", source)
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
            return SeriesResult("cancelled", "", tuple(kept), folder, "", source)
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
        shot_reason = refused[index] or (
            None if line == lines[index] else _refuse_prompt(age_gate, _with_idea(idea, line))
        )
        if shot_reason:
            blocked.append(index)
            _status(on_status, index, "blocked")
            continue
        _status(on_status, index, "drawing")
        if cancelled():
            _write_meta(folder, idea, scenario, style, mode, saved_lines, seeds, blocked, kept)
            return SeriesResult("cancelled", "", tuple(kept), folder, "", source)
        try:
            image = draw_frame(line, seed + index, hero, strength)
        except RunStopped as exc:
            blocked.append(index)
            _status(on_status, index, "blocked")
            if not exc.unavailable:
                continue
            _write_meta(folder, idea, scenario, style, mode, saved_lines, seeds, blocked, kept)
            return SeriesResult(
                "error",
                exc.message,
                tuple(kept),
                folder,
                exc.code or "internal",
                source,
            )
        released = None if image is None else _released_frame(age_gate, image, line)
        if released is None:
            blocked.append(index)
            _status(on_status, index, "blocked")
            continue
        path = _store_frame(folder, index, released, container)
        kept.append(path)
        saved_lines.append(line)
        seeds.append(seed + index)
        if hero is None:
            hero = image
        _status(on_status, index, "kept")
        _write_meta(folder, idea, scenario, style, mode, saved_lines, seeds, blocked, kept)
    _write_meta(folder, idea, scenario, style, mode, saved_lines, seeds, blocked, kept)
    if not kept:
        note_error("not_produced")
        return SeriesResult("", "", (), folder, "not_produced", source)
    return SeriesResult("", "", tuple(kept), folder, "", source)


def _stop(stopped: StopCode, reason: str, code: str, source: Source = "") -> SeriesResult:
    note_error(code)
    return SeriesResult(stopped, reason, (), None, code, source)


def _with_idea(idea: str, line: str) -> str:
    return f"{idea}\n{line}"


def _refuse_prompt(age_gate: AgeGate, prompt: str) -> str | None:
    """Refuse prompt."""
    try:
        age_gate.review_prompt(prompt)
    except SafetyError as exc:
        return exc.user_message
    except Exception:
        return text("gate.unavailable")
    return None


@dataclass(frozen=True)
class _Released:
    width: int
    height: int
    data: bytes
    safe: SafeImage | None


def _released_frame(age_gate: AgeGate, image: SeriesFrame, line: str) -> _Released | None:
    """Released frame."""
    raw = RawImage(image.width, image.height, image.to_bytes())
    try:
        released = age_gate.review_image(raw, line)
    except SafetyError:
        return None
    except Exception:
        return None
    if isinstance(released, SafeImage):
        return _Released(released.width, released.height, released.data, released)
    return _Released(image.width, image.height, image.to_bytes(), None)


def _store_frame(folder: Path, index: int, released: _Released, container: OutputFormat | None) -> Path:
    if container is None or released.safe is None:
        path = folder / f"{index + 1:02}.png"
        _write_png(path, released.width, released.height, released.data)
        return path
    path = folder / f"{index + 1:02}{extension(container)}"
    save_image(released.safe, path, container)
    return path


def _status(callback: Callable[[int, str], None] | None, index: int, status: str) -> None:
    if callback is not None:
        callback(index, status)


def _scenario(
    idea: str,
    count: int,
    style: str,
    mode: PlaceMode,
    rewrite: Callable[[str], str],
    *,
    adult: bool,
) -> tuple[Scenario | None, str, Source]:
    """Shots from the writer, then planned shots for any frame it did not cover."""
    subject = idea.strip()
    if not subject:
        return None, "scenario.format", ""
    collected: list[Shot] = []
    place = ""
    replies: list[str] = []
    while len(collected) < count:
        ask = min(SCENARIO_CHUNK, count - len(collected))
        previous = tuple(assemble_line(subject, shot, include_camera=True) for shot in collected)
        request = scenario_request(idea, ask, style, mode, adult=adult, previous=previous)
        if request is None:
            return None, "scenario.format", ""
        fresh, place, chunk_replies = _read_chunk(idea, subject, ask, mode, place, tuple(collected), request, rewrite)
        replies.extend(chunk_replies)
        collected.extend(fresh)
        if not fresh:
            break
    written = len(collected)
    if written < count:
        note_error(_reply_code(tuple(replies)) if written == 0 else "scenario.count")
        collected.extend(planned_shots(idea, count - written, mode, place=place, taken=tuple(collected), adult=adult))
    if len(collected) < count:
        return None, "scenario.count", ""
    source: Source = "writer" if written >= count else ("planned" if written == 0 else "mixed")
    return Scenario(subject, (subject,), place, tuple(collected[:count])), "", source


def _read_chunk(
    idea: str,
    subject: str,
    ask: int,
    mode: PlaceMode,
    place: str,
    taken: tuple[Shot, ...],
    request: str,
    rewrite: Callable[[str], str],
) -> tuple[list[Shot], str, tuple[str, ...]]:
    """Accepted shots for one chunk, the series place, and the raw replies."""
    replies: list[str] = []
    fresh: list[Shot] = []
    for prompt in (request, f"{request}\n{FORMAT_REMINDER}"):
        raw = rewrite(prompt)
        replies.append(raw)
        fresh, place = _accept_reply(idea, subject, ask, mode, place, taken, raw)
        if len(fresh) < ask:
            note_debug("scenario.raw", mask_numbers(raw)[:_RAW_LIMIT])
        if fresh or read_drafts(raw):
            break
    return fresh, place, tuple(replies)


def _accept_reply(
    idea: str,
    subject: str,
    ask: int,
    mode: PlaceMode,
    place: str,
    taken: tuple[Shot, ...],
    raw: str,
) -> tuple[list[Shot], str]:
    drafts = read_drafts(raw)
    if mode == "same" and not place:
        place = next((draft.place.strip() for draft in drafts if draft.place.strip()), "")
    fresh: list[Shot] = []
    for draft in drafts:
        if len(fresh) >= ask:
            break
        shot = accept_shot(idea, subject, mode, place, taken + tuple(fresh), draft)
        if shot is not None:
            fresh.append(shot)
    return fresh, place


def _reply_code(replies: tuple[str, ...]) -> str:
    if replies and all("{" not in reply for reply in replies) and any(reply_declined(reply) for reply in replies):
        return "scenario.declined"
    return "scenario.format"


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
    if not kept or any(path.suffix.lower() != ".png" for path in kept):
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
