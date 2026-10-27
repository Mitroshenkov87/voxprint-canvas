"""Run one Photoshoot series in phases: prep, render, decode, then check.

Prep writes the scenario with one writer session, reviews every prompt, and
encodes every accepted line with the text encoder alone. Those embeddings stay
in RAM, and prep fails when its models leave more than 0.5 GB on the card.
Render loads the picture pipeline once and samples every frame while the
denoiser stays on the GPU. Decode turns every kept latent into pixels in one
VAE session. The output check runs after the pipeline leaves the GPU, one frame
at a time. RAM copies stay when the tier keeps them. A frame is written only
after it passes.
"""
from __future__ import annotations

import json
import threading
import struct
import time
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from canvas.app.i18n import text
from canvas.app.shell_run import RunStopped
from canvas.core.exchange import handoff_image
from canvas.core.formats import EXPORT_FORMAT, OutputFormat, extension
from canvas.infra.errors import note_crash, note_debug, note_error, note_timing
from canvas.infra.phase_vram import Phase, ResidualVram, phase, require_released, start_phase
from canvas.infra.stage_timer import StageTimes, log_detail, note_peak_mib, set_current, timed
from canvas.infra.vram import peak_allocated_bytes
from canvas.models import hot_tier as hot
from canvas.models.runtime import (
    clear_series_embeds,
    decode_phase,
    park_pipeline,
    pipeline_scope,
    render_phase,
)
from canvas.pipeline.bf16_schedule import VramRefused
from canvas.pipeline.device_guard import DeviceRefused
from canvas.pipeline.enhance import mask_numbers, stated_ages
from canvas.pipeline.gpu_release import release_cuda_cache, release_gate_weights
from canvas.pipeline.adult_direction import ages_allowed, with_adult_anchor, youth_cues
from canvas.pipeline.frame_prompt import frame_identity, log_frame_prompt, strip_series_words, with_choices
from canvas.pipeline.photoshoot import (
    FORMAT_REMINDER,
    SCENARIO_CHUNK,
    PlaceMode,
    Scenario,
    Shot,
    accept_shot,
    assemble_line,
    describes_person,
    fixed_appearance,
    planned_shots,
    single_frame,
    read_drafts,
    reply_declined,
    scenario_request,
)
from canvas.safety.denylist import scan
from canvas.safety.gate import (
    AgeGate,
    GenerationBlocked,
    SafetyError,
    SafetyUnavailable,
    StylisedBlocked,
    YouthContextBlocked,
)
from canvas.safety.image import RawImage, SafeImage

StopCode = Literal["", "idea", "scenario", "cancelled", "error"]
Source = Literal["", "writer", "mixed", "planned"]
REFERENCE_STRENGTH = 0.80
REFERENCE_MIN = 0.75
_RAW_LIMIT = 6000
_PICTURE_CODES = frozenset({"image", "input"})


class SeriesFrame(Protocol):
    """Pixels a frame callback can return. The gate reviews a copy."""

    width: int
    height: int

    def to_bytes(self) -> bytes:
        """RGB bytes, three per pixel."""


@dataclass(frozen=True)
class _CookShot:
    """One frame whose prompt has already been accepted."""

    index: int
    line: str
    seed: int


@dataclass(frozen=True)
class _Drawn:
    """Pixels held in memory until the output check."""

    index: int
    line: str
    seed: int
    image: SeriesFrame


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
    strength: float | None,
    include_camera: Callable[[], bool],
    rewrite: Callable[[str], str],
    age_gate: AgeGate,
    draw_frame: Callable[[str, int, SeriesFrame | None, float | None], SeriesFrame | None],
    cancelled: Callable[[], bool],
    folder: Path,
    on_scenario: Callable[[Scenario], None] | None = None,
    line_for: Callable[[int], str] | None = None,
    on_status: Callable[[int, str], None] | None = None,
    adult: bool = False,
    container: OutputFormat | None = None,
    timings: StageTimes | None = None,
) -> SeriesResult:
    """Run series.

    ``timings`` collects per-stage seconds (stage names only). Without it the
    series keeps its own and writes one ``timing`` line to the generic log.
    """
    clock = timings if timings is not None else StageTimes()
    previous_clock = set_current(clock)
    noted: list[bool] = []
    prep = start_phase("prep")
    try:
        return _run_series(
            idea=idea, count=count, style=style, place_mode=place_mode, seed=seed, strength=strength,
            include_camera=include_camera, rewrite=rewrite, age_gate=age_gate, draw_frame=draw_frame,
            cancelled=cancelled, folder=folder, on_scenario=on_scenario, line_for=line_for,
            on_status=on_status, adult=adult, container=container, clock=clock, own_clock=timings is None,
            noted=noted, prep=prep,
        )
    except DeviceRefused as exc:
        note_error(exc.code)
        if timings is None and not noted:
            note_timing(clock.log_line())
        return SeriesResult("error", str(exc), (), None, exc.code)
    except ResidualVram as exc:
        note_error(exc.code)
        if timings is None and not noted:
            note_timing(clock.log_line())
        return SeriesResult("error", str(VramRefused()), (), None, exc.code)
    finally:
        _release_writer(rewrite)
        prep.finish()
        set_current(previous_clock)


def _run_series(
    *, idea: str, count: int, style: str, place_mode: str, seed: int, strength: float | None,
    include_camera: Callable[[], bool], rewrite: Callable[[str], str], age_gate: AgeGate,
    draw_frame: Callable[[str, int, SeriesFrame | None, float | None], SeriesFrame | None],
    cancelled: Callable[[], bool], folder: Path, on_scenario: Callable[[Scenario], None] | None,
    line_for: Callable[[int], str] | None, on_status: Callable[[int, str], None] | None,
    adult: bool, container: OutputFormat | None, clock: StageTimes, own_clock: bool,
    noted: list[bool], prep: Phase,
) -> SeriesResult:
    hot.note_tier()
    gate = SerialGate.wrap(age_gate)
    if place_mode == "same":
        mode: PlaceMode = "same"
    elif place_mode == "different":
        mode = "different"
    else:
        return _stop("scenario", "", "scenario.format")
    request = scenario_request(idea, count, style, mode, adult=adult)
    if request is None:
        return _stop("scenario", "", "scenario.format")
    # The prompt check is the part of prep that has to finish before the writer.
    # Picture-check weights are read SSD to RAM on a side thread during the
    # writer and the sampler. They stay off the GPU until the output check.
    with clock.measure("prep"):
        reason = _refuse_prompt(gate, idea)
        # The writer needs the GPU to itself. Anything the check put on the card is dropped first.
        release_gate_weights(gate)
    if reason:
        return _stop("idea", reason, "idea.policy")
    if adult and not ages_allowed(idea):
        return _stop("idea", "", "idea.minor")
    preload = _CheckPreload.open(gate)
    warm_error = preload.ready_error()
    if warm_error is not None:
        return _stop("error", warm_error.user_message, f"check.{warm_error.code}")
    appearance = fixed_appearance(idea, seed)
    # None is the caller's default. A same-place person series then follows the
    # first frame loosely enough that the pose can still change. With different
    # places every shot is drawn from its own line, so its place and pose lead.
    anchored = strength is None and mode == "same" and describes_person(idea)
    series_strength = REFERENCE_STRENGTH if anchored else strength
    with clock.measure("scenario"):
        try:
            scenario, fail_code, source = _scenario(
                idea, count, style, mode, rewrite, adult=adult, appearance=appearance, seed=seed,
            )
        except Exception:
            preload.close()
            raise
        finally:
            # Every chunk shared one writer. It leaves the card before the prompt checks.
            _release_writer(rewrite)
            hot.deactivate("writer")
    if scenario is None or len(scenario.shots) < count:
        preload.close()
        return _stop("scenario", "", fail_code if scenario is None else "scenario.count")
    lines = [
        single_frame(strip_series_words(assemble_line(
            scenario.subject_block, shot, include_camera=True, index=index, seed=seed,
        )))
        for index, shot in enumerate(scenario.shots)
    ]
    try:
        with clock.measure("prompt_check"):
            verdicts = [_prompt_verdict(gate, _with_idea(idea, line)) for line in lines]
    except Exception:
        preload.close()
        raise
    finally:
        # The picture model is next. GPU weights from the check are dropped first.
        release_gate_weights(gate)
    refused = [_verdict_message(verdict) for verdict in verdicts]
    hard = [
        reason for line, reason, verdict in zip(lines, refused, verdicts, strict=True)
        if reason and (_ends_series(verdict) or scan(line).minor)
    ]
    if hard or all(refused):
        code = "scenario.minor" if hard else "scenario.policy"
        preload.close()
        return _stop("scenario", (hard or refused)[0] or "", code, source)
    if adult and not ages_allowed("\n".join(lines)):
        preload.close()
        return _stop("scenario", "", "scenario.minor", source)
    if on_scenario is not None:
        on_scenario(scenario)
    folder.mkdir(parents=True, exist_ok=True)
    kept: list[Path] = []
    blocked: list[int] = []
    saved_lines: list[str] = []
    seeds: list[int] = []
    accepted, prep_stop, cancelled_prep = _accepted_shots(
        scenario, lines, verdicts, seed, idea, adult, include_camera, line_for, gate, clock, blocked,
        on_status, cancelled,
    )
    if cancelled_prep:
        release_gate_weights(gate)
        preload.close()
        _write_meta(folder, idea, scenario, style, mode, saved_lines, seeds, blocked, kept)
        if own_clock:
            note_timing(clock.log_line())
        return SeriesResult("cancelled", "", (), folder, "", source)
    try:
        try:
            _encode_lines(draw_frame, accepted, clock)
            # Prompt models are finished. The picture pipeline is the next thing on the GPU.
            release_gate_weights(gate)
            prep.finish()
            require_released("prep")
            drawn, cook_error, cancelled_run = _cook_frames(
                accepted, series_strength, draw_frame, cancelled, clock, blocked, on_status,
            )
        finally:
            park_pipeline()
            clear_series_embeds()
        release_gate_weights(gate)
        # Sampling overlapped the SSD read. The check still runs in order, and
        # nothing is written until it passes.
        load_error = preload.error()
        if load_error is not None:
            return _stop("error", load_error.user_message, f"check.{load_error.code}")
        check_code, check_reason, ending = _review_frames(
            drawn, gate, folder, container, clock, blocked, kept, saved_lines, seeds, on_status,
        )
        _write_meta(folder, idea, scenario, style, mode, saved_lines, seeds, blocked, kept)
        if cancelled_run:
            return SeriesResult("cancelled", "", tuple(kept), folder, "", source)
        if cook_error is not None and cook_error.unavailable:
            return SeriesResult(
                "error", cook_error.message, tuple(kept), folder, cook_error.code or "internal", source,
            )
        if cook_error is not None and (_ends_series(cook_error) or cook_error.code == "prompt"):
            note_error("scenario.minor")
            return SeriesResult("scenario", cook_error.message, tuple(kept), folder, "scenario.minor", source)
        if check_code:
            note_error(check_code)
            return SeriesResult("error", check_reason, tuple(kept), folder, check_code, source)
        if ending is not None:
            note_error("scenario.minor")
            return SeriesResult("scenario", ending, tuple(kept), folder, "scenario.minor", source)
        if prep_stop is not None:
            note_error("scenario.minor")
            return SeriesResult(
                "scenario", _verdict_message(prep_stop) or "", tuple(kept), folder, "scenario.minor", source,
            )
        if not kept:
            note_error("not_produced")
            return SeriesResult("", "", (), folder, "not_produced", source)
        return SeriesResult("", "", tuple(kept), folder, "", source)
    finally:
        preload.close()
        if own_clock:
            note_timing(clock.log_line())
            noted.append(True)


def _remember_vram_peak() -> None:
    """Record the CUDA allocation peak on the timing line."""
    peak = peak_allocated_bytes()
    if not peak or peak <= 0:
        return
    note_peak_mib(peak // (1024 * 1024))


def _accepted_shots(
    scenario: Scenario,
    lines: list[str],
    verdicts: list[Exception | None],
    seed: int,
    idea: str,
    adult: bool,
    include_camera: Callable[[], bool],
    line_for: Callable[[int], str] | None,
    gate: AgeGate,
    clock: StageTimes,
    blocked: list[int],
    on_status: Callable[[int, str], None] | None,
    cancelled: Callable[[], bool],
) -> tuple[list[_CookShot], Exception | None, bool]:
    """Every shot line and its prompt verdict, before the picture model loads.

    A series-ending prompt verdict keeps the shots accepted before it and stops
    the list there. A refusal that does not end the series drops that shot.
    """
    accepted: list[_CookShot] = []
    for index, shot in enumerate(scenario.shots):
        if cancelled():
            return accepted, None, True
        line = (
            assemble_line(
                scenario.subject_block, shot, include_camera=include_camera(), index=index, seed=seed,
            )
            if line_for is None
            else single_frame(strip_series_words(with_choices(line_for(index), index, seed)))
        )
        line = single_frame(strip_series_words(line))
        # The stored subject can still name the series and list every outfit.
        # The photograph only has to keep the appearance once those words are gone.
        identity = frame_identity(scenario.subject_block)
        if (identity and identity not in line) or (adult and youth_cues(line)):
            blocked.append(index)
            _status(on_status, index, "blocked")
            continue
        if adult:
            line = with_adult_anchor(line)
        if line == lines[index]:
            verdict = verdicts[index]
        else:
            with clock.measure("prompt_check"):
                verdict = _prompt_verdict(gate, _with_idea(idea, line))
        if _verdict_message(verdict):
            blocked.append(index)
            _status(on_status, index, "blocked")
            if _ends_series(verdict) or scan(line).minor:
                return accepted, verdict, False
            continue
        log_frame_prompt(line)
        accepted.append(_CookShot(index, line, seed + index))
    return accepted, None, False


def _release_writer(rewrite: Callable[[str], str]) -> None:
    """Close the writer session behind ``rewrite``. A plain callable has none."""
    release = getattr(rewrite, "release", None)
    if callable(release):
        release()


def _encode_lines(
    draw_frame: Callable[[str, int, SeriesFrame | None, float | None], SeriesFrame | None],
    accepted: list[_CookShot],
    clock: StageTimes,
) -> None:
    """Encode every accepted line with the text encoder alone. The embeddings stay in RAM."""
    prepare = getattr(draw_frame, "prepare", None)
    if not callable(prepare) or not accepted:
        return
    with clock.measure("encode_all"):
        prepare([shot.line for shot in accepted])


def _cook_frames(
    accepted: list[_CookShot],
    series_strength: float | None,
    draw_frame: Callable[[str, int, SeriesFrame | None, float | None], SeriesFrame | None],
    cancelled: Callable[[], bool],
    clock: StageTimes,
    blocked: list[int],
    on_status: Callable[[int, str], None] | None,
) -> tuple[list[_Drawn], RunStopped | None, bool]:
    """Render every accepted frame on one loaded pipeline, then decode them together.

    With a ``sample`` hook that returns True, the render phase only samples and
    the decode phase turns each kept latent into pixels. A series that follows its first frame
    draws that frame whole, because the later frames start from its pixels.
    Without the hook every frame is drawn whole in the render phase.
    """
    hero: SeriesFrame | None = None
    drawn: list[_Drawn] = []
    sampled: list[_CookShot] = []
    active = style_reference(series_strength)
    sample = getattr(draw_frame, "sample", None)
    stop: RunStopped | None = None
    stopped = False
    with pipeline_scope():
        with phase("render"), render_phase():
            for shot in accepted:
                if cancelled():
                    stopped = True
                    break
                _status(on_status, shot.index, "drawing")
                if cancelled():
                    stopped = True
                    break
                whole = not callable(sample) or (active is not None and hero is None)
                reference = hero if active is not None else None
                try:
                    started = time.perf_counter()
                    with clock.measure("render"):
                        image = None
                        if not whole and callable(sample):
                            # A hook that kept nothing leaves this frame to be drawn whole.
                            whole = sample(shot.line, shot.seed, reference, active) is not True
                        if whole:
                            image = draw_frame(shot.line, shot.seed, reference, active)
                    _remember_vram_peak()
                    _log_frame("render", shot.index, started)
                except RunStopped as exc:
                    _remember_vram_peak()
                    blocked.append(shot.index)
                    _status(on_status, shot.index, "blocked")
                    if exc.unavailable or _ends_series(exc) or (exc.code == "prompt" and scan(shot.line).minor):
                        stop = exc
                        break
                    continue
                if not whole:
                    sampled.append(shot)
                    continue
                if image is None:
                    blocked.append(shot.index)
                    _status(on_status, shot.index, "blocked")
                    continue
                if hero is None:
                    hero = image
                drawn.append(_Drawn(shot.index, shot.line, shot.seed, image))
        if sampled:
            decoded, decode_stop, decode_cancelled = _decode_frames(
                sampled, hero, active, draw_frame, cancelled, clock, blocked, on_status,
            )
            drawn.extend(decoded)
            drawn.sort(key=lambda item: item.index)
            stop = stop or decode_stop
            stopped = stopped or decode_cancelled
    return drawn, stop, stopped


def _decode_frames(
    sampled: list[_CookShot],
    hero: SeriesFrame | None,
    active: float | None,
    draw_frame: Callable[[str, int, SeriesFrame | None, float | None], SeriesFrame | None],
    cancelled: Callable[[], bool],
    clock: StageTimes,
    blocked: list[int],
    on_status: Callable[[int, str], None] | None,
) -> tuple[list[_Drawn], RunStopped | None, bool]:
    """Turn each sampled frame into pixels in one VAE session.

    Each call repeats the frame's own arguments, so it finds the latent its
    sample left and does not sample again.
    """
    drawn: list[_Drawn] = []
    reference = hero if active is not None else None
    with phase("decode"), decode_phase():
        for shot in sampled:
            if cancelled():
                return drawn, None, True
            try:
                started = time.perf_counter()
                with clock.measure("decode"):
                    image = draw_frame(shot.line, shot.seed, reference, active)
                _remember_vram_peak()
                _log_frame("decode", shot.index, started)
            except RunStopped as exc:
                _remember_vram_peak()
                blocked.append(shot.index)
                _status(on_status, shot.index, "blocked")
                if exc.unavailable or _ends_series(exc) or (exc.code == "prompt" and scan(shot.line).minor):
                    return drawn, exc, False
                continue
            if image is None:
                blocked.append(shot.index)
                _status(on_status, shot.index, "blocked")
                continue
            drawn.append(_Drawn(shot.index, shot.line, shot.seed, image))
    return drawn, None, False


def _log_frame(stage: str, index: int, started: float) -> None:
    """One per-frame timing line for the render or decode phase."""
    elapsed = max(0, int((time.perf_counter() - started) * 1000))
    log_detail("frame", stage, "cuda" if peak_allocated_bytes() is not None else "cpu", f"index={index + 1} ms={elapsed}")


def _review_frames(
    drawn: list[_Drawn],
    gate: AgeGate,
    folder: Path,
    container: OutputFormat | None,
    clock: StageTimes,
    blocked: list[int],
    kept: list[Path],
    saved_lines: list[str],
    seeds: list[int],
    on_status: Callable[[int, str], None] | None,
) -> tuple[str, str, str | None]:
    """Check every held frame in one batch, then write the ones that passed before a series stop.

    The picture pipeline has left the GPU. The check weights stay placed for
    the whole batch and leave the card once, before anything is written. A
    check that cannot finish is recorded once. Later frames are still checked
    and none of them are written.
    """
    verdicts = _check_frames(drawn, gate, clock)
    save_stop = False
    crash_noted = False
    check_code = ""
    check_reason = ""
    ending: str | None = None
    for item, released, refusal in verdicts:
        if released is None:
            blocked.append(item.index)
            _status(on_status, item.index, "blocked")
            if isinstance(refusal, SafetyUnavailable):
                if not crash_noted:
                    note_crash(f"check.{refusal.code}", refusal)
                    crash_noted = True
                if not check_code:
                    check_code = f"check.{refusal.code}"
                    check_reason = _verdict_message(refusal) or text("gate.unavailable")
                save_stop = True
                continue
            if _ends_series(refusal):
                save_stop = True
                if ending is None:
                    ending = _verdict_message(refusal) or ""
                continue
            continue
        if save_stop:
            blocked.append(item.index)
            _status(on_status, item.index, "blocked")
            continue
        with clock.measure("save"):
            path = _store_frame(folder, item.index, released, container)
        kept.append(path)
        saved_lines.append(item.line)
        seeds.append(item.seed)
        _status(on_status, item.index, "kept")
    return check_code, check_reason, ending


def _check_frames(
    drawn: list[_Drawn], gate: AgeGate, clock: StageTimes,
) -> list[tuple[_Drawn, _Released | None, Exception | None]]:
    """Every frame's verdict, in order. Nothing is written or shown here.

    The check weights leave VRAM once, after the last frame, also when a call raises.
    """
    verdicts: list[tuple[_Drawn, _Released | None, Exception | None]] = []
    try:
        with phase("output_check"):
            _release_check_cache()
            for item in drawn:
                started = time.perf_counter()
                with clock.measure("output_check"):
                    with timed("check"):
                        released, refusal = _released_frame(gate, item.image, item.line)
                verdicts.append((item, released, refusal))
                _log_frame("check", item.index, started)
    finally:
        release_gate_weights(gate)
    return verdicts


def _stated_age(idea: str) -> int | list[int] | None:
    """The age stated in the idea. Several ages stay a list. None when the idea states none."""
    stated = stated_ages(idea)
    if len(stated) == 1:
        return stated[0]
    if stated:
        return list(stated)
    return None


def style_reference(strength: float | None) -> float | None:
    """Loose style strength, or None when the frame is drawn from text.

    Args:
        strength: Requested series strength. Below ``REFERENCE_MIN`` draws from text.
    """
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        return None
    value = float(strength)
    if value < REFERENCE_MIN:
        return None
    return value


def _stop(stopped: StopCode, reason: str, code: str, source: Source = "") -> SeriesResult:
    note_error(code)
    return SeriesResult(stopped, reason, (), None, code, source)


def _stop_kept(
    stopped: StopCode,
    reason: str,
    code: str,
    source: Source,
    folder: Path,
    idea: str,
    scenario: Scenario,
    style: str,
    mode: PlaceMode,
    lines: list[str],
    seeds: list[int],
    blocked: list[int],
    kept: list[Path],
) -> SeriesResult:
    """Stop the series with frames already on disk. No later frame is drawn."""
    _write_meta(folder, idea, scenario, style, mode, lines, seeds, blocked, kept)
    note_error(code)
    return SeriesResult(stopped, reason, tuple(kept), folder, code, source)


def _with_idea(idea: str, line: str) -> str:
    return f"{idea}\n{line}"


_IMAGE_CHECK_NAMES = ("nsfw", "zeroshot", "faces", "ages", "fine")


class _CheckPreload:
    """SSD-to-RAM load of the picture checks, overlapped with the writer.

    Gates whose components have no ``load`` (the unit-test doubles) run
    ``warm`` on this thread, which is the old prep behaviour. A real stack
    loads on a side thread and stays on the CPU so the writer keeps the GPU.
    The output check still places and runs the models itself.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._error: SafetyUnavailable | None = None
        self._crash: Exception | None = None
        self._joined = False

    @classmethod
    def open(cls, gate: AgeGate) -> _CheckPreload:
        """Start the load, or run ``warm`` when this gate cannot load in the background."""
        job = cls()
        components = _image_loaders(gate)
        if not components:
            job._error = _warm_image_checks(gate)
            job._joined = True
            return job
        job._thread = threading.Thread(
            target=job._run, args=(components,), name="check-ram", daemon=True,
        )
        job._thread.start()
        return job

    def ready_error(self) -> SafetyUnavailable | None:
        """A ``warm`` failure. A background load is still running."""
        if self._thread is not None:
            return None
        return self._error

    def error(self) -> SafetyUnavailable | None:
        """Join, then the load failure. A crash from the thread is raised."""
        self.close()
        if self._crash is not None:
            crash = self._crash
            self._crash = None
            raise crash
        return self._error

    def close(self) -> None:
        """Join the side thread. A second call does nothing."""
        if self._joined:
            return
        self._joined = True
        thread = self._thread
        if thread is not None:
            thread.join()

    def _run(self, components: tuple[object, ...]) -> None:
        started = time.perf_counter()
        # (component, had _forced, previous value). Restored before the output check.
        saved: list[tuple[object, bool, object]] = []
        try:
            for component in components:
                had = hasattr(component, "_forced")
                previous = getattr(component, "_forced", None)
                try:
                    setattr(component, "_forced", "cpu")
                except (AttributeError, TypeError):
                    pass
                else:
                    saved.append((component, had, previous))
                load = getattr(component, "load", None)
                if callable(load):
                    load()
        except SafetyUnavailable as exc:
            self._error = exc
        except Exception as exc:
            self._crash = exc
        finally:
            for component, had, previous in saved:
                try:
                    if had:
                        setattr(component, "_forced", previous)
                    else:
                        delattr(component, "_forced")
                except (AttributeError, TypeError):
                    pass
            elapsed = max(0, int((time.perf_counter() - started) * 1000))
            log_detail("check_load", "preload", "cpu", f"ms={elapsed} models={len(components)}")


def _image_loaders(gate: AgeGate) -> tuple[object, ...]:
    """Picture-check components that can read their own weights. Guard stays put."""
    inner = gate._gate if isinstance(gate, SerialGate) else gate
    stack = getattr(inner, "_stack", None)
    if stack is None:
        return ()
    found: list[object] = []
    for name in _IMAGE_CHECK_NAMES:
        component = getattr(stack, name, None)
        if component is not None and callable(getattr(component, "load", None)):
            found.append(component)
    return tuple(found)


def _warm_image_checks(gate: AgeGate) -> SafetyUnavailable | None:
    """Load the picture checks into RAM. A frame then only moves them.

    The first output check was paying the SSD read. These models stay in the
    RAM tier, so prep does that read once. A load that fails stops the series
    before a frame is drawn: the check would have stopped it later.
    """
    warm = getattr(gate, "warm", None)
    if not callable(warm):
        return None
    try:
        warm()
    except SafetyUnavailable as exc:
        return exc
    return None


def _refuse_prompt(age_gate: AgeGate, prompt: str) -> str | None:
    """Refuse prompt."""
    return _verdict_message(_prompt_verdict(age_gate, prompt))


def _prompt_verdict(age_gate: AgeGate, prompt: str) -> Exception | None:
    """The stop a prompt review raised, or None when it may continue."""
    try:
        age_gate.review_prompt(prompt)
    except Exception as exc:
        return exc
    return None


def _verdict_message(verdict: BaseException | None) -> str | None:
    if verdict is None:
        return None
    if isinstance(verdict, SafetyError):
        return verdict.user_message
    return text("gate.unavailable")


def _ends_series(error: BaseException | None) -> bool:
    """True when this per-frame stop ends the series with ``scenario.minor``."""
    if isinstance(error, RunStopped):
        error = error.__cause__
    if not isinstance(error, GenerationBlocked):
        return False
    if error.code in _PICTURE_CODES:
        return True
    return error.code == "prompt" and not isinstance(error, (StylisedBlocked, YouthContextBlocked))


class SerialGate:
    """The same gate, one call at a time across the series threads.

    Every call goes to the wrapped gate unchanged. Only the concurrency differs.
    """

    def __init__(self, gate: AgeGate) -> None:
        self._gate = gate
        self._lock = threading.RLock()

    @classmethod
    def wrap(cls, gate: AgeGate) -> AgeGate:
        """The gate behind a lock. Typed as the gate: every call is passed through unchanged."""
        if isinstance(gate, SerialGate):
            return gate
        return cast(AgeGate, cls(gate))

    def __getattr__(self, name: str) -> object:
        attr = getattr(self._gate, name)
        if not callable(attr):
            return attr

        def call(*args: object, **kwargs: object) -> object:
            with self._lock:
                return attr(*args, **kwargs)

        return call


@dataclass(frozen=True)
class _Released:
    width: int
    height: int
    data: bytes
    safe: SafeImage | None


def _release_check_cache() -> None:
    """Return reserved GPU memory before the check asks how much is free.

    A decode that finished with a full reservation makes the device query wait.
    This is placement: the check still loads the same models and uses the same rule.
    """
    started = time.perf_counter()
    released = release_cuda_cache()
    elapsed = max(0, int((time.perf_counter() - started) * 1000))
    log_detail("check_cache", "empty", "cuda" if released else "cpu", f"ms={elapsed}")


def _released_frame(
    age_gate: AgeGate, image: SeriesFrame, line: str,
) -> tuple[_Released | None, Exception | None]:
    """The released frame, or None with the stop the review raised."""
    raw = RawImage(image.width, image.height, image.to_bytes())
    try:
        released = age_gate.review_image(raw, line)
    except Exception as exc:
        return None, exc
    if isinstance(released, SafeImage):
        return _Released(released.width, released.height, released.data, released), None
    return _Released(image.width, image.height, image.to_bytes(), None), None


def _store_frame(folder: Path, index: int, released: _Released, container: OutputFormat | None) -> Path:
    """Write one frame in the export format. PNG is the Forge default.

    PNG stays a filter-0 file so the contact sheet can read it. AVIF, JPEG and
    WebP go through the Forge hand-off. Nothing here runs before the check.
    """
    fmt = EXPORT_FORMAT if container is None else container
    with timed("encode"):
        if fmt is OutputFormat.PNG or released.safe is None:
            path = folder / f"{index + 1:02}.png"
            _write_png(path, released.width, released.height, released.data)
            return path
        path = folder / f"{index + 1:02}{extension(fmt)}"
        if released.safe is not None:
            handoff_image(released.safe, path, fmt)
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
    appearance: str = "",
    seed: int = 0,
) -> tuple[Scenario | None, str, Source]:
    """Shots from the writer, then planned shots for any shot it did not cover."""
    subject = idea.strip()
    if not subject:
        return None, "scenario.format", ""
    if appearance and appearance not in subject:
        subject = f"{appearance}, {subject}"
    collected: list[Shot] = []
    place = ""
    replies: list[str] = []
    while len(collected) < count:
        ask = min(SCENARIO_CHUNK, count - len(collected))
        previous = tuple(
            assemble_line(subject, shot, include_camera=True, index=index, seed=seed)
            for index, shot in enumerate(collected)
        )
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
    return Scenario(subject, (subject,), place, tuple(collected[:count]), appearance), "", source


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
        "appearance": scenario.appearance,
        "age": _stated_age(idea),
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
