"""Run every entry of a batch manifest through the phase-batched series pipeline.

The phases are the ones a single series uses, run once across all entries:

1. Every prompt (or shot line) and every negative prompt is checked like the
   prompt before anything else. A refused entry is recorded and draws nothing.
   The other entries go on.
2. One scenario writer session writes every prompt entry, then leaves the card.
   A shot entry needs no writer.
3. Every shot line of every entry is checked, then encoded in one text-encoder
   session.
4. The picture pipeline loads once, renders every entry, and decodes every
   entry in one VAE session. It stays loaded for the whole batch.
5. Every picture is checked before it is saved. Nothing unchecked is written.

A stop that belongs to one entry ends that entry only. A check that cannot
finish stops every write after it, in every entry. An unexpected error ends the
entry it happened in, or every entry that has no result yet when it is not tied
to one entry. No entry is left without a result.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from canvas.core.formats import OutputFormat
from canvas.infra import cuda_health
from canvas.infra.errors import note_error, note_timing
from canvas.infra.phase_vram import Phase, ResidualVram, start_phase
from canvas.infra.stage_timer import StageTimes, set_current
from canvas.models import hot_tier as hot
from canvas.models.runtime import pipeline_scope
from canvas.models.user_lora import apply_trigger, rewrite_alias
from canvas.pipeline.bf16_schedule import VramRefused, is_out_of_memory
from canvas.pipeline.device_guard import DeviceRefused
from canvas.pipeline.photoshoot import Scenario, scenario_request
from canvas.pipeline.series import (
    DrawFrame,
    SerialGate,
    SeriesResult,
    Source,
    _CheckPreload,
    _Plan,
    _release_writer,
    _stop,
    cook_plans,
    free_writer,
    place_mode_of,
    plan_series,
    review_plans,
    screen_idea,
    series_strength,
    shot_scenario,
    write_scenario,
)
from canvas.safety.gate import AgeGate, SafetyUnavailable


@dataclass(frozen=True)
class BatchEntry:
    """One manifest entry after validation.

    Exactly one of ``prompt`` and ``shot`` is set. ``name`` is already a safe
    folder name, unique within the batch.
    """

    name: str
    prompt: str
    shot: str
    negative: str
    seed: int
    count: int

    @property
    def text(self) -> str:
        """The prompt or the shot line, whichever the entry has."""
        return self.prompt or self.shot


@dataclass(frozen=True)
class EntryResult:
    """What one entry produced. ``scenario`` is None when the entry stopped before planning."""

    entry: BatchEntry
    folder: Path
    result: SeriesResult
    scenario: Scenario | None


def run_batch(
    *,
    entries: list[BatchEntry],
    folder: Path,
    style: str,
    place_mode: str,
    adult: bool,
    age_gate: AgeGate,
    rewrite: Callable[[str], str],
    frame_for: Callable[[BatchEntry], DrawFrame],
    container: OutputFormat | None = None,
    timings: StageTimes | None = None,
) -> list[EntryResult]:
    """Run every entry into its own subfolder of ``folder``. One result per entry, in order.

    Args:
        entries: Validated entries.
        folder: Batch output folder. Each entry writes into ``folder / entry.name``.
        style: Scenario style: photo, cinematic, or portrait.
        place_mode: ``same`` or ``different``.
        adult: Use the Expert adult-content prompt policy.
        age_gate: The gate every prompt, negative prompt and picture goes through.
        rewrite: One scenario writer session for the whole batch. It is released once.
        frame_for: The frame callback for one entry. Each entry carries its own negative.
        container: Export format. PNG when omitted.
        timings: Per-stage seconds. Without it one ``timing`` line goes to the log.
    """
    clock = timings if timings is not None else StageTimes()
    previous_clock = set_current(clock)
    prep = start_phase("prep")
    gate = SerialGate.wrap(age_gate)
    results: dict[int, SeriesResult] = {}
    scenarios: dict[int, Scenario] = {}
    try:
        # One RAM plan and one pipeline for every entry of the batch.
        with hot.job_scope(), pipeline_scope():
            hot.note_tier()
            _run(
                entries, folder, style, place_mode, adult, gate, rewrite, frame_for, container, clock,
                prep, results, scenarios,
            )
    except DeviceRefused as exc:
        note_error(exc.code)
        _fill(entries, results, SeriesResult("error", str(exc), (), None, exc.code))
    except ResidualVram as exc:
        note_error(exc.code)
        _fill(entries, results, SeriesResult("error", str(VramRefused()), (), None, exc.code))
    except Exception as exc:
        _fill(entries, results, _internal(exc))
    finally:
        _release_writer(rewrite)
        prep.finish()
        set_current(previous_clock)
        if timings is None:
            note_timing(clock.log_line())
    if len(results) < len(entries):
        _fill(entries, results, _internal(None))
    if cuda_health.lost():
        # Every entry that failed after the CUDA context broke reports the GPU, not a side effect.
        vram = SeriesResult("error", str(VramRefused()), (), None, "vram")
        for index, result in results.items():
            if result.stopped == "error" and not result.kept:
                results[index] = vram
    return [
        EntryResult(entry, folder / entry.name, results[index], scenarios.get(index))
        for index, entry in enumerate(entries)
    ]


def _run(
    entries: list[BatchEntry], folder: Path, style: str, place_mode: str, adult: bool, gate: AgeGate,
    rewrite: Callable[[str], str], frame_for: Callable[[BatchEntry], DrawFrame],
    container: OutputFormat | None, clock: StageTimes, prep: Phase,
    results: dict[int, SeriesResult], scenarios: dict[int, Scenario],
) -> None:
    mode = place_mode_of(place_mode)
    if mode is None:
        _fill(entries, results, _stop("scenario", "", "scenario.format"))
        return
    # One draw function per entry, made once. It carries the entry's LoRA (its ``lora`` attribute).
    frames: dict[int, DrawFrame] = {}

    def frame_of(index: int) -> DrawFrame:
        if index not in frames:
            frames[index] = frame_for(entries[index])
        return frames[index]

    # 1. Every prompt and every negative is checked before the writer and before any picture.
    for index, entry in enumerate(entries):
        if entry.prompt and scenario_request(entry.prompt, entry.count, style, mode, adult=adult) is None:
            results[index] = _stop("scenario", "", "scenario.format")
            continue
        # The gate sees the text with the alias swapped for the trigger and the trigger led in,
        # the same words the writer and the encoder get.
        try:
            lora = getattr(frame_of(index), "lora", None)
        except (DeviceRefused, ResidualVram, SafetyUnavailable):
            raise
        except Exception as exc:
            results[index] = _internal(exc)
            continue
        refused = screen_idea(gate, apply_trigger(entry.text, lora), entry.negative, adult, clock)
        if refused is not None:
            results[index] = refused
    # 2. One writer session for every prompt entry.
    written: dict[int, tuple[Scenario, Source]] = {}
    try:
        for index, entry in enumerate(entries):
            if index in results:
                continue
            if entry.shot:
                shot = shot_scenario(entry.shot, entry.count, entry.seed)
                if shot is None:
                    results[index] = _stop("scenario", "", "scenario.format")
                else:
                    written[index] = (shot, "")
                continue
            try:
                scenario, fail_code, source = write_scenario(
                    rewrite_alias(entry.prompt, getattr(frame_of(index), "lora", None)), entry.count, style, mode, rewrite, adult, entry.seed, clock,
                )
            except (DeviceRefused, ResidualVram, SafetyUnavailable):
                raise
            except Exception as exc:
                results[index] = _internal(exc)
                continue
            if scenario is None or len(scenario.shots) < entry.count:
                results[index] = _stop("scenario", "", fail_code if scenario is None else "scenario.count")
                continue
            written[index] = (scenario, source)
    finally:
        free_writer(rewrite)
    if not written:
        return
    # 3. The picture checks are read after the writer is gone, as in a single series.
    preload = _CheckPreload.open(gate)
    try:
        warm_error = preload.ready_error()
        if warm_error is not None:
            _fill(entries, results, _stop("error", warm_error.user_message, f"check.{warm_error.code}"))
            return
        plans: list[tuple[int, _Plan]] = []
        for index, (scenario, source) in written.items():
            entry = entries[index]

            def keep(found: Scenario, at: int = index) -> None:
                scenarios[at] = found

            try:
                planned = plan_series(
                    gate=gate, idea=entry.text, scenario=scenario, source=source, style=style, mode=mode,
                    seed=entry.seed, strength=series_strength(entry.text, mode, None), adult=adult,
                    include_camera=lambda: True, line_for=None, clock=clock, on_scenario=keep, on_status=None,
                    cancelled=lambda: False, folder=folder / entry.name, draw_frame=frame_of(index),
                )
            except (DeviceRefused, ResidualVram, SafetyUnavailable):
                raise
            except Exception as exc:
                results[index] = _internal(exc)
                continue
            if isinstance(planned, SeriesResult):
                results[index] = planned
            else:
                plans.append((index, planned))
        if not plans:
            return
        # 4. Render and decode every entry on one loaded pipeline.
        cook_plans([plan for _index, plan in plans], gate, lambda: False, clock, prep)
        load_error = preload.error()
        if load_error is not None:
            _fill(entries, results, _stop("error", load_error.user_message, f"check.{load_error.code}"))
            return
        # 5. Every picture is checked before it is saved.
        reviewed = review_plans([plan for _index, plan in plans], gate, container, clock)
        for (index, _plan), result in zip(plans, reviewed, strict=True):
            results[index] = result
    finally:
        preload.close()


def _fill(entries: list[BatchEntry], results: dict[int, SeriesResult], result: SeriesResult) -> None:
    """Give every entry that has no result yet this one."""
    for index in range(len(entries)):
        results.setdefault(index, result)


def _internal(exc: Exception | None) -> SeriesResult:
    """The result for an entry that ended on an unexpected error. The reason names the error type.

    A GPU memory failure, or any error once the CUDA context is lost, is a ``vram`` result instead.
    """
    if exc is not None and cuda_health.is_cuda_fault(exc):
        cuda_health.mark_lost(exc)
    if cuda_health.lost() or isinstance(exc, (VramRefused, MemoryError)) or (exc is not None and is_out_of_memory(exc)):
        note_error("vram")
        return SeriesResult("error", str(VramRefused()), (), None, "vram")
    kind = type(exc).__name__ if exc is not None else "no result"
    note_error("internal")
    return SeriesResult("error", f"The entry failed with an unexpected error ({kind}).", (), None, "internal")
