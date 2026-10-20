"""Write one Photoshoot series from the command line."""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from voxprint_common.i18n import add_catalogs, tr_lang
from voxprint_common.ui.safety import adult_prompt_policy

from canvas.app.i18n import text
from canvas.app.info import ENLARGE_NAME
from canvas.app.shell_run import ShellRunner
from canvas.core.formats import DEFAULT_FORMAT, extension
from canvas.infra.enlarge import EnlargeClient
from canvas.infra.paths import models_dir
from canvas.models.modes import Mode, Tier
from canvas.models.registry import load_catalog
from canvas.pipeline.adult_direction import scenario_system
from canvas.pipeline.enhance import open_rewriter
from canvas.pipeline.photoshoot import MAX_FRAMES, SCENARIO_TOKENS, SYSTEM_PROMPT, Scenario, assemble_line
from canvas.infra.errors import note_timing
from canvas.infra.stage_timer import StageTimes, note_stage
from canvas.pipeline.schedule import last_image_seconds, reset_image_seconds
from canvas.pipeline.series import SeriesFrame, run_series
from canvas.safety.gate import AgeGate
from canvas.safety.image import RawImage
from canvas.safety.inputs import ImageRole, PendingInput
from canvas.models.user_lora import UserLora
from canvas.ui.shell_layout import PictureJob
from canvas.ui.styles import look_for

_LOCALES = Path(__file__).resolve().parents[1] / "ui" / "locales"
_SCENARIO_STYLES = frozenset({"photo", "cinematic", "portrait"})


def catalogs() -> None:
    """Load the Canvas UI catalogs so command output can use them."""
    add_catalogs(_LOCALES)


def reject_frames(count: int) -> dict[str, object]:
    """JSON body when ``--frames`` is outside 1..MAX_FRAMES.

    Args:
        count: The number the command received.
    """
    catalogs()
    return _body(
        ok=False,
        error="bad_frames",
        message=tr_lang("photoshoot.frames.invalid", "en", limit=MAX_FRAMES),
        frames=(),
        scenario=None,
        requested=count,
        made=0,
        shots=(),
    )


def model_rewrite(tier: Tier, adult: bool) -> Callable[[str], str]:
    """The local scenario writer. The model stays closed between calls.

    Args:
        tier: The tier chosen for this machine.
        adult: Use the Expert adult-content prompt policy.
    """
    catalog = load_catalog()

    def rewrite(prompt: str) -> str:
        rewriter = open_rewriter(
            catalog, tier, models_dir(),
            lambda message: print(message, file=sys.stderr),
            max_new_tokens=SCENARIO_TOKENS,
        )
        try:
            return rewriter.complete(prompt, system=scenario_system(SYSTEM_PROMPT, adult))
        finally:
            rewriter.close()

    return rewrite


def model_generate(
    age_gate: AgeGate, tier: Tier, style: str, adult: bool,
    lora: UserLora | None = None, lora_scale: float = 0.7,
) -> Callable[[str, int, SeriesFrame | None, float | None], SeriesFrame | None]:
    """One frame through the same runner the window uses.

    Args:
        age_gate: The gate the series also calls.
        tier: The tier chosen for this machine.
        style: Look id from the app style row.
        adult: Use the Expert adult-content prompt policy.
        lora: The LoRA for every frame of this series. None uses the base model.
        lora_scale: Strength for that LoRA.
    """
    runner = ShellRunner(gate=age_gate, tier=tier, enlarge=EnlargeClient())
    runner.bound_lora = lora
    look = look_for(style)
    policy = adult_prompt_policy(allowed=adult)

    def generate(line: str, seed: int, hero: SeriesFrame | None, strength: float | None) -> RawImage | None:
        prompt = " ".join(part for part in (policy.template, line.strip(), look.suffix) if part)
        negative = ", ".join(part for part in (look.negative, policy.negatives) if part)
        mode = Mode.TEXT if hero is None else Mode.RESTYLE
        job = PictureJob(
            prompt, negative, mode, "1:1", "native", DEFAULT_FORMAT, seed, 8, 0.0, "euler", 1,
            adult_content=adult,
            lora_path="" if lora is None else str(lora.weights),
            lora_weight=lora_scale,
        )
        runner.img2img_strength = None if hero is None else strength
        fresh: tuple[PendingInput, ...] = ()
        if hero is not None:
            raw = RawImage(hero.width, hero.height, hero.to_bytes())
            fresh = (PendingInput(ImageRole.INIT, raw),)
        reset_image_seconds()
        try:
            return runner(job, fresh)
        finally:
            seconds = last_image_seconds()
            if seconds is not None:
                note_stage("denoise", seconds)

    return generate


def run_command(
    *,
    idea: str,
    folder: Path,
    frames: int,
    seed: int,
    style: str,
    places: str,
    adult: bool,
    age_gate: AgeGate,
    rewrite: Callable[[str], str],
    generate: Callable[[str, int, SeriesFrame | None, float | None], SeriesFrame | None],
    strength: float | None = None,
    timings: StageTimes | None = None,
) -> tuple[dict[str, object], int]:
    """Run one series into ``folder`` and describe what was written.

    Args:
        idea: Theme or idea.
        folder: Directory for the frames and ``scenario.txt``.
        frames: How many frames were requested.
        seed: First frame seed.
        style: Look id. Scenario text uses photo, cinematic, or portrait.
        places: ``same`` or ``different``.
        adult: Use the Expert adult-content prompt policy.
        age_gate: Passed through to the series.
        rewrite: Scenario writer.
        generate: One frame. A stopped frame returns None.
        strength: Loose style reference. None draws every frame from text.
    """
    catalogs()
    if frames < 1 or frames > MAX_FRAMES:
        return reject_frames(frames), 2
    clock = timings if timings is not None else StageTimes()
    body, code = _run_command(
        idea=idea, folder=folder, frames=frames, seed=seed, style=style, places=places, adult=adult,
        age_gate=age_gate, rewrite=rewrite, generate=generate, strength=strength, clock=clock,
    )
    body["timings"] = clock.summary()
    note_timing(clock.log_line())
    return body, code


def _run_command(
    *, idea: str, folder: Path, frames: int, seed: int, style: str, places: str, adult: bool,
    age_gate: AgeGate, rewrite: Callable[[str], str],
    generate: Callable[[str, int, SeriesFrame | None, float | None], SeriesFrame | None],
    strength: float | None, clock: StageTimes,
) -> tuple[dict[str, object], int]:
    scenario_style = style if style in _SCENARIO_STYLES else "photo"
    held: dict[str, Scenario] = {}
    statuses: dict[int, str] = {}

    def on_scenario(scenario: Scenario) -> None:
        held["scenario"] = scenario

    def on_status(index: int, status: str) -> None:
        statuses[index] = status

    result = run_series(
        idea=idea,
        count=frames,
        style=scenario_style,
        place_mode=places,
        seed=seed,
        strength=strength,
        include_camera=lambda: True,
        rewrite=rewrite,
        age_gate=age_gate,
        draw_frame=generate,
        cancelled=lambda: False,
        folder=folder,
        on_scenario=on_scenario,
        on_status=on_status,
        adult=adult,
        container=DEFAULT_FORMAT,
        timings=clock,
    )
    scenario = held.get("scenario")
    if (result.stopped in {"idea", "scenario"} and not result.kept) or scenario is None or result.folder is None:
        error = result.stopped if result.stopped in {"idea", "scenario"} else "scenario"
        return _body(
            False, error, _series_message(result.code), (), None, frames, 0, (),
            result.code, result.source,
        ), 1
    lines = [assemble_line(scenario.subject_block, shot, include_camera=True) for shot in scenario.shots]
    scenario_path = result.folder / "scenario.txt"
    scenario_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shots = _shot_rows(scenario, statuses, result.kept, result.code or "internal" if result.stopped == "error" else "")
    frame_paths = tuple(str(shot["path"]) for shot in shots if shot["code"] == "ok")
    made = len(frame_paths)
    if result.stopped == "error":
        return _body(
            False, "error", _english_stop_message(result.code or "internal"),
            frame_paths, str(scenario_path), frames, made, tuple(shots),
            result.code or "internal", result.source,
        ), 1
    if result.stopped == "scenario":
        return _body(
            False, "scenario", _series_message(result.code),
            frame_paths, str(scenario_path), frames, made, tuple(shots),
            result.code, result.source,
        ), 1
    if made == 0:
        return _body(
            False, "not_produced", _series_message(result.code or "not_produced"),
            frame_paths, str(scenario_path), frames, made, tuple(shots),
            result.code or "not_produced", result.source,
        ), 1
    return _body(
        True, "", "", frame_paths, str(scenario_path), frames, made, tuple(shots), "", result.source,
    ), 0


def _shot_rows(
    scenario: Scenario, statuses: dict[int, str], kept: tuple[Path, ...], error: str = "",
) -> list[dict[str, object]]:
    """One row per shot. After an error stop, unkept shots carry the error code, not ``not_produced``."""
    by_name = {path.name: path for path in kept}
    suffix = extension(DEFAULT_FORMAT)
    blocked = tr_lang("photoshoot.frame.blocked", "en")
    failed = tr_lang("photoshoot.frame.error", "en")
    rows: list[dict[str, object]] = []
    for index, _shot in enumerate(scenario.shots):
        path = by_name.get(f"{index + 1:02}{suffix}")
        if statuses.get(index) == "kept" and path is not None:
            rows.append({"index": index + 1, "code": "ok", "path": str(path)})
        elif error:
            rows.append({"index": index + 1, "code": error, "message": failed})
        else:
            rows.append({"index": index + 1, "code": "not_produced", "message": blocked})
    return rows


def _series_message(code: str) -> str:
    if code:
        return tr_lang(f"photoshoot.code.{code}", "en")
    return tr_lang("photoshoot.scenario.stopped", "en")


def _english_stop_message(code: str) -> str:
    """English sentence for a series stop code. ``--json`` stays in English."""
    if code == "internal":
        return text("photoshoot.code.internal", "en")
    if code.startswith("check."):
        return text("gate.unavailable", "en")
    if code == "gpu_busy":
        return text("startup.lock", "en")
    if code == "enlarge_unavailable":
        return text("enlarge.hint", "en", name=ENLARGE_NAME)
    return _series_message(code)


def _body(
    ok: bool,
    error: str,
    message: str,
    frames: tuple[str, ...],
    scenario: str | None,
    requested: int,
    made: int,
    shots: tuple[dict[str, object], ...],
    code: str = "",
    scenario_source: str = "",
) -> dict[str, object]:
    body: dict[str, object] = {
        "ok": ok,
        "frames": list(frames),
        "scenario": scenario,
        "requested": requested,
        "made": made,
        "shots": list(shots),
        "code": code,
        "scenario_source": scenario_source,
    }
    if error:
        body["error"] = error
        body["message"] = message
    return body
