"""Time one 6-frame Photoshoot series.

Proving Ground runs the live entry on a 16 GB machine. This module does not
rent a GPU and it does not fill in a duration that was not measured.
"""
from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

from canvas.app.shell_run import RunStopped, ShellRunner
from canvas.core.formats import DEFAULT_FORMAT
from canvas.infra.enlarge import EnlargeClient
from canvas.infra.paths import models_dir
from canvas.models.modes import Mode, Tier
from canvas.models.registry import load_catalog
from canvas.pipeline.enhance import open_rewriter
from canvas.pipeline.photoshoot import SCENARIO_TOKENS, SYSTEM_PROMPT
from canvas.pipeline.series import SeriesFrame, SeriesResult, run_series
from canvas.safety.gate import AgeGate
from canvas.safety.image import RawImage
from canvas.safety.inputs import ImageRole, PendingInput
from canvas.ui.shell_layout import PictureJob
from canvas.ui.styles import look_for

SERIES_FRAMES = 6


def measure_series(
    *,
    idea: str,
    folder: Path,
    seed: int,
    rewrite: Callable[[str], str],
    age_gate: AgeGate,
    generate: Callable[[str, int, SeriesFrame | None, float | None], SeriesFrame | None],
    style: str = "photo",
    place_mode: str = "same",
    strength: float | None = None,
) -> dict[str, object]:
    """Time one 6-frame series and return the measured seconds.

    Args:
        idea: Prompt-field text.
        folder: Series directory.
        seed: Hero seed.
        rewrite: Scenario writer.
        age_gate: The minors gate.
        generate: One frame. Receives the series strength.
        style: Everyday look id.
        place_mode: ``same`` or ``different``.
        strength: Loose style reference. None draws every frame from text.
    """
    started = time.perf_counter()
    result = run_series(
        idea=idea,
        count=SERIES_FRAMES,
        style=style,
        place_mode=place_mode,
        seed=seed,
        strength=strength,
        include_camera=lambda: True,
        rewrite=rewrite,
        age_gate=age_gate,
        draw_frame=generate,
        cancelled=lambda: False,
        folder=folder,
    )
    return _report(result, time.perf_counter() - started)


def live_rewrite(tier: Tier) -> Callable[[str], str]:
    """The local scenario writer. The model stays closed between calls.

    Args:
        tier: The tier chosen for this machine.
    """
    catalog = load_catalog()

    def rewrite(prompt: str) -> str:
        rewriter = open_rewriter(
            catalog, tier, models_dir(),
            lambda message: print(message, file=sys.stderr),
            max_new_tokens=SCENARIO_TOKENS,
        )
        try:
            return rewriter.complete(prompt, system=SYSTEM_PROMPT)
        finally:
            rewriter.close()

    return rewrite


def live_generate(age_gate: AgeGate, tier: Tier) -> Callable[[str, int, SeriesFrame | None, float | None], SeriesFrame | None]:
    """One frame. A style reference, when set, uses the first kept frame.

    Args:
        age_gate: The same gate the series calls by name.
        tier: The tier chosen for this machine.
    """
    runner = ShellRunner(gate=age_gate, tier=tier, enlarge=EnlargeClient())
    look = look_for("photo")

    def generate(line: str, seed: int, hero: SeriesFrame | None, strength: float | None) -> RawImage | None:
        prompt = " ".join(part for part in (line.strip(), look.suffix) if part)
        mode = Mode.TEXT if hero is None else Mode.RESTYLE
        job = PictureJob(prompt, look.negative, mode, "1:1", "native", DEFAULT_FORMAT, seed, 8, 0.0, "euler", 1)
        runner.img2img_strength = None if hero is None else strength
        fresh: tuple[PendingInput, ...] = ()
        if hero is not None:
            raw = RawImage(hero.width, hero.height, hero.to_bytes())
            fresh = (PendingInput(ImageRole.INIT, raw),)
        try:
            return runner(job, fresh)
        except RunStopped as exc:
            if exc.unavailable:
                return None
            raise

    return generate


def run_bench(
    *,
    prompt: str,
    folder: Path,
    seed: int,
    tier: Tier,
    age_gate: AgeGate,
    rewrite: Callable[[str], str],
    generate: Callable[[str, int, SeriesFrame | None, float], SeriesFrame | None],
) -> dict[str, object]:
    """Measure a 6-frame series with the callbacks the caller built.

    Args:
        prompt: The idea.
        folder: Series directory.
        seed: Hero seed.
        tier: Recorded on the report. The caller already chose the machine.
        age_gate: The minors gate.
        rewrite: Scenario writer.
        generate: Frame generator, including the hero frame.
    """
    report = measure_series(
        idea=prompt,
        folder=folder,
        seed=seed,
        rewrite=rewrite,
        age_gate=age_gate,
        generate=generate,
    )
    report["tier"] = tier.value
    return report


def _report(result: SeriesResult, seconds: float) -> dict[str, object]:
    kept = len(result.kept)
    return {
        "ok": result.stopped == "" and kept == SERIES_FRAMES,
        "frames": SERIES_FRAMES,
        "kept": kept,
        "stopped": result.stopped,
        "seconds": round(seconds, 3),
        "folder": None if result.folder is None else str(result.folder),
    }
