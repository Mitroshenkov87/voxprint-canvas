"""Generate a Photoshoot series from the prompt field."""
from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import assert_never

from voxprint_common.i18n import tr
from voxprint_common.ui.simple_image import Phase

from canvas.app.i18n import text
from canvas.app.info import ENLARGE_NAME
from canvas.app.shell_run import RunStopped
from canvas.infra.paths import models_dir
from canvas.models.mdl_registry import selectable
from canvas.models.modes import Mode
from canvas.models.registry import load_catalog
from canvas.models.snapshots import SnapshotError
from canvas.pipeline.enhance import open_rewriter
from canvas.pipeline.photoshoot import SYSTEM_PROMPT
from canvas.pipeline.series import SeriesResult, run_series
from canvas.safety.image import RawImage
from canvas.safety.inputs import ImageRole, PendingInput
from canvas.ui.shell_layout import job_from
from canvas.ui.styles import look_for


class _Halt(Exception):
    """The series cannot continue because the machine stopped the picture."""

    def __init__(self, message: str, *, unavailable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.unavailable = unavailable


def run_photoshoot(window: object) -> None:
    """Write the scenario, check it, and generate the series.

    Args:
        window: The open shell window. Its Photoshoot option is selected.
    """
    layout = window.shell.layout
    bar = window.photoshoot
    if not selectable():
        layout.unavailable(text("canvas.models.missing"))
        window.view.refresh()
        bar.sync()
        return
    idea = layout.prompt.strip()
    style = str(layout.get("canvas.style"))
    if style not in {"photo", "cinematic", "portrait"}:
        style = "photo"
    try:
        job = job_from(layout)
    except (ValueError, KeyError):
        layout.block(text("gate.unavailable"))
        window.view.refresh()
        bar.sync()
        return
    if job.output == "4k" and not window._enlarge_ready:
        layout.unavailable(text("enlarge.hint", name=ENLARGE_NAME))
        window.view.refresh()
        bar.sync()
        return
    folder = window.shell.state_dir / "pictures" / "series" / uuid.uuid4().hex
    bar.folder = folder
    bar._stop = False
    base_seed = int(layout.seed)

    def rewrite(prompt: str) -> str:
        if bar.writer is not None:
            return bar.writer(prompt)
        return _model_reply(window, prompt)

    def generate(line: str, seed: int, hero: object | None, strength: float) -> RawImage | None:
        used = _seed(bar.seed_mode, base_seed, seed)
        prompt = _compose(layout, line, style)
        # The first kept frame is the init image. Simple mode uses strength 0.45.
        picture = job._replace(
            prompt=prompt,
            seed=used,
            mode=Mode.TEXT if hero is None else Mode.RESTYLE,
            copies=1,
        )
        runner = window._runner
        fresh: tuple[PendingInput, ...] = ()
        if hero is None:
            if hasattr(runner, "img2img_strength"):
                runner.img2img_strength = None
        else:
            if hasattr(runner, "img2img_strength"):
                runner.img2img_strength = strength
            raw_hero = hero if isinstance(hero, RawImage) else RawImage(hero.width, hero.height, hero.to_bytes())
            fresh = (PendingInput(ImageRole.INIT, raw_hero),)
        try:
            raw = runner(picture, fresh)
        except RunStopped as exc:
            if exc.unavailable:
                raise _Halt(exc.message, unavailable=True) from exc
            return None
        window._photoshoot_frame = raw
        return raw

    def on_scenario(scenario: object) -> None:
        bar.show_scenario(scenario.subject_block, scenario.shots)
        window.view.refresh()
        bar.sync()

    def on_status(index: int, status: str) -> None:
        total = max(len(bar.rows), 1)
        if status == "drawing":
            window._advance(index, total, tr("photoshoot.progress.frame", current=index + 1, total=total))
        bar.mark(index, status)
        if status == "kept":
            _remember(window, bar.line_for(index), folder / f"{index + 1:02}.png")
            if not bar._stop and index + 1 < len(bar.rows) and layout.phase is not Phase.BUSY:
                layout.set_prompt(idea)
                layout.generate()
        window.view.refresh()
        bar.sync()

    try:
        result = run_series(
            idea=idea,
            count=bar.count,
            style=style,
            place_mode=bar.place,
            seed=base_seed,
            strength=_strength(bar.strength),
            include_camera=window._expert.enabled,
            rewrite=rewrite,
            age_gate=window.shell.gate.age_gate,
            generate=generate,
            cancelled=lambda: bar._stop or layout.phase is not Phase.BUSY,
            folder=folder,
            on_scenario=on_scenario,
            line_for=bar.line_for,
            on_status=on_status,
        )
    except _Halt as exc:
        if exc.unavailable:
            layout.unavailable(exc.message)
        else:
            layout.block(exc.message)
        window.view.refresh()
        bar.sync()
        return
    _finish(window, result, idea)


def _finish(window: object, result: SeriesResult, idea: str) -> None:
    layout = window.shell.layout
    layout.set_prompt(idea)
    if result.stopped == "idea" or result.stopped == "scenario":
        layout.block(result.reason or tr("photoshoot.scenario.stopped"))
    elif result.stopped == "cancelled":
        pass
    elif result.stopped == "":
        if layout.phase is Phase.BUSY:
            layout.cancel()
    else:
        assert_never(result.stopped)
    window.view.refresh()
    window.photoshoot.sync()


def _remember(window: object, line: str, picture: Path) -> None:
    layout = window.shell.layout
    idea = layout.prompt
    raw = getattr(window, "_photoshoot_frame", None)
    if not isinstance(raw, RawImage):
        return
    if layout.phase is not Phase.BUSY:
        return
    item_id = uuid.uuid4().hex
    window.shell.gate.stage(item_id, RawImage(raw.width, raw.height, raw.to_bytes()))
    layout.set_prompt(line)
    try:
        kept = layout.finish(item_id)
    except ValueError:
        layout.set_prompt(idea)
        return
    if kept is not None:
        window._pictures[item_id] = str(picture)
        window._series_folders[item_id] = str(picture.parent)
    layout.set_prompt(idea)


def _compose(layout: object, line: str, style: str) -> str:
    look = look_for(style)
    policy = layout.prompt_policy()
    return " ".join(part for part in (policy.template, line.strip(), look.suffix) if part)


def _seed(mode: str, base: int, series_seed: int) -> int:
    if mode == "series":
        return series_seed
    if mode == "locked":
        return base
    if mode == "random":
        return random.SystemRandom().randrange(1, 2**31)
    raise ValueError(mode)


def _strength(name: str) -> float:
    if name == "low":
        return 0.65
    if name == "medium":
        return 0.45
    if name == "high":
        return 0.30
    raise ValueError(name)


def _model_reply(window: object, prompt: str) -> str:
    try:
        rewriter = open_rewriter(load_catalog(), window.tier, models_dir(), max_new_tokens=3072)
    except (SnapshotError, OSError) as exc:
        raise _Halt(str(exc), unavailable=True) from exc
    try:
        return rewriter.complete(prompt, system=SYSTEM_PROMPT)
    finally:
        rewriter.close()
