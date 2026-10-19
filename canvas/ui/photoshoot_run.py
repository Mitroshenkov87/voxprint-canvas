"""Generate a Photoshoot series from the prompt field."""
from __future__ import annotations

import json
import random
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from voxprint_common.i18n import tr
from voxprint_common.ui.simple_image import Phase

from canvas.app.i18n import text
from canvas.app.info import ENLARGE_NAME
from canvas.app.shell_gate import reviewed_output
from canvas.app.shell_run import RunStopped, load_rgb
from canvas.infra.gpu_lock import lock_path, release
from canvas.infra.paths import models_dir
from canvas.models.mdl_registry import selectable
from canvas.models.modes import Mode
from canvas.models.registry import load_catalog
from canvas.models.snapshots import SnapshotError
from canvas.pipeline.adult_direction import scenario_system
from canvas.pipeline.enhance import open_rewriter
from canvas.pipeline.photoshoot import SYSTEM_PROMPT, Scenario
from canvas.pipeline.series import REFERENCE_STRENGTH, SeriesFrame, SeriesResult, run_series

if TYPE_CHECKING:
    from canvas.ui.shell_screen import ShellWindow
from canvas.safety.image import RawImage
from canvas.safety.inputs import ImageRole, PendingInput
from canvas.ui.shell_layout import SimpleLayout, job_from
from canvas.ui.styles import look_for
from canvas.ui.ui_call import on_ui, run_off_ui


class _Halt(Exception):
    """The series cannot continue because the machine stopped the picture."""

    def __init__(self, message: str, *, unavailable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.unavailable = unavailable


def run_photoshoot(window: ShellWindow) -> None:
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
    adult = adult_content(layout)

    def rewrite(prompt: str) -> str:
        if bar.writer is not None:
            return bar.writer(prompt)
        return _model_reply(window, prompt, adult)

    def generate(line: str, seed: int, hero: SeriesFrame | None, strength: float | None) -> RawImage | None:
        used = _seed(bar.seed_mode, base_seed, seed)
        prompt = _compose(layout, line, style)
        # High consistency may pass the first kept frame as a loose style reference.
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
            # Always a copy: the pipeline wipes its inputs, and the hero is reused for every shot.
            raw_hero = RawImage(hero.width, hero.height, hero.to_bytes())
            fresh = (PendingInput(ImageRole.INIT, raw_hero),)
        try:
            raw = runner(picture, fresh)
        except RunStopped as exc:
            if exc.unavailable:
                raise _Halt(exc.message, unavailable=True) from exc
            raise
        return raw

    def on_scenario(scenario: Scenario) -> None:
        def show(scenario: Scenario = scenario) -> None:
            bar.show_scenario(scenario.subject_block, scenario.shots)
            window.view.refresh()
            bar.sync()

        on_ui(show)

    def on_status(index: int, status: str) -> None:
        def show(index: int = index, status: str = status) -> None:
            total = max(len(bar.rows), 1)
            if status == "drawing":
                window._advance(index, total, tr("photoshoot.progress.frame", current=index + 1, total=total))
            bar.mark(index, status)
            window.view.refresh()
            bar.sync()

        on_ui(show)

    def stopped() -> bool:
        return bool(on_ui(lambda: bar._stop or layout.phase is not Phase.BUSY))

    try:
        result = run_off_ui(
            lambda: run_series(
                idea=idea,
                count=bar.count,
                style=style,
                place_mode=bar.place,
                seed=base_seed,
                strength=_strength(bar.strength),
                include_camera=lambda: True,
                rewrite=rewrite,
                age_gate=window.shell.gate.age_gate,
                draw_frame=generate,
                cancelled=stopped,
                folder=folder,
                on_scenario=on_scenario,
                line_for=bar.line_for,
                on_status=on_status,
                adult=adult,
            ),
        )
    except _Halt as exc:
        if exc.unavailable:
            layout.unavailable(exc.message)
        else:
            layout.block(exc.message)
        window.view.refresh()
        bar.sync()
        return
    finally:
        release(lock_path())
    _finish(window, result, idea)


def _finish(window: ShellWindow, result: SeriesResult, idea: str) -> None:
    layout = window.shell.layout
    layout.set_prompt(idea)
    if result.stopped == "cancelled":
        if not _record_series(window, result, idea) and layout.phase is Phase.BUSY:
            layout.cancel()
    elif result.stopped == "error":
        layout.unavailable(result.reason or tr("photoshoot.code.internal"))
    elif result.code:
        layout.block(tr(f"photoshoot.code.{result.code}"))
    elif result.stopped == "idea" or result.stopped == "scenario":
        layout.block(result.reason or tr("photoshoot.scenario.stopped"))
    elif result.stopped == "":
        if not _record_series(window, result, idea) and layout.phase is Phase.BUSY:
            layout.cancel()
    else:
        assert_never(result.stopped)
    window.view.refresh()
    window.photoshoot.sync()


def _record_series(window: ShellWindow, result: SeriesResult, idea: str) -> bool:
    """Add the kept frames to the history as one grouped series entry.

    The hero frame is staged so the content gate reviews it again through
    ``finish``; every frame was already released by the gate in ``run_series``.

    Returns:
        True when the series entry was added.
    """
    if not result.kept or result.folder is None:
        return False
    layout = window.shell.layout
    hero = result.kept[0]
    try:
        raw = load_rgb(hero)
    except OSError:
        return False
    if layout.phase is not Phase.BUSY and not layout.generate():
        return False
    item_id = f"series-{result.folder.name}"
    window.shell.gate.stage(item_id, RawImage(raw.width, raw.height, raw.to_bytes()))
    try:
        kept = layout.finish(item_id, reviewed_output(raw))
    except ValueError:
        return False
    if kept is None:
        return False
    window._pictures[item_id] = str(hero)
    window._series_folders[item_id] = str(result.folder)
    record_series(window.shell.state_dir, item_id, result.folder, result.kept, idea)
    return True


def load_series_folders(state_dir: Path) -> dict[str, str]:
    """History item ids mapped to the series folder that holds their frame.

    Args:
        state_dir: The Canvas state directory.
    """
    payload = _series_map(state_dir)
    folders: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            folder = value.get("folder")
            if isinstance(folder, str) and folder:
                folders[key] = folder
    return folders


def record_series(state_dir: Path, item_id: str, folder: Path, frames: tuple[Path, ...], idea: str) -> None:
    """Remember that history item ``item_id`` is one grouped series.

    Args:
        state_dir: The Canvas state directory.
        item_id: History item id of the series entry.
        folder: The series folder.
        frames: Kept frames, in order. The first is the hero.
        idea: The idea the series was written from.
    """
    path = _series_map_path(state_dir)
    payload = _series_map(state_dir)
    payload[item_id] = {
        "kind": "series",
        "folder": str(folder),
        "series": folder.name,
        "idea": idea,
        "frames": [frame.name for frame in frames],
        "file": frames[0].name if frames else "",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _series_map_path(state_dir: Path) -> Path:
    return state_dir / "pictures" / "series-map.json"


def _series_map(state_dir: Path) -> dict[str, object]:
    path = _series_map_path(state_dir)
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _compose(layout: SimpleLayout, line: str, style: str) -> str:
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


def _strength(name: str) -> float | None:
    if name == "low" or name == "medium":
        return None
    if name == "high":
        return REFERENCE_STRENGTH
    raise ValueError(name)


def busy_progress(window: ShellWindow) -> Callable[[str], None]:
    """Show download text on the window thread while a job is running."""

    def report(message: str) -> None:
        def show() -> None:
            if window.shell.layout.phase is Phase.BUSY:
                window._advance(0, 1, message)
            else:
                window.view.set_note(message)

        on_ui(show)

    return report


def adult_content(layout: SimpleLayout) -> bool:
    """True when the Expert «Для взрослых (18+)» prompt policy is on."""
    try:
        return layout.get("image.adult_content") is True
    except KeyError:
        return False


def _model_reply(window: ShellWindow, prompt: str, adult: bool = False) -> str:
    try:
        rewriter = open_rewriter(
            load_catalog(), window.tier, models_dir(), busy_progress(window), max_new_tokens=3072,
        )
    except (SnapshotError, OSError) as exc:
        raise _Halt(str(exc), unavailable=True) from exc
    try:
        return rewriter.complete(prompt, system=scenario_system(SYSTEM_PROMPT, adult))
    finally:
        rewriter.close()
