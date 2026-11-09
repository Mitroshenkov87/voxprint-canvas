"""Start a Photoshoot series from the prompt field. The series runs in the worker.

The window sends one ``series`` request. The worker writes the scenario,
runs its checks, draws the frames and writes the frames it released into the
window's private session folder. The window shows the scenario and the frame
status from worker events, then moves the released frames into the pictures
folder.
"""
from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from voxprint_common.i18n import tr
from voxprint_common.ui.safety import ReviewedOutput
from voxprint_common.ui.simple_image import Phase

from canvas.app.i18n import text
from canvas.core.formats import EXPORT_FORMAT, OutputFormat, UnsupportedFormat, parse_export_format
from canvas.app.info import ENLARGE_NAME
from canvas.ipc.schema import Reply, decode_event, inside, series_fields
from canvas.ui.shell_chrome import refresh_pictures
from canvas.models.mdl_registry import selectable
from canvas.ui.worker_checks import WorkerChecks

if TYPE_CHECKING:
    from canvas.ui.shell_screen import ShellWindow
from canvas.pipeline import adult_boundary
from canvas.ui.shell_layout import SimpleLayout, job_from
from canvas.ui.styles import look_for


@dataclass(frozen=True)
class SeriesOutcome:
    """A finished or stopped series, as the worker reported it."""

    stopped: str
    reason: str
    kept: tuple[Path, ...]
    folder: Path | None
    code: str = ""
    source: str = ""
    hero_released: bool = False
    hero_reason: str = ""


class _SeriesRun:
    """The series the window is waiting for."""

    def __init__(self, series_id: str, idea: str) -> None:
        self.series_id = series_id
        self.idea = idea
        self.job_id = ""


def run_photoshoot(window: ShellWindow) -> bool:
    """Send the series to the worker. True while the worker has it.

    Args:
        window: The open shell window. Its Photoshoot option is selected.
    """
    layout = window.shell.layout
    bar = window.photoshoot
    if not selectable():
        layout.unavailable(text("canvas.models.missing"))
        window.view.refresh()
        bar.sync()
        return False
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
        return False
    if job.output == "4k" and not window._enlarge_ready:
        layout.unavailable(text("enlarge.hint", name=ENLARGE_NAME))
        window.view.refresh()
        bar.sync()
        return False
    from canvas.ui.shell_screen import job_fields

    series_id = uuid.uuid4().hex
    bar.folder = window.shell.state_dir / "pictures" / "series" / series_id
    bar._stop = False
    base_seed = int(layout.seed)
    run = _SeriesRun(series_id, idea)
    window._series = run
    fields = {
        "idea": idea,
        "count": bar.count,
        "style": style,
        "place": bar.place,
        "seed_mode": bar.seed_mode,
        "strength": bar.strength,
        "seed": base_seed,
        "adult": adult_content(layout),
        "container": _export_format(layout).value,
        "job": job_fields(job),
        "template": layout.prompt_policy().template,
        "suffix": look_for(style).suffix,
        "series_id": series_id,
    }

    def progress(event) -> None:
        if window._series is not run:
            return
        decoded = decode_event(event.message)
        if decoded is None:
            return
        kind, data = decoded
        if kind == "scenario":
            from canvas.pipeline.photoshoot import Shot

            listed = data.get("shots")
            shots = tuple(Shot(**shot) for shot in (listed if isinstance(listed, list) else []) if isinstance(shot, dict))
            bar.show_scenario(str(data.get("subject", "")), shots, seed=base_seed)
            window.view.refresh()
            bar.sync()
        elif kind == "status":
            found = data.get("index")
            index = found if isinstance(found, int) and not isinstance(found, bool) else -1
            status = str(data.get("status", ""))
            total = max(len(bar.rows), 1)
            if status == "drawing":
                window._advance(index, total, tr("photoshoot.progress.frame", current=index + 1, total=total))
            bar.mark(index, status)
            window.view.refresh()
            bar.sync()
        elif kind == "note":
            window.show_progress_note(str(data.get("text", "")))

    def done(reply: Reply) -> None:
        _series_done(window, run, reply)

    def go() -> None:
        if window._series is not run:
            return
        if layout.phase is not Phase.BUSY:
            _end(window, run)
            return
        run.job_id = window.worker.submit("series", fields, on_done=done, on_progress=progress)

    def fail(reason: str) -> None:
        done(Reply("series", "refused", "worker_unavailable", reason, True))

    window.worker.when_ready(go, fail)
    return True


def stop_series(window: ShellWindow) -> None:
    """Ask the worker to stop after the frame that already finished. Finished frames stay."""
    run = window._series
    if isinstance(run, _SeriesRun) and run.job_id:
        window.worker.cancel(run.job_id)


def _end(window: ShellWindow, run: _SeriesRun) -> None:
    if window._series is run:
        window._series = None
        window._busy = False
    window.view.refresh()
    window.photoshoot.sync()


def _series_done(window: ShellWindow, run: _SeriesRun, reply: Reply) -> None:
    layout = window.shell.layout
    if window._series is not run:
        return
    data: dict[str, object] | None = None
    if reply.ok:
        data = dict(reply.data)
    elif reply.code == "cancelled":
        data = _cancelled_manifest(window, reply)
    if data is None:
        if layout.phase is Phase.BUSY:
            message = reply.message or text("gate.unavailable")
            if reply.unavailable:
                layout.unavailable(message)
            else:
                layout.block(message)
        _end(window, run)
        return
    outcome = _adopt(window, run, data)
    _end(window, run)
    _finish(window, outcome, run.idea)


def _cancelled_manifest(window: ShellWindow, reply: Reply) -> dict[str, object] | None:
    """The fields a cancelled series noted before it stopped, or None when they cannot be trusted."""
    session = window.worker.session_dir
    path = inside(reply.data.get("checkpoint"), session)
    if path is None or not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    finally:
        path.unlink(missing_ok=True)
    if not isinstance(body, dict) or body.get("op") != "series" or body.get("status") != "ok":
        return None
    if not series_fields(body, session):
        return None
    return body


def _adopt(window: ShellWindow, run: _SeriesRun, data: dict[str, object]) -> SeriesOutcome:
    """Move the released frames out of the session folder into the pictures folder."""
    session = window.worker.session_dir
    listed = data.get("kept")
    names = [str(name) for name in (listed if isinstance(listed, list) else [])]
    source = inside(data.get("folder"), session)
    folder: Path | None = None
    kept: tuple[Path, ...] = ()
    if source is not None and source.is_dir():
        folder = window.shell.state_dir / "pictures" / "series" / run.series_id
        folder.parent.mkdir(parents=True, exist_ok=True)
        if folder.exists():
            shutil.rmtree(folder)
        shutil.move(str(source), folder)
        kept = tuple(folder / name for name in names)
    return SeriesOutcome(
        stopped=str(data.get("stopped", "")),
        reason=str(data.get("reason", "")),
        kept=kept,
        folder=folder,
        code=str(data.get("code", "")),
        source=str(data.get("source", "")),
        hero_released=data.get("hero_released") is True,
        hero_reason=str(data.get("hero_reason", "")),
    )


def _finish(window: ShellWindow, result: SeriesOutcome, idea: str) -> None:
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
        layout.unavailable(text("gate.unavailable"))
    window.view.refresh()
    window.photoshoot.sync()


def _record_series(window: ShellWindow, result: SeriesOutcome, idea: str) -> bool:
    """Add the kept frames to the history as one grouped series entry.

    The screen keeps the hero frame only when the worker released it again
    as the series entry; every frame was already released in the worker.

    Returns:
        True when the series entry was added.
    """
    if not result.kept or result.folder is None:
        return False
    layout = window.shell.layout
    hero = result.kept[0]
    if not hero.is_file():
        return False
    if layout.phase is not Phase.BUSY and not layout.generate():
        return False
    item_id = f"series-{result.folder.name}"
    checks = window.shell.gate
    if isinstance(checks, WorkerChecks):
        if result.hero_released:
            checks.release(item_id, hero)
        else:
            checks.refuse(item_id, result.hero_reason or text("gate.unavailable"))
    try:
        kept = layout.finish(item_id, ReviewedOutput(path=hero))
    except ValueError:
        return False
    if kept is None:
        return False
    window._pictures[item_id] = str(hero)
    refresh_pictures(window)
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


def _export_format(layout: SimpleLayout) -> OutputFormat:
    """Forge file type for this series. WebP when the setting is missing."""
    raw = layout.get("canvas.export")
    if not isinstance(raw, str) or not raw.strip():
        return EXPORT_FORMAT
    try:
        return parse_export_format(raw)
    except UnsupportedFormat:
        return EXPORT_FORMAT


def adult_content(layout: SimpleLayout) -> bool:
    """True when the Expert «Для взрослых (18+)» prompt policy is on."""
    return adult_boundary.enabled(layout)
