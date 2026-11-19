"""The Canvas window: the shared image screen, plus the tools it does not draw."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import assert_never

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QMessageBox,
    QApplication,
    QButtonGroup,
    QDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
)

from voxprint_common.i18n import add_catalogs, set_language, tr
from voxprint_common.mdl.names import CLI_NAME
from voxprint_common.suite_settings import MODEL_IDS, model_default, set_model_default
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.safety import ReviewedOutput
from voxprint_common.ui.simple_image import Phase, SimpleImageWidget as SimpleWidget, HISTORY_CELL_SIZE, HISTORY_CELL_GAP
from voxprint_common.worker.qt import InfoBar
from canvas.app.i18n import text
from canvas.app.info import APP_VERSION, ENLARGE_NAME
from canvas.app.updates import check_for_update
from canvas.core.seeds import as_seed, field_value
from canvas.infra.enlarge import EnlargeClient
from canvas.infra.paths import projects_dir
from canvas.ipc.client import WorkerClient, default_host
from canvas.ipc.schema import Reply, decode_event
from canvas.models.mdl_layout import on_disk_manifest
from canvas.models.mdl_registry import selectable
from canvas.models.forge_retrain import request_retrain
from canvas.models.generate_picker import implied_family, stored_generate_model
from canvas.models.lora_library import LibraryEntry
from canvas.models.user_lora import LoraIndex, UserLora, chip_word, discover, installed_loras, resolve_selected, route_lora
from canvas.ui.lora_panel import ImportedDialog, LibraryDialog, dark, import_with_dialog
from canvas.models.modes import Mode, Tier
from canvas.models.sizes import closest_aspect
from canvas.ui.settings_dialog import SettingsDialog
from canvas.ui.picture_file import RawImage, load_rgb
from canvas.ui.enlarge_flow import (
    EnlargeConfirm,
    EnlargeError,
    EnlargeProgress,
    Installer,
    enlarge_question,
    run_shared_enlarge,
)
from canvas.ui.shell_chrome import (
    SHOT_HEIGHT,
    SHOT_WIDTH,
    _fourk_available,
    fit_panel,
    mount,
    polish,
    refresh_pictures,
    shell_style,
    show_real_pictures,
)
from canvas.ui.photoshoot_bar import PhotoshootBar
from canvas.ui.photoshoot_run import adult_content, load_series_folders, run_photoshoot, stop_series
from canvas.ui.shell_layout import CanvasShell, PictureJob, job_from
from canvas.ui.wizards import AddonsWizard, DiagnosticsWizard, MaskWizard, PictureWizard, SetupWizard
from canvas.ui.worker_checks import WorkerChecks

LORA_NOTE_MS = 12_000
"""How long the queued retrain note stays up."""
_RETRAIN_TARGET = "Z-Image-Turbo"


class _NeedImage(Exception):
    """A picture mode was started without the image it needs."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass
class _Run:
    """One Generate press: the job, and which copy the worker is drawing."""

    job: PictureJob
    total: int
    base_seed: int
    layout_prompt: str
    # Every picture of the run as ``(prompt, seed)``, built from the prompt
    # box, the picture count, and the optional batch fields. The seed is None
    # when the run plans it (unlocked random or a locked count-up).
    items: list[tuple[str, int | None]] | None = None
    # True when the batch box carried the run and the first line was mirrored
    # into the layout prompt so the shared empty-prompt checks pass.
    mirrored: bool = False
    index: int = 0
    job_id: str = ""
    seed: int = 0


def _batch_items(self) -> list[tuple[str, int | None]]:
    """The ``--prompts`` / ``--seeds`` parity: one ``(prompt, seed)`` per batch picture.

    Both fields live in the Options disclosure and stay empty for a normal
    run. A seed list sets the batch size, exactly like the CLI.

    Raises:
        ValueError: the two fields disagree with each other or with the typed prompt.
    """
    from canvas.app.generate_batch import GeneratePlanError, parse_seeds
    from canvas.ui.batch_panel import batch_fields

    seeds_text, lines = batch_fields(self)
    seeds: list[int] = []
    try:
        seeds = parse_seeds(seeds_text)
    except GeneratePlanError as exc:
        raise ValueError(str(exc)) from exc
    typed = self.shell.layout.prompt.strip()
    # The first line may sit in the prompt box through the batch mirror; that
    # is the prompts mode, not a mix. Any other typed words are a mix.
    if lines and typed and typed != lines[0]:
        raise ValueError(text("canvas.batch.mixed"))
    if lines:
        if seeds and len(seeds) != len(lines):
            raise ValueError(text("canvas.batch.mismatch"))
        return [(line, seeds[index] if seeds else None) for index, line in enumerate(lines)]
    if seeds:
        return [("", seed) for seed in seeds]
    return []


def _batch_lines(window) -> list[str]:
    """The batch prompt lines, or empty when the section is closed or blank."""
    from canvas.ui.batch_panel import batch_fields

    _seeds, lines = batch_fields(window)
    return lines


def job_fields(job: PictureJob) -> dict[str, object]:
    """``job`` as plain values the worker request can carry."""
    fields = job._asdict()
    fields["mode"] = job.mode.value
    fields["fmt"] = job.fmt.value
    return fields


def watch_history_select(view: SimpleWidget, listener: Callable[[str], None]) -> None:
    """Call ``listener`` after the shared screen selects a history item.

    The shared widget keeps history selection to itself; the Photoshoot
    script links its rows to the selected picture through this hook.
    """
    select = getattr(view, "_select")

    def watched(item_id: str) -> None:
        select(item_id)
        listener(item_id)

    setattr(view, "_select", watched)


class ShellWindow(QMainWindow):
    """Simple mode by default. Expert opens the side panel. The old window stays unused."""

    def __init__(
        self,
        shell: CanvasShell,
        *,
        tier: Tier,
        enlarge: object | None = None,
        worker: WorkerClient | None = None,
        installer: Installer | None = None,
    ) -> None:
        """Place the shared screen in the Canvas window.

        Args:
            shell: Headless state from :func:`canvas.ui.shell_layout.open_worker_layout`.
            tier: The tier chosen at startup.
            enlarge: Enlarge client. Only its ``ready`` answer is read here; the worker runs Enlarge.
            worker: The model worker. The default spawns ``canvas.worker_jobs`` on first use.
            installer: Installs Enlarge. The default uses the shared model queue.
        """
        super().__init__()
        add_catalogs(Path(__file__).resolve().parent / "locales")
        set_language("en")
        self.shell = shell
        self.tier = tier
        self._enlarge = enlarge if enlarge is not None else EnlargeClient()
        self._enlarge_ready = bool(getattr(self._enlarge, "ready", lambda: False)())
        self.worker = worker if worker is not None else WorkerClient(default_host(), tier=tier.value)
        if isinstance(shell.gate, WorkerChecks) and shell.gate.client is None:
            shell.gate.client = self.worker
        self._installer = installer if installer is not None else run_shared_enlarge
        self._run: _Run | None = None
        self._series: object | None = None
        self._diagnostics: DiagnosticsWizard | None = None
        self._picture: Path | None = None
        self._mask: RawImage | None = None
        self._mask_file: Path | None = None
        self._pictures: dict[str, str] = {}
        # Exact seed of each kept picture, shown on the result line.
        self._seeds: dict[str, int] = {}
        self._series_folders: dict[str, str] = load_series_folders(shell.state_dir)
        # The MDL GUI process this window started, reaped by _mdl_timer so it
        # never lingers as a zombie, and stopped with the window. The suite
        # download worker the GUI may spawn is shared with other apps and is
        # never signalled.
        self._mdl_gui: subprocess.Popen | None = None
        self._mdl_timer = QTimer(self)
        self._mdl_timer.setInterval(1000)
        self._mdl_timer.timeout.connect(self._reap_mdl)
        self._busy = False
        # The worker keeps the loaded pipeline between generations while this window is open.
        self._warm = False
        self._chrome_ready = False
        self.setObjectName("shell")
        self.setWindowTitle(tr("splash.app.canvas"))
        self.setStyleSheet(shell_style())
        self.setMinimumSize(shell.surface.min_width, shell.surface.min_height)
        self.resize(SHOT_WIDTH, SHOT_HEIGHT)
        self._restore_geometry()
        from PySide6.QtGui import QKeySequence, QShortcut

        generate_keys = QShortcut(QKeySequence("Ctrl+Return"), self)
        generate_keys.setObjectName("generateShortcut")
        generate_keys.activated.connect(self._shortcut_generate)
        self._generate_shortcut = generate_keys
        self.view = SimpleWidget(
            shell.layout,
            shell.settings,
            on_generate=self._on_generate,
            on_command=self._on_command,
            on_improve=self._on_improve,
            on_history=self._on_history,
            on_refresh=self._on_refresh,
            surface=shell.surface,
            cell_size=HISTORY_CELL_SIZE,
            cell_gap=HISTORY_CELL_GAP,
        )
        self._expert = ExpertMode()
        self._expert.subscribe(self.view.set_expert)
        if self._expert.enabled():
            self.view.set_expert(True)
        _busy_first(self, shell.layout)
        self._wire_batch_generate()
        self._split = mount(self)
        self.setCentralWidget(self._split)
        self.photoshoot = PhotoshootBar(self)
        watch_history_select(self.view, self.photoshoot.script.select_image)
        show_real_pictures(self)
        self._chrome_ready = True
        polish(self)

    def _wire_batch_generate(self) -> None:
        """Let the batch box carry a run whose prompt box is empty.

        The shared Generate button and the shared layout refuse an empty
        prompt box before the window's hook runs. With one prompt per line in
        the Batch disclosure the box is meant to be empty: the first line is
        mirrored into the layout state so those checks pass, every line gets
        the same review a typed prompt gets, and the mirror is cleared when
        the run ends.
        """
        from canvas.ui.batch_panel import batch_fields

        shared_generate = self.view._generate

        def pressed() -> None:
            _seeds, lines = batch_fields(self)
            if not lines or self.shell.layout.phase is Phase.BUSY:
                shared_generate()
                return
            layout = self.shell.layout
            layout.set_prompt(lines[0])
            for line in lines:
                if not layout._review_prompt(line):
                    layout.set_prompt("")
                    self.view.refresh()
                    return
            if not layout.generate(self._on_generate):
                layout.set_prompt("")
                self.view.refresh()
                return
            self.view.set_note("")
            self.view.refresh()

        self.view._generate = pressed

    def set_expert(self, enabled: bool) -> None:
        """Store Expert mode and show or hide the side panel.

        The shared screen hears about a stored change once. Filling the
        panel again piles the previous fields on the new ones, so a repeat
        of the same value fills it only when the panel is still in the
        other state.

        Args:
            enabled: True shows the expert values beside the picture.
        """
        previous = self._expert.enabled()
        shown = self.view._expert_panel.isVisible()
        self._expert.set_enabled(enabled)
        if previous == enabled and shown != enabled:
            self.view.set_expert(enabled)
        fit_panel(self, enabled)

    def _on_refresh(self) -> None:
        if self._chrome_ready:
            polish(self)
        if getattr(self, "photoshoot", None) is not None:
            self.photoshoot.sync()
        if self.shell.layout.phase is not Phase.BUSY:
            # Cancel in the shared screen: the worker stops; a picture that still arrives is deleted.
            run = getattr(self, "_run", None)
            if run is not None and run.job_id:
                self.worker.cancel(run.job_id)
            if getattr(self, "_series", None) is not None:
                stop_series(self)

    def use_picture(self, path: Path) -> None:
        """Use ``path`` as the source picture and match its shape.

        Args:
            path: An image file.
        """
        image = load_rgb(path)
        self._picture = path
        self.shell.layout.set_source_path(str(path))
        self.shell.layout.set_value("canvas.source", "picture")
        self.shell.layout.choose("size", closest_aspect(image.width, image.height))
        self.view.refresh()

    def use_mask(self, image: RawImage) -> None:
        """Keep ``image`` as the inpaint mask and select Paint over.

        Args:
            image: White pixels are painted over.
        """
        self._mask = image
        self._mask_file = self._write_mask(image)
        self.shell.layout.set_value("canvas.source", "picture")
        self.shell.layout.set_value("canvas.change", "inpaint")
        self.view.refresh()

    def setup_wizard(self) -> SetupWizard:
        """Start from, variation, change, and file type."""
        layout = self.shell.layout
        values = {
            "canvas.source": str(layout.get("canvas.source")),
            "canvas.variation": str(layout.get("canvas.variation")),
            "canvas.change": str(layout.get("canvas.change")),
            "canvas.format": str(layout.get("canvas.format")),
        }
        return SetupWizard(self, values, self._choose_setup)

    def picture_wizard(self) -> PictureWizard:
        """The picture picker."""
        return PictureWizard(self, self.use_picture)

    def mask_wizard(self) -> MaskWizard:
        """The mask painter. It asks for a picture when none is open."""
        picture = None if self._picture is None else _safe_load(self._picture)
        return MaskWizard(self, picture, self.use_mask)

    def addons_wizard(self) -> AddonsWizard:
        """The add-ons screen. It reads and writes the shared expert values."""
        layout = self.shell.layout
        strength = layout.get("shell.lora_weight")
        number = float(strength) if isinstance(strength, int | float) and not isinstance(strength, bool) else None
        found = discover()
        index = LoraIndex(installed_loras()[0], found.skipped)
        return AddonsWizard(
            self, index, str(layout.get("shell.addons") or ""), number, self._save_addons, self._import_from_addons,
        )

    def lora_library(self) -> LibraryDialog:
        """The LoRA library: «Import LoRA», previews and remove."""
        return LibraryDialog(self, self.worker, self._library_changed)

    def _library_changed(self) -> None:
        self.refresh_loras()
        self.view.refresh()

    def import_lora(self, path: Path | None = None, on_done: Callable[..., None] | None = None) -> None:
        """Import one Model Forge export into the library in the worker, then show what was added.

        Args:
            path: The file. A file dialog asks when omitted.
            on_done: Called with the entry, a refusal code, or None when nothing was imported.
        """

        def finished(result: LibraryEntry | str | None) -> None:
            if isinstance(result, str):
                box = QMessageBox(QMessageBox.Icon.Warning, text("canvas.lora.import"), text(result),
                                  QMessageBox.StandardButton.Ok, self)
                box.setObjectName("loraRefused")
                dark(box)
                box.exec()
            elif result is not None:
                self.refresh_loras()
                ImportedDialog(self, result).exec()
            if on_done is not None:
                on_done(result)

        import_with_dialog(self, self.worker, path, on_done=finished)

    def refresh_loras(self) -> None:
        """Re-read the library into the generate panel's LoRA pick."""
        picker = getattr(self, "_lora_picker", None)
        if picker is None:
            return
        layout = self.shell.layout
        strength = layout.get("shell.lora_weight")
        number = float(strength) if isinstance(strength, int | float) and not isinstance(strength, bool) else None
        picker.reload(str(layout.get("shell.addons") or ""), number)

    def _import_from_addons(self, add=None):
        def finished(result: LibraryEntry | str | None) -> None:
            if add is not None and isinstance(result, LibraryEntry):
                try:
                    add(result.as_lora())
                except RuntimeError:
                    return  # the add-ons screen was closed while the worker checked the file

        self.import_lora(on_done=finished)

    def _pick_model(self, model_id: str) -> None:
        """Store the generate-model pick. The shared family follows when the pick implies one."""
        layout = self.shell.layout
        chosen = stored_generate_model(model_id)
        if not chosen:
            return
        if str(layout.get("canvas.generate_model") or "") != chosen:
            layout.set_value("canvas.generate_model", chosen)
        family = implied_family(model_id)
        if family and family in MODEL_IDS and model_default() != family:
            set_model_default(family)

    def refresh_models(self) -> None:
        """Re-read the installed manifest into the generate-model pick."""
        picker = getattr(self, "_model_picker", None)
        if picker is None:
            return
        from canvas.ui.shell_chrome import _reload_model_picker

        _reload_model_picker(self)

    def _pick_lora(self, key: str, strength: float) -> None:
        changed = str(self.shell.layout.get("shell.addons") or "") != key
        self.shell.layout.set_value("shell.lora_weight", strength)
        if changed:
            # Only a new pick changes the trigger chip; a strength step does not redraw.
            self.shell.layout.set_value("shell.addons", key)
            self.view.refresh()
            self._show_lora_base_note(key)

    def _hide_lora_bar(self) -> None:
        old = getattr(self, "_lora_bar", None)
        if old is None:
            return
        old.widget.hide()
        old.widget.deleteLater()
        self._lora_bar = None

    def _selected_retrain(self, key: str | None = None) -> tuple[UserLora, str, str] | None:
        """The selected LoRA, trained base, and notice code when it cannot run as-is."""
        if key is None:
            key = str(self.shell.layout.get("shell.addons") or "")
        if not key:
            return None
        try:
            selected = resolve_selected(key)
            route = route_lora(selected) if selected is not None else None
        except Exception:  # noqa: BLE001 - a lookup failure shows nothing and does not start a job
            return None
        if selected is None or route is None:
            return None
        if not route.base:
            return selected, route.retrain_from, "lora.retrain"
        return None

    def _show_lora_base_note(self, key: str) -> None:
        """Offer a Forge retrain when the pick was trained on a base Canvas cannot run.

        The job does not run with that LoRA. A later matching pick hides the bar.
        """
        found = self._selected_retrain(key)
        if found is None:
            self._hide_lora_bar()
            self._show_alias_note(key)
            return
        lora, trained, code = found
        self._present_retrain(lora, trained, code)

    def _show_alias_note(self, key: str) -> None:
        """Say so when the pick's alias was renamed because another LoRA already had it."""
        if not key:
            return
        try:
            selected = resolve_selected(key)
            notes = installed_loras()[1]
        except Exception:  # noqa: BLE001 - a lookup failure shows nothing
            return
        if selected is None:
            return
        for lora_id, wanted, given in notes:
            if lora_id == selected.id and given == selected.alias:
                bar = InfoBar(
                    self,
                    text("canvas.lora.alias_clash_title"),
                    text("canvas.lora.alias_clash", lora=lora_id, wanted=wanted, given=given),
                )
                bar.present()
                self._lora_bar = bar
                QTimer.singleShot(LORA_NOTE_MS, bar.widget.hide)
                return

    def _present_retrain(self, lora: UserLora, trained: str, code: str = "lora.retrain") -> None:
        """Show the retrain title, note, and Forge button for ``lora``."""
        self._hide_lora_bar()
        self._retrain_lora = lora
        self._retrain_from = trained
        title = text("canvas.lora.retrain_title")
        note = text("canvas.lora.retrain_note", base=trained, target=_RETRAIN_TARGET)
        bar = InfoBar(
            self,
            title,
            note,
        )
        button = QPushButton(text("canvas.lora.retrain_action"), bar.widget)
        button.setObjectName("loraRetrain")
        button.clicked.connect(self._queue_retrain)
        layout = bar.widget.layout()
        if layout is not None:
            layout.addWidget(button)
        bar.present()
        self._lora_bar = bar

    def _queue_retrain(self) -> None:
        """Write a Forge inbox request, then show the short queued note."""
        lora = getattr(self, "_retrain_lora", None)
        trained = str(getattr(self, "_retrain_from", "") or "")
        if not isinstance(lora, UserLora):
            return
        try:
            request_retrain(lora.weights, trained)
        except OSError:
            return
        bar = getattr(self, "_lora_bar", None)
        if bar is None:
            return
        body = bar.widget.findChild(QLabel, "InfoBarBody")
        if body is not None:
            body.setText(text("canvas.lora.retrain_queued"))
        button = bar.widget.findChild(QPushButton, "loraRetrain")
        if button is not None:
            button.hide()
        bar.present()
        QTimer.singleShot(LORA_NOTE_MS, bar.widget.hide)

    def lora_trigger(self) -> str:
        """The word the chip shows and inserts: the alias when the LoRA has one, else its trigger.

        Empty when nothing is selected. The rare trigger itself is added to the
        final prompt on its own; the prompt box keeps what the user typed.
        """
        return chip_word(resolve_selected(str(self.shell.layout.get("shell.addons") or "")))

    def insert_trigger(self) -> None:
        """Insert the selected trigger word at the cursor. Nothing is inserted on its own."""
        word = self.lora_trigger()
        if not word:
            return
        box = self.findChild(QPlainTextEdit, "promptBox")
        if box is None:
            return
        cursor = box.textCursor()
        current = box.toPlainText()
        position = cursor.position()
        prefix = " " if position > 0 and current[position - 1] not in " \n" else ""
        suffix = " " if position < len(current) and current[position:position + 1] not in " \n" else ""
        cursor.insertText(f"{prefix}{word}{suffix}")
        box.setTextCursor(cursor)
        self.shell.layout.set_prompt(box.toPlainText())

    def diagnostics_wizard(self) -> DiagnosticsWizard:
        """The PC check. It does not run until a button is pressed; then it runs in the worker."""
        wizard = DiagnosticsWizard(self, self._run_check, self._run_update)
        self._diagnostics = wizard
        return wizard

    def settings_dialog(self) -> SettingsDialog:
        """The persisted settings: folder, file type, metadata, 18+, Expert mode, seed."""
        return SettingsDialog(self)

    def enlarge_confirm(self) -> EnlargeConfirm:
        """The question shown when 4K is chosen before Enlarge is installed."""
        return EnlargeConfirm(self, enlarge_question())

    def enlarge_progress(self) -> EnlargeProgress:
        """The install progress, with Cancel."""
        return EnlargeProgress(self)

    def _open_setup(self) -> None:
        self.setup_wizard().exec()

    def _open_settings(self) -> None:
        self.settings_dialog().exec()

    def _choose_setup(self, key: str, option: str) -> None:
        self.shell.layout.set_value(key, option)
        self.view.refresh()

    def _open_picture(self) -> None:
        self.picture_wizard().exec()

    def _open_mask(self) -> None:
        self.mask_wizard().exec()

    def _open_addons(self) -> None:
        self.addons_wizard().exec()
        self.refresh_loras()

    def _open_lora_import(self) -> None:
        self.import_lora()

    def _open_lora_library(self) -> None:
        self.lora_library().exec()

    def _open_check(self) -> None:
        self.diagnostics_wizard().exec()

    def _save_addons(self, path: str, strength: float) -> None:
        self.shell.layout.set_value("shell.addons", path)
        self.shell.layout.set_value("shell.lora_weight", strength)
        self.view.refresh()
        self.refresh_loras()
        self._show_lora_base_note(path)

    def _run_check(self, deep: bool) -> str:
        """Queue the self-test in the worker. The wizard shows the report when it arrives."""
        wizard = self._diagnostics

        def done(reply: Reply) -> None:
            if wizard is None:
                return
            try:
                if reply.ok and isinstance(reply.data.get("report"), dict):
                    wizard.show_report(_check_text(reply.data["report"]))  # type: ignore[arg-type]
                else:
                    wizard.show_report(text("canvas.check.failed"))
            except RuntimeError:
                return  # the wizard was closed

        self.worker.submit("selftest", {"deep": bool(deep)}, on_done=done)
        return ""

    def _run_update(self) -> str:
        tag = check_for_update(APP_VERSION)
        if tag is None:
            return text("canvas.check.current")
        return text("canvas.check.newer", tag=tag)

    def _on_improve(self, current: str) -> str:
        """Queue the rewrite in the worker. The field changes when the reply arrives."""
        if self._busy:
            return current
        self._busy = True
        adult = adult_content(self.shell.layout)

        def progress(event) -> None:
            decoded = decode_event(event.message)
            if decoded is not None and decoded[0] == "note":
                self.show_progress_note(str(decoded[1].get("text", "")))

        def done(reply: Reply) -> None:
            self._busy = False
            if not reply.ok:
                self.shell.layout.block(reply.message or text("gate.unavailable"))
                self.view.refresh()
                return
            improved = str(reply.data.get("prompt", current))
            self._apply_improved(current, improved)

        def fail(reason: str) -> None:
            done(Reply("improve", "refused", "worker_unavailable", reason, True))

        def go() -> None:
            self.worker.submit("improve", {"prompt": current, "adult": adult}, on_done=done, on_progress=progress)

        self.worker.when_ready(go, fail)
        return current

    def _apply_improved(self, current: str, improved: str) -> None:
        """Put the rewrite in the field the way the shared screen does, while the field still holds ``current``."""
        view = self.view
        box = getattr(view, "_prompt", None)
        if improved == current or box is None or box.toPlainText() != current:
            return
        view._previous_prompt = current
        self.shell.layout.set_prompt(improved)
        box.setPlainText(improved)
        view._revert.setEnabled(True)
        view.set_note("")
        joined = getattr(self, "_after_improve", None)
        if callable(joined):
            joined()

    def show_progress_note(self, message: str) -> None:
        """Show a worker status line: in the progress row while busy, else as a note."""
        if not message:
            return
        if self.shell.layout.phase is Phase.BUSY:
            self._advance(0, 1, message)
        else:
            self.view.set_note(message)

    def _write_mask(self, image: RawImage) -> Path | None:
        """Save the mask beside the state and store it in the Yitro mask slot."""
        from PIL import Image

        path = self.shell.state_dir / "mask.png"
        try:
            Image.frombytes("RGB", (image.width, image.height), image.to_bytes()).save(path)
        except OSError:
            return None
        self.shell.layout.set_mask_path(str(path))
        return path

    def _adopt_slots(self, live: dict[str, object]) -> None:
        """Use a picture or mask set through the Yitro slots (drop or Expert path)."""
        source = str(live.get("image.source") or "")
        if source and (self._picture is None or Path(source) != self._picture) and Path(source).is_file():
            # The chips still choose words or picture; the slot only names the file.
            self._picture = Path(source)
        mask = str(live.get("image.mask") or "")
        if mask and (self._mask_file is None or Path(mask) != self._mask_file) and Path(mask).is_file():
            loaded = _safe_load(Path(mask))
            if loaded is not None:
                self._mask = loaded
                self._mask_file = Path(mask)

    def _shortcut_generate(self) -> None:
        """Ctrl+Enter: the same phase setup and hook as the Generate button."""
        if self._busy:
            return
        self.shell.layout.generate(lambda settings: self._on_generate(settings))

    def _on_generate(self, settings: object = None) -> None:
        if self._busy:
            return
        self._busy = True
        if not self._generate(settings):
            self._busy = False

    def release_warm(self) -> None:
        """Ask the worker to drop the pipeline it kept for this window. A second call does nothing.

        The worker exiting releases it too, so a window that closes with the app needs nothing more.
        """
        if not self._warm:
            return
        self._warm = False
        if self.worker.state == "ready":
            self.worker.submit("warm", {"action": "drop"}, on_done=lambda _reply: None)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Release the warm pipeline, remember the size, and stop the MDL GUI with the window."""
        self.release_warm()
        self._save_geometry()
        self._stop_mdl()
        super().closeEvent(event)

    def _geometry_path(self) -> Path:
        return self.shell.state_dir / "window-geometry.json"

    def _restore_geometry(self) -> None:
        """The size the person left last time, when it is known and fits the minimum."""
        try:
            payload = json.loads(self._geometry_path().read_text(encoding="utf-8"))
            width = int(payload.get("width") or 0)
            height = int(payload.get("height") or 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, AttributeError):
            return
        minimum = self.minimumSize()
        if width >= minimum.width() and height >= minimum.height():
            self.resize(width, height)

    def _save_geometry(self) -> None:
        """Remember the window size for the next open. A maximized window keeps its saved size."""
        if self.isMaximized() or self.isFullScreen():
            return
        payload = {"width": int(self.width()), "height": int(self.height())}
        try:
            self._geometry_path().write_text(json.dumps(payload) + "\n", encoding="utf-8")
        except OSError:
            pass

    def _generate(self, settings: object = None) -> bool:
        """Start one Generate press in the worker. True while the worker has the job."""
        blocked = self._selected_retrain()
        if blocked is not None:
            lora, trained, code = blocked
            self._present_retrain(lora, trained, code)
            self.shell.layout.block(text(code, base=trained))
            self.view.refresh()
            return False
        if self.photoshoot.enabled:
            return run_photoshoot(self)
        layout = self.shell.layout
        live = dict(settings) if isinstance(settings, dict) else layout.live_settings()
        self._adopt_slots(live)
        if not selectable():
            layout.unavailable(text("canvas.models.missing"))
            self.view.refresh()
            return False
        try:
            job = job_from(layout, live)
        except (ValueError, KeyError):
            layout.block(text("gate.unavailable"))
            self.view.refresh()
            return False
        try:
            items = _batch_items(self)
        except ValueError as exc:
            layout.block(str(exc))
            self.view.refresh()
            return False
        total = max(1, len(items)) if items else max(1, job.copies)
        box = getattr(self.view, "_prompt", None)
        typed = box.toPlainText().strip() if box is not None else layout.prompt.strip()
        run = _Run(
            job=job, total=total, base_seed=job.seed, layout_prompt=layout.prompt.strip(), items=items,
            mirrored=bool(items) and not typed,
        )
        self._run = run
        self.worker.when_ready(lambda: self._check_prompts(run), lambda reason: self._stop_run(run, reason))
        return True

    def _check_prompts(self, run: _Run) -> None:
        items = run.items or []
        for line, _seed in items:
            if line.strip() and not self.shell.layout._review_prompt(line):
                self._end_run(run)
                self.view.refresh()
                return
        prompts = [prompt for prompt, _seed in items]
        checked = [run.layout_prompt, *prompts] if prompts else [run.layout_prompt, run.job.prompt]
        self.worker.check_prompts(checked, on_done=lambda reply: self._prompts_checked(run, reply))

    def _prompts_checked(self, run: _Run, reply: Reply) -> None:
        layout = self.shell.layout
        if self._run is not run:
            return
        if layout.phase is not Phase.BUSY:
            self._end_run(run)
            self.view.refresh()
            return
        if not reply.ok:
            self._stop_run(run, reply.message, unavailable=reply.unavailable, code=reply.code)
            return
        if run.job.output == "4k" and not self._enlarge_ready:
            self._stop_run(run, text("enlarge.hint", name=ENLARGE_NAME))
            return
        self._warm = True
        self._start_copy(run)

    def _start_copy(self, run: _Run) -> None:
        layout = self.shell.layout
        index = run.index
        items = run.items or []
        item = items[index] if index < len(items) else None
        from canvas.ui.shell_layout import composed_line

        prompt = run.job.prompt if item is None or not item[0] else composed_line(layout, item[0])
        # Unlocked: a fresh random seed for every copy. Locked: the base counts up, so a locked run repeats.
        # A batch seed list pins one seed per picture, like ``--seeds``.
        seed = layout.plan_seed(index, run.base_seed) if item is None or item[1] is None else item[1]
        run.seed = seed
        run.job = run.job._replace(seed=seed, prompt=prompt)
        layout.note_seed_used(seed)
        if index:
            if not layout.generate():
                self.view.refresh()
                self._end_run(run)
                return
        count_label = text("canvas.progress.count", current=index + 1, total=run.total)
        if not self._advance(index, run.total, count_label):
            self._end_run(run)
            return
        try:
            inputs = self._input_files(run.job.mode)
        except _NeedImage as exc:
            layout.block(exc.message)
            self.view.refresh()
            self._end_run(run)
            return
        run.job_id = self.worker.submit(
            "generate",
            {
                "job": job_fields(run.job),
                "layout_prompt": run.layout_prompt,
                "inputs": inputs,
                "vram": str(layout.get("shell.vram")),
                "warm": True,
            },
            on_done=lambda reply: self._copy_done(run, reply),
            on_progress=lambda event: self._on_run_progress(run, event),
        )


    def _on_run_progress(self, run: _Run, event) -> None:
        """Worker progress for one copy: stage lines and notes keep the busy bar legible."""
        if self._run is not run:
            return
        decoded = decode_event(event.message)
        if decoded is None:
            return
        kind, data = decoded
        if kind == "note" and data.get("stage") == "waiting_gpu":
            self._stage_label(run, text("canvas.stage.wait"))
            return
        if kind == "note":
            self.show_progress_note(str(data.get("text", "")))
            return
        if kind != "stage":
            return
        event_name = str(data.get("event", ""))
        if event_name == "step" and str(data.get("stage", "")) == "sample":
            step = _step_number(str(data.get("suffix", "")))
            if step is not None:
                total = max(1, int(run.job.steps))
                label = text("canvas.stage.step", current=min(step, total), total=total)
                self._step_progress(run, min(step, total), total, label)
            return
        if event_name not in {"start", "heartbeat"}:
            return
        key = _stage_label_key(str(data.get("stage", "")))
        if key:
            self._stage_label(run, text(key))

    def _step_progress(self, run: _Run, step: int, total: int, label: str) -> None:
        """Advance the busy bar with the sampler step, so it moves during the draw."""
        if self._run is not run:
            return
        layout = self.shell.layout
        try:
            layout.progress(step, total, label)
        except ValueError:
            return
        self.view.refresh()

    def _stage_label(self, run: _Run, label: str) -> None:
        """Swap the busy line for one stage without touching the step count."""
        if not label:
            return
        layout = self.shell.layout
        try:
            layout.progress(run.index, run.total, label)
        except ValueError:
            return  # the run moved on while the stage line was on its way
        self.view.refresh()

    def _copy_done(self, run: _Run, reply: Reply) -> None:
        layout = self.shell.layout
        run.job_id = ""
        if self._run is not run or layout.phase is not Phase.BUSY:
            self._discard(reply)
            self._end_run(run)
            self.view.refresh()
            return
        if not reply.ok:
            self._stop_run(run, reply.message, unavailable=reply.unavailable, code=reply.code)
            return
        count_label = text("canvas.progress.count", current=run.index + 1, total=run.total)
        if not self._advance(run.index + 1, run.total, count_label):
            self._discard(reply)
            self._end_run(run)
            return
        if not self._keep(reply, run.job):
            self._end_run(run)
            return
        run.index += 1
        if run.index >= run.total:
            self.view.refresh()
            self._end_run(run)
            return
        self._start_copy(run)

    def _stop_run(self, run: _Run, message: str, *, unavailable: bool = True, code: str = "") -> None:
        """End ``run`` with a reason. Unavailable is the default for anything the worker could not answer.

        A worker the machine killed (SIGKILL, exit 137) names memory instead of
        the generic unavailable sentence: that is the cold-load OOM, not a gate.
        """
        layout = self.shell.layout
        if layout.phase is Phase.BUSY:
            if not message and code == "engine_killed":
                message = text("canvas.engine.killed")
            reason = message or text("gate.unavailable")
            if unavailable:
                layout.unavailable(reason)
            else:
                layout.block(reason)
        self.view.refresh()
        self._end_run(run)

    def _end_run(self, run: _Run) -> None:
        if self._run is run:
            self._run = None
            self._busy = False
            layout = self.shell.layout
            if run.mirrored:
                # The batch box carried the run; give the mirrored first line back.
                layout.set_prompt("")
            if layout.seed_locked():
                # A locked seed stays in the field, so the next run repeats it.
                layout.used_seed = run.base_seed
                layout.seed = field_value(run.base_seed)

    def _discard(self, reply: Reply) -> None:
        if reply.ok:
            self.worker.discard(reply.data)

    def _advance(self, step: int, total: int, label: str) -> bool:
        try:
            self.shell.layout.progress(step, total, label)
        except ValueError:
            self.view.refresh()
            return False
        if self.shell.layout.phase is not Phase.BUSY:
            self.view.refresh()
            return False
        self.view.refresh()
        return True

    def _keep(self, reply: Reply, job: PictureJob) -> bool:
        """Store a picture the worker released. The screen accepts only that file."""
        layout = self.shell.layout
        item_id = str(reply.data["item_id"])
        used = as_seed(reply.data.get("seed"))
        self._seeds[item_id] = job.seed if used is None else used
        path = Path(str(reply.data["path"]))
        checks = self.shell.gate
        if isinstance(checks, WorkerChecks):
            checks.release(item_id, path)
        if layout.finish(item_id, ReviewedOutput(path=path)) is None:
            self.view.refresh()
            self.worker.discard(reply.data)
            return False
        folder = pictures_dir()
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{item_id}{path.suffix}"
        shutil.move(str(path), target)
        self._pictures[item_id] = str(target)
        refresh_pictures(self)
        return True

    def picture_path(self, item_id: str) -> Path | None:
        """The saved file for one history item, across sessions.

        The window map holds this session's pictures. An item from an earlier
        session resolves by its id in the pictures folder, or in the state
        pictures folder where earlier releases kept them. None when no file is
        there: the strip shows the neutral placeholder then.

        Args:
            item_id: One history item id.
        """
        found = self._pictures.get(item_id)
        if found:
            return Path(found)
        for folder in (pictures_dir(), self.shell.state_dir / "pictures"):
            for suffix in (".webp", ".avif", ".jxl", ".jpg", ".jpeg"):
                candidate = folder / f"{item_id}{suffix}"
                if candidate.is_file():
                    return candidate
        return None

    def _input_files(self, mode: Mode) -> list[dict[str, str]]:
        """The source picture and mask as files the worker reads. The worker decodes and checks them."""
        if mode is Mode.TEXT:
            return []
        if self._picture is None:
            raise _NeedImage(text("error.job"))
        source = {"role": "init", "path": str(self._picture)}
        if mode is Mode.INPAINT or mode is Mode.OUTPAINT:
            if self._mask is None:
                raise _NeedImage(text("error.job"))
            from PIL import Image

            mask = self.worker.session_dir / f"mask-{uuid.uuid4().hex}.png"
            try:
                Image.frombytes("RGB", (self._mask.width, self._mask.height), self._mask.to_bytes()).save(mask)
            except OSError as exc:
                raise _NeedImage(text("error.job")) from exc
            return [source, {"role": "mask", "path": str(mask)}]
        return [{"role": _role(mode), "path": str(self._picture)}]

    def _on_command(self, command_id: str) -> None:
        if command_id == "file.save":
            self._save_as()
            return
        if command_id == "file.open_folder":
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(pictures_dir())))
            return
        if command_id == "models.manage":
            self._manage_models()
            return
        if command_id == "view.settings":
            self._open_settings()
            return
        if command_id == "view.toggle_expert_mode":
            self.set_expert(not self._expert.enabled())
            return
        if command_id == "install.enlarge":
            self._offer_enlarge()

    def _offer_enlarge(self) -> None:
        if _fourk_available(self):
            self._select_4k()
            return
        dialog = self.enlarge_confirm()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._install_enlarge()

    def _install_enlarge(self) -> None:
        progress = self.enlarge_progress()
        progress.show()
        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.processEvents()
        outcome = self._installer(progress)
        progress.close()
        if outcome == "ready":
            self._select_4k()
            return
        if outcome == "failed":
            EnlargeError(self).exec()
            return
        if outcome == "cancelled":
            return
        assert_never(outcome)

    def _select_4k(self) -> None:
        state = self.shell.layout.state
        if state is None:
            return
        state._presets = state._presets.with_available("output", "4k", available=True)
        chip = self.view.root.findChild(QPushButton, "chip-output-4k")
        if chip is not None:
            chip.setEnabled(True)
            if not chip.isCheckable():
                chip.setCheckable(True)
            group = self.view._chip_groups.get("output")
            if isinstance(group, QButtonGroup) and chip not in group.buttons():
                group.addButton(chip)
        self.shell.layout.choose("output", "4k")
        self.view.refresh()

    def _on_history(self, action: str, item_id: str) -> None:
        path = self.picture_path(item_id)
        if path is not None:
            self._pictures[item_id] = str(path)
        if action == "copy":
            # The shared menu already copied the prompt; add the seed so the
            # pair lands in the clipboard together, from the file when the
            # history carries no seed.
            item = self.shell.layout.item(item_id)
            seed = self._seeds.get(item_id)
            prompt = "" if item is None else item.prompt
            if seed is None and item is not None:
                seed = item.seed or None
            if (seed is None or not prompt) and path:
                from canvas.core.formats import read_metadata

                meta = read_metadata(Path(path))
                seed = seed or _meta_seed(meta)
                prompt = prompt or str(meta.get("prompt") or "")
            lines = [part for part in (prompt, f"seed={seed}" if seed is not None else "") if part]
            app = QApplication.instance()
            clipboard = None if app is None else app.clipboard()
            if clipboard is not None and lines:
                clipboard.setText("\n".join(lines))
            return
        if action == "use_input" and path:
            self.use_picture(Path(path))
            return
        if action == "open" and path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            return
        if action == "show_folder":
            series = self._series_folders.get(item_id)
            folder = series if series else str(Path(path).parent) if path else str(pictures_dir())
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _save_as(self) -> None:
        item_id = self.shell.layout.selected
        source = None if item_id is None else self.picture_path(item_id)
        if source is None:
            return
        chosen, _name = QFileDialog.getSaveFileName(self, "", str(source))
        if chosen:
            shutil.copyfile(source, save_as_target(source, Path(chosen)))

    def _manage_models(self) -> None:
        """Ask the worker whether a GPU is visible, then start MDL. The window never touches CUDA."""

        def done(reply: Reply) -> None:
            manifest = None
            if not (reply.ok and reply.data.get("gpu") is True):
                # No CUDA: list the models that are on disk whatever card they were made for.
                manifest = on_disk_manifest(self.shell.state_dir / "mdl-on-disk.json")
            self._start_mdl(mdl_argv(manifest))

        self.worker.submit("probe_gpu", {}, on_done=done)

    def _start_mdl(self, argv: list[str]) -> None:
        """Start the MDL GUI once. A running one is reported, a finished one is reaped first."""
        self._reap_mdl()
        if self._mdl_gui is not None:
            self.view.set_note(text("canvas.models.manage_running"))
            return
        try:
            process = subprocess.Popen(argv)
        except OSError:
            self.view.set_note(text("canvas.models.missing"))
            return
        if process is not None:
            self._mdl_gui = process
            self._mdl_timer.start()

    def _reap_mdl(self) -> None:
        """Collect the MDL child once it exited; say so when it failed."""
        process = self._mdl_gui
        if process is None:
            self._mdl_timer.stop()
            return
        try:
            code = process.poll()
        except Exception:  # noqa: BLE001 - a test double may not know poll()
            self._mdl_gui = None
            self._mdl_timer.stop()
            return
        if code is None:
            return
        self._mdl_gui = None
        self._mdl_timer.stop()
        if code != 0:
            self.view.set_note(text("canvas.models.manage_failed"))

    def _stop_mdl(self) -> None:
        """Stop the MDL GUI this window started.

        Only the process recorded here is touched: the GUI child this window
        Popen'ed. The ``voxprint_common.mdl worker`` it may spawn is the shared
        suite download queue (other apps download through it too), so it is
        never signalled — not by create time, pid scan, or any other heuristic.
        """
        self._mdl_timer.stop()
        process = self._mdl_gui
        self._mdl_gui = None
        if process is not None and _process_running(process):
            _terminate(process)



def _meta_seed(meta: Mapping[str, object]) -> int | None:
    """The seed recorded on a saved picture, or None when it carries none."""
    value = meta.get("seed")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _step_number(suffix: str) -> int | None:
    """The finished step count in a ``step=N`` stage suffix."""
    for part in suffix.split():
        name, _, value = part.partition("=")
        if name == "step" and value.isdigit():
            return int(value)
    return None


def _stage_label_key(stage: str) -> str:
    """The progress sentence key for one stage code, or empty to keep the count line."""
    name = stage.strip().lower()
    if not name:
        return ""
    if "encod" in name:
        return "canvas.stage.encode"
    if "decod" in name:
        return "canvas.stage.decode"
    if "save" in name:
        return "canvas.stage.save"
    if "hot" in name or "wait" in name:
        return "canvas.stage.wait"
    if "check" in name:
        return "canvas.stage.check"
    if "sample" in name or "generate" in name:
        return "canvas.stage.draw"
    if (
        "load" in name or "cast" in name or "place" in name or "pipeline" in name
        or "warm" in name or "transformer" in name or "encoder" in name or "tokenizer" in name
        or "unet" in name or "vae" in name or "model" in name
    ):
        return "canvas.stage.load"
    return ""


def _process_running(process: subprocess.Popen) -> bool:
    try:
        return process.poll() is None
    except Exception:  # noqa: BLE001 - a test double may not know poll()
        return False


def _terminate(process: subprocess.Popen, *, grace_s: float = 2.0) -> None:
    """SIGTERM a child, wait, then SIGKILL. Silence when it already exited."""
    import signal as _signal

    try:
        os.kill(process.pid, _signal.SIGTERM)
    except (OSError, AttributeError):
        pass
    deadline = time.monotonic() + grace_s
    while _process_running(process) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_running(process):
        try:
            # Windows has no SIGKILL; os.kill with 9 still ends the process there.
            os.kill(process.pid, getattr(_signal, "SIGKILL", 9))
        except (OSError, AttributeError):
            pass
        with contextlib.suppress(Exception):
            process.wait(timeout=grace_s)


def save_as_target(source: Path, chosen: Path) -> Path:
    """Keep the picture's own container: a copy renamed to ``.png`` would be WebP bytes under a PNG name."""
    if chosen.suffix.lower() == source.suffix.lower():
        return chosen
    if not chosen.suffix:
        return chosen.with_name(chosen.name + source.suffix)
    return chosen.with_suffix(source.suffix)


def mdl_argv(manifest: Path | None = None) -> list[str]:
    """``voxprint-mdl --gui --app canvas``, preferring the interpreter's folder.

    The installed venv is not on PATH, so a bare name does not start MDL there.
    """
    name = f"{CLI_NAME}.exe" if sys.platform == "win32" else CLI_NAME
    # absolute() keeps a venv symlink in place. resolve() would follow it into the base interpreter.
    bundled = Path(sys.executable).absolute().parent / name
    program = str(bundled) if bundled.is_file() else CLI_NAME
    argv = [program, "--gui", "--app", "canvas"]
    if manifest is not None:
        argv += ["--manifest", str(manifest)]
    return argv


def _role(mode: Mode) -> str:
    if mode is Mode.RESTYLE:
        return "init"
    if mode is Mode.POSE:
        return "pose"
    if mode is Mode.DEPTH:
        return "depth"
    if mode is Mode.EDGE:
        return "edge"
    if (
        mode is Mode.TEXT
        or mode is Mode.INPAINT
        or mode is Mode.OUTPAINT
    ):
        raise _NeedImage(text("error.job"))
    assert_never(mode)


def _safe_load(path: Path) -> RawImage | None:
    try:
        return load_rgb(path)
    except OSError:
        return None


def _check_text(report: dict[str, object]) -> str:
    head = text("canvas.check.ready") if report.get("ok") else text("canvas.check.failed")
    system = report.get("os")
    gpu = report.get("gpu")
    encoders = report.get("encoders")
    system_line = system.get("message", "") if isinstance(system, dict) else ""
    gpu_line = gpu.get("message", "") if isinstance(gpu, dict) else ""
    marks = ""
    if isinstance(encoders, dict):
        marks = ", ".join(f"{name}: {encoders[name]}" for name in encoders)
    lines = [
        head,
        f"{text('canvas.check.os')}: {system_line}",
        f"{text('canvas.check.gpu')}: {gpu_line}",
        f"{text('canvas.check.encoders')}: {marks}",
    ]
    return "\n".join(str(line) for line in lines)


def _busy_first(window, layout) -> None:
    """Keep the Canvas order: the job runs after the layout is busy.

    Yitro calls the generate hook before the busy phase. Canvas runs the whole
    job inside the hook and finishes or refuses it there, so the hook must run
    after ``generate`` has reviewed the prompt and set the busy phase.
    """
    original = layout.generate

    def generate(hook=None) -> bool:
        if hook is None:
            return original()
        if layout.phase is Phase.BUSY:
            return True
        if not str(layout.prompt).strip() and _batch_lines(window):
            # A batch from the prompts list with an empty prompt box: set the
            # busy phase the way the shared screen does. Every line is
            # reviewed before its picture, in _check_prompts.
            layout._clear_run()
            layout.phase = Phase.BUSY
            hook(layout.live_settings())
            return True
        if not original():
            return False
        hook(layout.live_settings())
        return True

    layout.generate = generate


def pictures_dir() -> Path:
    """Where the window keeps finished pictures: ``<Documents>/VOXPrint Projects/Canvas/pictures``."""
    path = projects_dir() / "pictures"
    path.mkdir(parents=True, exist_ok=True)
    return path
