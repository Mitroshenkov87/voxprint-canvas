"""The Canvas window: the shared image screen, plus the tools it does not draw."""
from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from PIL import Image
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QMessageBox,
    QApplication,
    QButtonGroup,
    QDialog,
    QFileDialog,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
)

from voxprint_common.i18n import add_catalogs, set_language, tr
from voxprint_common.mdl.names import CLI_NAME
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.safety import ReviewedOutput
from voxprint_common.ui.simple_image import Phase, SimpleImageWidget as SimpleWidget
from canvas.app.i18n import text
from canvas.app.info import APP_VERSION, ENLARGE_NAME
from canvas.app.updates import check_for_update
from canvas.infra.enlarge import EnlargeClient
from canvas.ipc.client import WorkerClient, default_host
from canvas.ipc.schema import Reply, decode_event
from canvas.models.mdl_layout import on_disk_manifest
from canvas.models.mdl_registry import selectable
from canvas.models.lora_library import LibraryEntry
from canvas.models.user_lora import LoraIndex, discover, resolve_selected
from canvas.ui.lora_panel import ImportedDialog, LibraryDialog, dark, import_with_dialog, library_loras
from canvas.models.modes import Mode, Tier
from canvas.models.sizes import closest_aspect
from canvas.ui.about import AboutDialog
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
    index: int = 0
    job_id: str = ""


def job_fields(job: PictureJob) -> dict[str, object]:
    """``job`` as plain values the worker request can carry."""
    fields = job._asdict()
    fields["mode"] = job.mode.value
    fields["fmt"] = job.fmt.value
    return fields


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
        self._series_folders: dict[str, str] = load_series_folders(shell.state_dir)
        self._busy = False
        # The worker keeps the loaded pipeline between generations while this window is open.
        self._warm = False
        self._chrome_ready = False
        self.setObjectName("shell")
        self.setWindowTitle(tr("splash.app.canvas"))
        self.setStyleSheet(shell_style())
        self.setMinimumSize(shell.surface.min_width, shell.surface.min_height)
        self.resize(SHOT_WIDTH, SHOT_HEIGHT)
        self.view = SimpleWidget(
            shell.layout,
            shell.settings,
            on_generate=self._on_generate,
            on_command=self._on_command,
            on_improve=self._on_improve,
            on_history=self._on_history,
            on_refresh=self._on_refresh,
            surface=shell.surface,
        )
        self._expert = ExpertMode()
        self._expert.subscribe(self.view.set_expert)
        if self._expert.enabled():
            self.view.set_expert(True)
        _busy_first(shell.layout)
        self._split = mount(self)
        self.setCentralWidget(self._split)
        self.photoshoot = PhotoshootBar(self)
        show_real_pictures(self)
        self._chrome_ready = True
        polish(self)

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
        number = float(strength) if isinstance(strength, int | float) and not isinstance(strength, bool) else 0.7
        found = discover()
        index = LoraIndex(tuple(library_loras()) + found.available, found.skipped)
        return AddonsWizard(
            self, index, str(layout.get("shell.addons") or ""), number, self._save_addons, self._import_from_addons,
        )

    def lora_library(self) -> LibraryDialog:
        """The LoRA library: «Import LoRA», previews and remove."""
        return LibraryDialog(self, self.worker, self._library_changed)

    def _library_changed(self) -> None:
        self.refresh_loras()
        self.view.refresh()

    def import_lora(self, path: Path | None = None, on_done=None) -> None:
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

    def _pick_lora(self, key: str, strength: float) -> None:
        changed = str(self.shell.layout.get("shell.addons") or "") != key
        self.shell.layout.set_value("shell.lora_weight", strength)
        if changed:
            # Only a new pick changes the trigger chip; a strength step does not redraw.
            self.shell.layout.set_value("shell.addons", key)
            self.view.refresh()

    def lora_trigger(self) -> str:
        """The trigger word of the selected LoRA, or empty when nothing is selected."""
        selected = resolve_selected(str(self.shell.layout.get("shell.addons") or ""))
        if selected is None:
            return ""
        return selected.trigger

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

    def enlarge_confirm(self) -> EnlargeConfirm:
        """The question shown when 4K is chosen before Enlarge is installed."""
        return EnlargeConfirm(self, enlarge_question())

    def enlarge_progress(self) -> EnlargeProgress:
        """The install progress, with Cancel."""
        return EnlargeProgress(self)

    def _open_setup(self) -> None:
        self.setup_wizard().exec()

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
        """Release the warm pipeline with the window."""
        self.release_warm()
        super().closeEvent(event)

    def _generate(self, settings: object = None) -> bool:
        """Start one Generate press in the worker. True while the worker has the job."""
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
        run = _Run(job=job, total=max(1, job.copies), base_seed=layout.seed, layout_prompt=layout.prompt.strip())
        self._run = run
        self.worker.when_ready(lambda: self._check_prompts(run), lambda reason: self._stop_run(run, reason))
        return True

    def _check_prompts(self, run: _Run) -> None:
        self.worker.check_prompts(
            [run.layout_prompt, run.job.prompt], on_done=lambda reply: self._prompts_checked(run, reply),
        )

    def _prompts_checked(self, run: _Run, reply: Reply) -> None:
        layout = self.shell.layout
        if self._run is not run:
            return
        if layout.phase is not Phase.BUSY:
            self._end_run(run)
            self.view.refresh()
            return
        if not reply.ok:
            self._stop_run(run, reply.message, unavailable=reply.unavailable)
            return
        if run.job.output == "4k" and not self._enlarge_ready:
            self._stop_run(run, text("enlarge.hint", name=ENLARGE_NAME))
            return
        self._warm = True
        self._start_copy(run)

    def _start_copy(self, run: _Run) -> None:
        layout = self.shell.layout
        index = run.index
        if index:
            layout.seed = run.base_seed + index
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
        )

    def _copy_done(self, run: _Run, reply: Reply) -> None:
        layout = self.shell.layout
        run.job_id = ""
        if self._run is not run or layout.phase is not Phase.BUSY:
            self._discard(reply)
            self._end_run(run)
            self.view.refresh()
            return
        if not reply.ok:
            self._stop_run(run, reply.message, unavailable=reply.unavailable)
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

    def _stop_run(self, run: _Run, message: str, *, unavailable: bool = True) -> None:
        """End ``run`` with a reason. Unavailable is the default for anything the worker could not answer."""
        layout = self.shell.layout
        if layout.phase is Phase.BUSY:
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
        del job
        layout = self.shell.layout
        item_id = str(reply.data["item_id"])
        path = Path(str(reply.data["path"]))
        checks = self.shell.gate
        if isinstance(checks, WorkerChecks):
            checks.release(item_id, path)
        if layout.finish(item_id, ReviewedOutput(path=path)) is None:
            self.view.refresh()
            self.worker.discard(reply.data)
            return False
        folder = self.shell.state_dir / "pictures"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{item_id}{path.suffix}"
        shutil.move(str(path), target)
        self._pictures[item_id] = str(target)
        refresh_pictures(self)
        return True

    def _input_files(self, mode: Mode) -> list[dict[str, str]]:
        """The source picture and mask as files the worker reads. The worker decodes and checks them."""
        if mode is Mode.TEXT or mode is Mode.TEXT_CREATIVE:
            return []
        if self._picture is None:
            raise _NeedImage(text("error.job"))
        source = {"role": "init", "path": str(self._picture)}
        if mode is Mode.INPAINT or mode is Mode.OUTPAINT:
            if self._mask is None:
                raise _NeedImage(text("error.job"))
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
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.shell.state_dir / "pictures")))
            return
        if command_id == "models.manage":
            self._manage_models()
            return
        if command_id == "view.settings":
            AboutDialog(self).exec()
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
        path = self._pictures.get(item_id)
        if action == "use_input" and path:
            self.use_picture(Path(path))
            return
        if action == "open" and path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            return
        if action == "show_folder":
            series = self._series_folders.get(item_id)
            folder = series if series else str(Path(path).parent) if path else str(self.shell.state_dir / "pictures")
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _save_as(self) -> None:
        item_id = self.shell.layout.selected
        source = None if item_id is None else self._pictures.get(item_id)
        if source is None:
            return
        chosen, _name = QFileDialog.getSaveFileName(self, "", source)
        if chosen:
            shutil.copyfile(source, chosen)

    def _manage_models(self) -> None:
        """Ask the worker whether a GPU is visible, then start MDL. The window never touches CUDA."""

        def done(reply: Reply) -> None:
            manifest = None
            if not (reply.ok and reply.data.get("gpu") is True):
                # No CUDA: list the models that are on disk whatever card they were made for.
                manifest = on_disk_manifest(self.shell.state_dir / "mdl-on-disk.json")
            try:
                subprocess.Popen(mdl_argv(manifest))
            except OSError:
                self.view.set_note(text("canvas.models.missing"))

        self.worker.submit("probe_gpu", {}, on_done=done)


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
        or mode is Mode.TEXT_CREATIVE
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


def _busy_first(layout) -> None:
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
        if not original():
            return False
        hook(layout.live_settings())
        return True

    layout.generate = generate
