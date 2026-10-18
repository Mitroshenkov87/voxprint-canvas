"""The Canvas window: the shared image screen, plus the tools it does not draw."""
from __future__ import annotations

import shutil
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import assert_never

from PIL import Image
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QFileDialog,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
)

from voxprint_common.i18n import add_catalogs, tr
from voxprint_common.mdl.names import CLI_NAME
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.simple_image import Phase, SimpleImageWidget as SimpleWidget
from canvas.app.i18n import text
from canvas.app.info import APP_VERSION, ENLARGE_NAME
from canvas.app.selftest import selftest_report
from canvas.app.shell_gate import reviewed_output
from canvas.app.shell_run import RunStopped, ShellRunner, improve_prompt, load_rgb
from canvas.app.updates import check_for_update
from canvas.core.formats import extension, save_image
from canvas.infra.enlarge import EnlargeClient
from canvas.infra.vram import set_vram_cap_gb
from canvas.models.mdl_registry import selectable
from canvas.models.user_lora import discover, resolve_selected
from canvas.models.modes import Mode, Tier
from canvas.models.sizes import closest_aspect
from canvas.safety.image import RawImage
from canvas.safety.inputs import ImageRole, PendingInput
from canvas.ui.about import AboutDialog
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
    shell_style,
)
from canvas.ui.photoshoot_bar import PhotoshootBar
from canvas.ui.photoshoot_run import adult_content, load_series_folders, run_photoshoot
from canvas.ui.ui_call import run_off_ui
from canvas.ui.shell_layout import CanvasShell, PictureJob, job_from
from canvas.ui.wizards import AddonsWizard, DiagnosticsWizard, MaskWizard, PictureWizard, SetupWizard

Runner = Callable[[PictureJob, tuple[PendingInput, ...]], RawImage]
Improver = Callable[[str], str]


class _NeedImage(Exception):
    """A picture mode was started without the image it needs."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ShellWindow(QMainWindow):
    """Simple mode by default. Expert opens the side panel. The old window stays unused."""

    def __init__(
        self,
        shell: CanvasShell,
        *,
        tier: Tier,
        enlarge: object | None = None,
        runner: Runner | None = None,
        improver: Improver | None = None,
        installer: Installer | None = None,
    ) -> None:
        """Place the shared screen in the Canvas window.

        Args:
            shell: Headless state from :func:`canvas.ui.shell_layout.open_layout`.
            tier: The tier chosen at startup.
            enlarge: Enlarge client. A missing client is opened here.
            runner: Generates one picture. The default uses the pipeline.
            improver: Rewrites the prompt in place. The default uses the same gate.
            installer: Installs Enlarge. The default uses the shared model queue.
        """
        super().__init__()
        add_catalogs(Path(__file__).resolve().parent / "locales")
        self.shell = shell
        self.tier = tier
        self._enlarge = enlarge if enlarge is not None else EnlargeClient()
        self._enlarge_ready = bool(getattr(self._enlarge, "ready", lambda: False)())
        self._runner = runner if runner is not None else ShellRunner(
            gate=shell.gate.age_gate,
            tier=tier,
            enlarge=self._enlarge,
        )
        if improver is None:
            def improver(prompt: str) -> str:
                return improve_prompt(
                    gate=shell.gate.age_gate, prompt=prompt, tier=tier, adult=adult_content(shell.layout),
                )
        self._improver = improver
        self._installer = installer if installer is not None else run_shared_enlarge
        self._picture: Path | None = None
        self._mask: RawImage | None = None
        self._mask_file: Path | None = None
        self._pictures: dict[str, str] = {}
        self._series_folders: dict[str, str] = load_series_folders(shell.state_dir)
        self._busy = False
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
        return AddonsWizard(self, discover(), str(layout.get("shell.addons") or ""), number, self._save_addons)

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
        """The PC check. It does not run until a button is pressed."""
        return DiagnosticsWizard(self, self._run_check, self._run_update)

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

    def _open_check(self) -> None:
        self.diagnostics_wizard().exec()

    def _save_addons(self, path: str, strength: float) -> None:
        self.shell.layout.set_value("shell.addons", path)
        self.shell.layout.set_value("shell.lora_weight", strength)
        self.view.refresh()

    def _run_check(self, deep: bool) -> str:
        report = selftest_report(deep=deep)
        return _check_text(report)

    def _run_update(self) -> str:
        tag = check_for_update(APP_VERSION)
        if tag is None:
            return text("canvas.check.current")
        return text("canvas.check.newer", tag=tag)

    def _on_improve(self, current: str) -> str:
        if self._busy:
            return current
        self._busy = True
        try:
            return run_off_ui(lambda: self._improver(current))
        except Exception as exc:
            message = getattr(exc, "user_message", None)
            self.shell.layout.block(message if isinstance(message, str) and message else text("gate.unavailable"))
            self.view.refresh()
            return current
        finally:
            self._busy = False

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
        try:
            self._generate(settings)
        finally:
            self._busy = False

    def _generate(self, settings: object = None) -> None:
        if self.photoshoot.enabled:
            run_photoshoot(self)
            return
        layout = self.shell.layout
        live = dict(settings) if isinstance(settings, dict) else layout.live_settings()
        self._adopt_slots(live)
        if not selectable():
            layout.unavailable(text("canvas.models.missing"))
            self.view.refresh()
            return
        try:
            job = job_from(layout, live)
        except (ValueError, KeyError):
            layout.block(text("gate.unavailable"))
            self.view.refresh()
            return
        reason = self.shell.gate.review_prompt(job.prompt)
        if reason:
            layout.block(reason)
            self.view.refresh()
            return
        if job.output == "4k" and not self._enlarge_ready:
            layout.unavailable(text("enlarge.hint", name=ENLARGE_NAME))
            self.view.refresh()
            return
        self._apply_vram()
        total = max(1, job.copies)
        base_seed = layout.seed
        for index in range(total):
            if index:
                layout.seed = base_seed + index
                if not layout.generate():
                    self.view.refresh()
                    return
            if not self._advance(index, total, text("canvas.stage.prepare")):
                return
            try:
                fresh = self._fresh_inputs(job.mode)
                raw = run_off_ui(lambda fresh=fresh: self._runner(job, fresh))
            except _NeedImage as exc:
                layout.block(exc.message)
                self.view.refresh()
                return
            except RunStopped as exc:
                if exc.unavailable:
                    layout.unavailable(exc.message)
                else:
                    layout.block(exc.message)
                self.view.refresh()
                return
            if not self._advance(index + 1, total, text("canvas.stage.draw")):
                return
            if not self._keep(raw, job):
                return
        self.view.refresh()

    def _advance(self, step: int, total: int, label: str) -> bool:
        try:
            self.shell.layout.progress(step, total, label)
        except ValueError:
            self.view.refresh()
            return False
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        if self.shell.layout.phase is not Phase.BUSY:
            self.view.refresh()
            return False
        return True

    def _keep(self, raw: RawImage, job: PictureJob) -> bool:
        layout = self.shell.layout
        item_id = uuid.uuid4().hex
        self.shell.gate.stage(item_id, RawImage(raw.width, raw.height, raw.to_bytes()))
        if layout.finish(item_id, reviewed_output(raw)) is None:
            self.view.refresh()
            return False
        safe = self.shell.gate.kept.pop(item_id, None)
        if safe is None:
            layout.block(text("gate.unavailable"))
            self.view.refresh()
            return False
        folder = self.shell.state_dir / "pictures"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{item_id}{extension(job.fmt)}"
        save_image(safe, path, job.fmt)
        self._pictures[item_id] = str(path)
        return True

    def _fresh_inputs(self, mode: Mode) -> tuple[PendingInput, ...]:
        if mode is Mode.TEXT or mode is Mode.TEXT_CREATIVE:
            return ()
        if self._picture is None:
            raise _NeedImage(text("error.job"))
        source = load_rgb(self._picture)
        if mode is Mode.INPAINT or mode is Mode.OUTPAINT:
            if self._mask is None:
                raise _NeedImage(text("error.job"))
            mask = RawImage(self._mask.width, self._mask.height, self._mask.to_bytes())
            return (
                PendingInput(ImageRole.INIT, source),
                PendingInput(ImageRole.MASK, mask),
            )
        return (PendingInput(_role(mode), source),)

    def _apply_vram(self) -> None:
        value = str(self.shell.layout.get("shell.vram"))
        if value == "auto":
            set_vram_cap_gb(None)
            return
        if value in {"8", "12", "16", "24"}:
            set_vram_cap_gb(float(value))

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
        try:
            subprocess.Popen([CLI_NAME, "--gui", "--app", "canvas"])
        except OSError:
            self.view.set_note(text("canvas.models.missing"))


def _role(mode: Mode) -> ImageRole:
    if mode is Mode.RESTYLE:
        return ImageRole.INIT
    if mode is Mode.POSE:
        return ImageRole.POSE
    if mode is Mode.DEPTH:
        return ImageRole.DEPTH
    if mode is Mode.EDGE:
        return ImageRole.EDGE
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
