"""The Canvas window: the shared image screen, plus the tools it does not draw."""
from __future__ import annotations

import shutil
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import assert_never

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow

from voxprint_common.i18n import add_catalogs, tr
from voxprint_common.mdl.names import CLI_NAME
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.simple_image import Phase, SimpleImageWidget as SimpleWidget
from canvas.app.i18n import text
from canvas.app.info import APP_VERSION, ENLARGE_NAME, ENLARGE_RELEASES
from canvas.app.selftest import selftest_report
from canvas.app.shell_run import RunStopped, ShellRunner, improve_prompt, load_rgb
from canvas.app.updates import check_for_update
from canvas.core.formats import extension, save_image
from canvas.infra.enlarge import EnlargeClient
from canvas.infra.vram import set_vram_cap_gb
from canvas.models.mdl_registry import selectable
from canvas.models.modes import Mode, Tier
from canvas.models.sizes import closest_aspect
from canvas.safety.image import RawImage
from canvas.safety.inputs import ImageRole, PendingInput
from canvas.ui.about import AboutDialog
from canvas.ui.shell_chrome import SHOT_HEIGHT, SHOT_WIDTH, fit_panel, mount, polish, shell_style
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
    ) -> None:
        """Place the shared screen in the Canvas window.

        Args:
            shell: Headless state from :func:`canvas.ui.shell_layout.open_layout`.
            tier: The tier chosen at startup.
            enlarge: Enlarge client. A missing client is opened here.
            runner: Generates one picture. The default uses the pipeline.
            improver: Rewrites the prompt in place. The default uses the same gate.
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
                return improve_prompt(gate=shell.gate.age_gate, prompt=prompt, tier=tier)
        self._improver = improver
        self._picture: Path | None = None
        self._mask: RawImage | None = None
        self._pictures: dict[str, str] = {}
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
        self._split = mount(self)
        self.setCentralWidget(self._split)
        self._chrome_ready = True
        polish(self)

    def set_expert(self, enabled: bool) -> None:
        """Store Expert mode and show or hide the side panel.

        The shared screen is told once. ``set_enabled`` already notifies it
        when the stored value changes, and telling it again leaves the
        previous fields on top of the new ones.

        Args:
            enabled: True shows the expert values beside the picture.
        """
        previous = self._expert.enabled()
        self._expert.set_enabled(enabled)
        if previous == enabled:
            self.view.set_expert(enabled)
        fit_panel(self, self._expert.enabled())

    def _on_refresh(self) -> None:
        if self._chrome_ready:
            polish(self)

    def use_picture(self, path: Path) -> None:
        """Use ``path`` as the source picture and match its shape.

        Args:
            path: An image file.
        """
        image = load_rgb(path)
        self._picture = path
        self.shell.layout.set_value("canvas.source", "picture")
        self.shell.layout.choose("size", closest_aspect(image.width, image.height))
        self.view.refresh()

    def use_mask(self, image: RawImage) -> None:
        """Keep ``image`` as the inpaint mask and select Paint over.

        Args:
            image: White pixels are painted over.
        """
        self._mask = image
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
        number = float(strength) if isinstance(strength, int | float) else 0.7
        return AddonsWizard(self, str(layout.get("shell.addons") or ""), number, self._save_addons)

    def diagnostics_wizard(self) -> DiagnosticsWizard:
        """The PC check. It does not run until a button is pressed."""
        return DiagnosticsWizard(self, self._run_check, self._run_update)

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
        try:
            return self._improver(current)
        except Exception as exc:
            message = getattr(exc, "user_message", None)
            self.shell.layout.block(message if isinstance(message, str) and message else text("gate.unavailable"))
            self.view.refresh()
            return current

    def _on_generate(self) -> None:
        layout = self.shell.layout
        if not selectable():
            layout.unavailable(text("canvas.models.missing"))
            self.view.refresh()
            return
        try:
            job = job_from(layout)
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
                raw = self._runner(job, fresh)
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
        if layout.finish(item_id) is None:
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
            QDesktopServices.openUrl(QUrl(ENLARGE_RELEASES))

    def _on_history(self, action: str, item_id: str) -> None:
        path = self._pictures.get(item_id)
        if action == "use_input" and path:
            self.use_picture(Path(path))
            return
        if action == "open" and path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            return
        if action == "show_folder":
            folder = str(Path(path).parent) if path else str(self.shell.state_dir / "pictures")
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
        or mode is Mode.INSTRUCTION
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
