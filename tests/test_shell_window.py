"""The new window keeps the picture tools and does not offer instruction editing."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QWidget,
)

from canvas.app.i18n import text
from canvas.models.mdl_registry import enlarge_download_bytes
from canvas.ui.enlarge_flow import enlarge_question
from canvas.models.modes import Tier
from canvas.ui.shell_layout import open_layout
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import World
from tools.grab_shell_screens import prepare
from voxprint_common.i18n import set_language, tr
from voxprint_common.mdl.present import format_size
from voxprint_common.suite_settings import set_expert_mode
from voxprint_common.ui.simple_image import Phase


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _close_shell_windows():
    """Close the windows a test showed, so later tests do not paint them."""
    yield
    instance = QApplication.instance()
    if instance is None:
        return
    for widget in instance.topLevelWidgets():
        if isinstance(widget, ShellWindow) and widget.isVisible():
            widget.close()


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    set_language("en")
    yield tmp_path
    set_language("en")


def _assert_chips_whole(window, chips) -> None:
    panel = window.findChild(QWidget, "promptPanel")
    assert panel is not None
    for index, chip in enumerate(chips):
        font = QFont(chip.font())
        font.setWeight(QFont.Weight.DemiBold)
        assert chip.width() >= QFontMetrics(font).horizontalAdvance(chip.text()) + 12
        right = chip.mapTo(panel, chip.rect().topRight()).x()
        assert 0 <= chip.mapTo(panel, chip.rect().topLeft()).x() and right < panel.width()
        for other in chips[index + 1:]:
            assert not chip.geometry().intersects(other.geometry())


def _window(home: Path, *, runner=None, improver=None, installer=None, enlarge_installed: bool = False) -> ShellWindow:
    shell = open_layout(home / "state", gate=World().gate(), enlarge_installed=enlarge_installed)
    return ShellWindow(
        shell,
        tier=Tier.GB16,
        enlarge=object(),
        runner=runner,
        improver=improver,
        installer=installer,
    )


def test_simple_mode_hides_expert_words_and_keeps_picture_tools(app, home: Path) -> None:
    window = _window(home)
    window.show()
    app.processEvents()
    assert window.view.root.findChild(QPushButton, "shell.steps") is None
    setup = window.setup_wizard()
    assert window.findChild(QPushButton, "chip-change-restyle") is None
    assert setup.findChild(QPushButton, "setup-change-restyle") is not None
    assert setup.findChild(QPushButton, "setup-change-inpaint") is not None
    assert setup.findChild(QPushButton, "setup-change-pose") is not None
    assert setup.findChild(QPushButton, "setup-change-depth") is not None
    assert setup.findChild(QPushButton, "setup-change-edge") is not None
    assert setup.findChild(QLabel, "setupTitle") is not None
    assert setup.findChild(QLabel, "setup-hint-source").text()
    assert setup.findChild(QLabel, "setup-hint-variation").text()
    assert setup.findChild(QLabel, "setup-hint-restyle").text()
    assert setup.findChild(QPushButton, "setup-instruction-on") is None
    setup.findChild(QPushButton, "setup-change-inpaint").click()
    assert window.shell.layout.get("canvas.change") == "inpaint"
    assert window.findChild(QPushButton, "chip-instruction-on") is None
    picture = window.findChild(QPushButton, "actionPicture")
    mask = window.findChild(QPushButton, "actionMask")
    assert picture is not None and picture.isVisible()
    assert mask is not None and not mask.isVisible()
    assert window.findChild(QPushButton, "actionAddons") is None
    assert window.findChild(QAction, "actionAddons") is not None
    assert window.findChild(QAction, "actionCheck") is not None
    assert window.findChild(QAction, "actionSetup") is not None
    help_menu = window.findChild(QMenu, "menuHelp")
    assert help_menu is not None and len(help_menu.actions()) == 2
    proportions = window.findChild(QPushButton, "proportionsMenu")
    assert proportions is not None and proportions.isVisible() and proportions.text().startswith("Proportions:")
    image_size = window.findChild(QPushButton, "imageSizeMenu")
    assert image_size is not None and image_size.isVisible() and image_size.text().startswith("Image size:")
    assert window.findChild(QPushButton, "revertPrompt").isHidden()
    square = window.findChild(QPushButton, "chip-size-1-1")
    assert square is not None and not square.isVisible()
    detail = window.findChild(QPushButton, "chip-detail-light")
    assert detail is not None and not detail.isVisible()
    fourk = window.findChild(QPushButton, "chip-output-4k")
    assert fourk is not None and fourk.isEnabled()
    assert fourk.text() == "4K (needs Enlarge)"
    reason = window.findChild(QLabel, "reason-output-4k")
    assert reason is not None and not reason.isVisible()
    install = window.findChild(QPushButton, "unavailable-output-4k")
    assert install is not None and not install.isVisible()
    footer = window.findChild(QLabel, "footer")
    assert footer is not None and footer.text() == text("notice.responsibility")
    window.resize(1280, 800)
    app.processEvents()
    stage = window.findChild(QWidget, "stage")
    history = window.findChild(QWidget, "historyStrip")
    assert stage is not None and stage.height() >= 300
    assert history is not None and history.isVisible() and history.height() >= 88
    assert history.mapTo(window, history.rect().bottomLeft()).y() <= window.height()
    window.set_expert(True)
    app.processEvents()
    panel = window.findChild(QWidget, "expertPanel")
    assert panel is not None and panel.isVisible() and panel.width() >= 300
    assert panel.height() >= window.height() - 4
    stage_at = stage.mapTo(window, stage.rect().topLeft())
    panel_at = panel.mapTo(window, panel.rect().topLeft())
    assert panel_at.x() >= stage_at.x() + stage.width() - 8
    assert window.findChild(QLabel, "expertChip") is None
    assert window.findChild(QPushButton, "expertResetAll") is None
    style_names = ("photo", "cinematic", "portrait", "illustration", "anime", "3d", "painting", "sketch")
    style_chips = [window.findChild(QPushButton, f"chip-style-{name}") for name in style_names]
    assert all(chip is not None and chip.isVisible() for chip in style_chips)
    # The left panel may wrap the style chips (line count depends on the system font);
    # each chip keeps its full label inside the panel.
    _assert_chips_whole(window, style_chips)
    spins = [item for item in window.findChildren(QSpinBox, "shell.steps") if item.isVisible()]
    assert len(spins) == 1
    spin = spins[0]
    assert spin.width() < 400
    spin.setValue(33)
    scroll = window.findChild(QScrollArea, "expertScroll")
    form = window.findChild(QWidget, "expertForm")
    assert scroll is not None and form is not None
    if form.height() > scroll.viewport().height():
        assert scroll.verticalScrollBar().maximum() > 0
    window.set_expert(False)
    app.processEvents()
    panel = window.findChild(QWidget, "expertPanel")
    assert panel is not None and not panel.isVisible()
    window.set_expert(True)
    app.processEvents()
    assert window.findChild(QSpinBox, "shell.steps").value() == 33


def test_improve_and_revert_stay_in_the_same_field(app, home: Path) -> None:
    del app
    window = _window(home, improver=lambda prompt: prompt + " in warm light")
    window.view.set_prompt_text("a red bicycle")
    improve = window.findChild(QPushButton, "improvePrompt")
    back = window.findChild(QPushButton, "revertSegment")
    assert improve is not None and back is not None
    assert back.toolTip() == back.accessibleName() == "Revert to my prompt"
    assert back.isHidden()
    improve.click()
    box = window.findChild(QPlainTextEdit, "promptBox")
    assert "warm light" in box.toPlainText()
    assert not back.isHidden()
    back.click()
    assert box.toPlainText() == "a red bicycle"
    assert back.isHidden()


def test_a_manual_edit_hides_the_back_segment(app, home: Path) -> None:
    del app
    window = _window(home, improver=lambda prompt: prompt + " in warm light")
    window.view.set_prompt_text("a red bicycle")
    window.findChild(QPushButton, "improvePrompt").click()
    back = window.findChild(QPushButton, "revertSegment")
    assert not back.isHidden()
    box = window.findChild(QPlainTextEdit, "promptBox")
    box.setPlainText(box.toPlainText() + ", at dusk")
    assert back.isHidden()
    assert not window.findChild(QPushButton, "revertPrompt").isEnabled()


def test_generate_without_models_says_they_are_not_ready(app, home: Path) -> None:
    del app
    window = _window(home)
    window.view.set_prompt_text("a red bicycle")
    window.findChild(QPushButton, "generate").click()
    assert window.shell.layout.phase is Phase.UNAVAILABLE
    assert window.shell.layout.reason == text("canvas.models.missing")


def test_mask_and_picture_wizards_open(app, home: Path, tmp_path: Path) -> None:
    del app
    window = _window(home)
    picture = window.picture_wizard()
    assert picture.findChild(QPushButton, "pictureOpen") is not None
    path = tmp_path / "dot.png"
    from PIL import Image

    Image.new("RGB", (8, 8), (20, 30, 40)).save(path)
    picture.load_path(path)
    picture.findChild(QPushButton, "pictureUse").click()
    assert window._picture == path
    assert window.shell.layout.get("canvas.source") == "picture"
    mask = window.mask_wizard()
    board = mask.findChild(QWidget, "maskSurface")
    assert board is not None
    assert mask.findChild(QLabel, "maskTitle") is not None
    assert mask.findChild(QLabel, "maskHint").text() == text("canvas.mask.hint")
    slider = mask.findChild(QSlider, "maskSize")
    assert slider is not None and slider.minimum() < slider.value() < slider.maximum()
    mask.findChild(QPushButton, "maskUse").click()
    assert window._mask is not None
    assert window.shell.layout.get("canvas.change") == "inpaint"
    addons = window.addons_wizard()
    assert addons.findChild(QLabel, "thirdPartyNotice").text() == text("notice.third_party")
    check = window.diagnostics_wizard()
    assert check.findChild(QPushButton, "checkRun") is not None
    assert check.findChild(QLabel, "checkReport").text() == ""


def test_a_finished_picture_keeps_the_result_actions(app, home: Path) -> None:
    state = home / "state"
    state.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "items": [{"id": "pic-01", "prompt": "a red boat", "presets": [], "seed": 3}]}
    (state / "image-history.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    window = _window(home)
    labels = [action.text() for action in window.view.history_menu("pic-01").actions()]
    assert tr("simple.history.delete") in labels
    assert tr("simple.history.show_folder") in labels
    window.resize(1280, 800)
    window.show()
    window.view._select("pic-01")
    app.processEvents()
    bar = window.findChild(QWidget, "actionBar")
    assert bar is not None and bar.isVisible()
    for name in ("moreLike", "savePicture", "openFolder", "usePrompt", "useInput", "deletePicture"):
        button = bar.findChild(QPushButton, name)
        assert button is not None and button.isVisible() and button.text()
    result = window.findChild(QLabel, "result")
    assert result is not None and result.pixmap() is not None and not result.pixmap().isNull()


def test_a_hand_edit_uses_the_shared_reset(app, home: Path) -> None:
    window = _window(home)
    window.resize(1280, 800)
    window.show()
    window.set_expert(True)
    app.processEvents()
    assert window.findChild(QLabel, "expertChip") is None
    assert window.findChild(QPushButton, "expertResetAll") is None
    window.shell.layout.set_value("shell.steps", 33)
    window.view.refresh()
    app.processEvents()
    spins = [item for item in window.findChildren(QSpinBox, "shell.steps") if item.isVisible()]
    assert len(spins) == 1 and spins[0].value() == 33
    chip = window.findChild(QPushButton, "chip-quality-balanced")
    assert chip.text() == tr("simple.preset.changed", name=tr("simple.quality.balanced"))
    row = window.findChild(QPushButton, "reset-quality")
    assert row is not None and not row.isVisible()
    resets = [item for item in window.findChildren(QPushButton, "expertReset") if item.isVisible()]
    assert len(resets) == 1
    assert resets[0].text() == tr("simple.preset.reset", name=tr("simple.quality.balanced"))


def test_simple_shots_keep_the_expert_panel_closed(app, home: Path, monkeypatch) -> None:
    state = home / "state"
    state.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "items": [{"id": "pic-01", "prompt": "a red boat", "presets": [], "seed": 3}]}
    (state / "image-history.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    set_expert_mode(True)
    for language, locale in (("en", "en_US.UTF-8"), ("ru", "ru_RU.UTF-8")):
        monkeypatch.setenv("LANG", locale)
        monkeypatch.setenv("LC_ALL", locale)
        set_language(language)
        for name in ("simple", "simple-result"):
            set_expert_mode(True)
            window = _window(home)
            prepare(window, name)
            window.resize(1280, 800)
            window.show()
            app.processEvents()
            panel = window.findChild(QWidget, "expertPanel")
            toggle = window.findChild(QPushButton, "expertToggle")
            assert panel is not None and not panel.isVisible()
            assert toggle is not None and not toggle.isChecked()
            if name == "simple-result":
                result = window.findChild(QLabel, "result")
                assert result is not None and result.pixmap() is not None and not result.pixmap().isNull()
            window.close()


def test_russian_buttons_keep_their_full_labels(app, home: Path, monkeypatch) -> None:
    monkeypatch.setenv("LANG", "ru_RU.UTF-8")
    monkeypatch.setenv("LC_ALL", "ru_RU.UTF-8")
    set_language("ru")
    window = _window(home)
    window.resize(1280, 800)
    window.show()
    window.set_expert(True)
    app.processEvents()
    expected = {
        "chip-quality-balanced": "Сбалансировано",
        "expertToggle": "Режим эксперта",
    }
    for name, words in expected.items():
        button = window.findChild(QPushButton, name)
        assert button.text() == words
        font = QFont(button.font())
        font.setWeight(QFont.Weight.DemiBold)
        advance = QFontMetrics(font).horizontalAdvance(words)
        assert button.width() >= advance + 28
    style_names = ("photo", "cinematic", "portrait", "illustration", "anime", "3d", "painting", "sketch")
    style_chips = [window.findChild(QPushButton, f"chip-style-{name}") for name in style_names]
    assert all(chip is not None and chip.isVisible() for chip in style_chips)
    # The left panel may wrap the style chips (line count depends on the system font);
    # each chip keeps its full label inside the panel.
    _assert_chips_whole(window, style_chips)
    assert not style_chips[3].geometry().intersects(style_chips[4].geometry())
    combo = window.findChild(QComboBox, "canvas.output_size")
    assert combo is not None and combo.isVisible()
    assert combo.property("fittedText") == combo.currentText()
    assert "…" not in str(combo.property("fittedText"))
    setup = window.setup_wizard()
    assert setup.findChild(QPushButton, "setup-source-words").text() == "С текста"
    assert setup.findChild(QPushButton, "setup-source-picture").text() == "С картинки"
    assert setup.findChild(QPushButton, "setup-variation-quick").text() == "Обычно"
    assert setup.findChild(QPushButton, "setup-variation-varied").text() == "Смелее"
    back = window.findChild(QPushButton, "revertSegment")
    assert back is not None and back.toolTip() == "Вернуть мой запрос"
    assert window.findChild(QPushButton, "proportionsMenu").text().startswith("Пропорции:")
    assert window.findChild(QPushButton, "imageSizeMenu").text().startswith("Размер изображения:")
    fourk = window.findChild(QPushButton, "chip-output-4k")
    assert fourk is not None and fourk.isEnabled() and fourk.text() == "4K (нужен Enlarge)"
    install = window.findChild(QPushButton, "unavailable-output-4k")
    assert install is not None and not install.isVisible()
    question = window.enlarge_confirm()
    amount = enlarge_download_bytes()
    expected = (
        text("canvas.enlarge.confirm")
        if amount is None
        else text("canvas.enlarge.confirm.size", size=format_size(amount))
    )
    assert question.findChild(QLabel, "enlargePrompt").text() == expected
    assert question.findChild(QPushButton, "enlargeInstall").text() == "Установить"
    assert question.findChild(QPushButton, "enlargeCancel").text() == "Отмена"
    question.close()
    restyle = setup.findChild(QLabel, "setup-hint-restyle")
    depth = setup.findChild(QLabel, "setup-hint-depth")
    assert restyle is not None and depth is not None
    assert restyle.width() == depth.width() >= 200
    mask = window.mask_wizard()
    assert mask.findChild(QLabel, "maskNeed").text() == text("canvas.mask.need")


def test_the_new_screen_does_not_mention_temporary_editing() -> None:
    root = Path("canvas/ui")
    blob = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("shell_*.py"))
    blob += "\n" + (root / "wizards.py").read_text(encoding="utf-8")
    assert "edit.unavailable" not in blob
    assert "canvas.instruction" not in blob
    ids = json.loads(Path("canvas/mdl.json").read_text(encoding="utf-8"))
    folded = blob.casefold()
    for row in ids["models"]:
        assert row["id"].casefold() not in folded


def test_ui_sources_do_not_name_the_placeholder_models() -> None:
    blob = "\n".join(path.read_text(encoding="utf-8") for path in Path("canvas/ui").rglob("*.py")).casefold()
    catalog = json.loads(Path("canvas/mdl.json").read_text(encoding="utf-8"))
    for row in catalog["models"]:
        assert row["id"].casefold() not in blob


def _dismiss(window: ShellWindow, button_name: str) -> None:
    def press() -> None:
        dialog = window.findChild(QDialog, "enlargeConfirm")
        assert dialog is not None
        dialog.findChild(QPushButton, button_name).click()

    QTimer.singleShot(0, press)
    window.findChild(QPushButton, "chip-output-4k").click()


def test_cancelling_enlarge_stays_on_standard(app, home: Path) -> None:
    called: list[str] = []
    window = _window(home, installer=lambda _progress: called.append("ran") or "ready")
    window.show()
    app.processEvents()
    seen: dict[str, str] = {}

    def cancel() -> None:
        dialog = window.findChild(QDialog, "enlargeConfirm")
        assert dialog is not None
        seen["prompt"] = dialog.findChild(QLabel, "enlargePrompt").text()
        assert dialog.findChild(QPushButton, "enlargeInstall").text() == "Install"
        assert dialog.findChild(QPushButton, "enlargeCancel").text() == "Cancel"
        dialog.findChild(QPushButton, "enlargeCancel").click()

    QTimer.singleShot(0, cancel)
    window.findChild(QPushButton, "chip-output-4k").click()
    assert called == []
    assert seen["prompt"] == text("canvas.enlarge.confirm")
    assert window.shell.layout.presets["output"] == "standard"
    standard = window.findChild(QPushButton, "chip-output-standard")
    fourk = window.findChild(QPushButton, "chip-output-4k")
    assert standard is not None and standard.isChecked()
    assert fourk is not None and not fourk.isChecked()


def test_confirming_enlarge_selects_4k_after_install(app, home: Path) -> None:
    seen: dict[str, str] = {}

    def installer(progress) -> str:
        cancel = progress.findChild(QPushButton, "enlargeProgressCancel")
        label = progress.findChild(QLabel, "enlargeProgressLabel")
        seen["cancel"] = "" if cancel is None else cancel.text()
        seen["label"] = "" if label is None else label.text()
        return "ready"

    window = _window(home, installer=installer)
    window.show()
    _dismiss(window, "enlargeInstall")
    app.processEvents()
    assert seen["cancel"] == "Cancel"
    assert seen["label"] == text("canvas.enlarge.progress")
    fourk = window.findChild(QPushButton, "chip-output-4k")
    assert fourk is not None and fourk.text() == "4K" and fourk.isEnabled() and fourk.isChecked()
    assert window.shell.layout.presets["output"] == "4k"
    install = window.findChild(QPushButton, "unavailable-output-4k")
    assert install is not None and not install.isVisible()


def test_a_failed_enlarge_install_stays_on_standard(app, home: Path) -> None:
    found: dict[str, str] = {}

    def installer(_progress) -> str:
        def close() -> None:
            error = window.findChild(QDialog, "enlargeError")
            label = None if error is None else error.findChild(QLabel, "enlargeErrorText")
            found["text"] = "" if label is None else label.text()
            if error is not None:
                error.findChild(QPushButton, "enlargeErrorClose").click()

        QTimer.singleShot(0, close)
        return "failed"

    window = _window(home, installer=installer)
    window.show()
    _dismiss(window, "enlargeInstall")
    assert found["text"] == text("canvas.enlarge.failed")
    assert window.shell.layout.presets["output"] == "standard"
    fourk = window.findChild(QPushButton, "chip-output-4k")
    standard = window.findChild(QPushButton, "chip-output-standard")
    assert fourk is not None and fourk.text() == "4K (needs Enlarge)" and not fourk.isChecked()
    assert standard is not None and standard.isChecked()
    del app


def test_an_installed_enlarge_chip_reads_4k(app, home: Path) -> None:
    window = _window(home, enlarge_installed=True)
    window.show()
    app.processEvents()
    fourk = window.findChild(QPushButton, "chip-output-4k")
    assert fourk is not None and fourk.text() == "4K" and fourk.isEnabled()
    assert window.findChild(QPushButton, "unavailable-output-4k") is None
    fourk.click()
    app.processEvents()
    dialog = window.findChild(QDialog, "enlargeConfirm")
    assert dialog is None or not dialog.isVisible()
    assert window.shell.layout.presets["output"] == "4k"
    assert fourk.isChecked()


def test_the_enlarge_question_uses_the_registry_size(home: Path) -> None:
    assert text("canvas.enlarge.confirm", "en") == "4K needs VOXPrint Enlarge. Install it?"
    assert text("canvas.enlarge.confirm", "ru") == "Для 4K нужен VOXPrint Enlarge. Установить?"
    assert text("canvas.enlarge.confirm.size", "en") == "4K needs VOXPrint Enlarge (≈ {size}). Install it?"
    assert enlarge_download_bytes() is None
    folder = home / "enlarge-app"
    folder.mkdir()
    payload = {
        "schema": 1,
        "app": "enlarge",
        "models": [
            {
                "id": "weights",
                "version": "1",
                "vram_tier": "8gb",
                "files": [
                    {
                        "path": "model.bin",
                        "size": 1572864,
                        "sha256": "ab" * 32,
                        "mirrors": ["https://example.com/model.bin"],
                    }
                ],
            }
        ],
    }
    (folder / "mdl.json").write_text(json.dumps(payload), encoding="utf-8")
    state = home / "state"
    state.mkdir(parents=True, exist_ok=True)
    registry = {"apps": [{"id": "enlarge", "path": str(folder)}]}
    (state / "installed-apps.json").write_text(json.dumps(registry) + "\n", encoding="utf-8")
    assert enlarge_download_bytes() == 1572864
    assert enlarge_question() == text("canvas.enlarge.confirm.size", size=format_size(1572864))


def test_yitro_slots_and_live_settings_reach_the_job(app, home: Path, tmp_path: Path) -> None:
    del app
    from PIL import Image

    from canvas.safety.image import RawImage
    from canvas.ui.shell_layout import job_from

    window = _window(home)
    layout = window.shell.layout
    assert layout.source_picture and layout.mask
    path = tmp_path / "dot.png"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(path)
    window.use_picture(path)
    assert layout.get("image.source") == str(path)
    window.use_mask(RawImage(8, 8, b"\xff" * 192))
    assert Path(str(layout.get("image.mask"))).is_file()
    live = dict(layout.live_settings())
    live["shell.steps"] = 7
    live["shell.sampler"] = "euler"
    job = job_from(layout, live)
    assert job.steps == 7
    assert job.sampler == "euler"
    assert window.shell.gate.age_gate is not None


def test_picture_count_is_typed_and_cancel_keeps_finished_pictures(app, home: Path, monkeypatch) -> None:
    from tests.fakes import rgb

    from canvas.ui.shell_screen import ShellWindow

    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    drawn = {"n": 0}
    labels: list[str] = []

    def runner(job, inputs):
        del job, inputs
        drawn["n"] += 1
        return rgb(8, 8)

    window = _window(home, runner=runner)
    window.resize(1280, 800)
    window.show()
    app.processEvents()
    field = window.findChild(QLineEdit, "pictureCount")
    assert field is not None and field.isVisible() and field.text() == "1"
    for name in ("chip-count-1", "chip-count-2", "chip-count-4"):
        chip = window.findChild(QPushButton, name)
        assert chip is not None and not chip.isVisible()
    field.setText("12")
    field.editingFinished.emit()
    assert field.text() == "12"
    assert window.shell.layout.batch_size() == 12
    field.clear()
    field.editingFinished.emit()
    assert field.text() == "12"
    assert window.shell.layout.batch_size() == 12
    field.setText("12")
    assert not field.insert("abc")
    assert field.text() == "12"
    field.setText("3")
    field.editingFinished.emit()
    assert window.shell.layout.batch_size() == 3
    original = ShellWindow._advance

    def wrapped(self, step, total, label):
        labels.append(label)
        if self is window and drawn["n"] and self.shell.layout.history:
            self.shell.layout.cancel()
        return original(self, step, total, label)

    monkeypatch.setattr(ShellWindow, "_advance", wrapped)
    window.view.set_prompt_text("a red bicycle")
    window.findChild(QPushButton, "generate").click()
    app.processEvents()
    assert drawn["n"] == 1
    assert labels[0] == text("canvas.progress.count", current=1, total=3)
    assert len(window.shell.layout.history) == 1
    saved = Path(window._pictures[window.shell.layout.history[0].item_id])
    assert saved.is_file()
    stage = window.findChild(QWidget, "stage")
    history = window.findChild(QWidget, "historyStrip")
    assert stage is not None and stage.height() >= 300
    assert history is not None and history.height() >= 88


def test_generate_keeps_the_picture_through_the_gate(app, home: Path, monkeypatch) -> None:
    del app
    from tests.fakes import rgb

    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    window = _window(home, runner=lambda job, inputs: rgb(8, 8))
    window.view.set_prompt_text("a red bicycle")
    window.findChild(QPushButton, "generate").click()
    layout = window.shell.layout
    assert layout.phase is Phase.READY, layout.reason
    assert len(layout.history) == 1
    assert Path(window._pictures[layout.history[0].item_id]).is_file()


def test_proportions_and_image_size_menus_choose_presets(app, home: Path) -> None:
    window = _window(home)
    window.show()
    app.processEvents()
    proportions = window.findChild(QPushButton, "proportionsMenu")
    proportions.menu().aboutToShow.emit()
    names = [action.objectName() for action in proportions.menu().actions() if action.objectName()]
    assert names == [
        "proportion-16-9", "proportion-21-9", "proportion-3-2", "proportion-4-3",
        "proportion-1-1", "proportion-9-16", "proportion-2-3", "proportion-3-4",
    ]
    next(a for a in proportions.menu().actions() if a.objectName() == "proportion-16-9").trigger()
    app.processEvents()
    assert window.shell.layout.presets["size"] == "16:9"
    assert proportions.text() == "Proportions: 16:9  ▾"
    image_size = window.findChild(QPushButton, "imageSizeMenu")
    image_size.menu().aboutToShow.emit()
    labels = [action.text() for action in image_size.menu().actions()]
    assert labels == ["Standard", "4K (needs Enlarge)"]


@pytest.mark.parametrize("language", ["ru", "en"])
@pytest.mark.parametrize("width", [1280, 1100, 960])
def test_result_buttons_show_every_label_whole(app, home: Path, language: str, width: int) -> None:
    set_language(language)
    state = home / "state"
    state.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "items": [{"id": "pic-01", "prompt": "a red boat", "presets": [], "seed": 3}]}
    (state / "image-history.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    window = _window(home)
    window.resize(width, 800)
    window.show()
    window.view._select("pic-01")
    for _ in range(3):
        app.processEvents()
    bar = window.findChild(QWidget, "actionBar")
    assert bar is not None and bar.isVisible()
    names = ("moreLike", "savePicture", "openFolder", "usePrompt", "useInput", "deletePicture")
    buttons = [bar.findChild(QPushButton, name) for name in names]
    for index, button in enumerate(buttons):
        assert button is not None and button.isVisible() and button.text()
        font = QFont(button.font())
        font.setWeight(QFont.Weight.DemiBold)
        assert button.width() >= QFontMetrics(font).horizontalAdvance(button.text()) + 16, button.objectName()
        geometry = button.geometry()
        assert geometry.left() >= 0 and geometry.right() < bar.width(), button.objectName()
        assert geometry.bottom() < bar.height(), button.objectName()
        right = button.mapTo(window, button.rect().topRight()).x()
        assert right < window.width(), button.objectName()
        holder = bar.parentWidget()
        assert button.mapTo(holder, button.rect().topRight()).x() < holder.width(), button.objectName()
        for other in buttons[index + 1:]:
            assert not geometry.intersects(other.geometry()), (button.objectName(), other.objectName())
