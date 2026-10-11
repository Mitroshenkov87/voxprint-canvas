"""The two tabs, the format list, and the neutral status line."""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QPlainTextEdit, QTabWidget

from canvas.app.identity import splash_line, version_label
from canvas.app.info import APP_NAME, ENLARGE_HINT, ENLARGE_RELEASES, RESPONSIBILITY, THIRD_PARTY_NOTICE
from canvas.core.formats import OutputFormat
from canvas.models.modes import Mode, Tier
from canvas.models.sizes import ASPECT_GROUPS
from canvas.infra.enlarge import EnlargeError
from canvas.safety.gate import GenerationBlocked
from canvas.ui.about import AboutDialog
from canvas.ui.main_window import EnhanceReply, MainWindow
from canvas.ui.splash import Splash


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _labels(box):
    return [box.itemText(index) for index in range(box.count())]


def _aspect_buttons(window):
    return {
        button.property("ratio"): button
        for button in window.findChildren(type(window.generate_button), "aspect")
    }


def _aspect_groups(window):
    groups = []
    for label in window.findChildren(type(window.status), "aspect-group"):
        parent = label.parentWidget()
        ratios = tuple(
            button.property("ratio")
            for button in parent.findChildren(type(window.generate_button), "aspect")
        )
        groups.append((label.text(), ratios))
    return groups


def _ink_span(icon) -> tuple[int, int]:
    image = icon.pixmap(22, 22).toImage()
    xs = []
    ys = []
    for y in range(image.height()):
        for x in range(image.width()):
            if image.pixelColor(x, y).alpha() > 0:
                xs.append(x)
                ys.append(y)
    return max(xs) - min(xs), max(ys) - min(ys)


def test_16gb_offers_instruction_and_hides_creative(app):
    del app
    window = MainWindow(lambda prompt, mode, fmt: "ok", Tier.GB16)
    tabs = window.findChild(QTabWidget)
    assert tabs is not None
    assert tabs.tabText(0) == "Text to image"
    assert tabs.tabText(1) == "Image to image"
    assert _labels(window.creative) == ["Text to image"]
    assert "Instruction edit" in _labels(window.image_mode)
    assert "Creative" not in _labels(window.creative)
    formats = _labels(window.format_box)
    assert formats == ["AVIF", "JPEG XL", "WebP"]
    assert window.format_box.currentData() is OutputFormat.AVIF
    assert window.windowTitle() == APP_NAME
    footer = window.findChild(type(window.status), "footer")
    assert footer is not None and footer.text() == RESPONSIBILITY
    window.show()
    assert window.addons_box.isChecked() is False
    assert window.addons_notice.isHidden()
    window.addons_box.setChecked(True)
    assert window.addons_notice.isVisible()
    assert window.addons_notice.text() == THIRD_PARTY_NOTICE
    window.addons_box.setChecked(False)
    assert window.addons_notice.isHidden()
    assert _aspect_groups(window) == list(ASPECT_GROUPS)
    buttons = _aspect_buttons(window)
    assert all(not button.icon().isNull() for button in buttons.values())
    wide_w, wide_h = _ink_span(buttons["16:9"].icon())
    tall_w, tall_h = _ink_span(buttons["9:16"].icon())
    square_w, square_h = _ink_span(buttons["1:1"].icon())
    assert wide_w > wide_h
    assert tall_h > tall_w
    assert abs(square_w - square_h) <= 1
    assert window.current_aspect() == "1:1"
    assert window.findChild(type(window.generate_button), "orientation") is None
    assert window.fourk.text() == "Output size: 4K"
    assert window.fourk.isEnabled() is False
    assert window.fourk_hint.text() == ENLARGE_HINT
    assert window.fourk_hint.isHidden() is False
    assert window.install_enlarge.text() == "Install Enlarge"
    assert window.install_enlarge.isHidden() is False
    labels = [label.text() for label in tabs.widget(0).findChildren(QLabel)]
    assert "Quality" not in labels
    assert window.creative.isVisible() is False
    tabs.setCurrentIndex(1)
    assert window.aspect_row.isEnabled()
    window.close()


def test_24gb_offers_creative_and_instruction_and_reports_status(app):
    del app
    seen = {}

    def on_generate(prompt, mode, fmt):
        seen["prompt"] = prompt
        seen["mode"] = mode
        seen["fmt"] = fmt
        return "ready"

    window = MainWindow(on_generate, Tier.GB24)
    tabs = window.findChild(QTabWidget)
    assert tabs is not None
    labels = [label.text() for label in tabs.widget(0).findChildren(QLabel)]
    assert "Quality" not in labels
    assert "Creative" in _labels(window.creative)
    assert "Instruction edit" in _labels(window.image_mode)
    window.prompt.setPlainText("a red boat")
    window.generate_button.click()
    assert seen == {"prompt": "a red boat", "mode": Mode.TEXT, "fmt": OutputFormat.AVIF}
    assert window.status.text() == "ready"
    tabs = window.findChild(QTabWidget)
    assert tabs is not None
    tabs.setCurrentIndex(1)
    window.generate_button.click()
    assert seen["mode"] is Mode.RESTYLE

    def refuse(prompt, mode, fmt):
        del prompt, mode, fmt
        raise GenerationBlocked("prompt")

    window.on_generate = refuse
    window.generate_button.click()
    assert window.status.text() == "This request cannot be completed."

    def enlarge_failed(prompt, mode, fmt):
        del prompt, mode, fmt
        raise EnlargeError("bad_input", "bad_input: cannot identify image file", False)

    window.on_generate = enlarge_failed
    window.generate_button.click()
    assert window.status.text() == "bad_input: cannot identify image file"
    window.close()


def test_aspect_groups_and_input_default_and_fourk_when_enlarge_is_installed(app):
    del app
    window = MainWindow(lambda prompt, mode, fmt: "ok", Tier.GB24, enlarge_ready=True)
    window.show()
    assert window.fourk.isEnabled() is True
    assert window.fourk_hint.isHidden() is True
    assert window.install_enlarge.isHidden() is True
    buttons = _aspect_buttons(window)
    buttons["16:9"].click()
    assert window.current_aspect() == "16:9"
    assert buttons["16:9"].isChecked()
    buttons["9:16"].click()
    assert window.current_aspect() == "9:16"
    assert buttons["16:9"].isChecked() is False
    buttons["21:9"].click()
    assert window.current_aspect() == "21:9"
    fresh = MainWindow(lambda prompt, mode, fmt: "ok", Tier.GB16)
    fresh.set_input_size(928, 1664)
    assert fresh.current_aspect() == "9:16"
    fresh_buttons = _aspect_buttons(fresh)
    fresh_buttons["1:1"].click()
    fresh.set_input_size(1664, 928)
    assert fresh.current_aspect() == "1:1"
    window.close()
    fresh.close()


def test_install_enlarge_opens_the_releases_page(app, monkeypatch):
    del app
    opened: list[str] = []
    monkeypatch.setattr(
        "canvas.ui.main_window.QDesktopServices.openUrl",
        lambda url: opened.append(url.toString()) or True,
    )
    window = MainWindow(lambda prompt, mode, fmt: "ok", Tier.GB16)
    window.install_enlarge.click()
    assert opened == [ENLARGE_RELEASES]
    window.close()


def test_about_and_splash_show_the_release_identity(app, monkeypatch):
    del app
    window = MainWindow(lambda prompt, mode, fmt: "ok", Tier.GB24)
    opened: list[str] = []

    def _exec(dialog: AboutDialog) -> int:
        opened.append(dialog.version.text())
        return 0

    monkeypatch.setattr(AboutDialog, "exec", _exec)
    window.about_button.click()
    assert opened == [version_label()]
    dialog = AboutDialog(window)
    assert dialog.version.text() == version_label()
    assert dialog.version.objectName() == "about-version"
    splash = Splash()
    assert splash.version_line == splash_line()
    assert splash.version_line != version_label()
    splash.close()
    window.close()


def test_enhance_rewrites_the_prompt_in_place_and_revert_restores_it(app):
    del app
    seen: list[dict[str, object]] = []
    busy: list[tuple[str, bool]] = []

    def on_generate(prompt, mode, fmt, original=None):
        seen.append({"prompt": prompt, "mode": mode, "fmt": fmt, "original": original})
        return "ready"

    def on_enhance(text: str) -> EnhanceReply:
        busy.append((window.enhance_button.text(), window.enhance_button.isEnabled()))
        return EnhanceReply(f"english: {text}", "")

    window = MainWindow(on_generate, Tier.GB16, on_enhance=on_enhance)
    window.show()
    window.prompt.setPlainText("красная лодка")
    assert window.enhance_button.text() == "Enhance"
    assert window.enhance_button.toolTip() == "Improve the prompt"
    assert "translat" not in window.enhance_button.text().casefold()
    assert "translat" not in window.enhance_button.toolTip().casefold()
    assert window.enhance_button.parentWidget() is window.prompt.parentWidget()
    assert window.findChild(QPlainTextEdit, "enhanced") is None
    assert window.findChild(type(window.enhance_button), "use-enhanced") is None
    assert window.revert_button.toolTip() == "Revert"
    assert window.revert_button.text() == ""
    assert window.revert_button.icon().isNull() is False
    assert window.revert_button.isEnabled() is False
    window.enhance_button.click()
    assert busy == [("Enhancing…", False)]
    assert window.enhance_button.text() == "Enhance"
    assert window.enhance_button.isEnabled() is True
    assert window.prompt.toPlainText() == "english: красная лодка"
    assert window.revert_button.isEnabled() is True
    window.prompt.setFocus()
    QTest.keyClick(window.prompt, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert window.prompt.toPlainText() == "красная лодка"
    window.prompt.redo()
    assert window.prompt.toPlainText() == "english: красная лодка"
    window.prompt.setPlainText("english: красная лодка, edited")
    window.generate_button.click()
    assert seen[-1]["prompt"] == "english: красная лодка, edited"
    assert seen[-1]["original"] == "красная лодка"
    window.revert_button.click()
    assert window.prompt.toPlainText() == "красная лодка"
    assert window.revert_button.isEnabled() is False
    window.generate_button.click()
    assert seen[-1]["prompt"] == "красная лодка"
    assert seen[-1]["original"] is None
    window.close()


def test_enhance_keeps_the_prompt_and_shows_a_note_when_the_rewrite_is_refused(app):
    del app

    def on_enhance(text: str) -> EnhanceReply:
        return EnhanceReply(text, "The enhanced prompt changed a stated age, so the original prompt was used.")

    window = MainWindow(lambda prompt, mode, fmt: "ok", Tier.GB16, on_enhance=on_enhance)
    window.prompt.setPlainText("a 25-year-old woman")
    window.enhance_button.click()
    assert window.prompt.toPlainText() == "a 25-year-old woman"
    assert window.status.text() == "The enhanced prompt changed a stated age, so the original prompt was used."
    assert window.revert_button.isEnabled() is False
    window.close()


def test_confirm_download_uses_the_notice(app, monkeypatch):
    del app
    seen: list[str] = []

    def question(parent, title, notice, buttons):
        del parent, buttons
        seen.append(f"{title}:{notice}")
        return QMessageBox.StandardButton.No

    monkeypatch.setattr("canvas.ui.main_window.QMessageBox.question", question)
    window = MainWindow(lambda prompt, mode, fmt: "ok", Tier.GB16)
    assert window.confirm_download("ControlNet lite is 1.0 GB. Estimated time 1 minute.") is False
    assert seen == ["Download:ControlNet lite is 1.0 GB. Estimated time 1 minute."]
    window.close()
