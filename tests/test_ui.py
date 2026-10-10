"""The two tabs, the format list, and the neutral status line."""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QTabWidget

from canvas.app.identity import version_label
from canvas.app.info import APP_NAME, ENLARGE_HINT, ENLARGE_RELEASES, RESPONSIBILITY, THIRD_PARTY_NOTICE
from canvas.core.formats import OutputFormat
from canvas.models.modes import Mode, Tier
from canvas.models.sizes import ASPECT_GROUPS
from canvas.safety.gate import GenerationBlocked
from canvas.ui.about import AboutDialog
from canvas.ui.main_window import MainWindow
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


def test_16gb_hides_creative_and_instruction(app):
    del app
    window = MainWindow(lambda prompt, mode, fmt: "ok", Tier.GB16)
    tabs = window.findChild(QTabWidget)
    assert tabs is not None
    assert tabs.tabText(0) == "Text to image"
    assert tabs.tabText(1) == "Image to image"
    assert _labels(window.creative) == ["Text to image"]
    assert "Instruction edit" not in _labels(window.image_mode)
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
    assert splash.version_line == version_label()
    splash.close()
    window.close()
