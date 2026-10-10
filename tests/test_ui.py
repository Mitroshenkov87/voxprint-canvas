"""The two tabs, the format list, and the neutral status line."""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QTabWidget

from canvas.app.info import APP_NAME, ENLARGE_HINT, RESPONSIBILITY, THIRD_PARTY_NOTICE
from canvas.core.formats import OutputFormat
from canvas.models.modes import Mode, Tier
from canvas.safety.gate import GenerationBlocked
from canvas.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _labels(box):
    return [box.itemText(index) for index in range(box.count())]


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
    assert [window.aspect_box.itemText(index) for index in range(window.aspect_box.count())] == [
        "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9",
    ]
    assert window.current_aspect() == "1:1"
    assert window.orientation.text() == "Landscape"
    assert window.fourk.text() == "Output size: 4K"
    assert window.fourk.isEnabled() is False
    assert window.fourk_hint.text() == ENLARGE_HINT
    assert window.fourk_hint.isHidden() is False
    tabs.setCurrentIndex(1)
    assert window.aspect_box.isEnabled()
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


def test_aspect_toggle_and_input_default_and_fourk_when_enlarge_is_installed(app):
    del app
    window = MainWindow(lambda prompt, mode, fmt: "ok", Tier.GB24, enlarge_ready=True)
    window.show()
    assert window.fourk.isEnabled() is True
    assert window.fourk_hint.isHidden() is True
    window.aspect_box.setCurrentIndex(window.aspect_box.findData("16:9"))
    assert window.current_aspect() == "16:9"
    assert window.orientation.text() == "Landscape"
    window.orientation.click()
    assert window.current_aspect() == "9:16"
    assert window.orientation.text() == "Portrait"
    window.orientation.click()
    assert window.current_aspect() == "16:9"
    window.aspect_box.setCurrentIndex(window.aspect_box.findData("21:9"))
    window.orientation.click()
    assert window.aspect_box.currentData() == "21:9"
    assert window.current_aspect() == "9:21"
    fresh = MainWindow(lambda prompt, mode, fmt: "ok", Tier.GB16)
    fresh.set_input_size(928, 1664)
    assert fresh.current_aspect() == "9:16"
    fresh.aspect_box.setCurrentIndex(fresh.aspect_box.findData("1:1"))
    fresh.set_input_size(1664, 928)
    assert fresh.current_aspect() == "1:1"
    window.close()
    fresh.close()
