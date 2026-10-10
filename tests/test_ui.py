"""The two tabs, the format list, and the neutral status line."""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QTabWidget

from canvas.app.info import APP_NAME, RESPONSIBILITY, THIRD_PARTY_NOTICE
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
