"""Photoshoot is one option, a read-only shot list, and Cancel."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QCheckBox, QLabel, QLineEdit, QPushButton, QSpinBox, QWidget

from canvas.models.modes import Mode, Tier
from canvas.safety.gate import SafetyUnavailable
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import rgb
from tests.worker_support import inline_window, settle
from tests.test_photoshoot import IDEA, _payload
from voxprint_common.i18n import set_language


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    set_language("en")
    yield tmp_path
    set_language("en")


def _window(home: Path, runner) -> ShellWindow:
    return inline_window(
        home / "state", runner=runner, improver=lambda text: text, tier=Tier.GB16, enlarge=object(),
    )


def test_the_frame_count_starts_at_six(app, home: Path) -> None:
    window = _window(home, lambda job, inputs: rgb())
    window.show()
    app.processEvents()
    frames = window.findChild(QSpinBox, "photoshootFrames")
    label = window.findChild(QLabel, "photoshootFramesLabel")
    assert frames is not None and label is not None
    assert frames.value() == 6
    assert frames.minimum() == 1
    assert frames.maximum() == 99
    assert frames.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.UpDownArrows
    assert label.text() == "Frames"
    assert window.photoshoot.count == 6
    window.findChild(QCheckBox, "photoshootMode").click()
    app.processEvents()
    assert frames.isVisible()
    assert label.isVisible()


def test_the_frame_count_shows_its_widest_value(app, home: Path) -> None:
    window = _window(home, lambda job, inputs: rgb())
    window.setFixedSize(1920, 1080)
    window.show()
    window.findChild(QCheckBox, "photoshootMode").click()
    frames = window.findChild(QSpinBox, "photoshootFrames")
    frames.setValue(frames.maximum())
    for _ in range(3):
        app.processEvents()
    need = frames.fontMetrics().horizontalAdvance(frames.text())
    edit = frames.findChild(QLineEdit)
    room = edit.contentsRect().width() - edit.textMargins().left() - edit.textMargins().right()
    # The box border and the edit field both have to leave the digits whole.
    assert frames.width() >= need + 2 * 8
    assert room >= need


def test_the_option_sits_centred_under_generate(app, home: Path) -> None:
    window = _window(home, lambda job, inputs: rgb())
    window.setFixedSize(1920, 1080)
    window.show()
    for _ in range(3):
        app.processEvents()
    mode = window.findChild(QCheckBox, "photoshootMode")
    generate = window.findChild(QPushButton, "generate")
    panel = window._action_row

    def box(widget):
        return widget.geometry().translated(widget.parentWidget().mapTo(panel, widget.parentWidget().rect().topLeft()))

    for ticked in (False, True):
        if mode.isChecked() != ticked:
            mode.click()
            for _ in range(3):
                app.processEvents()
        button = box(generate)
        option = box(mode)
        assert option.top() >= button.bottom(), (ticked, option, button)
        assert abs(option.center().x() - button.center().x()) <= 2, (ticked, option, button)
        # Generate is centred in the panel and keeps its whole label.
        assert abs(button.center().x() - panel.rect().center().x()) <= 2
        assert generate.fontMetrics().horizontalAdvance(generate.text()) + 16 <= generate.width()
    frames = box(window.findChild(QSpinBox, "photoshootFrames"))
    assert frames.left() > option.right() and frames.right() <= panel.width()
    assert abs(frames.center().y() - option.center().y()) <= 2


def test_progress_uses_the_chosen_frame_count(app, home: Path, monkeypatch) -> None:
    def runner(job, inputs):
        del job, inputs
        return rgb()

    def writer(prompt: str) -> str:
        body = json.loads(prompt)
        return _payload(IDEA, int(body["count"]))

    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    window = _window(home, runner)
    window.session.series_writer = writer
    window.view.set_prompt_text(IDEA)
    window.show()
    window.findChild(QCheckBox, "photoshootMode").click()
    window.findChild(QSpinBox, "photoshootFrames").setValue(4)
    window.findChild(QPushButton, "generate").click()
    settle(window)
    rows = window.findChildren(QWidget, "photoshootScriptRow")
    assert len(rows) == 4
    assert window.photoshoot.script_window.isVisible()
    assert window.findChild(QLabel, "photoshootFrame").text() == "Shot 4 of 4"


def test_the_option_leaves_the_shell_create_button_in_place(app, home: Path) -> None:
    window = _window(home, lambda job, inputs: rgb())
    window.show()
    app.processEvents()
    mode = window.findChild(QCheckBox, "photoshootMode")
    assert mode is not None and mode.text() == "Photoshoot"
    assert window.findChild(QPushButton, "photoshootMode") is None
    generate = window.findChild(QPushButton, "generate")
    assert generate is not None and generate.isVisible()
    mode.click()
    app.processEvents()
    assert mode.isChecked()
    assert generate.isVisible()
    assert window.findChild(QPushButton, "photoshootCreate") is None
    assert window.findChild(QPushButton, "photoshootWrite") is None
    assert window.findChild(QPushButton, "chip-photoshoot-count-6") is None
    assert window.findChild(QPushButton, "photoshootHandle") is None
    assert window.findChild(QPushButton, "photoshootMenu") is None


def test_generate_writes_a_read_only_series(app, home: Path, monkeypatch) -> None:
    prompts: list[str] = []

    def runner(job, inputs):
        del inputs
        prompts.append(job.prompt)
        return rgb()

    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    window = _window(home, runner)
    window.session.series_writer = lambda _prompt: _payload(IDEA)
    window.view.set_prompt_text(IDEA)
    window.show()
    window.findChild(QCheckBox, "photoshootMode").click()
    window.findChild(QPushButton, "generate").click()
    settle(window)
    assert len(prompts) == 6
    assert all(
        "A single photograph, " in prompt and "50 mm, still, " in prompt and prompt.index("50 mm, still, ") < prompt.index(IDEA)
        for prompt in prompts
    )
    assert all(IDEA in prompt for prompt in prompts)
    lines = window.findChildren(QWidget, "photoshootScriptRow")
    assert len(lines) == 6
    assert all(row.isVisible() for row in lines)
    assert window.photoshoot.script_window.isVisible()
    assert window.findChild(QLineEdit, "photoshootCamera") is None
    assert window.findChild(QPushButton, "generate").isVisible()
    assert (home / "state" / "pictures" / "series").exists()
    assert window._series_folders
    assert all(path.endswith(".webp") for path in window._pictures.values())
    recorded = json.loads((home / "state" / "pictures" / "series-map.json").read_text(encoding="utf-8"))
    assert set(recorded) == set(window._series_folders)
    again = _window(home, runner)
    assert again._series_folders == window._series_folders


def test_expert_does_not_add_a_camera_control(app, home: Path, monkeypatch) -> None:
    prompts: list[str] = []

    def runner(job, inputs):
        del inputs
        prompts.append(job.prompt)
        return rgb()

    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    window = _window(home, runner)
    window.session.series_writer = lambda _prompt: _payload(IDEA)
    window.view.set_prompt_text(IDEA)
    window.set_expert(True)
    window.show()
    window.findChild(QCheckBox, "photoshootMode").click()
    window.findChild(QPushButton, "generate").click()
    settle(window)
    assert prompts and all("50 mm" in prompt for prompt in prompts)
    assert window.findChild(QLineEdit, "photoshootCamera") is None
    assert window.findChild(QPushButton, "photoshootMenu") is None


def test_a_blocked_frame_is_labelled_and_is_not_the_hero(app, home: Path, monkeypatch) -> None:
    modes: list[Mode] = []

    def runner(job, inputs):
        del inputs
        modes.append(job.mode)
        return rgb()

    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    window = _window(home, runner)
    gate = window.session.gate
    real = gate.review_image

    def review_image(raw, prompt=""):
        if "three-quarter" in prompt:
            raise SafetyUnavailable("classifier_error")
        return real(raw, prompt)

    gate.review_image = review_image
    window.session.series_writer = lambda _prompt: _payload(IDEA)
    window.view.set_prompt_text(IDEA)
    window.show()
    window.findChild(QCheckBox, "photoshootMode").click()
    window.findChild(QPushButton, "generate").click()
    settle(window)
    labels = [label.text() for label in window.findChildren(QLabel, "photoshootScriptStatus")]
    assert any("Shot blocked" in text for text in labels)
    assert modes[0] is Mode.TEXT
    assert all(mode is Mode.TEXT for mode in modes)
    folder = next(path for path in (home / "state" / "pictures" / "series").iterdir() if path.is_dir())
    assert not (folder / "02.png").exists()
    assert (folder / "01.webp").is_file()


def test_cancel_keeps_the_frames_already_finished(app, home: Path, monkeypatch) -> None:
    prompts: list[str] = []

    def runner(job, inputs):
        del inputs
        prompts.append(job.prompt)
        if len(prompts) == 1:
            window.photoshoot.cancel_series()
        return rgb()

    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    window = _window(home, runner)
    window.session.series_writer = lambda _prompt: _payload(IDEA)
    window.view.set_prompt_text(IDEA)
    window.show()
    window.findChild(QCheckBox, "photoshootMode").click()
    window.findChild(QPushButton, "generate").click()
    settle(window)
    assert len(prompts) == 1
    folder = next(path for path in (home / "state" / "pictures" / "series").iterdir() if path.is_dir())
    assert (folder / "01.webp").is_file()
    assert not (folder / "02.png").exists()
    cancel = window.findChild(QPushButton, "photoshootCancel")
    assert cancel is not None and cancel.text() == "Cancel"


def test_the_frame_stepper_stays_next_to_photoshoot_across_modes(app, home: Path) -> None:
    """The count stepper appears beside Photoshoot in simple, a preset, and Expert."""
    window = _window(home, lambda job, inputs: rgb())
    window.setFixedSize(1920, 1080)
    window.show()
    mode = window.findChild(QCheckBox, "photoshootMode")
    frames = window.findChild(QSpinBox, "photoshootFrames")
    label = window.findChild(QLabel, "photoshootFramesLabel")
    assert mode is not None and frames is not None
    assert not frames.isVisible()
    mode.click()
    for _ in range(4):
        app.processEvents()
    assert frames.isVisible() and label.isVisible()
    assert frames.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.UpDownArrows
    start = frames.value()
    frames.stepUp()
    assert frames.value() == start + 1
    frames.stepDown()
    assert frames.value() == start

    def geometry():
        for _ in range(4):
            app.processEvents()
        mode_box = mode.mapTo(window, mode.rect().topLeft())
        frames_box = frames.mapTo(window, frames.rect().topLeft())
        return (frames.isVisible(), label.isVisible(),
                mode_box.x(), mode_box.y(), frames_box.x(), frames_box.y(),
                frames.width(), frames.height(), frames.value())

    simple = geometry()
    assert simple[0] and simple[1]
    assert simple[4] > simple[2]  # stepper to the right of the checkbox
    assert abs(simple[5] - simple[3]) <= 4
    for button in window.findChildren(QPushButton):
        if button.text() == "Best" and button.isVisible():
            button.click()
            break
    preset = geometry()
    window.set_expert(True)
    expert = geometry()
    window.set_expert(False)
    back = geometry()
    assert simple[0:2] == preset[0:2] == expert[0:2] == back[0:2] == (True, True)
    # Same row relationship in every mode (x/y of the pair may shift slightly
    # if the left column wraps, but the stepper stays immediately right).
    for row in (simple, preset, expert, back):
        assert row[4] > row[2]
        assert abs(row[5] - row[3]) <= 4
