"""CLI parity controls: batch seeds/prompts, copy prompt+seed, Ctrl+Generate, window size memory."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLineEdit, QPlainTextEdit, QPushButton, QToolButton

from canvas.app.i18n import text
from canvas.models.modes import Tier
from canvas.ui.shell_screen import ShellWindow, _meta_seed
from tests.fakes import rgb
from tests.worker_support import inline_window, settle


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    yield tmp_path


def _window(home: Path, runner=None):
    return inline_window(home / "state", tier=Tier.GB16, enlarge=object(), runner=runner)


def _batch(window: ShellWindow, prompts: str = "", seeds: str = "") -> None:
    toggle = window.findChild(QToolButton, "batchToggle")
    assert toggle is not None
    if not toggle.isChecked():
        toggle.click()
    window.findChild(QPlainTextEdit, "batchPrompts").setPlainText(prompts)
    window.findChild(QLineEdit, "batchSeeds").setText(seeds)


def test_a_seed_list_sets_the_batch_payload(app, home: Path, monkeypatch) -> None:
    from tests.fakes import rgb as fake_rgb

    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    jobs: list[dict] = []

    def runner(job, inputs):
        jobs.append({"prompt": job.prompt, "seed": job.seed})
        return fake_rgb(8, 8)

    window = _window(home, runner)
    window.view.set_prompt_text("a red bicycle")
    _batch(window, seeds="11, 22, 33")
    window.findChild(QPushButton, "generate").click()
    settle(window)
    assert [row["seed"] for row in jobs] == [11, 22, 33]
    assert all("a red bicycle" in row["prompt"] for row in jobs)
    window.close()


def test_batch_prompts_draw_one_line_each(app, home: Path, monkeypatch) -> None:
    from tests.fakes import rgb as fake_rgb

    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    jobs: list[dict] = []

    def runner(job, inputs):
        jobs.append({"prompt": job.prompt, "seed": job.seed})
        return fake_rgb(8, 8)

    window = _window(home, runner)
    window.view.set_prompt_text("")
    _batch(window, prompts="a red bicycle\na blue boat", seeds="7, 9")
    window.findChild(QPushButton, "generate").click()
    settle(window)
    assert [row["seed"] for row in jobs] == [7, 9]
    assert [row["prompt"].split(". ")[-1] for row in jobs] == [
        "a red bicycle photorealistic photograph",
        "a blue boat photorealistic photograph",
    ]
    window.close()


def test_batch_prompts_refuse_a_typed_prompt_and_bad_pairs(app, home: Path, monkeypatch) -> None:
    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    window = _window(home, runner=lambda job, inputs: rgb(8, 8))
    layout = window.shell.layout
    window.view.set_prompt_text("a red bicycle")
    _batch(window, prompts="a blue boat")
    window.findChild(QPushButton, "generate").click()
    assert layout.phase is not None
    assert layout.reason == text("canvas.batch.mixed")
    window.view.set_prompt_text("")
    window.findChild(QPlainTextEdit, "batchPrompts").setPlainText("one\ntwo")
    window.findChild(QLineEdit, "batchSeeds").setText("5")
    window.shell.layout._clear_run()  # reset the phase
    window.view.refresh()
    window.findChild(QPushButton, "generate").click()
    assert layout.reason == text("canvas.batch.mismatch")
    window.close()


def test_copy_places_the_prompt_and_seed_on_the_clipboard(app, home: Path, monkeypatch) -> None:
    state = home / "state"
    state.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "items": [{"id": "pic-01", "prompt": "a red boat", "presets": [], "seed": 3}]}
    (state / "image-history.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    window = _window(home)
    window.show()
    app.processEvents()
    window._seeds["pic-01"] = 123456
    window._on_history("copy", "pic-01")
    clipboard = app.clipboard().text()
    assert "a red boat" in clipboard
    assert "seed=123456" in clipboard
    window.close()


def test_meta_seed_reads_an_int_and_nothing_else() -> None:
    assert _meta_seed({"seed": 42}) == 42
    assert _meta_seed({"seed": "42"}) is None
    assert _meta_seed({"seed": True}) is None
    assert _meta_seed({}) is None


def test_ctrl_enter_starts_a_generate(app, home: Path, monkeypatch) -> None:
    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    drawn: list[int] = []

    def runner(job, inputs):
        drawn.append(1)
        return rgb(8, 8)

    window = _window(home, runner)
    window.view.set_prompt_text("a red bicycle")
    shortcut = window.findChild(type(window._generate_shortcut), "generateShortcut")
    assert shortcut is not None
    assert shortcut.key().toString() in {"Ctrl+Return", "Ctrl+Enter"}
    shortcut.activated.emit()  # the key press itself is Qt's; this is the binding's target
    settle(window)
    assert drawn == [1]
    window.close()


def test_the_window_remembers_its_size(app, home: Path) -> None:
    first = _window(home)
    first.resize(1377, 855)
    first.close()
    saved = home / "state" / "window-geometry.json"
    assert saved.is_file()
    second = _window(home)
    assert second.width() == 1377 and second.height() == 855
    second.close()
