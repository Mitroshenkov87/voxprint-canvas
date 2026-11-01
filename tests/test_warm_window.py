"""The window keeps the loaded pipeline between generations, and lets it go when it must."""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QPushButton

from canvas.models import runtime
from canvas.models.modes import Tier
from canvas.pipeline.pipeline import Pipeline
from canvas.ui.shell_layout import open_layout
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import World
from tests.test_photoshoot import IDEA
from tests.test_runtime_reuse import _GB64, _install, _render


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _window(tmp_path: Path, monkeypatch, gate) -> ShellWindow:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    weights = tmp_path / "weights"
    shell = open_layout(tmp_path / "state", gate=gate)
    return ShellWindow(
        shell, tier=Tier.GB16, enlarge=object(),
        runner=lambda job, inputs: _render(gate, weights, job.prompt, None),
    )


def _click(window: ShellWindow) -> None:
    window.view.set_prompt_text(IDEA)
    button = window.findChild(QPushButton, "generate")
    assert button is not None
    button.click()


def test_the_window_keeps_the_pipeline_between_two_generations(app, tmp_path: Path, monkeypatch) -> None:
    del app
    loads, _places = _install(monkeypatch)
    window = _window(tmp_path, monkeypatch, World().gate())
    _click(window)
    assert loads[0] == 1
    assert runtime._slot is not None
    assert runtime.warm_held()
    _click(window)
    assert loads[0] == 1
    assert len(window.shell.layout.history) == 2
    window.close()
    assert runtime._slot is None
    assert not runtime.warm_held()


def test_a_card_without_room_lets_the_pipeline_go(app, tmp_path: Path, monkeypatch) -> None:
    del app
    loads, _places = _install(monkeypatch)
    monkeypatch.setattr(runtime, "_warm_fits", lambda: False)
    window = _window(tmp_path, monkeypatch, World().gate())
    _click(window)
    assert runtime._slot is None
    _click(window)
    assert loads[0] == 2
    window.close()


def test_closing_releases_also_on_a_tier_that_parks(app, tmp_path: Path, monkeypatch) -> None:
    del app
    loads, _places = _install(monkeypatch)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: _GB64)
    window = _window(tmp_path, monkeypatch, World().gate())
    _click(window)
    _click(window)
    assert loads[0] == 1
    window.close()
    assert runtime._slot is None


def test_a_model_change_still_loads_again_inside_the_hold(tmp_path: Path, monkeypatch) -> None:
    loads, _places = _install(monkeypatch)
    gate = World().gate()
    runtime.hold_warm()
    try:
        _render(gate, tmp_path / "weights", IDEA, None)
        _render(gate, tmp_path / "weights", IDEA, None)
        assert loads[0] == 1
        _render(gate, tmp_path / "other", IDEA, None)
        assert loads[0] == 2
        _render(gate, tmp_path / "other", IDEA, None, Tier.GB24)
        assert loads[0] == 3
    finally:
        runtime.drop_warm(release=True)
    assert runtime._slot is None


def test_enlarge_parks_the_warm_pipeline_first(tmp_path: Path, monkeypatch) -> None:
    _install(monkeypatch)
    gate = World().gate()
    runtime.hold_warm()
    try:
        _render(gate, tmp_path / "weights", IDEA, None)
        assert runtime._slot is not None
        Pipeline(gate, object(), Tier.GB16)._release_models()
        # ram32 keeps no RAM copy, so parking drops the pipeline before Enlarge runs.
        assert runtime._slot is None
    finally:
        runtime.drop_warm(release=True)


def test_drop_without_a_hold_does_nothing(tmp_path: Path, monkeypatch) -> None:
    loads, _places = _install(monkeypatch)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: _GB64)
    gate = World().gate()
    _render(gate, tmp_path / "weights", IDEA, None)
    runtime.drop_warm()
    assert runtime._scope_depth == 0
    assert runtime._slot is not None
    assert loads[0] == 1
