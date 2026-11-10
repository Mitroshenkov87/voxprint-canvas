"""The Photoshoot script model and its non-modal window.

The model is headless: rows append as shots are written, carry the LoRA
alias, strength, and the injected trigger, and link to the kept frame. The
window is separate from the image stage, reopens from the Script (N) button,
and never stops the series when it closes.
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QLabel, QPushButton, QWidget

from canvas.models.modes import Tier
from canvas.models.user_lora import UserLora
from canvas.pipeline.photoshoot import Shot
from canvas.ui.photoshoot_script import PhotoshootScriptModel, ScriptLora, series_loras
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import rgb
from tests.worker_support import inline_window, settle
from tests.test_photoshoot import _payload
from voxprint_common.i18n import set_language

IDEA = "a ceramic bowl on a wooden table, studio light"
PLACE = "wooden table"
SHOTS = tuple(
    Shot(scene, "still", "50 mm", PLACE)
    for scene in ("close view", "three-quarter view", "from above", "from the side")
)


def _lora(**changes) -> UserLora:
    lora = UserLora(
        id="vox_bowl",
        kind="object",
        folder=Path("/models/user-models/lora/vox_bowl"),
        weights=Path("/models/user-models/lora/vox_bowl/vox_bowl.safetensors"),
        trigger="vox_bowl_trigger",
        default_strength=0.9,
        strength_range=(0.0, 2.0),
        adult_content=False,
        alias="vox_bowl",
    )
    return lora if not changes else replace(lora, **changes)


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


def _window(home: Path, runner=None) -> ShellWindow:
    return inline_window(
        home / "state", runner=runner or (lambda job, inputs: rgb()), improver=lambda text: text,
        tier=Tier.GB16, enlarge=object(),
    )


def _run_series(window: ShellWindow, monkeypatch, *, frames: int = 6) -> None:
    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    window.session.series_writer = lambda _prompt: _payload(IDEA, frames)
    window.view.set_prompt_text(IDEA)
    window.show()
    window.findChild(QCheckBox, "photoshootMode").click()
    window.findChild(QPushButton, "generate").click()
    settle(window)


# --- The headless model --------------------------------------------------------------------------------------------


def test_rows_append_one_shot_at_a_time() -> None:
    model = PhotoshootScriptModel()
    added: list[int] = []
    counts: list[int] = []
    model.item_added.connect(added.append)
    model.count_changed.connect(counts.append)
    first = model.append_shot("shot 1")
    second = model.append_shot("shot 2")
    assert (first, second) == (0, 1)
    assert model.count == 2
    assert added == [0, 1]
    assert counts == [1, 2]
    assert [item.prompt for item in model.items()] == ["shot 1", "shot 2"]


def test_a_row_stores_lora_alias_strength_and_trigger() -> None:
    model = PhotoshootScriptModel()
    lora = ScriptLora(alias="vox_bowl", strength=0.8, trigger="vox_bowl_trigger", injected=True)
    model.append_shot("a ceramic bowl on a wooden table, studio light", (lora,))
    row = model.item(0)
    assert row is not None
    assert row.loras == [lora]
    assert row.loras[0].alias == "vox_bowl"
    assert row.loras[0].strength == 0.8
    assert row.loras[0].trigger == "vox_bowl_trigger"
    assert row.loras[0].injected is True


def test_a_row_without_lora_still_works() -> None:
    model = PhotoshootScriptModel()
    model.append_shot("shot 1")
    row = model.item(0)
    assert row is not None
    assert row.loras == []


def test_series_loras_resolve_chip_word_strength_and_injection() -> None:
    lora = _lora()
    found = series_loras(lora, 0.7)
    assert len(found) == 1
    assert found[0].alias == "vox_bowl"
    assert found[0].strength == 0.7
    assert found[0].trigger == "vox_bowl_trigger"
    assert found[0].injected is True
    # No requested strength: the LoRA's own default, like the series draw.
    assert series_loras(lora, None)[0].strength == 0.9
    # No LoRA at all: no row entry.
    assert series_loras(None, None) == ()
    # A LoRA without a trigger is not injected.
    bare = series_loras(_lora(trigger=" ", alias=""), 0.7)
    assert bare[0].trigger == ""
    assert bare[0].injected is False


def test_a_shot_updates_its_text_and_status() -> None:
    model = PhotoshootScriptModel()
    model.append_shot("draft")
    updated: list[int] = []
    model.item_updated.connect(updated.append)
    model.update_shot(0, prompt="final", status="drawing")
    row = model.item(0)
    assert row is not None
    assert row.prompt == "final"
    assert row.status == "drawing"
    assert updated == [0]
    model.update_shot(9, status="kept")
    assert updated == [0]


def test_selecting_a_row_and_selecting_an_image_meet_on_the_row() -> None:
    model = PhotoshootScriptModel()
    model.append_shot("shot 1")
    model.append_shot("shot 2")
    selected: list[int] = []
    images: list[str] = []
    model.item_selected.connect(selected.append)
    model.image_selected.connect(images.append)
    model.select_item(1)
    assert model.selected_index == 1
    assert selected == [1]
    model.record_image(1, "series-x", "/pictures/series/02.webp")
    assert model.item(1) is not None
    assert model.item(1).image_id == "series-x"
    model.select_image("series-x")
    assert images == ["series-x"]
    assert selected[-1] == 1
    model.select_image("another-picture")
    assert selected[-1] == -1
    assert model.selected_index == -1


def test_closing_the_window_does_not_drop_the_script() -> None:
    model = PhotoshootScriptModel()
    model.append_shot("shot 1")
    model.append_shot("shot 2")
    closed: list[bool] = []
    model.window_closed.connect(lambda: closed.append(True))
    model.notify_window_closed()
    assert closed == [True]
    assert model.count == 2
    assert model.item(0) is not None


def test_a_new_scenario_clears_the_rows() -> None:
    model = PhotoshootScriptModel()
    model.append_shot("shot 1")
    counts: list[int] = []
    model.count_changed.connect(counts.append)
    model.begin()
    assert model.count == 0
    assert model.items() == ()
    assert counts[-1] == 0


# --- The window and the Script (N) button --------------------------------------------------------------------------


def test_the_script_window_opens_with_the_series_outside_the_stage(app, home: Path, monkeypatch) -> None:
    window = _window(home)
    _run_series(window, monkeypatch)
    script_window = window.findChild(QDialog, "photoshootScriptWindow")
    assert script_window is not None
    assert script_window.isVisible()
    # The script never docks into the image stage.
    assert window.view._stage.findChild(QObject, "photoshootScriptList") is None
    assert window.view._stage.findChild(QObject, "photoshootScriptWindow") is None
    rows = window.findChildren(QWidget, "photoshootScriptRow")
    assert len(rows) == 6
    prompts = [row.findChild(QLabel, "photoshootScriptPrompt").text() for row in rows]
    assert all("A single photograph" in prompt for prompt in prompts)


def test_every_kept_row_is_linked_to_the_series_entry(app, home: Path, monkeypatch) -> None:
    window = _window(home)
    _run_series(window, monkeypatch)
    item_id = next(iter(window._series_folders))
    rows = window.photoshoot.script.items()
    assert len(rows) == 6
    assert all(row.status == "kept" for row in rows)
    assert all(row.image_id == item_id for row in rows)
    assert all(row.image_path.endswith(".webp") for row in rows)


def test_the_script_button_reopens_the_same_window(app, home: Path, monkeypatch) -> None:
    window = _window(home)
    _run_series(window, monkeypatch)
    bar = window.photoshoot
    button = window.findChild(QPushButton, "photoshootScriptButton")
    assert button is not None
    # Open window: the button hides.
    assert bar.script_window.isVisible()
    assert not button.isVisible()
    # Closing keeps the script and brings the button back with the count.
    bar.script_window.close()
    app.processEvents()
    assert not bar.script_window.isVisible()
    assert bar.script.count == 6
    assert button.isVisible()
    assert button.text() == "Script (6)"
    # Reopen: same window, same model.
    button.click()
    app.processEvents()
    assert bar.script_window.isVisible()
    assert button is window.findChild(QPushButton, "photoshootScriptButton")
    assert not button.isVisible()


def test_the_script_button_stays_hidden_without_a_script(app, home: Path) -> None:
    window = _window(home)
    window.show()
    app.processEvents()
    button = window.findChild(QPushButton, "photoshootScriptButton")
    assert button is not None
    assert not button.isVisible()
    assert not window.photoshoot.script_window.isVisible()


def test_a_row_click_selects_its_picture_in_the_main_window(app, home: Path, monkeypatch) -> None:
    window = _window(home)
    _run_series(window, monkeypatch)
    item_id = next(iter(window._series_folders))
    bar = window.photoshoot
    window.shell.layout.selected = None
    bar.script_window._list.setCurrentRow(2)
    app.processEvents()
    assert bar.script.selected_index == 2
    assert window.shell.layout.selected == item_id


def test_selecting_a_row_without_a_frame_is_a_no_op_on_the_image(app, home: Path) -> None:
    window = _window(home)
    window.show()
    bar = window.photoshoot
    bar.begin_series(None, None)
    bar.show_scenario(IDEA, SHOTS)
    app.processEvents()
    assert bar.script.count == 4
    assert all(row.image_id == "" for row in bar.script.items())
    bar.script.select_item(0)
    app.processEvents()
    # The row is highlighted; the picture area is left alone.
    assert bar.script.selected_index == 0
    assert bar.script_window._list.currentRow() == 0
    assert window.shell.layout.selected is None


def test_selecting_the_picture_in_the_main_window_highlights_its_row(app, home: Path, monkeypatch) -> None:
    window = _window(home)
    _run_series(window, monkeypatch)
    item_id = next(iter(window._series_folders))
    bar = window.photoshoot
    window.shell.layout.selected = None
    # The path a history-cell click takes in the shared screen.
    window.view._select(item_id)
    app.processEvents()
    assert bar.script.selected_index == 0
    assert bar.script_window._list.currentRow() == 0
    # A picture that is not in the script clears the row highlight.
    bar.script.select_image("not-in-the-script")
    app.processEvents()
    assert bar.script.selected_index == -1
    assert bar.script_window._list.currentRow() == -1


def test_a_row_shows_the_lora_chip_strength_and_injected_trigger(app, home: Path) -> None:
    window = _window(home)
    window.show()
    bar = window.photoshoot
    bar.begin_series(_lora(), 0.8)
    bar.show_scenario(IDEA, SHOTS[:1])
    app.processEvents()
    row = window.findChild(QWidget, "photoshootScriptRow")
    assert row is not None
    lora_label = row.findChild(QLabel, "photoshootScriptLora")
    trigger_label = row.findChild(QLabel, "photoshootScriptTrigger")
    assert lora_label is not None and lora_label.isVisible()
    assert lora_label.text() == "LoRA vox_bowl · 0.8"
    assert trigger_label is not None and trigger_label.isVisible()
    assert trigger_label.text() == "Trigger vox_bowl_trigger injected"
    # The written line leads with the trigger: the same injection the series encodes.
    prompt = row.findChild(QLabel, "photoshootScriptPrompt")
    assert prompt is not None
    assert prompt.text().startswith("vox_bowl_trigger, ")


def test_a_row_without_lora_shows_no_lora(app, home: Path) -> None:
    window = _window(home)
    window.show()
    bar = window.photoshoot
    bar.begin_series(None, None)
    bar.show_scenario(IDEA, SHOTS[:1])
    app.processEvents()
    row = window.findChild(QWidget, "photoshootScriptRow")
    assert row is not None
    lora_label = row.findChild(QLabel, "photoshootScriptLora")
    trigger_label = row.findChild(QLabel, "photoshootScriptTrigger")
    assert lora_label is not None and not lora_label.isVisible()
    assert trigger_label is not None and not trigger_label.isVisible()
