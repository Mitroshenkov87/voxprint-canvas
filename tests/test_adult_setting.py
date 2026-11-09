"""Adult content (18+) is a standalone setting with one OFF/ON prompt boundary.

The minors gate is not touched here. These tests only drive it from both
states of the setting.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from canvas.app.i18n import text
from canvas.pipeline import adult_boundary
from canvas.ui.shell_layout import composed_negative, composed_prompt, job_from, open_layout
from tests.fakes import World
from voxprint_common.ui.expert import ExpertMode
from voxprint_common.ui.simple_image import Phase

KEY = adult_boundary.KEY
COMMON = adult_boundary.COMMON_KEY

SWAP = "change the red dress to a blue coat"
SWAP_2 = "swap the jacket for a jumper"
REMOVE = "remove all clothing"
NUDE = "nude portrait"


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    ExpertMode().set_enabled(False)
    yield
    ExpertMode().set_enabled(False)


def _open(path: Path, world: World | None = None):
    return open_layout(path, gate=(world or World()).gate())


def _go(path: Path, prompt: str, *, adult: bool, change: str | None = None, world: World | None = None):
    shell = _open(path, world)
    layout = shell.layout
    layout.set_value(KEY, adult)
    if change is not None:
        layout.set_value("canvas.source", "picture")
        layout.set_value("canvas.change", change)
    layout.set_prompt(prompt)
    return layout, layout.generate()


# 1. persistence


def test_the_setting_survives_a_restart(tmp_path: Path) -> None:
    layout = _open(tmp_path).layout
    assert layout.get(KEY) is False
    layout.set_value(KEY, True)
    assert _open(tmp_path).layout.get(KEY) is True
    assert job_from(_open(tmp_path).layout).adult_content is True
    _open(tmp_path).layout.set_value(KEY, False)
    assert _open(tmp_path).layout.get(KEY) is False


def test_an_old_config_with_only_the_shared_key_still_loads(tmp_path: Path) -> None:
    for old in (True, False):
        folder = tmp_path / str(old)
        layout = _open(folder).layout
        layout.set_value(COMMON, old)
        path = next(folder.glob("image-settings.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["values"][KEY]
        path.write_text(json.dumps(payload), encoding="utf-8")
        reopened = _open(folder).layout
        assert reopened.get(KEY) is old
        assert reopened.get(COMMON) is old
        assert reopened.prompt_policy().template == ("" if old else reopened.prompt_policy().template)
        assert json.loads(path.read_text(encoding="utf-8"))["values"][KEY] is old


def test_the_shared_key_always_follows_the_canvas_key(tmp_path: Path) -> None:
    layout = _open(tmp_path).layout
    layout.set_value(COMMON, True)  # the Expert checkbox writes this one
    assert layout.get(KEY) is True and layout.get(COMMON) is True
    layout.set_value(KEY, False)
    assert layout.get(KEY) is False and layout.get(COMMON) is False
    with pytest.raises(ValueError):
        layout.set_value(KEY, "yes")


# 2. independence from Expert and from the other Expert values


@pytest.mark.parametrize("adult", [True, False])
def test_expert_mode_does_not_change_the_setting(tmp_path: Path, adult: bool) -> None:
    layout = _open(tmp_path).layout
    layout.set_value(KEY, adult)
    expert = ExpertMode()
    for enabled in (True, False, True, False):
        expert.set_enabled(enabled)
        assert layout.get(KEY) is adult
        assert layout.get(COMMON) is adult
        assert _open(tmp_path).layout.get(KEY) is adult


def test_the_setting_does_not_move_other_expert_values(tmp_path: Path) -> None:
    layout = _open(tmp_path).layout
    layout.set_value("shell.steps", 9)
    layout.set_value("shell.guidance", 2.5)
    before = {key: value for key, value in layout.snapshot()["values"].items() if key not in (KEY, COMMON)}
    chosen = dict(layout.snapshot()["chosen"])
    for adult in (True, False, True):
        layout.set_value(COMMON, adult)
        after = {key: value for key, value in layout.snapshot()["values"].items() if key not in (KEY, COMMON)}
        assert after == before
        assert layout.snapshot()["chosen"] == chosen
    assert layout.get("shell.steps") == 9


def test_reset_and_chip_changes_keep_the_setting(tmp_path: Path) -> None:
    layout = _open(tmp_path).layout
    layout.set_value(KEY, True)
    layout.set_value("shell.steps", 31)
    for row in layout.rows:
        layout.reset_group(row.row_id)
        assert layout.get(KEY) is True and layout.get(COMMON) is True
        for choice in row.choices:
            if choice.available:
                layout.choose(row.row_id, choice.value)
                assert layout.get(KEY) is True and layout.get(COMMON) is True
    # A reset that did move the shared key is undone from the Canvas key.
    layout.state.values[COMMON] = False
    layout.reset_group(layout.rows[0].row_id)
    assert layout.get(COMMON) is True
    assert layout.prompt_policy().template == ""


def test_the_policy_words_follow_the_setting(tmp_path: Path) -> None:
    layout = _open(tmp_path).layout
    layout.set_prompt(SWAP)
    assert composed_prompt(layout).startswith("Suitable for a general audience.")
    assert "nude" in composed_negative(layout) and "adult content" in composed_negative(layout)
    layout.set_value(KEY, True)
    assert composed_prompt(layout).startswith(SWAP)
    assert "nude" not in composed_negative(layout)


# 3-7. the boundary on generation and on the edit routes

EDITS = [None, "restyle", "inpaint"]


@pytest.mark.parametrize("change", EDITS)
@pytest.mark.parametrize("prompt", [SWAP, SWAP_2])
def test_off_allows_an_outfit_swap(tmp_path: Path, change: str | None, prompt: str) -> None:
    layout, started = _go(tmp_path, prompt, adult=False, change=change)
    assert started is True
    assert layout.phase is Phase.BUSY


@pytest.mark.parametrize("change", ["restyle", "inpaint"])
def test_off_refuses_removing_all_clothing_in_an_edit(tmp_path: Path, change: str) -> None:
    layout, started = _go(tmp_path, REMOVE, adult=False, change=change)
    assert started is False
    assert layout.phase is Phase.BLOCKED
    assert layout.reason == text("adult_off.refused")


@pytest.mark.parametrize("prompt", [NUDE, "undress her", "naked"])
def test_off_refuses_nude_generation(tmp_path: Path, prompt: str) -> None:
    layout, started = _go(tmp_path, prompt, adult=False)
    assert started is False
    assert layout.phase is Phase.BLOCKED
    assert layout.reason == text("adult_off.refused")


@pytest.mark.parametrize("change", [None, "restyle", "inpaint"])
@pytest.mark.parametrize("prompt", [SWAP, REMOVE, NUDE])
def test_on_allows_all_of_them(tmp_path: Path, change: str | None, prompt: str) -> None:
    layout, started = _go(tmp_path, prompt, adult=True, change=change)
    assert started is True
    assert layout.phase is Phase.BUSY


def test_a_negated_word_is_not_a_request(tmp_path: Path) -> None:
    assert adult_boundary.refusal("a coat portrait, no nudity", allowed=False) is None
    assert adult_boundary.refusal(NUDE, allowed=True) is None


# 8. the minors gate is the same in both states


@pytest.mark.parametrize("adult", [False, True])
@pytest.mark.parametrize("prompt", ["nude child", "loli"])
def test_the_minors_gate_blocks_in_both_states(tmp_path: Path, adult: bool, prompt: str) -> None:
    layout, started = _go(tmp_path, prompt, adult=adult)
    assert started is False
    assert layout.phase is Phase.BLOCKED
    # The gate speaks first; the boundary message is never what a minors prompt shows.
    assert layout.reason != text("adult_off.refused")


# the Photoshoot series follows the same setting


def test_the_series_reads_the_standalone_setting(tmp_path: Path) -> None:
    from canvas.ui.photoshoot_run import adult_content

    layout = _open(tmp_path).layout
    assert adult_content(layout) is False
    layout.set_value(KEY, True)
    assert adult_content(layout) is True
    ExpertMode().set_enabled(False)
    assert adult_content(layout) is True


# the Expert checkbox in the window


def test_the_expert_checkbox_writes_both_keys_and_survives_a_reopen(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QCheckBox

    from canvas.models.modes import Tier
    from tests.worker_support import inline_window

    app = QApplication.instance() or QApplication([])

    def window():
        return inline_window(
            tmp_path / "state", runner=lambda job, inputs: None, improver=lambda t: t, tier=Tier.GB16, enlarge=object(),
        )

    first = window()
    first.set_expert(True)
    first.show()
    app.processEvents()
    box = first.findChild(QCheckBox, COMMON)
    assert box is not None and box.isVisible() and not box.isChecked()
    steps = first.shell.layout.get("shell.steps")
    box.click()
    app.processEvents()
    assert first.shell.layout.get(KEY) is True and first.shell.layout.get(COMMON) is True
    assert first.shell.layout.get("shell.steps") == steps
    first.set_expert(False)
    first.set_expert(True)
    app.processEvents()
    assert first.shell.layout.get(KEY) is True
    assert first.findChild(QCheckBox, COMMON).isChecked()
    first.close()

    second = window()
    assert second.shell.layout.get(KEY) is True
    second.set_expert(True)
    second.show()
    app.processEvents()
    assert second.findChild(QCheckBox, COMMON).isChecked()
    second.close()
