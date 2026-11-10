"""Every picture gets a fresh random seed unless the seed is locked or given.

The seed used is shown, recorded in the job result, and written into the
saved picture even when full metadata is off.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QLineEdit, QPushButton, QSpinBox

from canvas.core import seeds
from canvas.core.formats import read_metadata
from canvas.models.modes import Tier
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import rgb
from tests.worker_support import inline_window, settle
from voxprint_common.i18n import set_language, tr
from voxprint_common.ui.simple_image import Phase


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _metadata_default(monkeypatch):
    # Full metadata is off by default; another test may have left the switch in the environment.
    monkeypatch.delenv("VOXPRINT_CANVAS_SAVE_METADATA", raising=False)


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    set_language("en")
    yield tmp_path
    set_language("en")


@pytest.fixture(autouse=True)
def _close_windows():
    yield
    instance = QApplication.instance()
    for widget in instance.topLevelWidgets() if instance else ():
        if isinstance(widget, ShellWindow) and widget.isVisible():
            widget.close()


# --- the helpers ---------------------------------------------------------------------------------


def test_fresh_seed_range_is_zero_to_two_to_the_32_minus_one(monkeypatch) -> None:
    asked: list[int] = []

    def randbelow(limit: int) -> int:
        asked.append(limit)
        return limit - 1

    monkeypatch.setattr(secrets, "randbelow", randbelow)
    assert seeds.fresh_seed() == 2**32 - 1
    assert asked == [2**32]
    monkeypatch.setattr(secrets, "randbelow", lambda limit: 0)
    assert seeds.fresh_seed() == 0


def test_fresh_seeds_differ_and_stay_in_range() -> None:
    drawn = [seeds.fresh_seed() for _ in range(64)]
    assert len(set(drawn)) > 60
    assert all(0 <= value <= 2**32 - 1 for value in drawn)
    # Some land above the old 2**31 limit.
    assert any(value >= 2**31 for value in [seeds.fresh_seed() for _ in range(200)])


def test_pick_seed_counts_up_from_a_base_and_wraps() -> None:
    assert [seeds.pick_seed(10, index) for index in range(3)] == [10, 11, 12]
    assert seeds.pick_seed(2**32 - 1, 1) == 0
    assert seeds.pick_seed(0, 0) == 0


def test_seed_plan_modes() -> None:
    random_plan = seeds.SeedPlan("random", 5)
    drawn = {random_plan.seed_for(5 + index) for index in range(8)}
    assert len(drawn) == 8
    assert random_plan.seed_for(5) == random_plan.used[5]  # one shot, one answer
    series = seeds.SeedPlan("series", 100)
    assert [series.seed_for(100 + index) for index in range(3)] == [100, 101, 102]
    locked = seeds.SeedPlan("locked", 7)
    assert {locked.seed_for(7 + index) for index in range(3)} == {7}


# --- the command line ----------------------------------------------------------------------------


def test_cli_seed_defaults_to_unset_and_zero_is_a_real_seed() -> None:
    from canvas.app.main import _parser

    out = ["--out", "x"]
    assert _parser().parse_args(["generate", "--prompt", "a", *out]).seed is None
    assert _parser().parse_args(["generate", "--prompt", "a", *out, "--seed", "0"]).seed == 0
    assert _parser().parse_args(["photoshoot", "--prompt", "a", *out]).seed is None
    assert _parser().parse_args(["photoshoot", "--prompt", "a", *out, "--seed", "0"]).seed == 0
    helps = _parser().format_help() + json.dumps([
        action.help for sub in _parser()._subparsers._group_actions for action in sub.choices["generate"]._actions
    ])
    assert "new random seed" in helps


def test_render_batch_seed_default_is_unset() -> None:
    from canvas.tools.render_batch import _parser as build_parser

    parsed = build_parser().parse_args(["--adult-content", "off", "--out", "x", "--prompts", "p.json"])
    assert parsed.seed is None
    assert build_parser().parse_args(
        ["--adult-content", "off", "--out", "x", "--prompts", "p.json", "--seed", "0"]
    ).seed == 0


def _render_items(tmp_path: Path, rows: list, *, seed: int | None):
    from canvas.core.formats import OutputFormat
    from canvas.tools import render_batch as rb

    seen: list[int] = []

    class _Hooks:
        pass

    saved: list[dict] = []

    class _Session:
        def __init__(self, *args, **kwargs) -> None:
            pass

        last_error = None

        def generate(self, prompt: str, seed: int):
            seen.append(seed)
            return rgb_safe(), ""

    def rgb_safe():
        from canvas.safety.gate import issue

        return issue(rgb(8, 8))

    items = [rb.PromptItem(prompt=row.get("prompt", "p"), item_id=f"{n:02d}", seed=row.get("seed")) for n, row in enumerate(rows, 1)]
    out = tmp_path / "out"
    out.mkdir(parents=True)
    import io

    original = rb._Session
    rb._Session = _Session  # type: ignore[assignment]
    try:
        code = rb._run_render(
            items, adult=False, out=out, base_seed=seed, steps=8, guidance=0.0, sampler="euler", aspect="1:1",
            lora_bind=None, hooks=_Hooks(), log=io.StringIO(),
        )
    finally:
        rb._Session = original
    del saved, OutputFormat, code
    return seen, out


def test_render_batch_without_seed_gives_each_item_its_own_seed_and_saves_it(tmp_path: Path) -> None:
    seen, out = _render_items(tmp_path, [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}], seed=None)
    assert len(set(seen)) == 3
    for stem, seed in zip(("01", "02", "03"), seen):
        assert read_metadata(out / f"{stem}.webp") == {"seed": str(seed)}
        assert json.loads((out / f"{stem}.json").read_text())["seed"] == seed


def test_render_batch_with_seed_counts_up_and_rows_with_a_seed_keep_it(tmp_path: Path) -> None:
    seen, _ = _render_items(tmp_path, [{"prompt": "a"}, {"prompt": "b", "seed": 99}, {"prompt": "c"}], seed=1001)
    assert seen == [1001, 99, 1003]
    again, _ = _render_items(tmp_path / "again", [{"prompt": "a"}, {"prompt": "b", "seed": 99}, {"prompt": "c"}], seed=1001)
    assert again == seen
    zero, _ = _render_items(tmp_path / "zero", [{"prompt": "a"}], seed=0)
    assert zero == [0]


def test_batch_manifest_entry_without_a_seed_is_random_and_with_one_is_kept() -> None:
    from canvas.app.batch_cli import parse_manifest

    entries = parse_manifest({"entries": [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c", "seed": 0}]})
    assert [entry.seeded for entry in entries] == [False, False, True]
    assert entries[0].seed != entries[1].seed
    assert entries[2].seed == 0


def test_a_batch_entry_without_a_seed_draws_random_seeds_per_picture() -> None:
    from canvas.pipeline.batch import BatchEntry

    entry = BatchEntry("a", "idea", "", "", 123, 4, seeded=False)
    plan = seeds.SeedPlan("series" if entry.seeded else "random", entry.seed)
    assert len({plan.seed_for(entry.seed + index) for index in range(4)}) == 4
    fixed = BatchEntry("a", "idea", "", "", 123, 4)
    plan = seeds.SeedPlan("series" if fixed.seeded else "random", fixed.seed)
    assert [plan.seed_for(fixed.seed + index) for index in range(4)] == [123, 124, 125, 126]


# --- the worker ----------------------------------------------------------------------------------


def test_worker_series_seed_modes() -> None:
    from canvas.worker_jobs import _series_seed

    assert _series_seed("series", 5, 6) == 6
    assert _series_seed("locked", 5, 6) == 5
    assert len({_series_seed("random", 5, 6) for _ in range(20)}) > 15
    with pytest.raises(ValueError):
        _series_seed("nope", 1, 1)


# --- the window ----------------------------------------------------------------------------------


def _window(home: Path, monkeypatch):
    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    used: list[int] = []

    def runner(job, inputs):
        del inputs
        used.append(job.seed)
        return rgb(8, 8)

    window = inline_window(home / "state", tier=Tier.GB16, enlarge=object(), runner=runner)
    window.view.set_prompt_text("a red bicycle")
    return window, used


def _generate(window) -> None:
    window.findChild(QPushButton, "generate").click()
    settle(window)


def _copies(window, count: int) -> None:
    field = window.findChild(QLineEdit, "pictureCount")
    field.setText(str(count))
    field.editingFinished.emit()


def test_simple_random_seed_toggle_is_on_by_default(app, home: Path, monkeypatch) -> None:
    window, _ = _window(home, monkeypatch)
    window.show()
    app.processEvents()
    box = window.findChild(QCheckBox, "seedDice")
    assert box is not None and box.isChecked()
    assert box.text() == tr("canvas.seed.box")
    assert window.shell.layout.seed_locked() is False


def test_two_unlocked_runs_use_different_seeds_and_show_them(app, home: Path, monkeypatch) -> None:
    window, used = _window(home, monkeypatch)
    window.show()
    _generate(window)
    _generate(window)
    assert window.shell.layout.phase is Phase.READY, window.shell.layout.reason
    assert len(used) == 2 and used[0] != used[1]
    assert all(0 <= seed <= 2**32 - 1 for seed in used)
    # The newest picture's seed is on the result line, and kept per picture.
    newest = window.shell.layout.history[0].item_id
    assert window._seeds[newest] == used[1]
    app.processEvents()
    label = window.findChild(QLabel, "seedInfo")
    assert label is not None and label.text() == tr("canvas.seed.shown", seed=used[1])


def test_copies_have_distinct_seeds_when_unlocked(app, home: Path, monkeypatch) -> None:
    window, used = _window(home, monkeypatch)
    window.show()
    _copies(window, 4)
    _generate(window)
    assert len(used) == 4 and len(set(used)) == 4
    assert len({window._seeds[item.item_id] for item in window.shell.layout.history}) == 4


def test_a_locked_seed_repeats_and_copies_count_up_from_it(app, home: Path, monkeypatch) -> None:
    window, used = _window(home, monkeypatch)
    window.show()
    layout = window.shell.layout
    layout.seed = 1234
    window.findChild(QCheckBox, "seedDice").click()  # off = locked
    assert layout.seed_locked() is True
    _generate(window)
    _generate(window)
    assert used == [1234, 1234]
    assert layout.seed == 1234
    used.clear()
    _copies(window, 3)
    _generate(window)
    assert used == [1234, 1235, 1236]
    assert layout.seed == 1234  # the base stays in the field for the next repeat


def test_a_locked_seed_above_the_field_limit_repeats_exactly(app, home: Path, monkeypatch) -> None:
    window, used = _window(home, monkeypatch)
    window.show()
    layout = window.shell.layout
    monkeypatch.setattr(seeds, "fresh_seed", lambda: 3_999_999_999)
    monkeypatch.setattr("canvas.ui.shell_layout.fresh_seed", lambda: 3_999_999_999)
    _generate(window)
    assert used == [3_999_999_999]
    assert layout.seed == seeds.FIELD_MAX  # the Qt field cannot show more
    layout.set_seed_locked(True)
    _generate(window)
    assert used == [3_999_999_999, 3_999_999_999]


def test_a_typed_seed_wins_over_the_last_used_one_when_locked(app, home: Path, monkeypatch) -> None:
    window, used = _window(home, monkeypatch)
    window.show()
    _generate(window)
    layout = window.shell.layout
    layout.set_seed_locked(True)
    layout.seed = 77
    _generate(window)
    assert used[-1] == 77


def test_expert_field_shows_the_last_seed_and_lock_stays_in_sync(app, home: Path, monkeypatch) -> None:
    window, used = _window(home, monkeypatch)
    window.resize(1400, 900)
    window.show()
    calls: list[bool] = []
    original = window.set_expert
    monkeypatch.setattr(window, "set_expert", lambda value, *a, **k: (calls.append(value), original(value, *a, **k))[1])
    original(True)
    app.processEvents()
    field = window.findChild(QSpinBox, "shell.seed")
    lock = window.findChild(QCheckBox, "canvas.seed_lock")
    assert field is not None and lock is not None
    assert lock.isChecked() is False
    _generate(window)
    app.processEvents()
    assert field.value() == min(used[-1], seeds.FIELD_MAX)
    dice = window.findChild(QCheckBox, "seedDice")
    assert dice.isChecked() is True
    # Expert lock on -> Simple toggle off, same stored setting.
    lock.click()
    app.processEvents()
    assert window.shell.layout.seed_locked() is True
    assert dice.isChecked() is False
    # Simple toggle back on -> Expert lock off.
    dice.click()
    app.processEvents()
    assert window.shell.layout.seed_locked() is False
    assert lock.isChecked() is False
    assert calls == []  # neither box touched Expert mode


# --- the job result and the saved picture ---------------------------------------------------------


def test_the_job_result_records_the_seed_and_the_file_carries_it(app, home: Path, monkeypatch) -> None:
    window, used = _window(home, monkeypatch)
    window.show()
    replies: list = []
    original = window._keep
    monkeypatch.setattr(window, "_keep", lambda reply, job: (replies.append(reply), original(reply, job))[1])
    _copies(window, 2)
    _generate(window)
    assert [reply.data["seed"] for reply in replies] == used
    for item in window.shell.layout.history:
        saved = Path(window._pictures[item.item_id])
        # Full metadata is off by default: the seed is the only field in the file.
        assert read_metadata(saved) == {"seed": str(window._seeds[item.item_id])}


# --- Photoshoot -----------------------------------------------------------------------------------


def _shoot(home: Path, monkeypatch, *, lock: bool = False, seed: int | None = None, frames: int = 4):
    from canvas.worker_jobs import _series_seed  # noqa: F401 - the module under test is imported first
    from tests.test_photoshoot import IDEA, _payload
    from PySide6.QtWidgets import QSpinBox

    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    used: list[int] = []

    def runner(job, inputs):
        del inputs
        used.append(job.seed)
        return rgb()

    window = inline_window(home / "state", runner=runner, improver=lambda text: text, tier=Tier.GB16, enlarge=object())
    window.session.series_writer = lambda prompt: _payload(IDEA, int(json.loads(prompt)["count"]))
    window.view.set_prompt_text(IDEA)
    window.show()
    window.findChild(QCheckBox, "photoshootMode").click()
    window.findChild(QSpinBox, "photoshootFrames").setValue(frames)
    if seed is not None:
        window.shell.layout.seed = seed
    if lock:
        window.shell.layout.set_seed_locked(True)
    return window, used


def test_a_photoshoot_series_has_distinct_seeds_when_unlocked(app, home: Path, monkeypatch) -> None:
    window, used = _shoot(home, monkeypatch)
    _generate(window)
    assert len(used) == 4 and len(set(used)) == 4
    # Every saved frame carries its own seed and nothing else.
    frames = sorted((home / "state" / "pictures" / "series").glob("*/[0-9][0-9].webp"))
    assert len(frames) == 4
    assert [read_metadata(path) for path in frames] == [{"seed": str(seed)} for seed in used]
    meta = json.loads(next((home / "state" / "pictures" / "series").glob("*/series.json")).read_text())
    assert meta["seeds"] == used
    # The Expert field shows the last seed used.
    assert window.shell.layout.seed == min(used[-1], seeds.FIELD_MAX)


def test_two_unlocked_photoshoot_runs_differ(app, home: Path, monkeypatch) -> None:
    window, used = _shoot(home, monkeypatch, frames=2)
    _generate(window)
    first = list(used)
    used.clear()
    _generate(window)
    assert set(first).isdisjoint(used)


def test_a_locked_photoshoot_seed_repeats_the_series(app, home: Path, monkeypatch) -> None:
    window, used = _shoot(home, monkeypatch, lock=True, seed=500, frames=3)
    _generate(window)
    first = list(used)
    assert first == [500, 501, 502]
    used.clear()
    _generate(window)
    assert used == first
    assert window.shell.layout.seed == 500


def test_the_photoshoot_bar_defaults_to_random_not_series(app, home: Path, monkeypatch) -> None:
    window, _ = _shoot(home, monkeypatch)
    assert window.photoshoot.seed_mode == "random"
