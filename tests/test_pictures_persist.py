"""Canvas must not trash or delete the user's pictures on its own.

Only an explicit per-item Delete in the history menu may remove a picture.
A new batch, a Photoshoot, or opening the window must leave existing files.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton

from canvas.models.modes import Tier
from tests.fakes import rgb
from tests.test_photoshoot import IDEA, _payload
from tests.worker_support import inline_window, settle
from voxprint_common.i18n import set_language


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    set_language("en")
    yield tmp_path
    set_language("en")


def _trash_root(home: Path) -> Path:
    return home / "xdg" / "Trash"


def _seed_pictures(home: Path) -> list[Path]:
    folder = home / "state" / "pictures"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("alpha.webp", "beta.webp", "gamma.webp"):
        path = folder / name
        path.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
        paths.append(path)
    return paths


def _assert_kept(paths: list[Path], home: Path) -> None:
    for path in paths:
        assert path.is_file(), path
    trash = _trash_root(home)
    if trash.exists():
        leftover = [p for p in trash.rglob("*") if p.is_file()]
        assert leftover == []


def test_opening_the_window_leaves_existing_pictures(app, home: Path) -> None:
    paths = _seed_pictures(home)
    window = inline_window(home / "state", runner=lambda job, inputs: rgb(), tier=Tier.GB16, enlarge=object())
    window.show()
    app.processEvents()
    _assert_kept(paths, home)
    window.close()


def test_a_new_batch_leaves_existing_pictures(app, home: Path, monkeypatch) -> None:
    paths = _seed_pictures(home)
    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    window = inline_window(home / "state", runner=lambda job, inputs: rgb(), tier=Tier.GB16, enlarge=object())
    window.view.set_prompt_text("a red boat")
    window.show()
    app.processEvents()
    generate = window.findChild(QPushButton, "generate")
    assert generate is not None
    generate.click()
    settle(window)
    _assert_kept(paths, home)
    window.close()


def test_a_photoshoot_leaves_existing_pictures(app, home: Path, monkeypatch) -> None:
    paths = _seed_pictures(home)
    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    window = inline_window(
        home / "state", runner=lambda job, inputs: rgb(), tier=Tier.GB16, enlarge=object(),
    )
    window.session.series_writer = lambda _prompt: _payload(IDEA)
    window.view.set_prompt_text(IDEA)
    window.show()
    mode = window.findChild(QCheckBox, "photoshootMode")
    assert mode is not None
    mode.click()
    window.findChild(QPushButton, "generate").click()
    settle(window)
    _assert_kept(paths, home)
    window.close()
