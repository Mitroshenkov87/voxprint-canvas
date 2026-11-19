"""The gallery resolves pictures from earlier sessions and copies their making-of data."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication, QPushButton

from canvas.models.modes import Tier
from tests.worker_support import inline_window


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    yield tmp_path


def _history(state: Path, item_id: str = "aa" * 16) -> str:
    state.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "items": [{"id": item_id, "prompt": "a red boat", "presets": [], "seed": 3}]}
    (state / "image-history.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return item_id


def _write_picture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), (200, 30, 40)).save(path, "WEBP")


def test_a_past_session_picture_in_the_projects_folder_paints_again(app, home: Path) -> None:
    from canvas.ui.shell_screen import pictures_dir

    item_id = _history(home / "state")
    saved = pictures_dir() / f"{item_id}.webp"
    _write_picture(saved)
    window = inline_window(home / "state", tier=Tier.GB16, enlarge=object())
    assert window.picture_path(item_id) == saved
    window.resize(1280, 800)
    window.show()
    window.view._select(item_id)
    for _ in range(3):
        QApplication.instance().processEvents()
    result = window.findChild(__import__("PySide6.QtWidgets", fromlist=["QLabel"]).QLabel, "result")
    assert result is not None and result.pixmap() is not None and not result.pixmap().isNull()
    window.close()


def test_a_past_session_picture_in_the_state_folder_paints_again(app, home: Path) -> None:
    item_id = _history(home / "state")
    saved = home / "state" / "pictures" / f"{item_id}.webp"
    _write_picture(saved)
    window = inline_window(home / "state", tier=Tier.GB16, enlarge=object())
    assert window.picture_path(item_id) == saved
    window.close()


def test_an_item_without_a_file_stays_a_placeholder(app, home: Path) -> None:
    item_id = _history(home / "state")
    window = inline_window(home / "state", tier=Tier.GB16, enlarge=object())
    assert window.picture_path(item_id) is None
    window.close()


def test_save_as_offers_the_resolved_past_picture(app, home: Path, monkeypatch) -> None:
    from canvas.ui.shell_screen import pictures_dir

    item_id = _history(home / "state")
    saved = pictures_dir() / f"{item_id}.webp"
    _write_picture(saved)
    window = inline_window(home / "state", tier=Tier.GB16, enlarge=object())
    window.shell.layout.select(item_id)
    monkeypatch.setattr(
        "canvas.ui.shell_screen.QFileDialog.getSaveFileName", lambda *args, **kwargs: (str(home / "out.webp"), ""),
    )

    def _copied(source, target):
        assert source == saved
        return target

    monkeypatch.setattr("canvas.ui.shell_screen.shutil.copyfile", _copied)
    window._on_command("file.save")
    window.close()
