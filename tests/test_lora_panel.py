"""The LoRA library screen and the generate panel's pick and strength control."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QPushButton, QSlider

from canvas.app.i18n import text
from canvas.models.lora_library import import_lora, list_library
from canvas.models.modes import Tier
from canvas.ui import lora_panel
from canvas.ui.lora_panel import LibraryDialog
from canvas.ui.shell_layout import open_layout
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import World
from tests.voxlora_files import manifest, write_voxlora
from tests.worker_support import inline_client, inline_window, make_session, pump, settle
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


def _client(monkeypatch):
    """A worker client running the import in this process. The base model counts as installed."""
    from canvas import worker_jobs

    monkeypatch.setattr("canvas.models.lora_library._base_on_disk", lambda: True)
    worker_jobs.install_session(make_session())
    return inline_client()


def _entry(home: Path, **record):
    return import_lora(write_voxlora(home / "in", record=manifest(**record)), World().gate(), base_ready=lambda: True)


def test_the_library_lists_previews_and_removes(app, home: Path):
    entry = _entry(home)
    changed: list[int] = []
    dialog = LibraryDialog(None, None, lambda: changed.append(1))
    card = dialog.findChild(QLabel, "loraName")
    assert card is not None and card.text() == "Brass Lamp"
    assert dialog.findChild(QLabel, "loraPreview-0") is not None
    dialog.findChild(QPushButton, "loraRemove").click()
    assert list_library() == () and changed == [1]
    assert entry.id not in [item.id for item in list_library()]
    shiboken6.delete(dialog)


def test_a_refused_import_shows_the_catalog_line(app, home: Path, monkeypatch):
    package = write_voxlora(home / "in", record=manifest(kind="vehicle"))
    client = _client(monkeypatch)
    dialog = LibraryDialog(None, client)
    results: list[object] = []
    dialog.run_import(package, on_done=results.append)
    pump(client)
    assert results == ["lora.kind"]
    assert dialog.findChild(QLabel, "loraStatus").text() == text("lora.kind")
    assert list_library() == ()
    shiboken6.delete(dialog)


def test_the_pick_follows_the_library_and_the_strength_range(app, home: Path, monkeypatch):
    entry = _entry(home, scale_min=0.1, scale_max=1.0, scale_default=0.6)
    window = ShellWindow(open_layout(home / "state", gate=World().gate()), tier=Tier.GB16, enlarge=object())
    window.resize(1280, 800)
    window.show()
    app.processEvents()
    picker = window._lora_picker
    assert picker.isVisible()
    combo = window.findChild(QComboBox, "loraPick")
    assert [combo.itemText(index) for index in range(combo.count())] == [text("canvas.lora.none"), "Brass Lamp"]
    combo.setCurrentIndex(1)
    slider = window.findChild(QSlider, "loraStrength")
    assert (slider.minimum(), slider.maximum(), slider.value()) == (10, 100, 100)
    assert window.shell.layout.get("shell.addons") == entry.id
    assert window.lora_trigger() == "brsslmp"
    slider.setValue(90)
    assert window.shell.layout.get("shell.lora_weight") == pytest.approx(0.9)
    combo.setCurrentIndex(0)
    assert window.shell.layout.get("shell.addons") == ""
    window.close()


def test_the_window_import_adds_to_the_pick(app, home: Path, monkeypatch):
    monkeypatch.setattr("canvas.models.lora_library._base_on_disk", lambda: True)
    shown: list[str] = []
    monkeypatch.setattr(lora_panel.ImportedDialog, "exec", lambda self: shown.append("shown"))
    window = inline_window(home / "state", tier=Tier.GB16, enlarge=object())
    window.show()
    app.processEvents()
    assert not window._lora_picker.isVisible()
    results: list[object] = []
    window.import_lora(write_voxlora(home / "in"), on_done=results.append)
    settle(window)
    result = results[0]
    assert result is not None and not isinstance(result, str) and shown == ["shown"]
    assert window._lora_picker.combo.count() == 2
    window.close()


def test_an_import_from_add_ons_replaces_the_empty_line(app, home: Path):
    from canvas.models.user_lora import LoraIndex
    from canvas.ui.wizards import AddonsWizard

    entry = _entry(home)
    wizard = AddonsWizard(None, LoraIndex((), ()), "", 0.7, lambda _key, _strength: None, lambda add: add(entry.as_lora()))
    assert wizard.findChild(QLabel, "addonsEmpty") is not None
    wizard.findChild(QPushButton, "addonsImport").click()
    app.processEvents()
    assert wizard._empty is None
    shiboken6.delete(wizard)


def test_a_strength_step_does_not_redraw_the_window(app, home: Path, monkeypatch):
    entry = _entry(home)
    window = ShellWindow(open_layout(home / "state", gate=World().gate()), tier=Tier.GB16, enlarge=object())
    window.show()
    app.processEvents()
    window.findChild(QComboBox, "loraPick").setCurrentIndex(1)
    assert window.shell.layout.get("shell.addons") == entry.id
    calls: list[int] = []
    monkeypatch.setattr(window.view, "refresh", lambda: calls.append(1))
    window.findChild(QSlider, "loraStrength").setValue(50)
    assert calls == [] and window.shell.layout.get("shell.lora_weight") == pytest.approx(0.5)
    window.close()


def test_the_third_party_dialog_asks_before_storing(app, home: Path, monkeypatch):
    from tests.lora_files import write_safetensors

    path = home / "x.safetensors"
    write_safetensors(path, {"layers.0.attention.to_q.lora_A.weight": b"\x01", "layers.0.attention.to_q.lora_B.weight": b"\x02"}, {})
    client = _client(monkeypatch)
    results: list[object] = []
    monkeypatch.setattr(lora_panel.ThirdPartyDialog, "exec", lambda self: 0)
    lora_panel.import_with_dialog(None, client, path, on_done=results.append)
    pump(client)
    assert results == [None]
    assert list_library() == ()
    monkeypatch.setattr(lora_panel.ThirdPartyDialog, "exec", lambda self: 1)
    lora_panel.import_with_dialog(None, client, path, on_done=results.append)
    pump(client)
    entry = results[1]
    assert entry is not None and entry.manifest.kind == "unknown"
    dialog = LibraryDialog(None, client)
    assert dialog.findChild(QLabel, "loraPreviewPlaceholder") is not None
    from PySide6.QtWidgets import QLineEdit

    edit = dialog.findChild(QLineEdit, "loraTriggerEdit")
    edit.setText("mylora")
    edit.editingFinished.emit()
    assert list_library()[0].manifest.trigger == "mylora"
    shiboken6.delete(dialog)


def test_a_library_trigger_edit_updates_the_chip(app, home: Path, monkeypatch):
    from tests.lora_files import write_safetensors

    path = home / "x.safetensors"
    write_safetensors(path, {"layers.0.attention.to_q.lora_A.weight": b"\x01", "layers.0.attention.to_q.lora_B.weight": b"\x02"}, {})
    entry = import_lora(path, World().gate(), base_ready=lambda: True, confirm=lambda _manifest: True)
    window = ShellWindow(open_layout(home / "state", gate=World().gate()), tier=Tier.GB16, enlarge=object())
    window.show()
    app.processEvents()
    window._pick_lora(entry.id, 0.8)
    library = window.lora_library()
    from PySide6.QtWidgets import QLineEdit

    edit = library.findChild(QLineEdit, "loraTriggerEdit")
    edit.setText("mylora")
    edit.editingFinished.emit()
    app.processEvents()
    assert window._lora_chip.text() == "mylora"
    window.close()


def test_import_runs_off_the_ui_thread_and_the_window_keeps_working(app, home: Path, monkeypatch):
    """The import runs in the worker process; the call returns at once and the window keeps ticking."""
    import json
    import os
    import time

    from PySide6.QtCore import QTimer
    from voxprint_common.worker import Handlers, WorkerHost

    from canvas.ipc.client import WorkerClient
    from tests.worker_handlers import FIXTURE_ENV

    monkeypatch.setenv(FIXTURE_ENV, json.dumps({"base_ready": True, "pause": 0.3}))
    host = WorkerHost(Handlers(run="tests.worker_handlers:run", check="tests.worker_handlers:check"),
                      state_dir=home / "worker")
    client = WorkerClient(host, tier=Tier.GB16.value)
    try:
        client.start()
        pump(client, timeout=60)
        ticks: list[int] = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(10)
        results: list[object] = []
        started = time.monotonic()
        lora_panel.import_with_dialog(None, client, write_voxlora(home / "in"), on_done=results.append)
        assert time.monotonic() - started < 0.5 and results == []
        pump(client, timeout=60)
        timer.stop()
        assert host.pid is not None and host.pid != os.getpid()
        assert len(ticks) >= 5
        result = results[0]
        assert not isinstance(result, str) and result is not None
    finally:
        client.shutdown()


def test_a_third_party_question_is_asked_on_the_ui_thread(app, home: Path, monkeypatch):
    import threading

    from tests.lora_files import write_safetensors

    path = home / "y.safetensors"
    write_safetensors(path, {"layers.0.attention.to_q.lora_A.weight": b"\x01", "layers.0.attention.to_q.lora_B.weight": b"\x02"}, {})
    asked: list[bool] = []

    def answer(self):
        asked.append(threading.current_thread() is threading.main_thread())
        return 1

    monkeypatch.setattr(lora_panel.ThirdPartyDialog, "exec", answer)
    client = _client(monkeypatch)
    results: list[object] = []
    lora_panel.import_with_dialog(None, client, path, on_done=results.append)
    pump(client)
    assert asked == [True] and results[0] is not None
