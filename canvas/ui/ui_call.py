"""Run a blocking call away from the window thread."""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

from PySide6.QtCore import QEventLoop, QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication

T = TypeVar("T")


class _Bridge(QObject):
    """Queues a callable onto the window thread."""

    call = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.call.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def _run(self, payload: object) -> None:
        fn, box, done = payload  # type: ignore[misc]
        try:
            box["value"] = fn()
        except Exception as exc:
            box["error"] = exc
        finally:
            done.set()


_bridge: _Bridge | None = None


def _ui_bridge() -> _Bridge:
    global _bridge
    if _bridge is None:
        _bridge = _Bridge()
        app = QApplication.instance()
        if isinstance(app, QApplication):
            _bridge.moveToThread(app.thread())
    return _bridge


def on_ui(fn: Callable[[], T]) -> T:
    """Run ``fn`` on the window thread and return its value."""
    app = QApplication.instance()
    if not isinstance(app, QApplication) or QThread.currentThread() is app.thread():
        return fn()
    box: dict[str, object] = {}
    done = threading.Event()
    _ui_bridge().call.emit((fn, box, done))
    done.wait()
    error = box.get("error")
    if isinstance(error, Exception):
        raise error
    return box.get("value")  # type: ignore[return-value]


def run_off_ui(fn: Callable[[], T]) -> T:
    """Run ``fn`` off the window thread while that thread keeps handling events."""
    app = QApplication.instance()
    if not isinstance(app, QApplication) or QThread.currentThread() is not app.thread():
        return fn()
    _ui_bridge()
    holder: dict[str, object] = {}
    done = threading.Event()

    def work() -> None:
        try:
            holder["value"] = fn()
        except Exception as exc:
            holder["error"] = exc
        finally:
            done.set()

    threading.Thread(target=work, name="canvas-ui", daemon=True).start()
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(15)

    def poll() -> None:
        if done.is_set():
            loop.quit()

    timer.timeout.connect(poll)
    timer.start()
    if not done.is_set():
        loop.exec()
    timer.stop()
    timer.deleteLater()
    error = holder.get("error")
    if isinstance(error, Exception):
        raise error
    return holder.get("value")  # type: ignore[return-value]
