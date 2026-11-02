"""The window process never imports torch, diffusers or transformers, and never builds the gate.

A fresh interpreter imports ``canvas.app.main`` and every ``canvas.ui``
module, opens the window through ``_launch`` (offscreen), then runs the
self-test and an Improve through a real spawned worker. An import hook
records every attempt to import a model library in the window process,
including attempts that would fail because the library is not installed, so
the check holds on machines without torch too. Constructing the gate or the
screen's in-process review adapter in the window process raises.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent(
    r'''
    import importlib
    import json
    import pkgutil
    import sys
    import time
    import traceback

    HEAVY = ("torch", "diffusers", "transformers")
    attempts = {}

    class Recorder:
        def find_spec(self, name, path=None, target=None):
            top = name.split(".")[0]
            if top in HEAVY and top not in attempts:
                attempts[top] = "".join(traceback.format_stack(limit=25)[:-1])
            return None

    sys.meta_path.insert(0, Recorder())

    import canvas.app.main as app_main
    import canvas.ui

    for module in pkgutil.iter_modules(canvas.ui.__path__):
        importlib.import_module("canvas.ui." + module.name)

    built = []
    from canvas.safety import gate as gate_module
    from canvas.app import shell_gate

    def refuse(kind):
        def init(self, *args, **kwargs):
            built.append(kind)
            raise AssertionError(kind + " built in the window process")
        return init

    gate_module.AgeGate.__init__ = refuse("AgeGate")
    shell_gate.ContentCheck.__init__ = refuse("ContentCheck")

    from PySide6.QtWidgets import QApplication, QPushButton

    app = QApplication.instance() or QApplication([])

    class Splash:
        def set_stage(self, message): pass
        def finish(self, window): pass
        def close(self): pass

    class FakeApp:
        def processEvents(self): pass
        def exec(self): return 0

    from canvas.models.modes import Tier

    launched = app_main._launch(Tier.GB16, Splash(), FakeApp())
    for widget in QApplication.topLevelWidgets():
        widget.close()

    # The production wiring, kept open: the worker is a real spawned process.
    from canvas.ipc.client import WorkerClient, default_host
    from canvas.ui.shell_layout import canvas_state_dir, open_worker_layout
    from canvas.ui.shell_screen import ShellWindow
    from canvas.ui.worker_checks import WorkerChecks

    worker = WorkerClient(default_host(), tier=Tier.GB16.value)
    window = ShellWindow(open_worker_layout(canvas_state_dir(), checks=WorkerChecks(worker)),
                         tier=Tier.GB16, enlarge=object(), worker=worker)
    window.show()

    def pump(limit):
        end = time.monotonic() + limit
        while time.monotonic() < end:
            app.processEvents()
            if not worker.busy and worker.state != "starting" and not window._busy:
                return True
            time.sleep(0.02)
        return False

    worker.start()
    pump(120)
    wizard = window.diagnostics_wizard()
    wizard.findChild(QPushButton, "checkRun").click()
    settled_check = pump(300)
    from PySide6.QtWidgets import QLabel

    report = wizard.findChild(QLabel, "checkReport").text()
    window.view.set_prompt_text("a red bicycle")
    window.findChild(QPushButton, "improvePrompt").click()
    settled_improve = pump(300)
    worker_pid = worker.host.pid
    worker.shutdown()
    print(json.dumps({
        "launched": launched,
        "attempts": attempts,
        "loaded": sorted(name for name in HEAVY if name in sys.modules),
        "built": built,
        "settled": [settled_check, settled_improve],
        "report": report,
        "worker_pid": worker_pid,
        "ui_pid": __import__("os").getpid(),
    }))
    '''
)


def test_the_window_process_imports_no_model_library(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["VOXPRINT_HOME"] = str(tmp_path / "home")
    env["VOXPRINT_RAM_LEDGER"] = str(tmp_path / "ledger")
    env.pop("PYTHONPATH", None)
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["launched"] == 0
    assert result["attempts"] == {}, "\n".join(result["attempts"].values())
    assert result["loaded"] == []
    assert result["built"] == []
    assert result["settled"] == [True, True]
    # The self-test ran in the worker and its report reached the wizard.
    assert result["report"].strip()
    assert result["worker_pid"] not in (None, result["ui_pid"])
