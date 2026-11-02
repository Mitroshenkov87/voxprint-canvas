"""Build a Canvas window wired to a worker, for tests.

``inline_window`` runs the worker handlers in this process through
:class:`tests.inline_host.InlineHost`, with fakes for the gate, the runner and
the rewriter. ``settle`` pumps the worker until the window is idle.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtWidgets import QApplication

from canvas import worker_jobs
from canvas.app.shell_gate import ContentCheck
from canvas.ipc.client import WorkerClient
from canvas.models.modes import Tier
from canvas.ui.shell_layout import open_worker_layout
from canvas.ui.shell_screen import ShellWindow
from canvas.ui.worker_checks import WorkerChecks
from tests.fakes import World
from tests.inline_host import InlineHost


def make_session(
    gate=None,
    *,
    runner=None,
    improver: Callable[[str], str] | None = None,
    tier: Tier = Tier.GB16,
    enlarge: object | None = None,
    writer: Callable[[str], str] | None = None,
) -> worker_jobs.Session:
    """A worker session around ``gate`` (a fake ``World`` gate by default)."""
    from canvas.app.shell_run import ShellRunner

    gate = gate if gate is not None else World().gate()
    if runner is None:
        runner = ShellRunner(gate=gate, tier=tier, enlarge=enlarge if enlarge is not None else object())

    def improve(*, prompt: str, adult: bool, progress) -> str:
        del adult, progress
        if improver is None:
            return prompt
        return improver(prompt)

    return worker_jobs.Session(
        gate=gate, check=ContentCheck(gate=gate), tier=tier, runner=runner, improver=improve, series_writer=writer,
    )


def inline_client(tier: Tier = Tier.GB16, **deadlines: float) -> WorkerClient:
    """A worker client over an in-process host."""
    return WorkerClient(InlineHost(), tier=tier.value, interval_ms=5, deadlines=deadlines or None)


def inline_window(
    state: Path,
    *,
    gate=None,
    runner=None,
    improver: Callable[[str], str] | None = None,
    writer: Callable[[str], str] | None = None,
    tier: Tier = Tier.GB16,
    enlarge: object | None = None,
    enlarge_installed: bool = False,
    installer=None,
) -> ShellWindow:
    """A window whose worker runs in this process with the given fakes. ``window.session`` is the worker session."""
    session = make_session(gate, runner=runner, improver=improver, tier=tier, enlarge=enlarge, writer=writer)
    worker_jobs.install_session(session)
    client = inline_client(tier)
    shell = open_worker_layout(state, checks=WorkerChecks(client), enlarge_installed=enlarge_installed)
    window = ShellWindow(
        shell, tier=tier, enlarge=enlarge if enlarge is not None else object(), worker=client, installer=installer,
    )
    window.session = session  # type: ignore[attr-defined]
    return window


def pump(client: WorkerClient, *, timeout: float = 20.0) -> None:
    """Pump ``client``'s worker and the event loop until no job is open."""
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app is not None:
            app.processEvents()
        client.adapter._bridge._on_timeout()
        if app is not None:
            app.processEvents()
        if not client.busy and not getattr(client.host, "pending", 0) and client.state != "starting":
            return
        if not isinstance(client.host, InlineHost):
            time.sleep(0.02)
    raise AssertionError("the worker did not settle")


def settle(window: ShellWindow, *, timeout: float = 20.0) -> None:
    """Pump the worker and the event loop until the window has no open job."""
    app = QApplication.instance()
    client = window.worker
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app is not None:
            app.processEvents()
        client.adapter._bridge._on_timeout()  # one poll, delivered now
        if app is not None:
            app.processEvents()
        host = client.host
        pending = getattr(host, "pending", 0)
        if not client.busy and not pending and not window._busy and client.state != "starting":
            if app is not None:
                app.processEvents()
            return
        if not isinstance(host, InlineHost):
            time.sleep(0.02)
    raise AssertionError("the window did not settle")
