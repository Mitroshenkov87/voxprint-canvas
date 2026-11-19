"""Window close asks the worker to stop, then lets it leave cleanly (rc.45 finding 3).

A cancelled worker stops at its next cancel check and exits without being
killed, so nothing it registered (shared semaphores among the rest) is left
for the resource tracker to report at shutdown.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from canvas.ipc.client import WorkerClient
from canvas.models.modes import Tier
from voxprint_common.worker import JobKind, JobSpec


class _HoldingHost:
    """A host that accepts a job and never finishes it, recording calls."""

    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.shutdowns: list[float] = []
        self._next = 0

    def submit(self, spec: JobSpec) -> str:
        del spec
        self._next += 1
        return f"job-{self._next}"

    def check(self, payload: object, *, model: str | None = None, phase: str | None = None) -> str:
        del payload, model, phase
        self._next += 1
        return f"job-{self._next}"

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)

    def poll(self):
        return ()

    def shutdown(self, *, timeout: float = 2.0) -> None:
        self.shutdowns.append(timeout)


def _client(host: _HoldingHost) -> WorkerClient:
    return WorkerClient(host, tier=Tier.GB16.value, interval_ms=5)


def test_shutdown_cancels_open_jobs_before_ending_the_worker(qapp) -> None:
    del qapp
    host = _HoldingHost()
    client = _client(host)
    seen: list[str] = []
    first = client.submit("generate", {}, on_done=lambda reply: seen.append(reply.code))
    second = client.submit("prompts", {}, on_done=lambda reply: seen.append(reply.code))
    client.shutdown()
    # Every open job was asked to stop before the host was shut down.
    assert host.cancelled == [first, second]
    assert host.shutdowns == [pytest.approx(30.0)]
    assert sorted(seen) == ["worker_unavailable", "worker_unavailable"]
    # A second shutdown is a no-op.
    client.shutdown()
    assert host.shutdowns == [pytest.approx(30.0)]


def test_shutdown_passes_the_grace_to_the_host(qapp) -> None:
    del qapp
    host = _HoldingHost()
    client = _client(host)
    client.shutdown(grace=2.5)
    assert host.shutdowns == [pytest.approx(2.5)]
