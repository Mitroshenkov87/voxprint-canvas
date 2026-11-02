"""An in-process stand-in for ``voxprint_common.worker.WorkerHost``, for window tests.

It runs the same handler strings (``canvas.worker_jobs:run``/``check``) in
this process, one job per :meth:`poll`, so window tests stay fast and can use
fakes that hold test state. Payloads and results are pickled both ways like
the real host does. Tests that need a real process use ``WorkerHost``.
"""
from __future__ import annotations

import importlib
import os
import pickle
import tempfile
import uuid
from collections import deque
from pathlib import Path

from voxprint_common.worker import (
    EventKind,
    Handlers,
    JobCancelled,
    JobContext,
    JobKind,
    JobOutcome,
    JobSpec,
    JobState,
    WorkerEvent,
)

from canvas.ipc.client import HANDLERS


def _callable(path: str):
    module_name, _sep, qual = path.partition(":")
    target: object = importlib.import_module(module_name)
    for part in qual.split("."):
        target = getattr(target, part)
    return target


class InlineHost:
    """Same surface as ``WorkerHost``: submit, check, cancel, poll, wait, shutdown, kill_worker, pid."""

    def __init__(self, handlers: Handlers = HANDLERS) -> None:
        self._run = _callable(handlers.run)
        self._check = _callable(handlers.check) if handlers.check else self._run
        self._pending: deque[tuple[str, JobSpec]] = deque()
        self._events: list[WorkerEvent] = []
        self._cancelled: set[str] = set()
        self._current: str | None = None
        self._outcomes: dict[str, JobOutcome] = {}
        self._state = Path(tempfile.mkdtemp(prefix="inline-worker-"))
        self.submitted: list[JobSpec] = []
        self.closed = False

    @property
    def pid(self) -> int | None:
        return None if self.closed else os.getpid()

    def start(self) -> None:
        return None

    def submit(self, spec: JobSpec) -> str:
        if self.closed:
            raise RuntimeError("Worker is shut down")
        pickle.dumps(spec.payload)
        job_id = uuid.uuid4().hex
        self._pending.append((job_id, spec))
        self.submitted.append(spec)
        return job_id

    def check(self, payload: object, *, model: str | None = None, phase: str | None = None) -> str:
        return self.submit(JobSpec(JobKind.CHECK, payload=payload, model=model, phase=phase))

    def cancel(self, job_id: str) -> None:
        if self._current == job_id:
            self._cancelled.add(job_id)
            return
        for item in list(self._pending):
            if item[0] == job_id:
                self._pending.remove(item)
                self._end(job_id, item[1], EventKind.CANCELLED, JobState.CANCELLED, None, "cancelled", None)
                return
        raise LookupError(job_id)

    @property
    def pending(self) -> int:
        return len(self._pending) + (1 if self._current else 0)

    def poll(self) -> tuple[WorkerEvent, ...]:
        if self.closed:
            return ()
        if self._pending and self._current is None:
            job_id, spec = self._pending.popleft()
            self._execute(job_id, spec)
        events, self._events = self._events, []
        return tuple(events)

    def wait(self, job_id: str, *, timeout: float = 30.0) -> JobOutcome:
        del timeout
        while job_id not in self._outcomes:
            if not self._pending:
                raise TimeoutError(job_id)
            self.poll()
        return self._outcomes[job_id]

    def outcome(self, job_id: str) -> JobOutcome | None:
        return self._outcomes.get(job_id)

    def kill_worker(self) -> None:
        self._events.append(WorkerEvent(kind=EventKind.CRASHED, message="Worker crashed (exit -9).", exit_code=-9))

    def shutdown(self, *, timeout: float = 2.0) -> None:
        del timeout
        self.closed = True

    def _emit(self, message: object) -> None:
        name = type(message).__name__
        if name == "Progress":
            self._events.append(WorkerEvent(
                kind=EventKind.PROGRESS,
                job_id=message.job_id,  # type: ignore[attr-defined]
                message=message.message,  # type: ignore[attr-defined]
                step=message.step,  # type: ignore[attr-defined]
                total_steps=message.total_steps,  # type: ignore[attr-defined]
                fraction=message.fraction,  # type: ignore[attr-defined]
                phase=message.phase,  # type: ignore[attr-defined]
            ))

    def _execute(self, job_id: str, spec: JobSpec) -> None:
        self._current = job_id
        ctx = JobContext(
            job_id=job_id,
            kind=spec.kind,
            model=spec.model,
            phase=spec.phase,
            payload=pickle.loads(pickle.dumps(spec.payload)),
            checkpoint=spec.checkpoint,
            model_object=None,
            state_dir=self._state,
            emit=self._emit,
            poll_cancel=lambda: self._cancelled,
        )
        handler = self._check if spec.kind is JobKind.CHECK else self._run
        try:
            try:
                value = handler(ctx)
            except JobCancelled as exc:
                self._end(job_id, spec, EventKind.CANCELLED, JobState.CANCELLED, None, "cancelled",
                          exc.checkpoint or ctx.checkpoint_out)
                return
            except Exception as exc:  # noqa: BLE001 - mirrors the real host
                self._end(job_id, spec, EventKind.FAILED, JobState.FAILED, None, f"{type(exc).__name__}: {exc}",
                          ctx.checkpoint_out)
                return
            if ctx.cancelled():
                self._end(job_id, spec, EventKind.CANCELLED, JobState.CANCELLED, None, "cancelled", ctx.checkpoint_out)
                return
            try:
                value = pickle.loads(pickle.dumps(value))
            except Exception as exc:  # noqa: BLE001
                self._end(job_id, spec, EventKind.FAILED, JobState.FAILED, None, f"TypeError: {exc}", None)
                return
            self._end(job_id, spec, EventKind.FINISHED, JobState.FINISHED, value, "", ctx.checkpoint_out)
        finally:
            self._current = None
            self._cancelled.discard(job_id)

    def _end(self, job_id, spec, kind, state, value, message, checkpoint) -> None:
        self._outcomes[job_id] = JobOutcome(job_id, state, value=value, message=message, checkpoint=checkpoint)
        self._events.append(WorkerEvent(
            kind=kind, job_id=job_id, message=message, value=value, checkpoint=checkpoint, phase=spec.phase,
        ))
