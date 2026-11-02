"""The window's handle on the Canvas worker process.

:class:`WorkerClient` wraps a ``voxprint_common.worker.WorkerHost`` (or any
object with the same ``submit``/``check``/``cancel``/``poll``/``shutdown``
surface) and its Qt adapter. It never blocks the UI thread: replies arrive
through ``QtWorkerHost`` signals and are handed to the callback given at
submit time, exactly once.

Every way a job can end without a valid reply (the worker crashed, stopped
sending heartbeats, raised, returned something malformed, or missed its
deadline) is delivered as an unavailable :class:`~canvas.ipc.schema.Reply`
with no picture. A picture file that arrives for a job the window already
gave up on is deleted.

Nothing here imports torch, diffusers, transformers or the Canvas gate.
"""
from __future__ import annotations

import contextlib
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QTimer
from voxprint_common.worker import CheckHook, Handlers, JobKind, JobSpec, WorkerEvent, WorkerHost
from voxprint_common.worker.qt import QtWorkerHost

from canvas.ipc.schema import Reply, failure, inside, parse_reply, request

__all__ = ["DEADLINES", "HANDLERS", "WorkerClient", "default_host"]

HANDLERS = Handlers(run="canvas.worker_jobs:run", check="canvas.worker_jobs:check")

# Seconds a job may run before the window gives up on it. The heartbeat rule catches a stuck process sooner.
DEADLINES: dict[str, float] = {
    "hello": 300.0,
    "prompts": 300.0,
    "generate": 1800.0,
    "series": 4 * 3600.0,
    "improve": 1800.0,
    "lora_import": 1800.0,
    "selftest": 3600.0,
    "probe_gpu": 120.0,
    "warm": 300.0,
}

# Model jobs load weights; the heartbeat thread keeps running while torch releases the interpreter.
HEARTBEAT_TIMEOUT = 60.0
CANCEL_TIMEOUT = 30.0

Done = Callable[[Reply], None]
Progress = Callable[[WorkerEvent], None]


def default_host(handlers: Handlers = HANDLERS) -> WorkerHost:
    """The worker host the app uses: one spawned process, restarted after a crash."""
    return WorkerHost(handlers, heartbeat_interval=1.0, heartbeat_timeout=30.0, cancel_timeout=CANCEL_TIMEOUT)


@dataclass
class _Job:
    op: str
    on_done: Done
    on_progress: Progress | None
    deadline: float
    cancelled: bool = False
    extra: dict[str, object] = field(default_factory=dict)


class WorkerClient:
    """Submit window requests to the worker and route its events back.

    ``state`` is ``idle`` until :meth:`start`, ``starting`` while the hello
    job runs, then ``ready`` or ``failed``. Picture work waits for ``ready``
    through :meth:`when_ready`. A failed hello is tried again the next time
    work is asked for.
    """

    def __init__(
        self,
        host: object,
        *,
        tier: str,
        interval_ms: int = 30,
        deadlines: Mapping[str, float] | None = None,
        heartbeat_timeout: float | None = HEARTBEAT_TIMEOUT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wrap ``host``. Construct this on the UI thread.

        Args:
            host: A ``WorkerHost``, or a stand-in with the same methods.
            tier: The tier value the window chose at startup.
            interval_ms: How often the adapter polls the host.
            deadlines: Per-op seconds before a job is given up. Defaults to :data:`DEADLINES`.
            heartbeat_timeout: Per-job heartbeat timeout for model jobs. None uses the host's.
            clock: Monotonic clock.
        """
        self.host = host
        self.tier = tier
        self.adapter = QtWorkerHost(host, interval_ms=interval_ms)  # type: ignore[arg-type]
        self.adapter.progress.connect(self._on_progress)
        self.adapter.finished.connect(self._on_finished)
        self.adapter.failed.connect(self._on_failed)
        self.adapter.crashed.connect(self._on_crashed)
        self.hook = CheckHook(host)  # type: ignore[arg-type]
        self.state = "idle"
        self.failure = ""
        self.hello: dict[str, object] = {}
        self.crashes = 0
        self._deadlines = dict(DEADLINES)
        if deadlines:
            self._deadlines.update(deadlines)
        self._clock = clock
        self._heartbeat_timeout = heartbeat_timeout
        self._jobs: dict[str, _Job] = {}
        self._waiting: list[tuple[Callable[[], None], Callable[[str], None]]] = []
        self._session: Path | None = None
        self._closed = False
        self._timer = QTimer()
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.check_deadlines)
        self._timer.start()

    # --- lifecycle --------------------------------------------------------------------------------------------

    @property
    def session_dir(self) -> Path:
        """Private folder for this window session. Pictures cross here, after the worker's checks."""
        if self._session is None:
            self._session = Path(tempfile.mkdtemp(prefix="canvas-session-"))
        return self._session

    def start(self) -> None:
        """Start the worker and send hello. A second call while it runs or after it passed does nothing."""
        if self._closed or self.state in {"starting", "ready"}:
            return
        self.state = "starting"
        self.submit("hello", {}, on_done=self._on_hello)

    def when_ready(self, go: Callable[[], None], fail: Callable[[str], None]) -> None:
        """Run ``go`` once the worker said hello, or ``fail`` with a reason when it could not."""
        if self._closed:
            fail("")
            return
        if self.state == "ready":
            go()
            return
        self._waiting.append((go, fail))
        if self.state != "starting":
            self.state = "idle"
            self.start()

    @property
    def busy(self) -> bool:
        """True while any job is open."""
        return bool(self._jobs)

    def shutdown(self) -> None:
        """Stop polling, end the worker process and remove the session folder. A second call does nothing."""
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        pending = list(self._jobs.items())
        self._jobs.clear()
        with contextlib.suppress(Exception):
            self.adapter.shutdown(timeout=2.0)
        for _job_id, job in pending:
            with contextlib.suppress(Exception):
                job.on_done(failure(job.op, "worker_unavailable"))
        if self._session is not None:
            shutil.rmtree(self._session, ignore_errors=True)

    # --- jobs ------------------------------------------------------------------------------------------------

    def submit(
        self,
        op: str,
        fields: Mapping[str, object],
        *,
        on_done: Done,
        on_progress: Progress | None = None,
    ) -> str:
        """Queue one request and return its job id. ``on_done`` runs exactly once.

        A request the host refuses (for example a payload that does not
        pickle) ends as unavailable on the next event-loop turn.
        """
        payload = request(op, tier=self.tier, out_dir=str(self.session_dir), **fields)
        try:
            if self._closed:
                raise RuntimeError("the worker is shut down")
            if op == "prompts":
                job_id = self.hook.run(payload, phase=op)
            else:
                job_id = self.host.submit(  # type: ignore[attr-defined]
                    JobSpec(JobKind.INFERENCE, payload=payload, phase=op, heartbeat_timeout=self._heartbeat_timeout)
                )
        except Exception as exc:  # noqa: BLE001 - every refusal of the transport ends unavailable
            message = str(exc)
            QTimer.singleShot(0, lambda: on_done(failure(op, "worker_unavailable", message)))
            return ""
        self._jobs[job_id] = _Job(op, on_done, on_progress, self._clock() + self._deadlines.get(op, 1800.0))
        return job_id

    def check_prompts(self, prompts: list[str], *, on_done: Done) -> str:
        """Review ``prompts`` as a ``check`` job (``CheckHook``) before a picture starts."""
        return self.submit("prompts", {"prompts": list(prompts)}, on_done=on_done)

    def cancel(self, job_id: str) -> None:
        """Ask the worker to stop ``job_id``. The reply still arrives, as cancelled or as a result."""
        job = self._jobs.get(job_id)
        if job is None or job.cancelled:
            return
        job.cancelled = True
        with contextlib.suppress(LookupError):
            self.host.cancel(job_id)  # type: ignore[attr-defined]

    def check_deadlines(self) -> None:
        """End every job that is past its deadline as unavailable, and ask the worker to stop it."""
        now = self._clock()
        late = [job_id for job_id, job in self._jobs.items() if now >= job.deadline]
        for job_id in late:
            job = self._jobs.pop(job_id)
            with contextlib.suppress(LookupError):
                self.host.cancel(job_id)  # type: ignore[attr-defined]
            job.on_done(failure(job.op, "worker_timeout"))

    def discard(self, value: object) -> None:
        """Delete a picture file or a series folder named in a reply that will not be used."""
        if not isinstance(value, Mapping):
            return
        session = self._session
        for key in ("path", "folder"):
            target = inside(value.get(key), session)
            if target is None:
                continue
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                with contextlib.suppress(OSError):
                    target.unlink()

    # --- events ----------------------------------------------------------------------------------------------

    def _on_hello(self, reply: Reply) -> None:
        waiting, self._waiting = self._waiting, []
        if reply.ok:
            self.state = "ready"
            self.failure = ""
            self.hello = dict(reply.data)
            for go, _fail in waiting:
                go()
            return
        self.state = "failed"
        self.failure = reply.message
        for _go, fail in waiting:
            fail(reply.message)

    def _on_progress(self, event: WorkerEvent) -> None:
        job = self._jobs.get(event.job_id)
        if job is not None and job.on_progress is not None:
            job.on_progress(event)

    def _on_finished(self, event: WorkerEvent) -> None:
        job = self._jobs.pop(event.job_id, None)
        if job is None:
            self.discard(event.value)
            return
        reply = parse_reply(job.op, event.value, session_dir=self._session)
        if not reply.ok:
            self.discard(event.value)
        job.on_done(reply)

    def _on_failed(self, event: WorkerEvent) -> None:
        job = self._jobs.pop(event.job_id, None)
        if job is None:
            return
        if str(event.kind) == "cancelled":
            job.on_done(Reply(job.op, "refused", "cancelled", "", True, {"checkpoint": event.checkpoint}))
            return
        job.on_done(failure(job.op, "worker_failed", event.message))

    def _on_crashed(self, event: WorkerEvent) -> None:
        self.crashes += 1
        job = self._jobs.pop(event.job_id, None) if event.job_id else None
        if job is not None:
            job.on_done(failure(job.op, "worker_crashed", event.message))
