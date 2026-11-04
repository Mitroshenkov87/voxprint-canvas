"""rc.36: two heavy jobs on two GPUs end cleanly instead of aborting the process.

rc.34/35: both jobs hit ``VramRefused``, ``empty_cache`` then raised
``AcceleratorError``, and the next garbage collection aborted the process
(exit 3). Host RAM was full because both processes kept the text encoder and
the transformer parked.

Everything here uses fakes: no GPU, no model files, no downloads.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import pytest
from voxprint_common.ram_probe import RamTier
from voxprint_common.worker import EventKind, Handlers, JobKind, JobSpec, JobState, WorkerHost

from canvas import worker_jobs
from canvas.infra import cuda_health, ram_ledger
from canvas.ipc.schema import parse_reply, request
from canvas.models import hot_tier
from canvas.models.modes import Mode, Tier
from canvas.pipeline import batch, gpu_release
from canvas.pipeline.bf16_schedule import VramRefused
from canvas.pipeline.batch import BatchEntry
from canvas.ui.shell_layout import PictureJob
from canvas.core.formats import OutputFormat
from tests.worker_handlers import AcceleratorError, _BrokenTorch

GIB = 1024**3
PICTURES = {".avif", ".jxl", ".webp", ".png", ".jpg"}


# --- the worker process ---------------------------------------------------------------------------------------


@pytest.fixture
def worker(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    host = WorkerHost(
        Handlers(run="tests.worker_handlers:run", check="tests.worker_handlers:check"),
        state_dir=tmp_path / "worker",
        backoff_initial=0.05,
    )
    folder = tmp_path / "session"
    folder.mkdir()
    yield host, folder
    host.shutdown()


def _generate_fields(prompt: str = "a red bicycle") -> dict[str, object]:
    job = PictureJob(
        prompt=prompt, negative="", mode=Mode.TEXT, aspect="1:1", output="native", fmt=OutputFormat.WEBP, seed=7,
        steps=8, guidance=1.0, sampler="euler", copies=1, adult_content=False,
    )
    return {
        "job": {**job._asdict(), "mode": Mode.TEXT.value, "fmt": job.fmt.value},
        "layout_prompt": prompt, "inputs": [], "vram": "auto", "warm": False,
    }


def _submit(host: WorkerHost, folder: Path, fixture: dict[str, object]) -> str:
    payload = request("generate", tier=Tier.GB24.value, out_dir=str(folder), _fixture=fixture, **_generate_fields())
    return host.submit(JobSpec(JobKind.INFERENCE, payload=payload, phase="generate"))


def _until(host: WorkerHost, done, timeout: float = 60.0) -> list:
    """Poll ``host`` until ``done(events)`` is true. Returns every event seen."""
    seen: list = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        seen.extend(host.poll())
        if done(seen):
            return seen
        time.sleep(0.01)
    raise AssertionError(f"timed out; events: {seen}")


def _pictures(folder: Path) -> list[Path]:
    return [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in PICTURES]


def test_vram_refused_in_the_worker_is_a_vram_reply_and_the_worker_stays_up(worker) -> None:
    host, folder = worker
    job_id = _submit(host, folder, {"tier": Tier.GB24.value, "gpu_fault": "vram"})
    events = _until(host, lambda _seen: host.outcome(job_id) is not None)
    outcome = host.outcome(job_id)
    assert outcome is not None and outcome.state is JobState.FINISHED, outcome
    reply = parse_reply("generate", outcome.value, session_dir=folder)
    assert not reply.ok and reply.code == "vram" and reply.unavailable
    assert reply.message == str(VramRefused())
    assert _pictures(folder) == []
    assert not any(event.kind is EventKind.CRASHED for event in events)
    pid = host.pid
    # The same process takes the next picture.
    again = _submit(host, folder, {"tier": Tier.GB24.value})
    _until(host, lambda _seen: host.outcome(again) is not None)
    after = host.outcome(again)
    assert after is not None and after.state is JobState.FINISHED
    assert parse_reply("generate", after.value, session_dir=folder).ok
    assert host.pid == pid


def test_a_failed_cuda_release_sends_the_vram_reply_then_the_worker_leaves_with_its_own_code(worker) -> None:
    host, folder = worker
    host.start()
    _until(host, lambda _seen: host.pid is not None)
    process = host._process
    assert process is not None
    job_id = _submit(host, folder, {"tier": Tier.GB24.value, "gpu_fault": "release"})
    events = _until(host, lambda _seen: host.outcome(job_id) is not None)
    outcome = host.outcome(job_id)
    assert outcome is not None and outcome.state is JobState.FINISHED, outcome
    reply = parse_reply("generate", outcome.value, session_dir=folder)
    assert not reply.ok and reply.code == "vram" and reply.unavailable
    assert _pictures(folder) == []
    process.join(10)
    # A clean, distinct exit: not the driver abort (3 / -6) and not a job failure.
    assert process.exitcode == cuda_health.EXIT_CUDA_LOST
    # 0.6.1: a clean reply then exit 75 is a planned restart, not a crash.
    assert not [event for event in events if event.kind is EventKind.CRASHED]
    # The host starts a fresh worker, and it draws.
    again = _submit(host, folder, {"tier": Tier.GB24.value})
    _until(host, lambda _seen: host.outcome(again) is not None)
    after = host.outcome(again)
    assert after is not None and after.state is JobState.FINISHED
    assert parse_reply("generate", after.value, session_dir=folder).ok
    assert host.pid != process.pid


# --- the lost CUDA context in this process -----------------------------------------------------------------------


@pytest.fixture
def exits(monkeypatch) -> list[int]:
    codes: list[int] = []

    def fake_exit(code: int) -> None:
        codes.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(cuda_health, "_exit", fake_exit)
    yield codes
    cuda_health.reset()


def test_a_failed_release_marks_cuda_lost_and_stops_collection(monkeypatch, exits) -> None:
    monkeypatch.setattr(gpu_release, "_torch", lambda: _BrokenTorch())
    assert gpu_release.release_cuda_cache() is False
    assert cuda_health.lost() and cuda_health.reason() == "AcceleratorError"
    assert not gc.isenabled()
    assert gc.get_freeze_count() > 0
    collected: list[int] = []
    monkeypatch.setattr(gpu_release.gc, "collect", lambda *a: collected.append(1) or 0)
    # Nothing touches CUDA or collects again.
    assert gpu_release.release_cuda_cache() is False
    assert gpu_release.free_vram_gb() is None
    assert collected == []
    assert exits == []
    cuda_health.reset()
    assert gc.isenabled() and gc.get_freeze_count() == 0


def test_cuda_faults_are_told_apart_from_a_plain_out_of_memory() -> None:
    assert cuda_health.is_cuda_fault(AcceleratorError("CUDA error: out of memory"))
    assert cuda_health.is_cuda_fault(RuntimeError("CUDA error: an illegal memory access was encountered"))
    wrapped = RuntimeError("stage failed")
    wrapped.__cause__ = AcceleratorError("boom")
    assert cuda_health.is_cuda_fault(wrapped)
    assert not cuda_health.is_cuda_fault(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not cuda_health.is_cuda_fault(VramRefused())
    assert not cuda_health.is_cuda_fault(None)


def test_a_lost_context_moves_no_parked_weights(monkeypatch, exits) -> None:
    from tests.test_ram_park import _Module

    monkeypatch.setenv("VOXPRINT_RAM_MODE", "ram96")
    hot_tier.reset()
    try:
        assert hot_tier.remember("vae", _Module("bfloat16"), size_bytes=1024)
        cuda_health.mark_lost("test")
        assert hot_tier.activate("vae", "cuda:0") is False
        moved: list[str] = []
        monkeypatch.setattr(hot_tier, "_assign", lambda *_a: moved.append("assign"))
        hot_tier.deactivate("vae")
        assert moved == []
    finally:
        cuda_health.reset()
        hot_tier.reset()


class _Ctx:
    """The parts of ``JobContext`` the routes use."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.job_id = "job-1"
        self.checkpoint_out = None
        self.sent: list[object] = []
        self.codes: list[int] = []
        self.notes: list[str] = []

    def reply_and_exit(self, result: object, code: int) -> None:
        self.sent.append(result)
        self.codes.append(code)
        raise SystemExit(code)

    def progress(self, *, message: str = "", **_fields: object) -> None:
        self.notes.append(message)

    def cancelled(self) -> bool:
        return False


def test_a_lost_context_replies_vram_with_no_picture_then_leaves(monkeypatch, tmp_path: Path, exits) -> None:
    from voxprint_common.worker import EXIT_GPU_UNRECOVERABLE

    picture = tmp_path / ("a" * 32 + ".webp")

    def route(_ctx, _current, _payload):
        # The picture was drawn, then the release after it broke the context.
        picture.write_bytes(b"pixels")
        cuda_health.mark_lost(AcceleratorError("CUDA error"))
        return {"proto": "canvas.worker/1", "op": "generate", "status": "ok", "path": str(picture)}

    monkeypatch.setitem(worker_jobs._ROUTES, "generate", route)
    monkeypatch.setattr(worker_jobs, "session", lambda _tier: object())
    ctx = _Ctx(request("generate", tier=Tier.GB24.value))
    with pytest.raises(SystemExit):
        worker_jobs.run(ctx)  # type: ignore[arg-type]
    assert ctx.codes == [EXIT_GPU_UNRECOVERABLE] == [cuda_health.EXIT_CUDA_LOST]
    assert not picture.exists()
    [sent] = ctx.sent
    assert sent["status"] == "refused" and sent["code"] == "vram"
    assert "path" not in sent


def test_a_worker_with_a_lost_context_takes_no_new_job(monkeypatch, exits) -> None:
    cuda_health.mark_lost("earlier job")
    with pytest.raises(SystemExit):
        worker_jobs.run(_Ctx(request("hello")))  # type: ignore[arg-type]
    assert exits == [cuda_health.EXIT_CUDA_LOST]


# --- the batch command -------------------------------------------------------------------------------------------


def test_batch_rows_report_vram_and_the_cli_leaves_cleanly(monkeypatch, tmp_path: Path, capsys, exits) -> None:
    from tests.test_batch import CUP, LAMP, _rewrite
    from tests.test_photoshoot_cli import _ready
    from canvas.app.main import main

    _ready(monkeypatch)

    def frame_for(entry: BatchEntry):
        def draw(line: str, seed: int, hero: object | None, strength: float | None):
            raise AcceleratorError("CUDA error: an illegal memory access was encountered")

        return draw

    monkeypatch.setattr("canvas.app.main.model_rewrite", lambda *_args: _rewrite)
    monkeypatch.setattr("canvas.app.main.model_generate", lambda *_args: frame_for(None))
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    manifest = tmp_path / "shoot.json"
    manifest.write_text(json.dumps({"entries": [{"shot": CUP, "name": "cup"}, {"shot": LAMP, "name": "lamp"}]}),
                        encoding="utf-8")
    with pytest.raises(SystemExit) as left:
        main(["batch", "--manifest", str(manifest), "--json"])
    assert left.value.code == 1 and exits == [1]
    body = json.loads(capsys.readouterr().out)
    rows = {row["name"]: row for row in body["entries"]}
    assert set(rows) == {"cup", "lamp"}
    assert all(row["made"] == 0 and row["frames"] == [] for row in rows.values())
    assert any(row["code"] == "vram" for row in rows.values())
    assert all(row["code"] in {"vram", "not_produced"} for row in rows.values() if row["status"] == "error")
    summary = json.loads((tmp_path / "shoot_batch" / "batch.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in summary["entries"]] == ["cup", "lamp"]


def test_an_entry_error_after_a_cuda_fault_is_a_vram_row(exits) -> None:
    result = batch._internal(AcceleratorError("CUDA error"))
    assert result.code == "vram" and result.kept == ()
    assert cuda_health.lost()
    cuda_health.reset()
    assert batch._internal(VramRefused()).code == "vram"
    assert batch._internal(RuntimeError("boom")).code == "internal"


# --- two jobs and one RAM ledger ---------------------------------------------------------------------------------


def _machine(monkeypatch, *, free: int, rss: int = 4 * GIB) -> None:
    monkeypatch.setenv("VOXPRINT_RAM_MODE", "ram96")
    monkeypatch.setattr(hot_tier, "probed_bytes", lambda: 128 * GIB)
    monkeypatch.setattr(hot_tier, "free_bytes", lambda: free)
    monkeypatch.setattr(ram_ledger, "process_bytes", lambda pid: rss)
    hot_tier.reset()


def _other_job(peak: int) -> Path:
    path = ram_ledger.folder() / f"voxprint-ram-{os.getppid()}.json"
    path.write_text(json.dumps({"pid": os.getppid(), "peak": peak}), encoding="utf-8")
    return path


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += max(seconds, 0.5)


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(hot_tier, "_clock", fake.time)
    monkeypatch.setattr(hot_tier, "_sleep", fake.sleep)
    yield fake
    hot_tier.set_wait_listener(None)
    hot_tier.reset()


def test_the_second_job_down_tiers_when_the_first_holds_the_ram_plan(monkeypatch, clock) -> None:
    # The first job on the other card planned the full ram96 park and has not loaded yet.
    first_peak = 4 * GIB + hot_tier._FOOTPRINT["transformer"] + hot_tier._FOOTPRINT["text_encoder"] \
        + hot_tier._FOOTPRINT["vae"] + hot_tier._JOB_RESERVE
    _machine(monkeypatch, free=105 * GIB)
    _other_job(first_peak)
    assert hot_tier.begin_job() is RamTier.RAM64
    assert clock.slept == []
    own = json.loads((ram_ledger.folder() / f"voxprint-ram-{os.getpid()}.json").read_text(encoding="utf-8"))
    # Both plans together stay inside free RAM.
    assert (first_peak - 4 * GIB) + (own["peak"] - 4 * GIB) <= 105 * GIB - hot_tier._FREE_FLOOR


def test_the_second_job_waits_while_no_tier_fits_then_runs_once_ram_is_back(monkeypatch, clock) -> None:
    _machine(monkeypatch, free=100 * GIB)
    other = _other_job(90 * GIB)
    stages: list[tuple[str, float]] = []

    def listener(stage: str, left: float) -> None:
        stages.append((stage, left))
        if len(stages) == 3:
            other.unlink()

    hot_tier.set_wait_listener(listener)
    assert hot_tier.begin_job() is RamTier.RAM96
    assert [stage for stage, _left in stages] == ["waiting_gpu"] * 3
    assert len(clock.slept) == 3


def test_the_wait_is_bounded_and_then_reads_from_disk(monkeypatch, clock) -> None:
    monkeypatch.setenv("VOXPRINT_RAM_WAIT_S", "10")
    _machine(monkeypatch, free=100 * GIB)
    _other_job(90 * GIB)
    stages: list[str] = []
    hot_tier.set_wait_listener(lambda stage, _left: stages.append(stage))
    assert hot_tier.begin_job() is RamTier.RAM32
    assert hot_tier.keeps_main_tier() is False
    assert 1 <= len(stages) <= 10
    assert clock.now >= 10


def test_a_cancelled_wait_stops_the_job(monkeypatch, clock) -> None:
    _machine(monkeypatch, free=100 * GIB)
    _other_job(90 * GIB)

    class Stop(Exception):
        pass

    def listener(_stage: str, _left: float) -> None:
        raise Stop()

    hot_tier.set_wait_listener(listener)
    with pytest.raises(Stop):
        hot_tier.begin_job()


def test_a_tier_query_outside_a_job_never_waits(monkeypatch, clock) -> None:
    _machine(monkeypatch, free=100 * GIB)
    _other_job(90 * GIB)
    assert hot_tier.job_tier() is RamTier.RAM32
    assert clock.slept == []


def test_planning_holds_the_cross_process_lock(monkeypatch, clock) -> None:
    _machine(monkeypatch, free=120 * GIB)
    held: list[bool] = []
    real = ram_ledger.others_pending

    def watching() -> int:
        held.append((ram_ledger.folder() / ram_ledger._PLAN_LOCK).exists())
        return real()

    monkeypatch.setattr(ram_ledger, "others_pending", watching)
    hot_tier.begin_job()
    assert held == [True]
    assert not (ram_ledger.folder() / ram_ledger._PLAN_LOCK).exists()


def test_a_plan_lock_left_by_a_finished_process_is_taken_over() -> None:
    import subprocess
    import sys

    gone = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    lock = ram_ledger.folder() / ram_ledger._PLAN_LOCK
    lock.write_text(gone.stdout.strip(), encoding="ascii")
    with ram_ledger.plan_lock(timeout=1.0) as held:
        assert held
        assert lock.read_text(encoding="ascii") == str(os.getpid())
    assert not lock.exists()


def test_a_plan_lock_held_by_a_live_process_times_out_and_planning_goes_on() -> None:
    lock = ram_ledger.folder() / ram_ledger._PLAN_LOCK
    lock.write_text(str(os.getppid()), encoding="ascii")
    started = time.monotonic()
    with ram_ledger.plan_lock(timeout=0.2) as held:
        assert not held
    assert time.monotonic() - started < 5
    assert lock.exists()
    lock.unlink()


def test_the_worker_reports_waiting_gpu_while_a_picture_waits(monkeypatch, tmp_path: Path, clock) -> None:
    from canvas.ipc.schema import decode_event

    _machine(monkeypatch, free=100 * GIB)
    other = _other_job(90 * GIB)

    def route(ctx, _current, _payload):
        hot_tier.begin_job()
        return {"proto": "canvas.worker/1", "op": "generate", "status": "ok"}

    def listener_check(stage: str, left: float) -> None:
        raise AssertionError("replaced by the worker")

    hot_tier.set_wait_listener(listener_check)
    monkeypatch.setitem(worker_jobs._ROUTES, "generate", route)
    monkeypatch.setattr(worker_jobs, "session", lambda _tier: object())
    ctx = _Ctx(request("generate", tier=Tier.GB24.value))
    real_progress = ctx.progress

    def progress(*, message: str = "", **fields: object) -> None:
        real_progress(message=message, **fields)
        other.unlink(missing_ok=True)

    ctx.progress = progress  # type: ignore[method-assign]
    result = worker_jobs.run(ctx)  # type: ignore[arg-type]
    assert result["status"] == "ok"
    decoded = [decode_event(note) for note in ctx.notes]
    assert decoded and decoded[0] is not None
    kind, data = decoded[0]
    assert kind == "note" and data["stage"] == "waiting_gpu"
    # The window's own listener is put back after the job.
    assert hot_tier._wait_listener is listener_check
