"""The RAM plan covers the whole job and counts other Canvas jobs, so two ram96 jobs cannot fill RAM.

rc.35: two ram96 jobs each parked the text encoder and the transformer, RAM reached
128 of 128 GB, and the process aborted. Memory numbers here are fakes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from voxprint_common.ram_probe import RamTier

from canvas.infra import ram_ledger
from canvas.models import hot_tier
from tests.test_ram_park import _Module

GIB = 1024**3
TE = hot_tier._FOOTPRINT["text_encoder"]
DIT = hot_tier._FOOTPRINT["transformer"]
VAE = hot_tier._FOOTPRINT["vae"]


def _machine(monkeypatch, *, free: int, tier: str = "ram96", rss: int = 4 * GIB) -> dict[str, int]:
    memory = {"free": free, "rss": rss}
    monkeypatch.setenv("VOXPRINT_RAM_MODE", tier)
    monkeypatch.setattr(hot_tier, "probed_bytes", lambda: 128 * GIB)
    monkeypatch.setattr(hot_tier, "free_bytes", lambda: memory["free"])
    monkeypatch.setattr(ram_ledger, "process_bytes", lambda pid: memory["rss"])
    hot_tier.reset()
    return memory


def _other_job(peak: int) -> Path:
    """A ledger entry for another live process: this test's parent."""
    path = ram_ledger.folder() / f"voxprint-ram-{os.getppid()}.json"
    path.write_text(json.dumps({"pid": os.getppid(), "peak": peak}), encoding="utf-8")
    return path


def _park_all() -> dict[str, bool]:
    return {
        name: hot_tier.remember(name, _Module("bfloat16"), size_bytes=size)
        for name, size in (("transformer", DIT), ("text_encoder", TE), ("vae", VAE))
    }


def test_a_job_alone_on_a_large_machine_keeps_every_model(monkeypatch) -> None:
    _machine(monkeypatch, free=120 * GIB)
    assert hot_tier.plan_job() is RamTier.RAM96
    assert _park_all() == {"transformer": True, "text_encoder": True, "vae": True}
    entry = json.loads((ram_ledger.folder() / f"voxprint-ram-{os.getpid()}.json").read_text(encoding="utf-8"))
    assert entry["peak"] == 4 * GIB + TE + DIT + VAE + hot_tier._JOB_RESERVE


def test_the_second_job_counts_what_the_first_still_takes_and_reads_from_disk(monkeypatch) -> None:
    # The first job planned a 60 GB peak and uses 10 GB so far; 100 GB is free.
    memory = _machine(monkeypatch, free=100 * GIB)
    _other_job(60 * GIB)
    memory["rss"] = 10 * GIB
    assert ram_ledger.others_pending() == 50 * GIB
    assert hot_tier.plan_job() is RamTier.RAM32
    assert hot_tier.keeps_main_tier() is False
    assert _park_all() == {"transformer": False, "text_encoder": False, "vae": False}


def test_a_job_with_room_for_the_transformer_only_keeps_it_and_reads_the_rest(monkeypatch) -> None:
    room = DIT + 2 * GIB
    _machine(monkeypatch, free=room + hot_tier._JOB_RESERVE + hot_tier._FREE_FLOOR)
    assert hot_tier.plan_job() is RamTier.RAM64
    assert hot_tier.keeps_main_tier() is True
    parked = _park_all()
    assert parked["transformer"] is True
    assert parked["text_encoder"] is False


def test_every_park_checks_free_ram_again(monkeypatch) -> None:
    memory = _machine(monkeypatch, free=120 * GIB)
    assert hot_tier.plan_job() is RamTier.RAM96
    assert hot_tier.remember("transformer", _Module("bfloat16"), size_bytes=DIT) is True
    # Another job starts after the plan and expects 106 GB more.
    _other_job(110 * GIB)
    assert hot_tier.remember("text_encoder", _Module("bfloat16"), size_bytes=TE) is False
    memory["free"] = 120 * GIB
    _other_job(4 * GIB)
    assert hot_tier.remember("text_encoder", _Module("bfloat16"), size_bytes=TE) is True


def test_a_batch_keeps_one_plan_for_every_entry(monkeypatch) -> None:
    memory = _machine(monkeypatch, free=120 * GIB)
    with hot_tier.job_scope() as tier:
        assert tier is RamTier.RAM96
        memory["free"] = 10 * GIB
        assert hot_tier.begin_job() is RamTier.RAM96
    assert hot_tier.begin_job() is RamTier.RAM32


def test_unknown_free_ram_keeps_the_suite_tier(monkeypatch) -> None:
    _machine(monkeypatch, free=0)
    _other_job(200 * GIB)
    assert hot_tier.plan_job() is RamTier.RAM96


def test_an_entry_from_a_finished_process_is_ignored_and_removed(monkeypatch) -> None:
    _machine(monkeypatch, free=120 * GIB)
    gone = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    pid = int(gone.stdout.strip())
    stale = ram_ledger.folder() / f"voxprint-ram-{pid}.json"
    stale.write_text(json.dumps({"pid": pid, "peak": 500 * GIB}), encoding="utf-8")
    assert ram_ledger.others_pending() == 0
    assert not stale.exists()


def test_two_processes_see_each_other_through_the_ledger(monkeypatch) -> None:
    _machine(monkeypatch, free=100 * GIB, rss=0)
    script = (
        "import sys, time\n"
        "from canvas.infra import ram_ledger\n"
        f"ram_ledger.publish({70 * GIB})\n"
        "print('ready', flush=True)\n"
        "sys.stdin.readline()\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(Path.cwd()), *sys.path]))
    other = subprocess.Popen(
        [sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
    )
    try:
        assert other.stdout is not None and other.stdout.readline().strip() == "ready"
        assert ram_ledger.others_pending() == 70 * GIB
        assert hot_tier.plan_job() is RamTier.RAM32
    finally:
        assert other.stdin is not None
        other.stdin.write("\n")
        other.stdin.flush()
        other.wait(timeout=30)
    deadline = time.monotonic() + 5
    while ram_ledger.others_pending() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ram_ledger.others_pending() == 0
    assert hot_tier.plan_job() is RamTier.RAM96


@pytest.fixture(autouse=True)
def _clean() -> None:
    yield
    hot_tier.reset()
    ram_ledger.withdraw()
