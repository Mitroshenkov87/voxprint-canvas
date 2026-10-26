"""Per-stage wall-clock timings for one run. Stage names and seconds only.

No prompt text, picture data or check verdicts are stored, so the summary may go
to the ``--json`` report and to the generic log. Each measured stage also writes
a start line, an end line, and a heartbeat while it is still running.
"""
from __future__ import annotations

import re
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import psutil
from voxprint_common.paths import logs_dir

from canvas.infra.vram_trace import note_vram

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
# A long stage prints this often so a stall is visible without waiting for the end line.
HEARTBEAT_S = 30.0
_LOG_LOCK = threading.Lock()
_LOG_TRIES = 5


class StageTimes:
    """Thread-safe totals per stage name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._started = time.perf_counter()
        self._tags: dict[str, str] = {}

    def tag(self, key: str, value: str) -> None:
        """Record one short code (for example ``tier=gb16``) for the log line."""
        if not _NAME.match(key) or not _NAME.match(value):
            raise ValueError("tags are short lowercase codes")
        with self._lock:
            self._tags[key] = value

    def note_mib(self, key: str, mib: int) -> None:
        """Keep the larger MiB reading. The value is ``mib`` plus the number."""
        if not _NAME.match(key) or mib <= 0:
            raise ValueError("tags are short lowercase codes")
        with self._lock:
            current = self._tags.get(key, "")
            previous = int(current[3:]) if current.startswith("mib") and current[3:].isdigit() else 0
            best = max(previous, mib)
            value = f"mib{best}"
        self.tag(key, value)

    def add(self, stage: str, seconds: float) -> None:
        """Add one sample. ``stage`` must be a short lowercase code."""
        if not _NAME.match(stage):
            raise ValueError("stage names are short lowercase codes")
        with self._lock:
            self._totals[stage] = self._totals.get(stage, 0.0) + max(0.0, float(seconds))
            self._counts[stage] = self._counts.get(stage, 0) + 1

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Time the block under ``stage``, including when it raises."""
        note_vram(stage, "before")
        device, vram = _device_and_vram()
        _write_stage(stage, "start", device, vram)
        started = time.perf_counter()
        stop = threading.Event()
        beater = threading.Thread(target=_heartbeat, args=(stage, stop), name=f"stage-{stage}", daemon=True)
        beater.start()
        try:
            yield
        finally:
            stop.set()
            beater.join(timeout=1)
            self.add(stage, time.perf_counter() - started)
            note_vram(stage, "after")
            device, vram = _device_and_vram()
            _write_stage(stage, "end", device, vram)

    def summary(self) -> dict[str, object]:
        """``{"wall_s": .., "stages": {name: {"count", "total_s", "mean_s"}}}``."""
        with self._lock:
            stages = {
                name: {
                    "count": self._counts[name],
                    "total_s": round(total, 3),
                    "mean_s": round(total / self._counts[name], 3),
                }
                for name, total in sorted(self._totals.items())
            }
        return {"wall_s": round(time.perf_counter() - self._started, 3), "stages": stages}

    def log_line(self) -> str:
        """One generic-log line: ``timing wall=.. name=count/total ...``."""
        data = self.summary()
        stages = data["stages"]
        assert isinstance(stages, dict)
        parts = [f"wall={data['wall_s']}"]
        parts += [f"{name}={row['count']}x{row['total_s']}" for name, row in stages.items()]
        with self._lock:
            parts += [f"{key}={value}" for key, value in sorted(self._tags.items())]
        return "timing " + " ".join(parts)


_current: StageTimes | None = None


def set_current(times: StageTimes | None) -> StageTimes | None:
    """Make ``times`` the run-wide collector for library stages. Returns the previous one."""
    global _current
    previous, _current = _current, times
    return previous


@contextmanager
def timed(stage: str) -> Iterator[None]:
    """Time a library stage into the run-wide collector, when one is set."""
    times = _current
    if times is None:
        yield
        return
    with times.measure(stage):
        yield


def note_stage(stage: str, seconds: float) -> None:
    """Add one sample to the run-wide collector, when one is set."""
    times = _current
    if times is not None:
        times.add(stage, seconds)


def note_tag(key: str, value: str) -> None:
    """Tag the run-wide collector, when one is set."""
    times = _current
    if times is not None:
        times.tag(key, value)


def note_peak_mib(mib: int) -> None:
    """Remember the higher MiB peak on the timing line as ``peak_vram=mibN``."""
    times = _current
    if times is None or isinstance(mib, bool) or not isinstance(mib, int) or mib <= 0:
        return
    times.note_mib("peak_vram", mib)


def log_stage(
    stage: str, event: str, device: str, *, tokens: int | None = None, tok_s: int | None = None,
    dtype: str = "", kv_cache: str = "", prefill_ms: int | None = None,
    decode_tok_s: int | None = None, graph: str = "",
) -> None:
    """Write one ``stage`` line. ``device`` is ``cuda``, ``cpu``, or ``none``.

    A writer line carries ``tokens``, ``tok_s``, ``prefill_ms``, ``decode_tok_s``,
    ``dtype`` and ``kv_cache``. ``tok_s`` is the decode rate. Prefill is separate
    so a long prompt is not mistaken for a slow decode.
    """
    parts: list[str] = []
    if tokens is not None and tok_s is not None:
        parts.append(f"tokens={int(tokens)} tok_s={int(tok_s)}")
    if prefill_ms is not None:
        parts.append(f"prefill_ms={int(prefill_ms)}")
    if decode_tok_s is not None:
        parts.append(f"decode_tok_s={int(decode_tok_s)}")
    if graph:
        parts.append(f"graph={graph}")
    if dtype:
        parts.append(f"dtype={dtype}")
    if kv_cache:
        parts.append(f"kv_cache={kv_cache}")
    _device, vram = _device_and_vram()
    _write_stage(stage, event, device, vram, " ".join(parts))


def log_detail(stage: str, event: str, device: str, suffix: str = "") -> None:
    """Write one stage line with an extra suffix. RAM and VRAM are included."""
    _device, vram = _device_and_vram()
    _write_stage(stage, event, device, vram, suffix)


@contextmanager
def beat_while(stage: str) -> Iterator[None]:
    """Write heartbeats for one block, including when no ``StageTimes`` is current.

    The Photoshoot entry loads the prompt check before it opens a measured
    stage, and the window warms the same models on a side thread. Both run
    under ``pythonw``, whose stderr is ``None``. This uses the same log writer
    as ``StageTimes.measure``.
    """
    if not _NAME.match(stage):
        yield
        return
    stop = threading.Event()
    beater = threading.Thread(target=_heartbeat, args=(stage, stop), name=f"stage-{stage}", daemon=True)
    beater.start()
    try:
        yield
    finally:
        stop.set()
        beater.join(timeout=1)


def _heartbeat(stage: str, stop: threading.Event) -> None:
    """Write a heartbeat to the log every ``HEARTBEAT_S`` until the stage ends.

    ``pythonw`` has no stderr. A print there used to kill this thread after the
    first wait, so the log never grew. A Windows sharing violation on the log
    file is retried inside ``_write_stage``. A later failure in the device read
    must not kill the thread either: the next wait still writes.
    """
    delay = HEARTBEAT_S if HEARTBEAT_S > 0 else 30.0
    while not stop.wait(delay):
        try:
            device, vram = _device_and_vram()
            _write_stage(stage, "heartbeat", device, vram)
            _echo_heartbeat(stage, device, vram)
        except Exception:
            continue


def _echo_heartbeat(stage: str, device: str, vram: int) -> None:
    stream = sys.stderr
    if stream is None:
        return
    try:
        print(f"stage {stage} heartbeat device={device} vram_mib={vram}", file=stream, flush=True)
    except (AttributeError, OSError, ValueError, RuntimeError):
        return


def _write_stage(stage: str, event: str, device: str, vram: int, suffix: str = "") -> None:
    if not _NAME.match(stage) or not _NAME.match(event):
        return
    kind = device if device in {"cuda", "cpu", "none"} else "none"
    line = (
        f"stage {stage} {event} device={kind} "
        f"vram_mib={max(0, int(vram))} ram_mib={_ram_mib()}"
    )
    if suffix:
        line = f"{line} {suffix}"
    for attempt in range(_LOG_TRIES):
        try:
            with _LOG_LOCK:
                folder = logs_dir()
                folder.mkdir(parents=True, exist_ok=True)
                with (folder / "voxprint.log").open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()
            return
        except PermissionError:
            if attempt + 1 == _LOG_TRIES:
                return
            time.sleep(0.02 * (attempt + 1))
        except OSError:
            return


def _ram_mib() -> int:
    """RAM in use, in MiB. A failed read logs zero."""
    try:
        return max(0, int(psutil.virtual_memory().used) // (1024 * 1024))
    except (OSError, AttributeError, ValueError, TypeError):
        return 0


def _device_and_vram() -> tuple[str, int]:
    """``cuda`` when the card holds a model, ``cpu`` when CUDA is idle, else ``none``."""
    # torch is an optional extra. Importing it here keeps Canvas importable
    # in unit tests when the package is absent. This is not a circular import.
    try:
        import torch
    except ImportError:
        return "none", 0
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return "none", 0
    try:
        available = bool(cuda.is_available())
    except (TypeError, RuntimeError):
        return "none", 0
    if not available:
        return "none", 0
    try:
        used = int(cuda.memory_allocated())
    except (TypeError, RuntimeError, AttributeError):
        used = 0
    mib = max(0, used // (1024 * 1024))
    # A few hundred MiB is the idle baseline. A loaded model is several GiB.
    if mib >= 512:
        return "cuda", mib
    return "cpu", mib
