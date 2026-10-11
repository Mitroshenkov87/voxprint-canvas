"""Enlarge client. The public contract is the headless CLI, protocol 1.

stdout is JSON lines. stderr is a log: it is read and never parsed. A non-zero
exit code is the error the user sees. Canvas keeps the suite GPU lock and
passes its pid so Enlarge runs inside that lock.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from canvas.infra.discovery import AppInstall, find_app, probe_capabilities
from canvas.models.sizes import FOUR_K_LONG

PROTOCOL = 1
_STAGES = frozenset({"queued", "waiting_gpu", "loading", "tiles", "encoding"})
# name, user message, retryable. Exit 0 is success and is not in this table.
_EXITS: dict[int, tuple[str, str, bool]] = {
    1: ("internal", "Enlarge could not finish.", False),
    2: ("bad_input", "Enlarge could not read this picture.", False),
    3: ("unsupported_format", "Enlarge cannot write this format.", False),
    4: ("gpu_unsupported", "This computer cannot run Enlarge.", False),
    5: ("gpu_busy", "The GPU is busy.", True),
    6: ("oom", "Enlarge ran out of memory.", False),
    7: ("model_missing", "Download the upscale model in Voxprint AI Enlarge, then try again.", True),
    8: ("io_error", "Enlarge could not write the picture.", False),
    130: ("cancelled", "Enlarge was cancelled.", False),
}


class EnlargeError(RuntimeError):
    """Enlarge stopped. ``str`` is the message to show."""

    def __init__(self, code: str, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class EnlargeResult:
    """The file Enlarge wrote, and its size."""

    output: Path
    width: int
    height: int
    format: str


def exit_error(code: int) -> EnlargeError:
    """The user message for one Enlarge exit code. Unknown codes use internal."""
    name, message, retryable = _EXITS.get(code, _EXITS[1])
    return EnlargeError(name, message, retryable)


def _protocol_error(message: dict[str, Any]) -> EnlargeError:
    """The JSON ``code`` and ``message`` from an Enlarge ``error`` line.

    ``model_missing`` keeps the download hint. That line can name an internal
    weight, and the hint is the sentence the window shows.
    """
    raw_code = message.get("code")
    code = raw_code.strip() if isinstance(raw_code, str) and raw_code.strip() else "internal"
    raw_text = message.get("message")
    text = raw_text.strip() if isinstance(raw_text, str) else ""
    table = next((item for item in _EXITS.values() if item[0] == code), None)
    raw_retry = message.get("retryable")
    if isinstance(raw_retry, bool):
        retryable = raw_retry
    elif table is not None:
        retryable = table[2]
    else:
        retryable = False
    if code == "model_missing":
        body = _EXITS[7][1]
    elif text:
        body = text
    elif table is not None:
        body = table[1]
    else:
        body = _EXITS[1][1]
    return EnlargeError(code, f"{code}: {body}", retryable)


def enlarge_argv(
    program: list[str],
    *,
    source: Path,
    output: Path,
    long_side: int,
    format_name: str,
    parent_pid: int,
    preset: str = "photo",
    dry_run: bool = False,
) -> list[str]:
    """The only Enlarge command line Canvas builds."""
    argv = [
        *program,
        "upscale",
        "--input",
        str(source.resolve()),
        "--output",
        str(output),
        "--long-side",
        str(long_side),
        "--format",
        format_name,
        "--preset",
        preset,
        "--json-progress",
        "--parent-lock-pid",
        str(parent_pid),
    ]
    if dry_run:
        argv.append("--dry-run")
    return argv


def loads(line: bytes | bytearray | str) -> dict[str, Any]:
    """One protocol-1 JSON object from Enlarge's stdout."""
    text = line if isinstance(line, str) else bytes(line).decode("utf-8")
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("protocol") != PROTOCOL:
        raise EnlargeError("internal", "Enlarge sent a message this app does not understand.", False)
    return data


def message_kind(message: dict[str, Any]) -> str:
    """``type`` when it is set, otherwise ``op``. The release sends both."""
    kind = message.get("type")
    if isinstance(kind, str) and kind.strip():
        return kind.strip()
    kind = message.get("op")
    if isinstance(kind, str):
        return kind.strip()
    return ""


def capabilities_ready(body: dict[str, Any] | None) -> bool:
    """4K is offered when ``--capabilities --json`` reports protocol 1 and a usable GPU."""
    if not isinstance(body, dict) or body.get("protocol") != PROTOCOL:
        return False
    gpu = body.get("gpu")
    if isinstance(gpu, dict) and gpu.get("status") == "none":
        return False
    return True


class EnlargeClient:
    """Capability probe plus one enlarge job. Tests can pass ``command``."""

    def __init__(self, command: list[str] | None = None) -> None:
        if command is None:
            self.install = find_app("Enlarge", "enlarge")
        else:
            self.install = AppInstall(list(command), "", "voxprint-enlarge")
        self.command = list(self.install.command) if self.install is not None else []

    def installed(self) -> bool:
        """True when an Enlarge command was found."""
        return self.install is not None

    def capabilities(self) -> dict[str, Any] | None:
        """The ``--capabilities --json`` body, or None when Enlarge is absent."""
        if self.install is None:
            return None
        return probe_capabilities(self.install, "enlarge")

    def ready(self) -> bool:
        """True when 4K may be offered."""
        return capabilities_ready(self.capabilities())

    def enlarge(
        self,
        source: Path,
        *,
        output: Path,
        format_name: str = "avif",
        long_side: int = FOUR_K_LONG,
        dry_run: bool = False,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> EnlargeResult:
        """Run one upscale and return the written file."""
        if long_side != FOUR_K_LONG:
            raise exit_error(2)
        if format_name != "avif":
            raise exit_error(3)
        if not self.command:
            raise EnlargeError("io_error", "Enlarge is not installed.", False)
        parent_pid = os.getpid()
        command = enlarge_argv(
            self.command,
            source=source,
            output=output,
            long_side=long_side,
            format_name=format_name,
            parent_pid=parent_pid,
            dry_run=dry_run,
        )
        env = os.environ.copy()
        env["VOXPRINT_GPU_PARENT_PID"] = str(parent_pid)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        assert proc.stdout is not None

        def _capture_stderr() -> None:
            if proc.stderr is not None:
                proc.stderr.read()

        logs = threading.Thread(target=_capture_stderr, daemon=True)
        logs.start()
        result: EnlargeResult | None = None
        failure: EnlargeError | None = None
        try:
            for line in proc.stdout:
                if not line.strip():
                    continue
                try:
                    message = loads(line)
                except EnlargeError as exc:
                    failure = exc
                    break
                except json.JSONDecodeError:
                    failure = exit_error(1)
                    break
                if on_event is not None:
                    on_event(message)
                kind = message_kind(message)
                if kind == "done":
                    result = _result(message, output)
                elif kind == "error":
                    failure = _protocol_error(message)
                    break
                elif kind == "progress" and message.get("stage") not in _STAGES:
                    failure = exit_error(1)
                    break
                elif kind not in {"accepted", "progress"}:
                    failure = exit_error(1)
                    break
        finally:
            if failure is not None and proc.poll() is None:
                proc.kill()
            code = proc.wait()
            logs.join(timeout=5)
        if failure is not None:
            raise failure
        if code == 0 and result is not None:
            return result
        if code == 0:
            raise exit_error(1)
        raise exit_error(code)


def _result(message: dict[str, Any], fallback: Path) -> EnlargeResult:
    output = message.get("output")
    path = Path(output) if isinstance(output, str) and output else fallback
    return EnlargeResult(path, int(message.get("width") or 0), int(message.get("height") or 0), str(message.get("format") or ""))
