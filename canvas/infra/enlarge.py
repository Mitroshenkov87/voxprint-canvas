"""Enlarge client. The public contract is the headless CLI.

Progress lines and the final result are JSON on stdout. A non-zero exit code
is the error. Canvas does not open a socket and does not start Enlarge itself.
The argument list lives in :func:`enlarge_argv` so a rename of the command
(for example ``upscale --in`` / ``--out``) is a one-place change.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from canvas.infra.discovery import AppInstall, find_app, probe_capabilities

PROTOCOL = 1


class EnlargeError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class EnlargeResult:
    output: Path
    width: int
    height: int
    format: str


def enlarge_argv(program: list[str], *, source: Path, output: Path, scale: int, format_name: str) -> list[str]:
    """The only Enlarge command line Canvas builds.

    Current contract: ``voxprint-enlarge --input X --output Y --scale N --json-progress``.
    """
    return [
        *program,
        "--input",
        str(source.resolve()),
        "--output",
        str(output),
        "--scale",
        str(scale),
        "--preset",
        "photo",
        "--quality",
        "fast",
        "--format",
        format_name,
        "--json-progress",
    ]


def loads(line: bytes | bytearray | str) -> dict[str, Any]:
    text = line if isinstance(line, str) else bytes(line).decode("utf-8")
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("protocol") != PROTOCOL:
        raise EnlargeError("internal", "Enlarge sent a message this app does not understand.", False)
    return data


class EnlargeClient:
    """Capability probe plus one enlarge job. Tests can pass ``command``."""

    def __init__(self, command: list[str] | None = None) -> None:
        if command is None:
            self.install = find_app("Enlarge", "enlarge")
        else:
            self.install = AppInstall(list(command), "", "voxprint-enlarge")
        self.command = list(self.install.command) if self.install is not None else []

    def installed(self) -> bool:
        return self.install is not None

    def capabilities(self) -> dict[str, Any] | None:
        if self.install is None:
            return None
        return probe_capabilities(self.install, "enlarge")

    def enlarge(
        self,
        source: Path,
        *,
        scale: int,
        output: Path,
        format_name: str = "avif",
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> EnlargeResult:
        if scale not in (2, 4):
            raise EnlargeError("bad_input", "scale must be 2 or 4", False)
        if not self.command:
            raise EnlargeError("io_error", "Enlarge is not installed.", False)
        command = enlarge_argv(self.command, source=source, output=output, scale=scale, format_name=format_name)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        assert proc.stdout is not None
        result: EnlargeResult | None = None
        failure: EnlargeError | None = None
        for line in proc.stdout:
            if not line.strip():
                continue
            try:
                message = loads(line)
            except (EnlargeError, json.JSONDecodeError) as exc:
                failure = EnlargeError("internal", str(exc), False)
                break
            if on_event is not None:
                on_event(message)
            kind = message.get("type")
            if kind == "done":
                result = _result(message, output)
            elif kind == "error":
                failure = _error(message)
        code = proc.wait()
        if failure is not None:
            raise failure
        if result is None or code != 0:
            raise EnlargeError("internal", "Enlarge did not finish.", False)
        return result


def _result(message: dict[str, Any], fallback: Path) -> EnlargeResult:
    output = message.get("output")
    path = Path(output) if isinstance(output, str) and output else fallback
    return EnlargeResult(path, int(message.get("width") or 0), int(message.get("height") or 0), str(message.get("format") or ""))


def _error(message: dict[str, Any]) -> EnlargeError:
    code = message.get("code")
    text = message.get("message")
    return EnlargeError(
        code if isinstance(code, str) else "internal",
        text if isinstance(text, str) else "Enlarge failed.",
        bool(message.get("retryable", False)),
    )
