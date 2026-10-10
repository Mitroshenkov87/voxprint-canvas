"""Enlarge protocol 1 client.

Discovery and socket names follow the suite rules. Messages follow
IPC-ENLARGE protocol 1: a local JSON-lines socket, then
``voxprint-enlarge --input ... --json-progress`` when the socket is down.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtNetwork import QLocalSocket

from canvas.app.info import APP_VERSION
from canvas.infra.discovery import AppInstall, find_app, probe_capabilities, socket_path

PROTOCOL = 1
_SOCKET_WAIT_S = 3.0
_MESSAGE_WAIT_S = 30.0


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


def dumps(message: dict[str, Any]) -> bytes:
    body = dict(message)
    body["protocol"] = PROTOCOL
    return (json.dumps(body, separators=(",", ":")) + "\n").encode("utf-8")


def loads(line: bytes | bytearray | str) -> dict[str, Any]:
    text = line if isinstance(line, str) else bytes(line).decode("utf-8")
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("protocol") != PROTOCOL:
        raise EnlargeError("internal", "Enlarge sent a message this app does not understand.", False)
    return data


class EnlargeClient:
    """Capability probe plus one enlarge job. Tests can pass ``command`` and ``socket``."""

    def __init__(self, command: list[str] | None = None, socket: str | None = None) -> None:
        if command is None:
            self.install = find_app("Enlarge", "enlarge")
        else:
            self.install = AppInstall(list(command), "", "voxprint-enlarge")
        self.command = list(self.install.command) if self.install is not None else []
        self._socket_name = socket
        self._spawned: subprocess.Popen[bytes] | None = None
        self._buf = bytearray()

    def installed(self) -> bool:
        return self.install is not None

    def socket_name(self) -> str:
        if self._socket_name is not None:
            return self._socket_name
        override = os.environ.get("CANVAS_ENLARGE_SOCKET", "").strip()
        if override:
            return override
        return socket_path("enlarge", "voxprint-enlarge")

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
        transport: str = "auto",
    ) -> EnlargeResult:
        if scale not in (2, 4):
            raise EnlargeError("bad_input", "scale must be 2 or 4", False)
        if transport not in {"auto", "socket", "cli"}:
            raise EnlargeError("internal", "Unknown Enlarge transport.", False)
        if transport == "cli":
            return self._cli(source, scale=scale, output=output, format_name=format_name, on_event=on_event)
        try:
            return self._socket(source, scale=scale, output=output, format_name=format_name, on_event=on_event)
        except EnlargeError as exc:
            if transport == "socket" or exc.code != "io_error" or not self.command:
                raise
            return self._cli(source, scale=scale, output=output, format_name=format_name, on_event=on_event)

    def shutdown(self) -> None:
        spawned = self._spawned
        self._spawned = None
        if spawned is not None and spawned.poll() is None:
            spawned.terminate()

    def _socket(
        self,
        source: Path,
        *,
        scale: int,
        output: Path,
        format_name: str,
        on_event: Callable[[dict[str, Any]], None] | None,
    ) -> EnlargeResult:
        sock = self._connect()
        self._buf = bytearray()
        job_id = uuid.uuid4().hex
        self._write(sock, {"type": "hello", "client": "canvas", "version": APP_VERSION})
        hello = self._read(sock)
        self._emit(on_event, hello)
        protocols = hello.get("protocols")
        if hello.get("type") != "hello" or hello.get("server") != "enlarge" or not isinstance(protocols, list) or PROTOCOL not in protocols:
            raise EnlargeError("internal", "Enlarge did not answer hello.", False)
        self._write(sock, _request(job_id, source, scale, output, format_name))
        while True:
            message = self._read(sock)
            self._emit(on_event, message)
            kind = message.get("type")
            if kind == "done":
                return _result(message, output)
            if kind == "error":
                raise _error(message)
            if kind not in {"accepted", "progress"}:
                raise EnlargeError("internal", "Enlarge sent an unexpected message.", False)

    def _cli(
        self,
        source: Path,
        *,
        scale: int,
        output: Path,
        format_name: str,
        on_event: Callable[[dict[str, Any]], None] | None,
    ) -> EnlargeResult:
        if not self.command:
            raise EnlargeError("io_error", "Enlarge is not installed.", False)
        command = [
            *self.command,
            "--input",
            str(source.resolve()),
            "--scale",
            str(scale),
            "--preset",
            "photo",
            "--quality",
            "fast",
            "--format",
            format_name,
            "--output",
            str(output),
            "--json-progress",
        ]
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
            self._emit(on_event, message)
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

    def _connect(self) -> QLocalSocket:
        live = _try_connect(self.socket_name(), 300)
        if live is not None:
            return live
        if not self.command:
            raise EnlargeError("io_error", "Enlarge is not installed.", False)
        if self._spawned is None or self._spawned.poll() is not None:
            try:
                self._spawned = subprocess.Popen([*self.command, "--serve", "--hidden"])
            except OSError as exc:
                raise EnlargeError("io_error", "Enlarge is not installed.", False) from exc
        deadline = time.monotonic() + _SOCKET_WAIT_S
        while time.monotonic() < deadline:
            live = _try_connect(self.socket_name(), 200)
            if live is not None:
                return live
            if self._spawned.poll() is not None:
                break
        raise EnlargeError("io_error", "Enlarge did not open a socket.", False)

    def _read(self, sock: QLocalSocket) -> dict[str, Any]:
        deadline = time.monotonic() + _MESSAGE_WAIT_S
        while time.monotonic() < deadline:
            if b"\n" in self._buf:
                line, _, rest = self._buf.partition(b"\n")
                self._buf = bytearray(rest)
                if not line.strip():
                    continue
                return loads(line)
            remaining = int((deadline - time.monotonic()) * 1000)
            if remaining <= 0 or not sock.waitForReadyRead(min(remaining, 200)):
                continue
            self._buf.extend(sock.readAll().data())
        raise EnlargeError("io_error", "Enlarge did not answer.", False)

    def _write(self, sock: QLocalSocket, message: dict[str, Any]) -> None:
        written = sock.write(dumps(message))
        if written < 0 or not sock.waitForBytesWritten(1000):
            raise EnlargeError("io_error", "Could not write to Enlarge.", False)

    @staticmethod
    def _emit(on_event: Callable[[dict[str, Any]], None] | None, message: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(message)


def _request(job_id: str, source: Path, scale: int, output: Path, format_name: str) -> dict[str, Any]:
    return {
        "type": "enlarge.request",
        "id": job_id,
        "input": str(source.resolve()),
        "scale": scale,
        "preset": "photo",
        "quality": "fast",
        "model": None,
        "format": format_name,
        "output": str(output),
        "output_dir": None,
        "open_ui": False,
    }


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


def _try_connect(name: str, timeout_ms: int) -> QLocalSocket | None:
    sock = QLocalSocket()
    sock.connectToServer(name)
    if sock.waitForConnected(timeout_ms):
        return sock
    sock.abort()
    return None
