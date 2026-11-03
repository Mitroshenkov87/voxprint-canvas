"""Canvas worker protocol ``canvas.worker/1``.

The worker host (``voxprint_common.worker``) carries pickled payloads and
results and does not read them. This module is the Canvas side of that
contract: request builders for the window, and a strict reader for replies.

Every request is a plain ``dict`` of strings, numbers, booleans, lists and
dicts. Every reply is one too. Pictures never travel inside a reply. The
worker writes a file into the private session folder the window gave it and
names that file in the reply, and it does so only after its checks pass.

A reply that does not match the schema reads as ``unavailable``. So does a
crashed worker, a timeout, or a job that failed for any other reason. Those
replies never carry a picture.

Imports stay light: no Qt, no model libraries, no Canvas pipeline code.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "PROTO",
    "OPS",
    "TRANSPORT_CODES",
    "Reply",
    "decode_event",
    "encode_event",
    "failure",
    "parse_reply",
    "refused",
    "request",
    "reply",
]

PROTO = "canvas.worker/1"

OPS = frozenset({
    "hello",
    "prompts",
    "generate",
    "series",
    "improve",
    "lora_import",
    "selftest",
    "probe_gpu",
    "warm",
})

# Codes the window adds when the worker did not answer cleanly. Each one ends the job with no picture.
TRANSPORT_CODES = frozenset({
    "cancelled",
    "worker_crashed",
    "worker_timeout",
    "worker_failed",
    "worker_unavailable",
    "proto_mismatch",
    "malformed",
})

_STATUSES = frozenset({"ok", "refused", "needs_confirm"})
_ITEM_ID = re.compile(r"^[0-9a-f]{32}$")
_EVENT_PREFIX = PROTO + " "


@dataclass(frozen=True)
class Reply:
    """One reply from the worker, after :func:`parse_reply` checked it.

    ``status`` is ``ok``, ``refused`` or ``needs_confirm``. A refused reply
    has a ``code`` and a ``message``. ``unavailable`` marks a reply the
    window shows as unavailable rather than as a refusal. ``data`` holds the
    route's own fields, already checked.
    """

    op: str
    status: str
    code: str = ""
    message: str = ""
    unavailable: bool = False
    data: Mapping[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True for a reply that may be used."""
        return self.status == "ok"


def request(op: str, **fields: object) -> dict[str, object]:
    """Build one request for ``op``.

    Args:
        op: One of :data:`OPS`.
        fields: The route's fields. They must pickle.

    Raises:
        ValueError: ``op`` is not part of the protocol.
    """
    if op not in OPS:
        raise ValueError(f"unknown worker op {op!r}")
    body: dict[str, object] = {"proto": PROTO, "op": op}
    body.update(fields)
    return body


def reply(op: str, **fields: object) -> dict[str, object]:
    """A successful reply, built in the worker."""
    body: dict[str, object] = {"proto": PROTO, "op": op, "status": "ok"}
    body.update(fields)
    return body


def refused(op: str, code: str, message: str, *, unavailable: bool) -> dict[str, object]:
    """A refused reply, built in the worker. It carries no picture."""
    return {
        "proto": PROTO,
        "op": op,
        "status": "refused",
        "code": str(code or "internal"),
        "message": str(message),
        "unavailable": bool(unavailable),
    }


def failure(op: str, code: str, detail: str = "") -> Reply:
    """A transport failure the window builds itself. Always unavailable, never a picture.

    ``detail`` (for example the worker's exception text) is kept for logs in
    ``data`` and is never the sentence the window shows.
    """
    return Reply(op, "refused", code, "", True, {"detail": detail} if detail else {})


def parse_reply(op: str, value: object, *, session_dir: Path | None = None) -> Reply:
    """Check one worker reply against the schema.

    Anything unexpected becomes an unavailable reply with code ``malformed``
    or ``proto_mismatch``. A picture path outside ``session_dir`` is
    malformed too.

    Args:
        op: The op the window sent.
        value: What the worker returned.
        session_dir: The private folder the worker was told to write into.
    """
    if not isinstance(value, Mapping):
        return failure(op, "malformed")
    if value.get("proto") != PROTO:
        return failure(op, "proto_mismatch")
    if value.get("op") != op:
        return failure(op, "malformed")
    status = value.get("status")
    if status not in _STATUSES:
        return failure(op, "malformed")
    if status == "refused":
        code = value.get("code")
        message = value.get("message")
        unavailable = value.get("unavailable")
        if not isinstance(code, str) or not code or not isinstance(message, str) or not isinstance(unavailable, bool):
            return failure(op, "malformed")
        return Reply(op, "refused", code, message, unavailable, {})
    data = {key: item for key, item in value.items() if key not in {"proto", "op", "status"}}
    if status == "needs_confirm":
        if op != "lora_import" or not isinstance(data.get("name"), str):
            return failure(op, "malformed")
        return Reply(op, "needs_confirm", data=data)
    checker = _CHECKS.get(op)
    if checker is None or not checker(data, session_dir):
        return failure(op, "malformed")
    return Reply(op, "ok", data=data)


def encode_event(kind: str, **data: object) -> str:
    """A structured progress message, for the window to read back with :func:`decode_event`."""
    return _EVENT_PREFIX + json.dumps({"kind": kind, **data}, ensure_ascii=False, separators=(",", ":"))


def decode_event(message: str) -> tuple[str, dict[str, object]] | None:
    """Read a structured progress message. Plain status text gives None."""
    if not isinstance(message, str) or not message.startswith(_EVENT_PREFIX):
        return None
    try:
        body = json.loads(message[len(_EVENT_PREFIX):])
    except (ValueError, TypeError):
        return None
    if not isinstance(body, dict) or not isinstance(body.get("kind"), str):
        return None
    kind = str(body.pop("kind"))
    return kind, body


def inside(path: object, folder: Path | None) -> Path | None:
    """``path`` as a file inside ``folder``, or None when it is not one."""
    if not isinstance(path, str) or not path or folder is None:
        return None
    try:
        candidate = Path(path).resolve()
        root = folder.resolve()
    except (OSError, RuntimeError):
        return None
    if candidate == root or root not in candidate.parents:
        return None
    return candidate


def _number(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _hello(data: Mapping[str, object], _folder: Path | None) -> bool:
    return isinstance(data.get("tier"), str)


def _prompts(data: Mapping[str, object], _folder: Path | None) -> bool:
    return not data


def _generate(data: Mapping[str, object], folder: Path | None) -> bool:
    item_id = data.get("item_id")
    if not isinstance(item_id, str) or _ITEM_ID.match(item_id) is None:
        return False
    path = inside(data.get("path"), folder)
    if path is None or not path.name.startswith(item_id) or not path.is_file():
        return False
    return _number(data.get("width")) and _number(data.get("height"))


def _series(data: Mapping[str, object], folder: Path | None) -> bool:
    return series_fields(data, folder)


def series_fields(data: Mapping[str, object], folder: Path | None) -> bool:
    """Check a series result, either a reply or the manifest a cancelled series leaves."""
    stopped = data.get("stopped")
    if stopped not in {"", "idea", "scenario", "cancelled", "error"}:
        return False
    if not isinstance(data.get("reason"), str) or not isinstance(data.get("code"), str):
        return False
    if not isinstance(data.get("hero_released"), bool):
        return False
    kept = data.get("kept")
    if not isinstance(kept, list):
        return False
    series_folder = data.get("folder")
    if not kept:
        return series_folder is None or inside(series_folder, folder) is not None
    root = inside(series_folder, folder)
    if root is None or not root.is_dir():
        return False
    for name in kept:
        if not isinstance(name, str) or "/" in name or "\\" in name or name in {"", ".", ".."}:
            return False
        if not (root / name).is_file():
            return False
    return True


def _improve(data: Mapping[str, object], _folder: Path | None) -> bool:
    return isinstance(data.get("prompt"), str)


def _lora(data: Mapping[str, object], _folder: Path | None) -> bool:
    entry = data.get("entry_id")
    return entry is None or (isinstance(entry, str) and bool(entry))


def _selftest(data: Mapping[str, object], _folder: Path | None) -> bool:
    return isinstance(data.get("report"), Mapping)


def _probe(data: Mapping[str, object], _folder: Path | None) -> bool:
    return isinstance(data.get("gpu"), bool)


def _warm(data: Mapping[str, object], _folder: Path | None) -> bool:
    return isinstance(data.get("warm"), bool)


_CHECKS = {
    "hello": _hello,
    "prompts": _prompts,
    "generate": _generate,
    "series": _series,
    "improve": _improve,
    "lora_import": _lora,
    "selftest": _selftest,
    "probe_gpu": _probe,
    "warm": _warm,
}
