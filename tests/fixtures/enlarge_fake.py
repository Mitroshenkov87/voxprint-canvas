"""Stand-in for the Enlarge CLI contract v1.

``--dry-run`` and a normal ``upscale`` print the same stdout: protocol 1 JSON
lines for accepted, every progress stage, and done. Logs go to stderr.
``ENLARGE_FAKE_EXIT`` selects a non-zero exit after those lines.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_STAGES = (
    ("queued", 0),
    ("waiting_gpu", 5),
    ("loading", 15),
    ("tiles", 55),
    ("encoding", 90),
)
_CODES = {
    1: "internal",
    2: "bad_input",
    3: "unsupported_format",
    4: "gpu_unsupported",
    5: "gpu_busy",
    6: "oom",
    7: "model_missing",
    8: "io_error",
    130: "cancelled",
}
_PREFIX = b"VOXPRINT rgb "


def main(argv: list[str]) -> int:
    if "--capabilities" in argv and "--json" in argv:
        protocol = int(os.environ.get("ENLARGE_FAKE_PROTOCOL", "1"))
        status = os.environ.get("ENLARGE_FAKE_GPU", "ok")
        json.dump(
            {"protocol": protocol, "app": "enlarge", "version": "0.1.0", "gpu": {"status": status, "reason": None}},
            sys.stdout,
        )
        return 0
    if len(argv) < 2 or argv[1] != "upscale":
        print("enlarge: expected upscale", file=sys.stderr)
        return 2
    if "--scale" in argv or "--preset" in argv or "--quality" in argv:
        print("enlarge: old flags", file=sys.stderr)
        return 2
    source = _flag(argv, "--input")
    output = _flag(argv, "--output")
    long_side = _flag(argv, "--long-side")
    format_name = _flag(argv, "--format")
    parent = _flag(argv, "--parent-lock-pid")
    if source is None or output is None or long_side != "3840" or format_name != "avif" or parent is None:
        print("enlarge: bad arguments", file=sys.stderr)
        return 2
    if "--json-progress" not in argv:
        print("enlarge: missing json progress", file=sys.stderr)
        return 2
    env_pid = os.environ.get("VOXPRINT_GPU_PARENT_PID", "")
    if env_pid != parent:
        print("enlarge: parent pid mismatch", file=sys.stderr)
        return 2
    lock_file = os.environ.get("VOXPRINT_GPU_LOCK", "").strip()
    if lock_file:
        try:
            lock = json.loads(Path(lock_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print("enlarge: lock unreadable", file=sys.stderr)
            return 5
        if str(lock.get("pid")) != parent:
            print("enlarge: lock is not held by the parent", file=sys.stderr)
            return 5
    dry = "--dry-run" in argv
    print(f"enlarge: log dry_run={dry}", file=sys.stderr)
    Path(output + ".call.json").write_text(
        json.dumps({"argv": argv, "parent_env": env_pid, "dry_run": dry}),
        encoding="utf-8",
    )
    job = "1"
    _emit({"protocol": 1, "type": "accepted", "id": job, "queue_pos": 0})
    for stage, pct in _STAGES:
        _emit({"protocol": 1, "type": "progress", "id": job, "pct": pct, "stage": stage, "eta_s": 1.5})
    code = int(os.environ.get("ENLARGE_FAKE_EXIT", "0"))
    if code != 0:
        _emit(
            {
                "protocol": 1,
                "type": "error",
                "id": job,
                "code": _CODES.get(code, "internal"),
                "message": "raw cli detail model-id foo",
                "retryable": code in (5, 6),
            }
        )
        return code
    width, height = _write_output(source, output)
    _emit(
        {
            "protocol": 1,
            "type": "done",
            "id": job,
            "output": output,
            "width": width,
            "height": height,
            "format": "avif",
            "ms": 12,
        }
    )
    return 0


def _flag(argv: list[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def _emit(message: dict[str, object]) -> None:
    print(json.dumps(message), flush=True)


def _write_output(source: str, output: str) -> tuple[int, int]:
    raw = Path(source).read_bytes() if Path(source).is_file() else b""
    if not raw.startswith(_PREFIX):
        Path(output).write_bytes(b"dry-run")
        return 3840, 2160
    header, _, _body = raw.partition(b"\n")
    parts = header.decode("ascii").split()
    width, height = int(parts[2]), int(parts[3])
    factor = 3840 // max(width, height)
    out_w, out_h = width * factor, height * factor
    Path(output).write_bytes(_PREFIX + f"{out_w} {out_h}\n".encode("ascii") + bytes((4, 5, 6)) * (out_w * out_h))
    return out_w, out_h


if __name__ == "__main__":
    sys.exit(main(sys.argv))
