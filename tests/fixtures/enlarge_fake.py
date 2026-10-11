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

from PIL import Image

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
class _NotPng(Exception):
    """The upscale input is not a PNG Pillow can decode."""


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
    if "--scale" in argv or "--quality" in argv:
        print("enlarge: old flags", file=sys.stderr)
        return 2
    source = _flag(argv, "--input")
    output = _flag(argv, "--output")
    long_side = _flag(argv, "--long-side")
    format_name = _flag(argv, "--format")
    preset = _flag(argv, "--preset")
    parent = _flag(argv, "--parent-lock-pid")
    if (
        source is None
        or output is None
        or long_side != "3840"
        or format_name != "avif"
        or preset != "photo"
        or parent is None
    ):
        print("enlarge: bad arguments", file=sys.stderr)
        return 2
    if "--json-progress" not in argv:
        print("enlarge: missing json progress", file=sys.stderr)
        return 2
    env_pid = os.environ.get("VOXPRINT_GPU_PARENT_PID", "")
    if env_pid != parent:
        print("enlarge: parent pid mismatch", file=sys.stderr)
        return 2
    dirty = _dirty_env()
    if dirty is not None:
        print(f"enlarge: dirty environment {dirty}", file=sys.stderr)
        return 2
    lock_file = os.environ.get("VOXPRINT_GPU_LOCK", "").strip()
    if lock_file and Path(lock_file).is_file():
        try:
            lock = json.loads(Path(lock_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print("enlarge: lock unreadable", file=sys.stderr)
            return 5
        if str(lock.get("owner")) == "canvas" and str(lock.get("pid")) == parent:
            print("enlarge: canvas still holds the gpu lock", file=sys.stderr)
            return 5
    dry = "--dry-run" in argv
    print(f"enlarge: log dry_run={dry}", file=sys.stderr)
    Path(output + ".call.json").write_text(
        json.dumps({"argv": argv, "parent_env": env_pid, "dry_run": dry, "env": sorted(os.environ)}),
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
                "retryable": code in (5, 7),
            }
        )
        return code
    try:
        width, height = _png_size(source)
    except _NotPng:
        _emit(
            {
                "protocol": 1,
                "type": "error",
                "id": job,
                "code": "bad_input",
                "message": "cannot identify image file",
                "retryable": False,
            }
        )
        return 2
    width, height = _write_output(output, width, height)
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


_BANNED = {
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "CUDA_MODULE_LOADING",
}


def _dirty_env() -> str | None:
    """An inherited variable Enlarge must not see. The interpreter may add a locale."""
    if not os.environ.get("PATH"):
        return "PATH"
    if not os.environ.get("VOXPRINT_GPU_PARENT_PID"):
        return "VOXPRINT_GPU_PARENT_PID"
    for key in os.environ:
        upper = key.upper()
        if upper in _BANNED or upper.startswith(("PYTORCH_", "TORCH_", "CUDA_")):
            return key
    return None


def _flag(argv: list[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def _emit(message: dict[str, object]) -> None:
    body = dict(message)
    name = body.get("type")
    kind = os.environ.get("ENLARGE_FAKE_KIND", "both")
    if kind == "op":
        body.pop("type", None)
        body["op"] = name
    elif kind == "type":
        pass
    else:
        body["op"] = name
    print(json.dumps(body), flush=True)


def _png_size(source: str) -> tuple[int, int]:
    path = Path(source)
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                raise _NotPng()
            return image.size
    except (OSError, ValueError) as exc:
        raise _NotPng() from exc


def _write_output(output: str, width: int, height: int) -> tuple[int, int]:
    factor = 3840 // max(width, height)
    out_w, out_h = width * factor, height * factor
    Image.new("RGB", (out_w, out_h), (4, 5, 6)).save(output, format="AVIF")
    return out_w, out_h


if __name__ == "__main__":
    sys.exit(main(sys.argv))
