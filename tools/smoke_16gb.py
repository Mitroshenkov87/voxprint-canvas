"""16 GB smoke run for Proving Ground.

Generates five 1024x1024 pictures with Z-Image-Turbo BF16 on the 16 GB tier
(group offload), through the normal ``generate`` command. Every output goes
through the content gate inside that command, so a written file is a gate pass
and a blocked message is a gate fail. The prompts are neutral.

Exit codes: 0 every picture passed, 1 a picture failed or was blocked,
2 setup, 3 no GPU, 4 the content gate is unavailable, 5 a manifest entry
failed its download or hash check (the deep self-test re-hashes every file).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from PIL import Image

PROMPTS = (
    "a white porcelain teapot on a wooden table, soft daylight",
    "a mountain landscape with a lake at sunrise",
    "a busy city street at dusk with lit shop windows",
    "a ceramic teapot and two cups on a linen cloth, studio light",
    "a green valley with a river and distant hills, clear sky",
)
SIZE = (1024, 1024)
BLOCKED = "This request cannot be completed."
UNAVAILABLE = "Image generation is unavailable."
EXIT_GATE_UNAVAILABLE = 4
EXIT_MANIFEST = 5
ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smoke_16gb")
    parser.add_argument("--out", default="smoke-16gb", help="folder for pictures and result.json")
    parser.add_argument("--python", default=sys.executable, help="interpreter that has Canvas installed")
    parser.add_argument("--vram-cap-gb", type=float, default=16.0)
    return parser


def _canvas(python: str, cap: float, *args: str) -> list[str]:
    return [python, "-m", "canvas.app.main", "--vram-cap-gb", _cap_text(cap), *args]


def _cap_text(cap: float) -> str:
    return str(int(cap)) if float(cap) == int(cap) else str(cap)


def _commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def gpu_name() -> str:
    """First GPU name from nvidia-smi, or an empty string."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return lines[0] if lines else ""


def _sample_vram(stop: threading.Event, peak: list[int]) -> None:
    while not stop.wait(1):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        for line in proc.stdout.splitlines():
            token = line.strip().split(",")[0].strip()
            if token.isdigit():
                peak[0] = max(peak[0], int(token))


def gate_ready(report: dict[str, object]) -> bool:
    """True when every safety classifier row in the self-test is present and loaded."""
    rows = report.get("classifiers")
    if not isinstance(rows, list) or not rows:
        return False
    for row in rows:
        if not isinstance(row, dict) or not row.get("ok") or row.get("detail") == "not_downloaded":
            return False
    return True


def manifest_ids(path: Path | None = None) -> set[str]:
    """Model ids the installer downloads through MDL (``mdl.json``)."""
    source = ROOT / "mdl.json" if path is None else path
    try:
        body = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    rows = body.get("models") if isinstance(body, dict) else None
    if not isinstance(rows, list):
        return set()
    return {str(row.get("id")) for row in rows if isinstance(row, dict) and row.get("id")}


def manifest_failures(report: dict[str, object], wanted: set[str]) -> list[str]:
    """Problems from a deep self-test: an empty manifest, a missing file or a hash failure.

    Every row must pass its hash. Every id the manifest lists, and the 16 GB
    generator, must be downloaded. Optional packs that are not in the manifest
    may stay absent.
    """
    if not wanted:
        return ["mdl.json lists no models"]
    required = set(wanted) | {"z-image-turbo-bf16-16gb"}
    failed: list[str] = []
    seen: set[str] = set()
    for key in ("models", "classifiers"):
        rows = report.get(key)
        if not isinstance(rows, list):
            failed.append(f"{key}: missing from the self-test")
            continue
        for row in rows:
            if not isinstance(row, dict):
                failed.append(f"{key}: bad row")
                continue
            ident = str(row.get("id"))
            seen.add(ident)
            detail = str(row.get("detail") or "")
            if not row.get("ok"):
                failed.append(f"{ident}: {detail or 'failed'}")
            elif ident in required and detail == "not_downloaded":
                failed.append(f"{ident}: not_downloaded")
    for ident in sorted(required - seen):
        failed.append(f"{ident}: not in the self-test")
    return failed


def _json(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.strip().splitlines()):
        try:
            body = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(body, dict):
            return body
    return {}


def run_image(python: str, cap: float, index: int, prompt: str, dest: Path) -> dict[str, object]:
    """Generate one picture and describe the result."""
    started = time.perf_counter()
    proc = subprocess.run(
        _canvas(python, cap, "generate", "--prompt", prompt, "--aspect", "1:1", "--output-size", "native",
                "--seed", str(index), "--out", str(dest), "--json"),
        capture_output=True, text=True, timeout=7200, check=False,
    )
    wall = time.perf_counter() - started
    body = _json(proc.stdout)
    text = proc.stdout + proc.stderr
    row: dict[str, object] = {
        "index": index,
        "prompt": prompt,
        "exit_code": proc.returncode,
        "wall_seconds": round(wall, 3),
        "seconds_per_image": body.get("seconds_per_image"),
        "diffusion_seconds": body.get("diffusion_seconds"),
        "peak_allocated_bytes": body.get("peak_allocated_bytes"),
        "size": None,
        "gate": "fail",
        "detail": "",
    }
    if proc.returncode == 0 and body.get("ok") and dest.is_file():
        with Image.open(dest) as image:
            size = (image.width, image.height)
        row["size"] = list(size)
        row["gate"] = "pass"
        if size != SIZE:
            row["detail"] = "unexpected size"
    elif UNAVAILABLE in text:
        row["gate"] = "unavailable"
        row["detail"] = "gate unavailable"
    elif BLOCKED in text:
        row["detail"] = "blocked"
    else:
        row["gate"] = "error"
        row["detail"] = str(body.get("error") or proc.stderr.strip()[-400:])
    return row


def summarise(rows: list[dict[str, object]], nvidia_peak_mib: int, gpu: str, tier: str) -> dict[str, object]:
    """The JSON document Proving Ground keeps."""
    allocated: list[int] = []
    for row in rows:
        value = row.get("peak_allocated_bytes")
        if isinstance(value, int) and not isinstance(value, bool):
            allocated.append(value)
    return {
        "commit": _commit(),
        "gpu": gpu,
        "tier": tier,
        "model": "Z-Image-Turbo BF16 (16 GB group offload)",
        "size": list(SIZE),
        "images": rows,
        "peak_vram": {
            "torch_max_memory_allocated_bytes": max(allocated) if allocated else None,
            "nvidia_smi_peak_mib": nvidia_peak_mib,
        },
        "gate": {
            "pass": sum(1 for r in rows if r["gate"] == "pass"),
            "fail": sum(1 for r in rows if r["gate"] != "pass"),
        },
    }


def _write(folder: Path, payload: dict[str, object], code: int) -> int:
    payload["exit_code"] = code
    payload["ok"] = code == 0
    text = json.dumps(payload, indent=2)
    (folder / "result.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    return code


def main(argv: list[str] | None = None) -> int:
    """Run the smoke set and write ``result.json``."""
    args = _parser().parse_args(argv)
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    gpu = gpu_name()
    base: dict[str, object] = {"commit": _commit(), "gpu": gpu}
    if not gpu:
        return _write(folder, {**base, "error": "no GPU"}, 3)
    try:
        proc = subprocess.run(
            _canvas(args.python, args.vram_cap_gb, "--selftest", "--deep", "--json"),
            capture_output=True, text=True, timeout=7200, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _write(folder, {**base, "error": f"selftest: {exc}"}, 2)
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        report = None
    if not isinstance(report, dict):
        return _write(folder, {**base, "error": "selftest did not print JSON", "stderr": proc.stderr[-800:]}, 2)
    if not gate_ready(report):
        return _write(folder, {**base, "error": "content gate unavailable", "classifiers": report.get("classifiers")},
                      EXIT_GATE_UNAVAILABLE)
    broken = manifest_failures(report, manifest_ids())
    if broken:
        return _write(folder, {**base, "error": "a manifest entry failed its download or hash check",
                               "failed": broken}, EXIT_MANIFEST)
    gpu_report = report.get("gpu") if isinstance(report.get("gpu"), dict) else {}
    tier = str(gpu_report.get("tier") or "")  # type: ignore[union-attr]
    if tier != "16gb":
        return _write(folder, {**base, "error": f"expected the 16gb tier, got {tier or 'none'}"}, 2)
    peak = [0]
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_vram, args=(stop, peak), daemon=True)
    sampler.start()
    rows: list[dict[str, object]] = []
    try:
        for index, prompt in enumerate(PROMPTS, start=1):
            rows.append(run_image(args.python, args.vram_cap_gb, index, prompt, folder / f"smoke-{index}.png"))
            if rows[-1]["gate"] == "unavailable":
                break
    finally:
        stop.set()
        sampler.join(timeout=3)
    payload = summarise(rows, peak[0], gpu, tier)
    if any(r["gate"] == "unavailable" for r in rows):
        return _write(folder, payload, EXIT_GATE_UNAVAILABLE)
    ok = len(rows) == len(PROMPTS) and all(r["gate"] == "pass" and not r["detail"] for r in rows)
    return _write(folder, payload, 0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
