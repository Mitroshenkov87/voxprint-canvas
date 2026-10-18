"""16 GB / 24 GB smoke run for Proving Ground.

Generates five 1024x1024 pictures with Z-Image-Turbo BF16 through the normal
``generate`` command, in one of two passes:

* ``--pass 16gb``: ``--vram-cap-gb 16`` (default cap). Canvas applies
  ``torch.cuda.set_per_process_memory_fraction`` and takes the 16 GB tier, so a
  24 GB card runs the 16 GB group-offload path.
* ``--pass 24gb``: no cap. The card must be on the 24 GB tier.

Every output goes through the content gate inside that command, so a written
file is a gate pass and a blocked message is a gate fail. The prompts are neutral.

Afterwards a gate timing benchmark loads the safety stack in this process and
times each kept frame by stage: the prompt check, the NSFW pass and the age
stage (SigLIP labels, YuNet, FairFace, the estimator). The age stage is forced on these
neutral frames (``forced: true``), and when YuNet finds no face FairFace and the estimator read
one centred box, so the cost is measured even though the product gate would
not need it. The benchmark records timings only, never scores or decisions.
Each pass writes ``result-<pass>.json``.

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
PASSES = {"16gb": ("z-image-turbo-bf16-16gb", "Z-Image-Turbo BF16 (16 GB group offload)"),
          "24gb": ("z-image-turbo-bf16", "Z-Image-Turbo BF16 (24 GB)")}
ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smoke_16gb")
    parser.add_argument("--out", default="smoke-16gb", help="folder for pictures and result-<pass>.json")
    parser.add_argument("--python", default=sys.executable, help="interpreter that has Canvas installed")
    parser.add_argument("--pass", dest="tier_pass", choices=sorted(PASSES), default="16gb")
    parser.add_argument("--vram-cap-gb", type=float, default=None,
                        help="cap for the 16gb pass (default 16); the 24gb pass runs uncapped")
    parser.add_argument("--manifest", default=None, help="mdl.json to check (default: this checkout's)")
    parser.add_argument("--no-bench", action="store_true", help="skip the gate timing benchmark")
    return parser


def _cap_for(tier_pass: str, cap: float | None) -> float | None:
    if tier_pass == "24gb":
        return None
    return 16.0 if cap is None or cap <= 0 else float(cap)


def _canvas(python: str, cap: float | None, *args: str) -> list[str]:
    capped = [] if cap is None else ["--vram-cap-gb", _cap_text(cap)]
    return [python, "-m", "canvas.app.main", *capped, *args]


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


def _canvas_version(python: str) -> str:
    """Installed voxprint-canvas version in that interpreter, or an empty string."""
    try:
        proc = subprocess.run(
            [python, "-c", "import importlib.metadata as m; print(m.version('voxprint-canvas'))"],
            capture_output=True, text=True, timeout=60, check=False,
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
    """Required model ids in ``mdl.json``. Optional entries are fetched only on request."""
    source = ROOT / "mdl.json" if path is None else path
    try:
        body = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    rows = body.get("models") if isinstance(body, dict) else None
    if not isinstance(rows, list):
        return set()
    return {str(row.get("id")) for row in rows
            if isinstance(row, dict) and row.get("id") and not row.get("optional")}


def manifest_failures(report: dict[str, object], wanted: set[str],
                      generator: str = "z-image-turbo-bf16-16gb") -> list[str]:
    """Problems from a deep self-test: an empty manifest, a missing file or a hash failure.

    Every row must pass its hash. Every required manifest id, and the generator
    for this pass, must be downloaded. Optional entries may stay absent.
    """
    if not wanted:
        return ["mdl.json lists no models"]
    required = set(wanted) | {generator}
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


def run_image(python: str, cap: float | None, index: int, prompt: str, dest: Path) -> dict[str, object]:
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


def summarise(rows: list[dict[str, object]], nvidia_peak_mib: int, gpu: str, tier: str,
              model: str = PASSES["16gb"][1]) -> dict[str, object]:
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
        "model": model,
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


def _timed(fn):
    started = time.perf_counter()
    value = fn()
    return value, round(time.perf_counter() - started, 4)


def _fine_gpu(raw, boxes) -> dict[str, object] | None:
    """Estimator on CUDA in FP16: seconds per call and peak bytes. None without CUDA."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    from canvas.infra.paths import models_dir
    from canvas.models.registry import load_catalog
    from canvas.models.snapshots import snapshot_ready
    from canvas.safety.adapters import FaceAgeRegressor

    (spec,) = load_catalog().by_role("age-fine")
    folder = snapshot_ready(spec, models_dir())
    if folder is None:
        return {"error": "not_downloaded"}
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    adapter = FaceAgeRegressor(folder, device="cuda")
    adapter.load()
    adapter.estimate(raw, boxes)
    torch.cuda.synchronize()
    started = time.perf_counter()
    adapter.estimate(raw, boxes)
    torch.cuda.synchronize()
    seconds = round(time.perf_counter() - started, 4)
    peak = int(torch.cuda.max_memory_allocated()) - int(before)
    del adapter
    torch.cuda.empty_cache()
    return {"s": seconds, "faces": len(boxes), "peak_bytes": peak}


def gate_bench(prompts: list[str], images: list[Path], cap: float | None) -> dict[str, object]:
    """Time the gate by stage on kept frames, with the age stage forced. Timings only."""
    from canvas.infra.gpu import probe_gpu
    from canvas.infra.paths import models_dir
    from canvas.infra.vram import peak_allocated_bytes, set_vram_cap_gb
    from canvas.models.loader import load_catalog_stack
    from canvas.safety.classifiers import Face
    from canvas.safety.constants import IMAGE_LABELS
    from canvas.safety.gate import AgeGate
    from canvas.safety.image import RawImage

    set_vram_cap_gb(cap)
    probe_gpu()  # applies the per-process memory fraction for a capped pass
    stack, load_seconds = _timed(lambda: load_catalog_stack(models_dir()))
    gate = AgeGate(stack)
    frames: list[dict[str, object]] = []
    for prompt, path in zip(prompts, images):
        with Image.open(path) as opened:
            rgb = opened.convert("RGB")
            width, height = rgb.size
            pixels = rgb.tobytes()
        raw = RawImage(width, height, pixels)
        _, prompt_s = _timed(lambda: gate.review_prompt(prompt))
        _, nsfw_s = _timed(lambda: stack.nsfw.sexual_score(raw))
        _, siglip_s = _timed(lambda: stack.zeroshot.score_image(raw, IMAGE_LABELS))
        faces, yunet_s = _timed(lambda: tuple(stack.faces.detect(raw)))
        synthetic = not faces
        boxes = faces or (Face(width // 4, height // 4, width // 2, height // 2),)
        _, fairface_s = _timed(lambda: stack.ages.estimate(raw, boxes))
        _, fine_s = _timed(lambda: stack.fine.estimate(raw, boxes))
        fine_gpu = _fine_gpu(raw, boxes)
        raw.wipe()
        frames.append({
            "image": path.name,
            "prompt_check_s": prompt_s,
            "nsfw_pass_s": nsfw_s,
            "age_stage_s": round(siglip_s + yunet_s + fairface_s + fine_s, 4),
            "age_stage": {"siglip_s": siglip_s, "yunet_s": yunet_s, "fairface_s": fairface_s,
                          "fine_s": fine_s, "fine_gpu": fine_gpu,
                          "faces": len(faces), "synthetic_face": synthetic},
        })

    def mean(key: str) -> float | None:
        values = [float(f[key]) for f in frames]  # type: ignore[arg-type]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "forced": True,
        "stack_load_s": load_seconds,
        "frames": frames,
        "mean": {key: mean(key) for key in ("prompt_check_s", "nsfw_pass_s", "age_stage_s")},
        "torch_max_memory_allocated_bytes": peak_allocated_bytes(),
    }


def _write(folder: Path, payload: dict[str, object], code: int, name: str = "result.json") -> int:
    payload["exit_code"] = code
    payload["ok"] = code == 0
    text = json.dumps(payload, indent=2)
    (folder / name).write_text(text + "\n", encoding="utf-8")
    print(text)
    return code


def main(argv: list[str] | None = None) -> int:
    """Run one pass of the smoke set and write ``result-<pass>.json``."""
    args = _parser().parse_args(argv)
    tier_pass = args.tier_pass
    generator, model = PASSES[tier_pass]
    cap = _cap_for(tier_pass, args.vram_cap_gb)
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"result-{tier_pass}.json"
    gpu = gpu_name()
    base: dict[str, object] = {"commit": _commit(), "canvas_version": _canvas_version(args.python),
                               "gpu": gpu, "pass": tier_pass, "vram_cap_gb": cap}
    write = lambda payload, code: _write(folder, payload, code, name)  # noqa: E731
    if not gpu:
        return write({**base, "error": "no GPU"}, 3)
    try:
        proc = subprocess.run(
            _canvas(args.python, cap, "--selftest", "--deep", "--json"),
            capture_output=True, text=True, timeout=7200, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return write({**base, "error": f"selftest: {exc}"}, 2)
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        report = None
    if not isinstance(report, dict):
        return write({**base, "error": "selftest did not print JSON", "stderr": proc.stderr[-800:]}, 2)
    if not gate_ready(report):
        return write({**base, "error": "content gate unavailable", "classifiers": report.get("classifiers")},
                     EXIT_GATE_UNAVAILABLE)
    broken = manifest_failures(report, manifest_ids(Path(args.manifest) if args.manifest else None), generator)
    if broken:
        return write({**base, "error": "a manifest entry failed its download or hash check",
                      "failed": broken}, EXIT_MANIFEST)
    gpu_report = report.get("gpu") if isinstance(report.get("gpu"), dict) else {}
    tier = str(gpu_report.get("tier") or "")  # type: ignore[union-attr]
    if tier != tier_pass:
        return write({**base, "error": f"expected the {tier_pass} tier, got {tier or 'none'}"}, 2)
    peak = [0]
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_vram, args=(stop, peak), daemon=True)
    sampler.start()
    rows: list[dict[str, object]] = []
    try:
        for index, prompt in enumerate(PROMPTS, start=1):
            dest = folder / f"smoke-{tier_pass}-{index}.png"
            rows.append(run_image(args.python, cap, index, prompt, dest))
            if rows[-1]["gate"] == "unavailable":
                break
        payload = {**summarise(rows, peak[0], gpu, tier, model), "canvas_version": base["canvas_version"],
                   "pass": tier_pass, "vram_cap_gb": cap}
        kept = [(str(r["prompt"]), folder / f"smoke-{tier_pass}-{r['index']}.png") for r in rows if r["gate"] == "pass"]
        if kept and not args.no_bench:
            try:
                payload["gate_timing"] = gate_bench([k[0] for k in kept], [k[1] for k in kept], cap)
            except Exception as exc:  # the benchmark is a measurement; a failure is reported, not hidden
                payload["gate_timing"] = {"error": type(exc).__name__}
                return write(payload, EXIT_GATE_UNAVAILABLE)
    finally:
        stop.set()
        sampler.join(timeout=3)
    payload["peak_vram"]["nvidia_smi_peak_mib"] = peak[0]
    if any(r["gate"] == "unavailable" for r in rows):
        return write(payload, EXIT_GATE_UNAVAILABLE)
    ok = len(rows) == len(PROMPTS) and all(r["gate"] == "pass" and not r["detail"] for r in rows)
    return write(payload, 0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
