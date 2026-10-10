"""Compare text-to-image bases on one GPU.

The same prompts and seeds run for every base. The script records seconds and
peak VRAM, and it writes an image only after the content checks release it.
Images stay in the operating-system temp directory. Run it later on an RTX 40
series card; this environment does not need a GPU to import the script.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from canvas.infra.paths import models_dir
from canvas.models.loader import ModelNotReady, ZImageGenerator, load_catalog_stack
from canvas.models.modes import Mode, Tier
from canvas.models.registry import Catalog, load_catalog
from canvas.safety.gate import AgeGate, GenerationBlocked, SafetyError, SafetyUnavailable
from canvas.safety.image import RawImage, SafeImage, ScreenedInput

PROMPTS = (
    "a red boat on calm water",
    "a white cup on a wooden table",
    "a mountain ridge under a clear sky",
    "a rainy street with yellow street lamps",
)
SEEDS = (7, 21, 42)

Runner = Callable[[str, str, int, ScreenedInput], RawImage]
Probe = Callable[[], int | None]


@dataclass(frozen=True)
class BakeoffRow:
    base: str
    prompt: str
    seed: int
    seconds: float
    peak_vram_mb: int | None
    status: str
    path: str
    width: int
    height: int


def sample_vram_mb() -> int | None:
    """Read current GPU memory from nvidia-smi. Missing output means unknown."""
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=flags,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    token = completed.stdout.strip().splitlines()[0].split(",")[0].strip()
    try:
        return int(token)
    except ValueError:
        return None


def _temp_dir(out_dir: Path) -> Path:
    root = Path(tempfile.gettempdir()).resolve()
    target = out_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    inside_temp = target == root or root in target.parents
    inside_repo = target == repo or repo in target.parents
    if not inside_temp or inside_repo:
        raise ValueError("bake-off images stay in the OS temp directory")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _peak(sample: Probe, work: Callable[[], RawImage]) -> tuple[RawImage, int | None, float]:
    peak: int | None = None

    def consider() -> None:
        nonlocal peak
        value = sample()
        if value is None:
            return
        if peak is None or value > peak:
            peak = value

    consider()
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(0.05):
            consider()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        image = work()
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        thread.join(timeout=1)
        consider()
    return image, peak, elapsed


def _generator_runner(catalog: Catalog, tier: Tier) -> Runner:
    def run(base: str, prompt: str, seed: int, screened: ScreenedInput) -> RawImage:
        del prompt, seed
        selected = replace(catalog, default_text_base=base)
        return ZImageGenerator(selected, tier).generate(screened)

    return run


def run_bakeoff(
    catalog: Catalog,
    gate: AgeGate,
    tier: Tier,
    *,
    runner: Runner,
    out_dir: Path,
    prompts: tuple[str, ...] = PROMPTS,
    seeds: tuple[int, ...] = SEEDS,
    vram: Probe | None = None,
) -> tuple[BakeoffRow, ...]:
    """Run every base on the same prompt and seed pairs. Refusals write nothing."""
    folder = _temp_dir(out_dir)
    sample = vram or sample_vram_mb
    rows: list[BakeoffRow] = []
    index = 0
    for base in catalog.text_bases:
        for prompt in prompts:
            for seed in seeds:
                index += 1
                rows.append(_one(folder, gate, runner, sample, base, prompt, seed, index))
    report = [
        {
            "base": row.base,
            "prompt": row.prompt,
            "seed": row.seed,
            "seconds": row.seconds,
            "peak_vram_mb": row.peak_vram_mb,
            "status": row.status,
            "path": row.path,
            "width": row.width,
            "height": row.height,
        }
        for row in rows
    ]
    (folder / "bakeoff.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return tuple(rows)


def _one(
    folder: Path,
    gate: AgeGate,
    runner: Runner,
    sample: Probe,
    base: str,
    prompt: str,
    seed: int,
    index: int,
) -> BakeoffRow:
    empty = BakeoffRow(base, prompt, seed, 0.0, None, "unavailable", "", 0, 0)
    try:
        screened = gate.release_request(prompt, Mode.TEXT.value, ())
    except GenerationBlocked:
        return BakeoffRow(base, prompt, seed, 0.0, None, "refused", "", 0, 0)
    except SafetyError:
        return empty
    try:
        image, peak, elapsed = _peak(sample, lambda: runner(base, prompt, seed, screened))
    except ModelNotReady:
        screened.wipe()
        return BakeoffRow(base, prompt, seed, 0.0, None, "not_ready", "", 0, 0)
    except GenerationBlocked:
        screened.wipe()
        return BakeoffRow(base, prompt, seed, 0.0, None, "refused", "", 0, 0)
    except SafetyError:
        screened.wipe()
        return empty
    try:
        released = gate.review_image(image)
    except GenerationBlocked:
        screened.wipe()
        return BakeoffRow(base, prompt, seed, elapsed, peak, "refused", "", 0, 0)
    except SafetyError:
        screened.wipe()
        return BakeoffRow(base, prompt, seed, elapsed, peak, "unavailable", "", 0, 0)
    if type(released) is not SafeImage:
        screened.wipe()
        raise TypeError("the checks must release the image")
    path = folder / f"{index:02d}-{base}-seed{seed}.bin"
    path.write_bytes(released.data)
    screened.wipe()
    return BakeoffRow(base, prompt, seed, elapsed, peak, "kept", str(path), released.width, released.height)


def _tier(value: str) -> Tier:
    if value == Tier.GB16.value:
        return Tier.GB16
    if value == Tier.GB24.value:
        return Tier.GB24
    raise ValueError(value)


def main(argv: list[str] | None = None) -> int:
    """Load the checks, then compare bases. Exit 2 when the checks cannot load."""
    parser = argparse.ArgumentParser(description="Compare text-to-image bases")
    parser.add_argument("--tier", choices=(Tier.GB16.value, Tier.GB24.value), default=Tier.GB24.value)
    args = parser.parse_args(argv)
    try:
        stack = load_catalog_stack(models_dir())
    except SafetyUnavailable:
        print("Image generation is unavailable.")
        return 2
    catalog = load_catalog()
    tier = _tier(args.tier)
    folder = Path(tempfile.gettempdir()) / "voxprint-bakeoff"
    rows = run_bakeoff(catalog, AgeGate(stack), tier, runner=_generator_runner(catalog, tier), out_dir=folder)
    for row in rows:
        print(f"{row.base}\t{row.seed}\t{row.status}\t{row.seconds:.3f}\t{row.peak_vram_mb}")
    print(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
