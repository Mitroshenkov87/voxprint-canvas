"""Wall time of a mocked Photoshoot series: the old check order against the overlap.

No weights and no GPU. Render, decode and the output check are sleeps of the
given lengths, so the numbers show the schedule only. The old order checks
every frame after the pipeline leaves the GPU. The new order checks a frame
on a CPU worker while later frames render or decode, then checks the rest.

    python tools/bench_series_check.py [--frames 5] [--render-s 1.0] [--decode-s 0.2]
        [--check-gpu-s 0.3] [--check-cpu-s 0.8] [--cook]

``--cook`` samples every frame in the render phase and gives it pixels in
the decode phase, as the app does. Without it every frame is drawn whole.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from canvas.pipeline import series
from canvas.safety.classifiers import Face, GuardVerdict
from canvas.safety.constants import IMAGE_LABELS, PROMPT_LABELS
from canvas.safety.gate import AgeGate, SafetyStack
from canvas.safety.image import RawImage, SafeImage

IDEA = "Yellow ceramic teapot with a blue handle"
_SCENES = ("close view", "three-quarter view", "from above", "from the side", "by the window",
           "among cups", "on a tray", "beside a spoon", "under a lamp", "near the sink")


class _Guard:
    """Prompt check double that always passes."""

    def assess(self, text: str) -> GuardVerdict:
        """Safe verdict."""
        del text
        return GuardVerdict("safe", ())


class _Zero:
    """Zero-shot double with every score at zero."""

    def score_text(self, text: str, labels: object) -> dict[str, float]:
        """Zero for each prompt label."""
        del text, labels
        return {label: 0.0 for label in PROMPT_LABELS}

    def score_image(self, image: RawImage, labels: object) -> dict[str, float]:
        """Zero for each image label."""
        del image, labels
        return {label: 0.0 for label in IMAGE_LABELS}


class _Faces:
    """Finds no faces."""

    def detect(self, image: RawImage) -> list[Face]:
        """No faces."""
        del image
        return []


class _Estimates:
    """Age estimators for a picture without faces."""

    def estimate(self, image: RawImage, faces: object) -> list[float]:
        """No estimates."""
        del image, faces
        return []


class _Nsfw:
    """Scores every picture zero."""

    def sexual_score(self, image: RawImage) -> float:
        """Zero."""
        del image
        return 0.0


class _Frame:
    """A 2x2 RGB frame."""

    width = 2
    height = 2

    def __init__(self, index: int) -> None:
        self._data = bytes((index % 250, 7, 7)) * 4

    def to_bytes(self) -> bytes:
        """RGB bytes."""
        return self._data


def _scenario(count: int) -> str:
    shots = [
        {"scene": _SCENES[index % len(_SCENES)], "pose": "still", "camera": "50 mm", "place": "kitchen table"}
        for index in range(count)
    ]
    return json.dumps({
        "subject_block": IDEA, "descriptors": ["Yellow ceramic teapot", "blue handle"],
        "place": "kitchen table", "shots": shots,
    })


def _gate(check_gpu_s: float, check_cpu_s: float) -> AgeGate:
    """A gate whose image check takes the CPU length on the worker and the GPU length in the batch."""
    gate = AgeGate(SafetyStack(_Guard(), _Zero(), _Faces(), _Estimates(), _Nsfw(), _Estimates()))  # type: ignore[arg-type]
    real = gate.review_image

    def review_image(raw: RawImage, prompt: str = "") -> SafeImage:
        on_worker = threading.current_thread().name == "check-overlap"
        time.sleep(check_cpu_s if on_worker else check_gpu_s)
        return real(raw, prompt)

    gate.review_image = review_image  # type: ignore[method-assign]
    return gate


def _drawer(render_s: float, decode_s: float, cook: bool) -> object:
    """``draw_frame`` with an optional ``sample`` hook, as the window builds it."""
    sampled: set[int] = set()

    def draw(line: str, seed: int, hero: object | None, strength: float | None) -> _Frame:
        del line, hero, strength
        if seed in sampled:
            time.sleep(decode_s)
        else:
            time.sleep(render_s + decode_s)
        return _Frame(seed)

    def sample(line: str, seed: int, hero: object | None, strength: float | None) -> bool:
        del line, hero, strength
        time.sleep(render_s)
        sampled.add(seed)
        return True

    if cook:
        draw.sample = sample  # type: ignore[attr-defined]
    return draw


def run_once(*, overlap: bool, frames: int, render_s: float, decode_s: float,
             check_gpu_s: float, check_cpu_s: float, cook: bool) -> tuple[float, int]:
    """One mocked series. Returns the wall seconds and the number of frames kept."""
    previous = series._OVERLAP_CHECK
    series._OVERLAP_CHECK = overlap
    try:
        with tempfile.TemporaryDirectory() as folder:
            started = time.perf_counter()
            result = series.run_series(
                idea=IDEA, count=frames, style="photo", place_mode="same", seed=0, strength=None,
                include_camera=lambda: True, rewrite=lambda _prompt: _scenario(frames),
                age_gate=_gate(check_gpu_s, check_cpu_s),
                draw_frame=_drawer(render_s, decode_s, cook),  # type: ignore[arg-type]
                cancelled=lambda: False, folder=Path(folder) / "series",
            )
            wall = time.perf_counter() - started
    finally:
        series._OVERLAP_CHECK = previous
    return wall, len(result.kept)


def main(argv: list[str] | None = None) -> int:
    """Print the wall time of both orders."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--render-s", type=float, default=1.0)
    parser.add_argument("--decode-s", type=float, default=0.2)
    parser.add_argument("--check-gpu-s", type=float, default=0.3, help="one frame's check after the pipeline left")
    parser.add_argument("--check-cpu-s", type=float, default=0.8, help="one frame's check on the CPU worker")
    parser.add_argument("--cook", action="store_true", help="sample in render, pixels in decode")
    args = parser.parse_args(argv)
    # Bench log lines stay out of the user's log.
    os.environ["VOXPRINT_HOME"] = tempfile.mkdtemp(prefix="vx-bench-")
    options = dict(frames=args.frames, render_s=args.render_s, decode_s=args.decode_s,
                   check_gpu_s=args.check_gpu_s, check_cpu_s=args.check_cpu_s, cook=args.cook)
    old, old_kept = run_once(overlap=False, **options)
    new, new_kept = run_once(overlap=True, **options)
    mode = "cook" if args.cook else "whole"
    print(f"mode={mode} frames={args.frames} render_s={args.render_s} decode_s={args.decode_s} "
          f"check_gpu_s={args.check_gpu_s} check_cpu_s={args.check_cpu_s}")
    print(f"sequential  wall_s={old:.2f} kept={old_kept}")
    print(f"overlap     wall_s={new:.2f} kept={new_kept}")
    print(f"saved_s={old - new:.2f} speedup={old / new:.2f}x")
    return 0 if old_kept == new_kept == args.frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
