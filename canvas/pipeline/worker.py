"""Worker-process boundary.

The child screens its copies again (prompt and any input images), then calls
the generator. It does not write images to disk. The parent mints the finished
picture through ``release_generated`` before anything is shown or saved; no
classifier runs on generated pixels.
"""
from __future__ import annotations

import base64
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from canvas.infra.paths import models_dir
from canvas.models.loader import ModelNotReady, ZImageGenerator, load_catalog_stack
from canvas.infra.gpu import probe_gpu
from canvas.models.plan import BaseNotSelected, OptionalModelUnavailable
from canvas.models.modes import Tier, select_tier
from canvas.models.registry import load_catalog
from canvas.pipeline.bf16_schedule import VramRefused
from canvas.pipeline.device_guard import DeviceRefused
from canvas.pipeline.engine import run_generator
from canvas.safety.gate import AgeGate, SafetyError, SafetyUnavailable, release_generated
from canvas.safety.image import RawImage, SafeImage
from canvas.safety.inputs import ImageRole, PendingInput


def worker_generate(gate: AgeGate, generator: object, prompt: str, mode: str, inputs: list[PendingInput]) -> RawImage:
    """Screen, then generate. The screened value is wiped before this returns."""
    screened = gate.release_request(prompt, mode, inputs)
    try:
        return run_generator(generator, screened)
    finally:
        screened.wipe()


def handle_message(payload: dict[str, Any], gate: AgeGate, generator: object) -> dict[str, Any]:
    """One JSON request. Errors become a neutral message and no image bytes."""
    inputs: list[PendingInput] = []
    try:
        for item in payload.get("images") or []:
            raw = RawImage(int(item["width"]), int(item["height"]), base64.b64decode(item["data_b64"]))
            inputs.append(PendingInput(ImageRole(str(item["role"])), raw))
        image = worker_generate(gate, generator, str(payload.get("prompt") or ""), str(payload.get("mode") or ""), inputs)
    except (SafetyError, ModelNotReady, BaseNotSelected, OptionalModelUnavailable, VramRefused, DeviceRefused) as exc:
        for item in inputs:
            item.image.wipe()
        return {"ok": False, "message": str(exc)}
    except (KeyError, TypeError, ValueError):
        for item in inputs:
            item.image.wipe()
        return {"ok": False, "message": SafetyUnavailable("classifier_error").user_message}
    width, height = image.width, image.height
    data = base64.b64encode(image.to_bytes()).decode("ascii")
    image.wipe()
    return {"ok": True, "width": width, "height": height, "data_b64": data}


def release_worker_image(
    gate: AgeGate, payload: dict[str, Any], prompt: str = "", *, lora_files: Sequence[Path] = (),
) -> SafeImage:
    """Parent side. Worker JSON is not a picture until this returns a SafeImage.

    A failed worker reply raises and nothing is kept. ``prompt`` and
    ``lora_files`` are accepted for callers and ignored: the request was
    screened before generation and no classifier runs on the output.
    """
    if not payload.get("ok"):
        raise SafetyUnavailable("worker")
    raw = RawImage(int(payload["width"]), int(payload["height"]), base64.b64decode(str(payload["data_b64"])))
    image = release_generated(raw)
    if type(image) is not SafeImage:
        raise TypeError("the checks must release the image")
    return image


def _worker_tier() -> Tier:
    """The tier for a worker process, including a VRAM cap set by the parent."""
    info = probe_gpu()
    if info is None:
        return Tier.GB24
    tier = select_tier(info.total_gb)
    if tier is None:
        return Tier.GB24
    return tier


def main() -> int:
    """Read one JSON job from stdin and write one JSON result to stdout."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        payload = {}
    try:
        stack = load_catalog_stack(models_dir())
        gate = AgeGate(stack)
        gate.warm_prompt()
        catalog = load_catalog()
        generator = ZImageGenerator(catalog, _worker_tier())
        result = handle_message(payload if isinstance(payload, dict) else {}, gate, generator)
    except SafetyUnavailable as exc:
        result = {"ok": False, "message": str(exc)}
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
