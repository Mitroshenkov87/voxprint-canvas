"""Headless batch render through the normal Canvas pipeline.

PG runs this on a GPU machine. It never downloads weights, never bypasses the
minors gate, and writes one picture plus a log (or a Forge photo folder in
dataset mode) for each prompt.

    python -m canvas.tools.render_batch --prompts prompts.json \\
        --lora <path> --lora-strength 1.0 --preset balanced \\
        --adult-content on --out DIR --seed 1001

    python -m canvas.tools.render_batch --mode dataset \\
        --persona persona.json --prompts dataset-prompts.json \\
        --preset balanced --adult-content on --out DIR --seed 1001
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from canvas.core.formats import OutputFormat, save_image
from canvas.models.modes import Mode, Tier
from canvas.models.plan import OptionalModelUnavailable
from canvas.models.registry import load_catalog
from canvas.models.user_lora import (
    FAMILY_TURBO,
    ZIMAGE_LORA_DEFAULT_STRENGTH,
    LoraRejected,
    prepare_cli_lora,
    retrain_rejection,
    route_lora,
)
from canvas.safety.image import SafeImage
from canvas.safety.gate import AgeGate, SafetyError

PRESETS = ("fast", "balanced", "best")
SCHEMA = "voxprint-canvas-render-batch/1"
# Forge ``photos.photo_id``: first 12 hex characters of the file SHA-256.
_PHOTO_ID_LEN = 12


@dataclass
class Hooks:
    """Test seams. Production leaves every field at None."""

    weights_ready: Callable[[], bool] | None = None
    generate: Callable[..., SafeImage] | None = None
    log: TextIO = field(default_factory=lambda: sys.stderr)


@dataclass
class PromptItem:
    """One prompt from a manifest file."""

    prompt: str
    item_id: str = ""
    seed: int | None = None
    caption: str = ""
    short: str = ""
    explicit: bool | None = None


def preset_values(name: str) -> tuple[int, float, str]:
    """Steps, guidance, and sampler from common's shared Canvas quality presets."""
    from voxprint_common.ui.settings import reference_settings
    from voxprint_common.ui.variants import surface_for, variant_presets

    if name not in PRESETS:
        raise ValueError(f"preset must be one of {', '.join(PRESETS)}")
    document = variant_presets(surface_for("canvas"), reference_settings())
    for group in document.groups:
        if group.group_id != "quality":
            continue
        for option in group.options:
            if option.option_id == name:
                values = dict(option.values)
                return (
                    int(values["shell.steps"]),
                    float(values["shell.guidance"]),
                    str(values.get("shell.sampler") or "euler"),
                )
    raise ValueError(f"common has no quality preset {name!r}")


def photo_id(data: bytes) -> str:
    """Forge's photo id: ``sha256[:12]`` of the file bytes."""
    return hashlib.sha256(data).hexdigest()[:_PHOTO_ID_LEN]


def load_json(path: Path) -> Any:
    """Read a UTF-8 JSON file.

    Raises:
        ValueError: The file is missing or not JSON.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"The file is missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"The file is not valid JSON: {path}") from exc


def load_prompts(path: Path) -> list[PromptItem]:
    """Read a prompt list (a JSON array or an object with a ``prompts`` array)."""
    raw = load_json(path)
    rows = raw.get("prompts") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"The prompt file has no prompts: {path}")
    items: list[PromptItem] = []
    for index, row in enumerate(rows, start=1):
        if isinstance(row, str):
            text = row.strip()
            if not text:
                raise ValueError(f"Prompt {index} is empty.")
            items.append(PromptItem(prompt=text, item_id=f"{index:02d}"))
            continue
        if not isinstance(row, dict):
            raise ValueError(f"Prompt {index} must be a string or an object.")
        text = str(row.get("prompt") or "").strip()
        if not text:
            raise ValueError(f"Prompt {index} has no prompt text.")
        seed = row.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError(f"Prompt {index} has a bad seed.")
        explicit = row.get("explicit")
        if explicit is not None and not isinstance(explicit, bool):
            raise ValueError(f"Prompt {index} has a bad explicit flag.")
        items.append(
            PromptItem(
                prompt=text,
                item_id=str(row.get("id") or f"{index:02d}"),
                seed=seed,
                caption=str(row.get("caption") or "").strip(),
                short=str(row.get("short") or "").strip(),
                explicit=explicit,
            )
        )
    return items


def load_persona(path: Path) -> dict[str, Any]:
    """Read a persona description. The identity text is required."""
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"The persona file must be an object: {path}")
    description = str(raw.get("description") or "").strip()
    if not description:
        raise ValueError(f"The persona file has no description: {path}")
    return raw


def compose_prompt(item: PromptItem, persona: dict[str, Any] | None) -> str:
    """Put the fixed persona identity in front of a dataset shot when needed."""
    if persona is None:
        return item.prompt
    identity = str(persona.get("description") or "").strip()
    if not identity or identity in item.prompt:
        return item.prompt
    return f"{identity} {item.prompt}"


def apply_preset_steps(steps: int) -> None:
    """Make the process use this step count (canvas.toml ``steps``)."""
    from canvas.infra import settings as canvas_settings

    canvas_settings._cached = canvas_settings.Settings(steps=int(steps))


def turbo_weights_ready(tier: Tier) -> bool:
    """True when the default Z-Image-Turbo snapshot for ``tier`` is on disk."""
    from canvas.infra.paths import models_dir
    from canvas.models.snapshots import snapshot_ready

    catalog = load_catalog()
    model_id = "z-image-turbo-bf16-16gb" if tier is Tier.GB16 else "z-image-turbo-bf16"
    try:
        spec = catalog.by_id(model_id)
    except Exception:
        return False
    return snapshot_ready(spec, models_dir()) is not None


def bind_lora(path: Path, strength: float) -> tuple[Any, float]:
    """Open ``path`` and accept it only when it may load on Z-Image-Turbo."""
    lora = prepare_cli_lora(path)
    catalog = load_catalog()
    rejected = retrain_rejection(lora, catalog.text_bases, FAMILY_TURBO)
    if rejected is not None:
        raise rejected
    route = route_lora(lora, catalog.text_bases, FAMILY_TURBO)
    if not route.base:
        raise LoraRejected("lora.retrain", base=route.retrain_from)
    return lora, strength


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a prompt list through the Canvas pipeline (minors gate on).",
    )
    parser.add_argument("--mode", choices=("render", "dataset"), default="render")
    parser.add_argument("--prompts", required=True, type=Path, help="JSON prompt list")
    parser.add_argument("--persona", type=Path, default=None, help="persona.json (dataset mode)")
    parser.add_argument("--lora", type=Path, default=None, help="LoRA file, package, or folder")
    parser.add_argument(
        "--lora-strength",
        type=float,
        default=ZIMAGE_LORA_DEFAULT_STRENGTH,
        help="LoRA strength (default 1.0; no auto-scaling)",
    )
    parser.add_argument("--preset", choices=PRESETS, default="balanced")
    parser.add_argument("--adult-content", choices=("on", "off"), required=True)
    parser.add_argument("--out", required=True, type=Path, help="output folder")
    parser.add_argument("--seed", type=int, default=1001, help="base seed; dataset mode adds the image index")
    parser.add_argument("--aspect", default="1:1")
    return parser


def main(argv: Sequence[str] | None = None, hooks: Hooks | None = None) -> int:
    """CLI entry. Returns a process exit code. 2 means a missing input or weights."""
    args = _parser().parse_args(argv)
    hooks = hooks or Hooks()
    log = hooks.log
    try:
        items = load_prompts(args.prompts)
        persona = load_persona(args.persona) if args.persona is not None else None
        steps, guidance, sampler = preset_values(args.preset)
    except ValueError as exc:
        print(str(exc), file=log)
        return 2
    if args.mode == "dataset" and args.lora is not None:
        print("Dataset mode does not take a LoRA (the set is trained afterwards).", file=log)
        return 2
    if args.mode == "dataset" and args.persona is None:
        print("Dataset mode needs --persona persona.json.", file=log)
        return 2
    if args.lora is not None and not args.lora.exists():
        print(f"The LoRA file is missing: {args.lora}", file=log)
        return 2

    adult = args.adult_content == "on"
    out = args.out
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"The output folder could not be created: {out} ({exc})", file=log)
        return 2

    if hooks.weights_ready is not None:
        ready = hooks.weights_ready()
    else:
        ready = _live_ready(log)
    if not ready:
        print(
            "The Z-Image-Turbo weights are missing. Install them with VOXPrint MDL, then run this command again.",
            file=log,
        )
        return 2

    lora_bind: tuple[Any, float] | None = None
    if args.lora is not None:
        try:
            lora_bind = bind_lora(args.lora, float(args.lora_strength))
        except FileNotFoundError:
            print(f"The LoRA file is missing: {args.lora}", file=log)
            return 2
        except LoraRejected as exc:
            from canvas.app.i18n import text

            print(text(exc.code, **exc.details) if exc.details else str(exc), file=log)
            return 2
        except (OSError, ValueError) as exc:
            print(f"The LoRA could not be opened: {exc}", file=log)
            return 2

    apply_preset_steps(steps)
    if args.mode == "dataset":
        return _run_dataset(items, persona, adult=adult, out=out, base_seed=int(args.seed),
                            steps=steps, guidance=guidance, sampler=sampler, aspect=args.aspect,
                            hooks=hooks, log=log)
    return _run_render(items, adult=adult, out=out, base_seed=int(args.seed), steps=steps,
                       guidance=guidance, sampler=sampler, aspect=args.aspect, lora_bind=lora_bind,
                       hooks=hooks, log=log)


def _live_ready(log: TextIO) -> bool:
    from canvas.app.platform import check_gpu
    from canvas.infra.gpu import probe_gpu

    gpu = check_gpu(probe_gpu())
    if not gpu.ok or gpu.tier is None:
        print(gpu.message or "No supported GPU is available.", file=log)
        return False
    return turbo_weights_ready(gpu.tier)


def _run_render(
    items: list[PromptItem],
    *,
    adult: bool,
    out: Path,
    base_seed: int,
    steps: int,
    guidance: float,
    sampler: str,
    aspect: str,
    lora_bind: tuple[Any, float] | None,
    hooks: Hooks,
    log: TextIO,
) -> int:
    session = _Session(hooks, adult=adult, aspect=aspect, lora_bind=lora_bind)
    written = 0
    failed = 0
    for index, item in enumerate(items):
        seed = item.seed if item.seed is not None else base_seed + index
        stem = _safe_stem(item.item_id or f"{index + 1:02d}")
        image_path = out / f"{stem}.webp"
        log_path = out / f"{stem}.json"
        started = time.perf_counter()
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "id": item.item_id,
            "prompt": item.prompt,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "sampler": sampler,
            "base": FAMILY_TURBO,
            "adult_content": adult,
            "lora_strength": None if lora_bind is None else lora_bind[1],
            "gate": "pending",
        }
        try:
            image, gate = session.generate(item.prompt, seed)
            record["gate"] = gate
            if image is not None:
                save_image(image, image_path, OutputFormat.WEBP, metadata=None, purpose="picture")
                record["file"] = image_path.name
                written += 1
        except Exception as exc:  # noqa: BLE001 — batch log, then continue
            record["gate"] = record["gate"] if record["gate"] != "pending" else "error"
            record["error"] = str(exc)
            failed += 1
            print(f"{stem}: {exc}", file=log)
        record["seconds"] = round(time.perf_counter() - started, 3)
        log_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if written == 0:
        print("No pictures were written.", file=log)
        return 1 if failed or items else 2
    return 0 if failed == 0 else 1


def _run_dataset(
    items: list[PromptItem],
    persona: dict[str, Any] | None,
    *,
    adult: bool,
    out: Path,
    base_seed: int,
    steps: int,
    guidance: float,
    sampler: str,
    aspect: str,
    hooks: Hooks,
    log: TextIO,
) -> int:
    # Forge ``lora prepare --photos DIR`` reads only image files in a flat folder
    # (``.jpg``, ``.jpeg``, ``.png``, ``.webp``, ``.avif``). Captions go in a
    # JSONL of ``id``, ``long``, ``short`` (id = sha256[:12] of the file bytes)
    # and are applied later with ``--captions FILE``. Forge prepends its trigger.
    photos = out / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    session = _Session(hooks, adult=adult, aspect=aspect, lora_bind=None)
    manifest_items: list[dict[str, Any]] = []
    caption_rows: list[dict[str, str]] = []
    written = 0
    failed = 0
    for index, item in enumerate(items):
        seed = item.seed if item.seed is not None else base_seed + index
        prompt = compose_prompt(item, persona)
        stem = f"{index + 1:02d}"
        image_path = photos / f"{stem}.webp"
        started = time.perf_counter()
        row: dict[str, Any] = {
            "id": "",
            "file": f"photos/{stem}.webp",
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "gate": "pending",
        }
        try:
            image, gate = session.generate(prompt, seed)
            row["gate"] = gate
            if image is not None:
                save_image(image, image_path, OutputFormat.WEBP, metadata=None, purpose="master")
                digest = photo_id(image_path.read_bytes())
                long = item.caption or prompt
                short = item.short or long
                row["id"] = digest
                caption_rows.append({"id": digest, "long": long, "short": short})
                written += 1
        except Exception as exc:  # noqa: BLE001 — batch log, then continue
            row["gate"] = row["gate"] if row["gate"] != "pending" else "error"
            row["error"] = str(exc)
            failed += 1
            print(f"{stem}: {exc}", file=log)
        row["seconds"] = round(time.perf_counter() - started, 3)
        manifest_items.append(row)
    (out / "captions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in caption_rows),
        encoding="utf-8",
    )
    manifest = {
        "schema": SCHEMA,
        "mode": "dataset",
        "base": FAMILY_TURBO,
        "preset_steps": steps,
        "guidance": guidance,
        "sampler": sampler,
        "adult_content": adult,
        "lora": None,
        "persona": persona,
        "photos_dir": "photos",
        "captions": "captions.jsonl",
        "items": manifest_items,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if written == 0:
        print("No pictures were written.", file=log)
        return 1 if failed or items else 2
    return 0 if failed == 0 else 1


class _Session:
    """One AgeGate + generator for a batch. Tests inject ``hooks.generate``."""

    def __init__(
        self,
        hooks: Hooks,
        *,
        adult: bool,
        aspect: str,
        lora_bind: tuple[Any, float] | None,
    ) -> None:
        self.hooks = hooks
        self.adult = adult
        self.aspect = aspect
        self.lora_bind = lora_bind
        self._pipeline = None
        self._gate: AgeGate | None = None

    def generate(self, prompt: str, seed: int) -> tuple[SafeImage | None, str]:
        if self.hooks.generate is not None:
            image = self.hooks.generate(prompt, seed, explicit=self.adult)
            return image, "passed"
        return self._live(prompt, seed)

    def _live(self, prompt: str, seed: int) -> tuple[SafeImage | None, str]:
        from canvas.app.platform import check_gpu
        from canvas.infra.gpu import probe_gpu
        from canvas.infra.paths import models_dir
        from canvas.models.loader import ZImageGenerator, load_catalog_stack
        from canvas.models.plan import BaseNotSelected, ModeUnavailable
        from canvas.pipeline.pipeline import Pipeline
        from canvas.infra.enlarge import EnlargeClient

        if self._pipeline is None:
            gpu = check_gpu(probe_gpu())
            if not gpu.ok or gpu.tier is None:
                raise RuntimeError(gpu.message or "No supported GPU is available.")
            catalog = load_catalog()
            root = models_dir()
            stack = load_catalog_stack(root, catalog)
            gate = AgeGate(stack)
            gate.warm()
            generator = ZImageGenerator(catalog, gpu.tier)
            generator.models = root
            if self.lora_bind is not None:
                lora, strength = self.lora_bind
                generator.lora_path = str(lora.weights)
                generator.lora_strength = strength
                generator.bound_lora = lora
            self._gate = gate
            self._pipeline = Pipeline(gate, generator, gpu.tier, use_gpu_lock=True, catalog=catalog, enlarge=EnlargeClient())
        assert self._gate is not None
        try:
            self._pipeline.generator.seed = seed
            image = self._pipeline.generate(prompt, Mode.TEXT, (), aspect=self.aspect, output="native", explicit=self.adult)
        except SafetyError as exc:
            return None, getattr(exc, "code", None) or str(exc)
        except (ModeUnavailable, BaseNotSelected, OptionalModelUnavailable) as exc:
            raise RuntimeError(str(exc)) from exc
        return image, "passed"


def _safe_stem(item_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in item_id.strip())
    return cleaned.strip("-_") or "frame"


if __name__ == "__main__":
    sys.exit(main())
