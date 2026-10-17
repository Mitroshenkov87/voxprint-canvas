"""Cheap checks and budget sizing for a bake-off question. No GPU imports."""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

from bakeoff.prompt_lint import lint_text, stated_ages

SCHEMA = "voxprint-canvas-bakeoff/1"
CHECKS = ("prompt-lint", "dry-run", "schema")
PROMPT_ORDER = (
    "general", "text_en", "text_ru", "adult", "fp_young_adult",
    "scene", "hands", "adherence", "portrait",
)
SHOWCASE_COUNTS = {
    "text_en": 3,
    "text_ru": 3,
    "scene": 3,
    "hands": 2,
    "adherence": 2,
    "portrait": 2,
}
QUESTION_ARMS = {
    "showcase": ("z_bf16", "q_bf16"),
    "metrics": ("z_bf16", "q_bf16", "q_tuned", "z_fp8", "q_fp8"),
    "gate-measurement": ("z_bf16", "q_bf16", "z_fp8", "q_fp8", "q_tuned"),
}
ARM_FAMILY = {
    "z_bf16": ("z", "bf16"),
    "z_fp8": ("z", "fp8-forge"),
    "q_bf16": ("q", "bf16"),
    "q_tuned": ("q", "bf16"),
    "q_fp8": ("q", "fp8-forge"),
}
# Planning estimates only. Z-Image at 1024 square, Qwen at 1328 square, then scaled by pixels.
Z_BASE_S = 3.0
Z_BASE_PX = 1024 * 1024
Q_BASE_S = 50.0
Q_BASE_PX = 1328 * 1328
FAMILY_LOAD_S = {"z": 180.0, "q": 240.0}
QUANT_EXTRA_S = {"bf16": 0.0, "fp8-forge": 60.0}
_EXPLICIT = {"nude", "naked", "topless", "lingerie", "erotic", "erotica", "boudoir", "nudity", "nsfw", "porn"}
_SUMMARY_KEYS = (
    "schema", "generated_utc", "models", "scorers", "gate_rule", "prompt_set", "question",
    "budget_min", "prompt_sets", "subset_n", "arms", "pairs", "tiers", "notes",
)
_CATALOG = Path(__file__).resolve().parent.parent / "canvas" / "models" / "catalog.json"


def question_name(prompt_set: str) -> str:
    """Which question a prompt set answers."""
    if prompt_set == "showcase":
        return "showcase"
    if prompt_set == "fp_young_adults":
        return "gate-measurement"
    if prompt_set in ("v1", "v2"):
        return "metrics"
    raise ValueError(prompt_set)


def native_frames() -> dict[str, dict[str, tuple[int, int]]]:
    """Registry native frames. ``z`` and ``q`` differ."""
    raw = json.loads(_CATALOG.read_text(encoding="utf-8"))
    sizes = raw["sizes"]
    frames: dict[str, dict[str, tuple[int, int]]] = {}
    for family, key in (("z", "z-image-turbo"), ("q", "qwen-image-2512")):
        frames[family] = {aspect: (int(pair[0]), int(pair[1])) for aspect, pair in sizes[key]["native"].items()}
    return frames


def native_frame(family: str, aspect: str) -> tuple[int, int]:
    """The model's native frame for one aspect."""
    table = native_frames()[family]
    if aspect not in table:
        raise ValueError(aspect)
    return table[aspect]


def image_seconds(family: str, width: int, height: int) -> float:
    """Estimated seconds for one image at this frame."""
    pixels = width * height
    if family == "z":
        return Z_BASE_S * pixels / Z_BASE_PX
    if family == "q":
        return Q_BASE_S * pixels / Q_BASE_PX
    raise ValueError(family)


def render_size(family: str, prompt: dict[str, object], default: tuple[int, int]) -> tuple[int, int]:
    """Native aspect frame for a showcase prompt, otherwise the arm's square size."""
    aspect = prompt.get("aspect")
    if not aspect:
        return default
    return native_frame(family, str(aspect))


def image_name(prompt: dict[str, object], seed: int, width: int, height: int) -> str:
    """File name for one image. Showcase names include the full frame size."""
    if prompt.get("aspect"):
        return f"{prompt['id']}_s{seed}_{width}x{height}.webp"
    return f"{prompt['id']}_s{seed}.webp"


def preflight_errors(rows: list[dict[str, object]], prompt_set: str) -> list[str]:
    """Prompt-lint and showcase schema errors. Empty means the rows may run."""
    errors: list[str] = []
    seen: set[str] = set()
    frames = native_frames()["z"]
    for row in rows:
        ident = str(row.get("id") or "?")
        if not row.get("id") or not row.get("prompt") or not row.get("category"):
            errors.append(f"{ident}: missing id, prompt, or category")
            continue
        if ident in seen:
            errors.append(f"{ident}: duplicate id")
        seen.add(ident)
        text = str(row["prompt"])
        if row.get("target"):
            text = f"{text}\n{row['target']}"
        errors.extend(f"{ident}: {reason}" for reason in lint_text(text))
        if prompt_set == "showcase":
            errors.extend(_showcase_row_errors(ident, row, frames))
    if prompt_set == "showcase":
        counts: dict[str, int] = {}
        for row in rows:
            counts[str(row.get("category"))] = counts.get(str(row.get("category")), 0) + 1
        if counts != SHOWCASE_COUNTS:
            errors.append(f"showcase categories {counts}")
    return errors


def _showcase_row_errors(ident: str, row: dict[str, object], frames: dict[str, tuple[int, int]]) -> list[str]:
    errors: list[str] = []
    if row.get("category") == "adult" or row.get("content_level") == "explicit":
        errors.append(f"{ident}: showcase has no explicit prompts")
    aspect = str(row.get("aspect") or "")
    if aspect not in frames:
        errors.append(f"{ident}: missing native aspect")
    hits = _explicit_tokens(str(row.get("prompt") or ""))
    if hits:
        errors.append(f"{ident}: explicit term {hits[0]}")
    ages = stated_ages(str(row.get("prompt") or ""))
    if row.get("category") == "portrait":
        listed = row.get("ages")
        if listed != ages or not ages or any(age < 25 or age > 30 for age in ages):
            errors.append(f"{ident}: portrait age must be 25-30")
    elif any(age < 25 or age > 30 for age in ages):
        errors.append(f"{ident}: showcase person age must be 25-30")
    return errors


def _explicit_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    return [token for token in tokens if token in _EXPLICIT]


def _load_cost(family: str, quant: str, loaded_family: set[str], loaded_quant: dict[str, str]) -> float:
    cost = 0.0
    if family not in loaded_family:
        cost += FAMILY_LOAD_S[family]
    if loaded_quant.get(family) != quant:
        cost += QUANT_EXTRA_S[quant]
    return cost


def _commit_load(family: str, quant: str, loaded_family: set[str], loaded_quant: dict[str, str]) -> None:
    loaded_family.add(family)
    loaded_quant[family] = quant


def _take(prompts: list[dict[str, object]], cost, seconds: float) -> tuple[list[dict[str, object]], float]:
    buckets: dict[str, list[dict[str, object]]] = {}
    order: list[str] = []
    for prompt in prompts:
        category = str(prompt["category"])
        if category not in buckets:
            order.append(category)
            buckets[category] = []
        buckets[category].append(prompt)
    picked: list[dict[str, object]] = []
    used = 0.0
    progressed = True
    while progressed:
        progressed = False
        for category in order:
            queue = buckets[category]
            if not queue:
                continue
            nxt = queue[0]
            price = float(cost(nxt))
            if used + price <= seconds + 1e-9:
                queue.pop(0)
                picked.append(nxt)
                used += price
                progressed = True
    chosen = {str(prompt["id"]) for prompt in picked}
    return [prompt for prompt in prompts if str(prompt["id"]) in chosen], used


def size_run(
    prompt_set: str,
    prompts: list[dict[str, object]],
    selected: list[str],
    budget_min: float,
    seeds: int,
    reserve_min: float,
    subset_ids: set[str],
    defaults: dict[str, tuple[int, int]],
) -> dict[str, object]:
    """Keep the arms and a balanced prompt slice that fit in the budget.

    A budget of 0 keeps every selected arm and its prompts. Estimates are still filled in.
    """
    question = question_name(prompt_set)
    seconds_left = math.inf if budget_min <= 0 else max(0.0, (budget_min - reserve_min) * 60.0)
    order = [name for name in QUESTION_ARMS[question] if name in selected]
    for name in selected:
        if name not in order:
            order.append(name)
    loaded_family: set[str] = set()
    loaded_quant: dict[str, str] = {}
    kept: list[dict[str, object]] = []
    dropped: list[str] = []
    for name in order:
        family, quant = ARM_FAMILY[name]
        pool = [prompt for prompt in prompts if str(prompt["id"]) in subset_ids] if name == "q_tuned" else list(prompts)
        default = defaults[name]
        load = _load_cost(family, quant, loaded_family, loaded_quant)

        def cost(prompt: dict[str, object], family: str = family, default: tuple[int, int] = default) -> float:
            width, height = render_size(family, prompt, default)
            return image_seconds(family, width, height) * seeds

        picked, used = _take(pool, cost, seconds_left - load)
        if not picked:
            dropped.append(name)
            continue
        _commit_load(family, quant, loaded_family, loaded_quant)
        seconds_left -= load + used
        kept.append({
            "arm": name,
            "prompt_ids": [str(prompt["id"]) for prompt in picked],
            "estimate_min": round((load + used) / 60.0, 2),
        })
    notes: list[str] = []
    if budget_min > 0:
        kept_bits = [f"{row['arm']} ({len(row['prompt_ids'])} prompts, {row['estimate_min']} min)" for row in kept]
        note = f"budget {budget_min:g} min keeps " + (", ".join(kept_bits) if kept_bits else "nothing")
        if dropped:
            note += "; drops " + ", ".join(dropped)
        notes.append(note)
    return {
        "question": question,
        "kept": kept,
        "dropped": dropped,
        "notes": notes,
        "estimate_min": round(sum(float(row["estimate_min"]) for row in kept), 2),
    }


def validate_plan(plan: dict[str, object]) -> list[str]:
    """Schema errors for the dry-run plan."""
    errors: list[str] = []
    for key in ("prompt_set", "question", "prompts", "budget_min", "deadline_min", "checks", "arms"):
        if key not in plan:
            errors.append(f"missing {key}")
    if plan.get("checks") != list(CHECKS):
        errors.append("checks")
    arms = plan.get("arms")
    if not isinstance(arms, dict) or not arms:
        errors.append("arms")
    else:
        for name, spec in arms.items():
            if not isinstance(spec, dict) or "prompts" not in spec or "estimate_min" not in spec:
                errors.append(f"arm {name}")
    if plan.get("prompt_set") == "showcase":
        frames = plan.get("frames")
        if not isinstance(frames, list) or len(frames) != plan.get("prompts"):
            errors.append("frames")
        elif any(row.get("z") == row.get("q") for row in frames):
            errors.append("frames differ by model")
    return errors


def validate_summary(summary: dict[str, object]) -> list[str]:
    """Schema errors for summary.json."""
    errors: list[str] = []
    if summary.get("schema") != SCHEMA:
        errors.append("schema")
    for key in _SUMMARY_KEYS:
        if key not in summary:
            errors.append(f"missing {key}")
    return errors


def showcase_html(records: list[dict[str, object]], prompts: list[dict[str, object]]) -> str:
    """One full-size image per page, with prev/next, labelled for review."""
    by_id = {str(prompt["id"]): prompt for prompt in prompts}
    order = {prompt_id: index for index, prompt_id in enumerate(by_id)}
    arm_order = {name: index for index, name in enumerate(QUESTION_ARMS["showcase"])}
    rows = sorted(
        records,
        key=lambda row: (
            order.get(str(row.get("prompt_id")), 10**6),
            arm_order.get(str(row.get("arm")), 10**6),
            int(row.get("seed") or 0),
        ),
    )
    pages: list[str] = []
    for index, row in enumerate(rows):
        prompt = by_id.get(str(row.get("prompt_id")), {})
        model = str(row.get("model") or _model_name(str(row.get("arm") or "")))
        arm = str(row.get("arm") or "")
        prompt_id = str(row.get("prompt_id") or "")
        seed = int(row.get("seed") or 0)
        width = int(row.get("width") or 0)
        height = int(row.get("height") or 0)
        label = f"model {model} · arm {arm} · prompt {prompt_id} · seed {seed} · size {width}×{height}"
        image = "../" + str(row.get("image") or "").replace("\\", "/")
        text = html.escape(str(prompt.get("prompt") or ""))
        hidden = "" if index == 0 else " hidden"
        pages.append(
            f'<section class="page" data-page="{index}"{hidden}>'
            f"<p class=\"label\">{html.escape(label)}</p>"
            f"<p class=\"prompt\">{text}</p>"
            f'<img src="{html.escape(image)}" width="{width}" height="{height}" '
            f'style="width:{width}px;height:{height}px;max-width:none" alt="{html.escape(label)}">'
            "</section>"
        )
    body = "\n".join(pages) if pages else "<p>No showcase images yet.</p>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Showcase review</title>
<style>
  body {{ margin: 0; background: #111; color: #eee; font: 16px/1.4 sans-serif; }}
  nav {{ position: sticky; top: 0; background: #111; padding: 12px 16px; }}
  button {{ font: inherit; margin-right: 8px; }}
  img {{ display: block; }}
  .label, .prompt {{ padding: 0 16px; }}
</style>
</head>
<body>
<nav>
  <button id="prev" type="button">Prev</button>
  <button id="next" type="button">Next</button>
  <span id="pos"></span>
</nav>
{body}
<script>
const pages = Array.from(document.querySelectorAll("[data-page]"));
let index = 0;
function show(next) {{
  if (!pages.length) return;
  index = Math.max(0, Math.min(pages.length - 1, next));
  pages.forEach((page, n) => {{ page.hidden = n !== index; }});
  document.getElementById("pos").textContent = (index + 1) + " / " + pages.length;
  document.getElementById("prev").disabled = index === 0;
  document.getElementById("next").disabled = index === pages.length - 1;
}}
document.getElementById("prev").addEventListener("click", () => show(index - 1));
document.getElementById("next").addEventListener("click", () => show(index + 1));
document.addEventListener("keydown", (event) => {{
  if (event.key === "ArrowLeft") show(index - 1);
  if (event.key === "ArrowRight") show(index + 1);
}});
show(0);
</script>
</body>
</html>
"""


def _model_name(arm: str) -> str:
    if arm.startswith("z_"):
        return "Z-Image-Turbo"
    if arm.startswith("q_"):
        return "Qwen-Image-2512"
    return arm
