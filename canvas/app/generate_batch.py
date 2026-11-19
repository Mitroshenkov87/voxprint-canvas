"""Plan a multi-picture ``generate`` run: count, seeds, and a prompts file."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from canvas.core.formats import DEFAULT_FORMAT, extension, picture_path
from canvas.core.seeds import SEED_MAX, SEED_MIN, as_seed, pick_seed

MAX_COUNT = 100


class GeneratePlanError(ValueError):
    """The count, seeds, or prompts file cannot be used."""


@dataclass(frozen=True)
class GenerateItem:
    """One picture in a ``generate`` batch."""

    index: int
    prompt: str
    seed: int


def parse_seeds(text: str) -> list[int]:
    """Parse a comma-separated seed list. Empty text is an empty list."""
    raw = str(text or "").strip()
    if not raw:
        return []
    seeds: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            raise GeneratePlanError("A --seeds list cannot contain an empty entry.")
        try:
            value = int(token, 10)
        except ValueError as exc:
            raise GeneratePlanError(f"{token!r} is not a seed.") from exc
        if as_seed(value) is None:
            raise GeneratePlanError(f"Seed {value} is outside {SEED_MIN}..{SEED_MAX}.")
        seeds.append(value)
    return seeds


def load_prompt_file(path: Path) -> list[tuple[str, int | None]]:
    """Load prompts from a text file (one per line) or JSONL with per-line prompt/seed."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GeneratePlanError(f"The prompts file could not be read: {path}") from exc
    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        raise GeneratePlanError("The prompts file is empty.")
    if path.suffix.lower() == ".jsonl" or nonempty[0].lstrip().startswith("{"):
        return _jsonl(lines)
    return [(line.strip(), None) for line in lines if line.strip()]


def plan_items(
    *,
    prompt: str = "",
    count: int | None = None,
    seed: int | None = None,
    seeds: list[int] | None = None,
    prompts: list[tuple[str, int | None]] | None = None,
) -> list[GenerateItem]:
    """Build the (prompt, seed) list for one ``generate`` invocation.

    ``--seed S`` counts up (S, S+1, …). With no seed and no per-item seed, each
    picture draws a fresh random seed. ``--seeds`` sets the list and the count
    unless ``--count`` matches it.
    """
    if seed is not None and seeds:
        raise GeneratePlanError("Use --seed or --seeds, not both.")
    if seed is not None and as_seed(seed) is None:
        raise GeneratePlanError(f"Seed {seed} is outside {SEED_MIN}..{SEED_MAX}.")
    if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 1):
        raise GeneratePlanError("--count must be at least 1.")
    if count is not None and count > MAX_COUNT:
        raise GeneratePlanError(f"--count cannot be more than {MAX_COUNT}.")
    if seeds is not None and len(seeds) > MAX_COUNT:
        raise GeneratePlanError(f"--seeds cannot list more than {MAX_COUNT} seeds.")
    if prompts is not None and len(prompts) > MAX_COUNT:
        raise GeneratePlanError(f"The prompts file cannot have more than {MAX_COUNT} prompts.")

    if prompts is not None:
        if prompt.strip():
            raise GeneratePlanError("Use --prompt or --prompts, not both.")
        if seeds is not None and count is not None and count != len(seeds):
            raise GeneratePlanError("--count must match the number of --seeds.")
        if seeds is not None and len(seeds) != len(prompts):
            raise GeneratePlanError("The number of --seeds must match the number of prompts.")
        if count is not None and count != len(prompts):
            raise GeneratePlanError("--count must match the number of prompts in the file.")
        items: list[GenerateItem] = []
        for index, (line, line_seed) in enumerate(prompts):
            if not line.strip():
                raise GeneratePlanError(f"Prompt {index + 1} is empty.")
            chosen = line_seed
            if chosen is None and seeds is not None:
                chosen = seeds[index]
            items.append(GenerateItem(index + 1, line.strip(), pick_seed(seed if chosen is None else chosen, 0 if chosen is not None else index)))
        return items

    text = str(prompt or "").strip()
    if not text:
        raise GeneratePlanError("A prompt is required.")
    if seeds:
        if count is not None and count != len(seeds):
            raise GeneratePlanError("--count must match the number of --seeds.")
        return [GenerateItem(index + 1, text, value) for index, value in enumerate(seeds)]
    total = 1 if count is None else count
    return [GenerateItem(index + 1, text, pick_seed(seed, index)) for index in range(total)]


def item_path(out: Path, item: GenerateItem, *, total: int, out_arg: str) -> tuple[Path, str | None]:
    """The file for ``item``. Batch runs include the index and the seed in the name."""
    if _as_folder(out, total, out_arg):
        folder = out
        prepared = folder / f"{item.index:03d}-{item.seed}{extension(DEFAULT_FORMAT)}"
        path, warning = picture_path(prepared)
        return path, warning
    if total == 1:
        return picture_path(out)
    path, warning = picture_path(out)
    named = path.with_name(f"{path.stem}-{item.index:03d}-{item.seed}{path.suffix}")
    return named, warning


def _as_folder(out: Path, total: int, out_arg: str) -> bool:
    argument = str(out_arg or "")
    if argument.endswith(("/", "\\")) or out.is_dir():
        return True
    return total > 1 and out.suffix == ""


def _jsonl(lines: list[str]) -> list[tuple[str, int | None]]:
    items: list[tuple[str, int | None]] = []
    for number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GeneratePlanError(f"Line {number} of the prompts file is not JSON.") from exc
        if not isinstance(loaded, dict):
            raise GeneratePlanError(f"Line {number} of the prompts file must be a JSON object.")
        prompt = loaded.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise GeneratePlanError(f"Line {number} of the prompts file needs a prompt.")
        if "seed" not in loaded or loaded.get("seed") is None:
            items.append((prompt.strip(), None))
            continue
        seed = loaded.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or as_seed(seed) is None:
            raise GeneratePlanError(f"Line {number} of the prompts file has a seed outside {SEED_MIN}..{SEED_MAX}.")
        items.append((prompt.strip(), seed))
    if not items:
        raise GeneratePlanError("The prompts file is empty.")
    return items
