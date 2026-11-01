"""``canvas batch``: read a manifest, run every entry, and write a batch summary.

The manifest is a JSON object with an ``entries`` list::

    {"entries": [
        {"name": "cup", "prompt": "a red ceramic cup on a wooden table", "count": 2},
        {"shot": "close-up of an old fisherman mending a net", "negative": "blurry", "seed": 7}
    ]}

Each entry has exactly one of ``prompt`` (an idea the scenario writer turns
into ``count`` shots) or ``shot`` (one pose or shot line drawn ``count`` times
with consecutive seeds). ``negative``, ``seed``, ``count`` and ``name`` are
optional. Any other key, a wrong type, or a value out of range stops the
command with exit code 2 before any model loads.

Every prompt, shot line and negative prompt is checked like the prompt before
anything is drawn. A refused entry is recorded as ``refused`` in
``batch.json`` and draws nothing; the other entries continue. Every picture is
checked before it is saved.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from canvas.core.formats import EXPORT_FORMAT, OutputFormat, extension
from canvas.infra.stage_timer import StageTimes
from canvas.pipeline.batch import BatchEntry, EntryResult, run_batch
from canvas.pipeline.photoshoot import MAX_FRAMES, assemble_line
from canvas.pipeline.series import DrawFrame
from canvas.app.photoshoot_cli import _SCENARIO_STYLES, _english_stop_message, _series_message, catalogs
from canvas.safety.gate import AgeGate

MAX_ENTRIES = 100
# Pictures wait in RAM until the output check, so the batch has a ceiling.
MAX_BATCH_FRAMES = 400
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_NAME = 64
MAX_TEXT = 4000
SUMMARY_NAME = "batch.json"
_ENTRY_KEYS = frozenset({"name", "prompt", "shot", "negative", "seed", "count"})
_TOP_KEYS = frozenset({"entries"})
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"} | {f"com{n}" for n in range(1, 10)} | {f"lpt{n}" for n in range(1, 10)}
)


class ManifestInvalid(ValueError):
    """The manifest cannot be run. The message names the entry and the field."""


def load_manifest(path: Path) -> list[BatchEntry]:
    """Read and validate a manifest file.

    Raises:
        ManifestInvalid: The file is missing, unreadable, not JSON, or breaks a rule.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ManifestInvalid(f"The manifest cannot be read: {path} ({exc.strerror or exc})") from exc
    if not path.is_file():
        raise ManifestInvalid(f"The manifest is not a file: {path}")
    if size > MAX_MANIFEST_BYTES:
        raise ManifestInvalid(f"The manifest is larger than {MAX_MANIFEST_BYTES // 1024} KB: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ManifestInvalid(f"The manifest cannot be read as UTF-8 text: {path}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestInvalid(f"The manifest is not valid JSON (line {exc.lineno}, column {exc.colno}).") from exc
    return parse_manifest(document)


def parse_manifest(document: object) -> list[BatchEntry]:
    """Validate a decoded manifest.

    Raises:
        ManifestInvalid: The first rule the manifest breaks.
    """
    if not isinstance(document, dict):
        raise ManifestInvalid('The manifest must be a JSON object with an "entries" list.')
    extra = sorted(str(key) for key in document if key not in _TOP_KEYS)
    if extra:
        raise ManifestInvalid(f"Unknown manifest key: {extra[0]}.")
    items = document.get("entries")
    if not isinstance(items, list) or not items:
        raise ManifestInvalid('The manifest needs a non-empty "entries" list.')
    if len(items) > MAX_ENTRIES:
        raise ManifestInvalid(f"The manifest has {len(items)} entries; the limit is {MAX_ENTRIES}.")
    entries = [_entry(index, item) for index, item in enumerate(items)]
    total = sum(entry.count for entry in entries)
    if total > MAX_BATCH_FRAMES:
        raise ManifestInvalid(f"The batch asks for {total} pictures; the limit is {MAX_BATCH_FRAMES}.")
    seen: dict[str, int] = {}
    for index, entry in enumerate(entries):
        folded = entry.name.casefold()
        if folded in seen:
            raise ManifestInvalid(
                f"Entry {index + 1}: the folder name {entry.name!r} is already used by entry {seen[folded] + 1}."
            )
        seen[folded] = index
    return entries


def _entry(index: int, item: object) -> BatchEntry:
    where = f"Entry {index + 1}"
    if not isinstance(item, dict):
        raise ManifestInvalid(f"{where}: must be a JSON object.")
    extra = sorted(str(key) for key in item if key not in _ENTRY_KEYS)
    if extra:
        raise ManifestInvalid(f"{where}: unknown key {extra[0]!r}.")
    prompt = _text(item, "prompt", where)
    shot = _text(item, "shot", where)
    if bool(prompt) == bool(shot):
        raise ManifestInvalid(f'{where}: needs exactly one of "prompt" or "shot".')
    negative = _text(item, "negative", where)
    seed = _whole(item, "seed", where, default=1, low=0, high=2**32 - 1)
    count = _whole(item, "count", where, default=1, low=1, high=MAX_FRAMES)
    name = _folder_name(item, index, where)
    return BatchEntry(name=name, prompt=prompt, shot=shot, negative=negative, seed=seed, count=count)


def _text(item: dict[str, object], key: str, where: str) -> str:
    if key not in item:
        return ""
    value = item[key]
    if not isinstance(value, str):
        raise ManifestInvalid(f"{where}: {key!r} must be text.")
    value = value.strip()
    if key != "negative" and not value:
        raise ManifestInvalid(f"{where}: {key!r} is empty.")
    if len(value) > MAX_TEXT:
        raise ManifestInvalid(f"{where}: {key!r} is longer than {MAX_TEXT} characters.")
    return value


def _whole(item: dict[str, object], key: str, where: str, *, default: int, low: int, high: int) -> int:
    if key not in item:
        return default
    value = item[key]
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ManifestInvalid(f"{where}: {key!r} must be a whole number from {low} to {high}.")
    return value


def _folder_name(item: dict[str, object], index: int, where: str) -> str:
    """The entry's subfolder: its name made safe, or its 1-based index."""
    if "name" not in item:
        return f"{index + 1:03}"
    raw = item["name"]
    if not isinstance(raw, str):
        raise ManifestInvalid(f"{where}: 'name' must be text.")
    cleaned = sanitise_name(raw)
    if not cleaned:
        raise ManifestInvalid(f"{where}: 'name' has no letters or digits a folder can use.")
    if cleaned.casefold() == SUMMARY_NAME:
        raise ManifestInvalid(f"{where}: 'name' cannot be {SUMMARY_NAME}.")
    return cleaned


def sanitise_name(raw: str) -> str:
    """A folder name made of letters, digits, dot, dash and underscore. Empty when nothing is left."""
    cleaned = _UNSAFE.sub("_", raw.strip()).strip("._")[:MAX_NAME].strip("._")
    if not any(char.isalnum() for char in cleaned):
        return ""
    if cleaned.split(".")[0].casefold() in _RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned


def default_out(manifest: Path) -> Path:
    """``--out`` when it is omitted: a folder next to the manifest, named after it."""
    return manifest.with_name(f"{manifest.stem}_batch")


def run_batch_command(
    *,
    entries: list[BatchEntry],
    folder: Path,
    style: str,
    places: str,
    adult: bool,
    age_gate: AgeGate,
    rewrite: Callable[[str], str],
    frame_for: Callable[[BatchEntry], DrawFrame],
    container: OutputFormat | None = None,
    timings: StageTimes | None = None,
) -> tuple[dict[str, object], int]:
    """Run the batch into ``folder``, write ``batch.json``, and return it with the exit code.

    Exit code 0 when every entry wrote at least one picture, otherwise 1. A
    refused entry counts as not written. The other entries still run.
    """
    catalogs()
    fmt = EXPORT_FORMAT if container is None else container
    clock = timings if timings is not None else StageTimes()
    scenario_style = style if style in _SCENARIO_STYLES else "photo"
    outcomes = run_batch(
        entries=entries, folder=folder, style=scenario_style, place_mode=places, adult=adult,
        age_gate=age_gate, rewrite=rewrite, frame_for=frame_for, container=fmt, timings=clock,
    )
    rows = [_safe_row(index, outcome, fmt) for index, outcome in enumerate(outcomes)]
    made = sum(len(outcome.result.kept) for outcome in outcomes)
    ok = all(row["status"] == "ok" for row in rows)
    body: dict[str, object] = {
        "ok": ok,
        "folder": str(folder),
        "summary": str(folder / SUMMARY_NAME),
        "entries": rows,
        "requested": sum(entry.count for entry in entries),
        "made": made,
        "refused": sum(1 for row in rows if row["status"] == "refused"),
        "timings": clock.summary(),
    }
    (folder / SUMMARY_NAME).write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return body, 0 if ok else 1


def _safe_row(index: int, outcome: EntryResult, fmt: OutputFormat) -> dict[str, object]:
    """One row per entry, always. A failure while describing the entry becomes an error row."""
    try:
        return _row(index, outcome, fmt)
    except Exception as exc:
        entry = outcome.entry
        return {
            "index": index + 1, "name": entry.name, "kind": "prompt" if entry.prompt else "shot",
            "folder": None, "status": "error", "code": "internal", "requested": entry.count, "made": 0,
            "frames": [], "scenario": None, "seed": entry.seed, "format": extension(fmt).lstrip("."),
            "message": f"The entry summary could not be written ({type(exc).__name__}).",
        }


def _row(index: int, outcome: EntryResult, fmt: OutputFormat) -> dict[str, object]:
    entry = outcome.entry
    result = outcome.result
    frames = [str(path) for path in result.kept]
    scenario_path = _write_scenario(outcome)
    status = _status(result.stopped, result.code, bool(frames))
    if status == "ok" and len(frames) < entry.count:
        status = "partial"
    row: dict[str, object] = {
        "index": index + 1,
        "name": entry.name,
        "kind": "prompt" if entry.prompt else "shot",
        "folder": str(outcome.folder) if result.folder is not None else None,
        "status": status,
        "code": result.code,
        "requested": entry.count,
        "made": len(frames),
        "frames": frames,
        "scenario": scenario_path,
        "seed": entry.seed,
        "format": extension(fmt).lstrip("."),
    }
    if status != "ok":
        row["message"] = _message(result.stopped, result.code, result.reason)
        if not result.code and status == "partial":
            row["code"] = "frames_short"
            row["message"] = f"{len(frames)} of {entry.count} pictures were saved; the others were not."
    return row


def _status(stopped: str, code: str, wrote: bool) -> str:
    """``ok``, ``refused``, ``partial``, ``not_produced`` or ``error`` for one entry."""
    if stopped == "idea" or code in {"idea.policy", "idea.minor", "scenario.policy"}:
        return "refused"
    if stopped == "error":
        return "partial" if wrote else "error"
    if stopped == "scenario":
        if code == "scenario.minor":
            return "partial" if wrote else "refused"
        return "partial" if wrote else "error"
    if not wrote:
        return "not_produced"
    return "ok"


def _message(stopped: str, code: str, reason: str) -> str:
    if stopped == "idea" and reason.strip():
        return reason
    if stopped == "error":
        if code == "internal" and reason.strip():
            return reason
        return _english_stop_message(code or "internal")
    return _series_message(code or "not_produced")


def _write_scenario(outcome: EntryResult) -> str | None:
    """The entry's ``scenario.txt``: one line per shot. None when the entry stopped before its shots."""
    scenario = outcome.scenario
    if scenario is None or outcome.result.folder is None:
        return None
    entry = outcome.entry
    lines = [
        assemble_line(scenario.subject_block, shot, include_camera=True, index=index, seed=entry.seed)
        for index, shot in enumerate(scenario.shots)
    ]
    path = outcome.folder / "scenario.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
