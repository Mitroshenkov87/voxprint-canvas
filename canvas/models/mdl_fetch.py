"""Download pinned files with VOXPrint MDL and hardlink a Canvas view.

The weights CI jobs use this. The app itself only enqueues a fetch and does
not write a legacy ``<org>--<name>`` copy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voxprint_common.mdl import download
from voxprint_common.mdl.catalog import MODEL_PREFIX, PlannedFile, load_manifest

from canvas.app.i18n import text
from canvas.infra.paths import models_dir
from canvas.infra.stage_timer import log_detail
from canvas.models.registry import ModelSpec, load_catalog
from canvas.models.snapshots import SnapshotError, snapshot_ready
from canvas.models.views import ViewError, ensure_view

MANIFEST = Path(__file__).resolve().parents[2] / "mdl.json"
GATE_ROLES = ("prompt-guard", "zero-shot", "face-detector", "age", "nsfw", "age-fine")
ENCODER_ID = "z-image-turbo-bf16-16gb"
ENCODER_PREFIXES = ("text_encoder/", "tokenizer/")


def gate_specs() -> tuple[ModelSpec, ...]:
    """Content-check models, one row per role, in gate order."""
    catalog = load_catalog()
    found: list[ModelSpec] = []
    for role in GATE_ROLES:
        rows = catalog.by_role(role)
        if len(rows) != 1:
            raise SnapshotError(f"{role} needs one model")
        found.append(rows[0])
    return tuple(found)


def plan_blobs(
    specs: tuple[ModelSpec, ...], root: Path, prefixes: tuple[str, ...] = (),
) -> list[PlannedFile]:
    """MDL blob plan for ``specs``. ``prefixes`` keeps matching repo paths only.

    Destinations are ``<root>/blobs/<sha256>``. A legacy repo folder is not a
    destination.
    """
    manifest = load_manifest(MANIFEST)
    rows = {model.id: model for model in manifest.models}
    planned: list[PlannedFile] = []
    seen: dict[str, PlannedFile] = {}
    for spec in specs:
        model = rows.get(spec.id)
        if model is None:
            raise SnapshotError(f"{spec.id} is not in mdl.json")
        listed = {item.path: item for item in model.files}
        selected = _selected(spec, prefixes)
        if not selected:
            raise SnapshotError(f"{spec.id} has no files to fetch")
        for path, digest in selected.items():
            item = listed.get(path)
            if item is None or item.sha256.lower() != digest.lower():
                raise SnapshotError(f"{spec.id}/{path} is not pinned in mdl.json")
            current = seen.get(digest)
            if current is None:
                current = PlannedFile(
                    component_id=f"model-{digest[:MODEL_PREFIX]}",
                    sha256=digest,
                    size=item.size,
                    version=digest,
                    name=path,
                    destination=root / "blobs" / digest,
                    urls=item.urls,
                    apps=["canvas"],
                    model_ids=[spec.id],
                    model_id=spec.id,
                    display_name=spec.id,
                )
                seen[digest] = current
                planned.append(current)
                continue
            if current.size != item.size:
                raise SnapshotError(f"{digest} is declared with two sizes")
            extra = tuple(url for url in item.urls if url not in current.urls)
            if extra:
                current.urls = (*current.urls, *extra)
            if spec.id not in current.model_ids:
                current.model_ids.append(spec.id)
    return planned


def materialize(
    specs: tuple[ModelSpec, ...], *, prefixes: tuple[str, ...] = (), key_suffix: str = "",
) -> tuple[Path, ...]:
    """Fetch ``specs`` into the blob store and return one view per spec."""
    root = models_dir()
    _download(plan_blobs(specs, root, prefixes))
    folders: list[Path] = []
    for spec in specs:
        files = _selected(spec, prefixes)
        folder = ensure_view(root, f"{spec.id}{key_suffix}", files)
        if folder is None:
            raise SnapshotError("model missing", user_message=text("model.missing"))
        _note(f"{spec.id}{key_suffix}")
        folders.append(folder)
    return tuple(folders)


def encoder_view(spec: ModelSpec, root: Path) -> Path | None:
    """The text-encoder view for ``spec``, or None when a blob is missing."""
    files = _selected(spec, ENCODER_PREFIXES)
    if not files:
        return None
    return ensure_view(root, f"{spec.id}-encoder", files)


def fetch_gate() -> tuple[Path, ...]:
    """Download every content-check model and open it as a view."""
    specs = gate_specs()
    materialize(specs)
    root = models_dir()
    opened: list[Path] = []
    for spec in specs:
        folder = snapshot_ready(spec, root)
        if folder is None:
            raise SnapshotError("model missing", user_message=text("model.missing"))
        print(f"id={spec.id} kind=view path={folder}", flush=True)
        opened.append(folder)
    return tuple(opened)


def fetch_encoder() -> Path:
    """Download the 16 GB pack's text encoder and tokenizer, and open that view."""
    spec = load_catalog().by_id(ENCODER_ID)
    (folder,) = materialize((spec,), prefixes=ENCODER_PREFIXES, key_suffix="-encoder")
    print(f"id={spec.id}-encoder kind=view path={folder}", flush=True)
    return folder


def main(argv: list[str] | None = None) -> int:
    """``gate`` fetches the content-check models. ``encoder`` fetches the text encoder."""
    parser = argparse.ArgumentParser(prog="mdl_fetch")
    parser.add_argument("kind", choices=("gate", "encoder"))
    args = parser.parse_args(argv)
    try:
        if args.kind == "gate":
            fetch_gate()
        elif args.kind == "encoder":
            fetch_encoder()
        else:
            print(f"unknown kind {args.kind}", file=sys.stderr)
            return 2
    except (OSError, RuntimeError, SnapshotError, ViewError) as exc:
        detail = getattr(exc, "detail", "")
        print(detail or exc, file=sys.stderr)
        return 1
    return 0


def _selected(spec: ModelSpec, prefixes: tuple[str, ...]) -> dict[str, str]:
    if not prefixes:
        return dict(spec.files)
    return {path: digest for path, digest in spec.files.items() if path.startswith(prefixes)}


def _download(planned: list[PlannedFile]) -> None:
    """Write each planned file with MDL. Tests replace this."""
    if not planned:
        raise SnapshotError("nothing to fetch")

    def note(fraction: float | None, message: str) -> None:
        del fraction
        if message:
            print(message, flush=True)

    outcome = download(planned=planned, display=False, priority=False, on_progress=note)
    if outcome.code != 0:
        raise RuntimeError(outcome.message or "MDL download failed")


def _note(model_id: str) -> None:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in model_id)[:80]
    log_detail("model", "path", "cpu", f"id={safe} kind=view")


if __name__ == "__main__":
    raise SystemExit(main())
