"""Adopt legacy weight folders, then download only what is still missing.

The installer runs this before the app starts. Each legacy directory is handed
to ``voxprint-mdl adopt`` once. A non-zero adopt is logged and that model is
left out of the download, so a failed adopt cannot become a second copy.
Legacy folders are never deleted.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from voxprint_common.mdl.catalog import CatalogError, load_manifest, plan_files
from voxprint_common.mdl.job import detect_tier

from canvas.infra.paths import models_dir
from canvas.infra.stage_timer import log_detail
from canvas.models.registry import Catalog, ModelSpec, RegistryError, load_catalog
from canvas.models.required import required_specs, view_complete
from canvas.models.snapshots import weight_dir


def main(argv: list[str] | None = None) -> int:
    """Adopt, then download. Adopt failures stay out of the download."""
    parser = argparse.ArgumentParser(description="Adopt legacy models, then fetch what is missing.")
    parser.add_argument("--app", default="canvas")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", action="append", default=[])
    args = parser.parse_args(argv)
    extra = tuple(str(item) for item in args.model)
    try:
        return fetch(Path(args.manifest), app=str(args.app), extra=extra)
    except (OSError, CatalogError, RegistryError, ValueError) as exc:
        print(f"model adopt action=failed reason={_token(type(exc).__name__)}", flush=True)
        return 1


def fetch(manifest: Path, *, app: str = "canvas", extra: tuple[str, ...] = (), root: Path | None = None) -> int:
    """Adopt each legacy directory, then download ids that are still incomplete.

    A missing ``voxprint-mdl`` or a non-zero adopt logs ``action=failed`` and
    does not download those ids. The process still exits 0 when the remaining
    download is empty, so setup can finish and preflight can refuse the view.
    """
    models = root if root is not None else models_dir()
    catalog = load_catalog()
    specs = required_specs(catalog)
    binary = _binary()
    failed = _adopt(binary, app, manifest, models, specs)
    wanted = _missing(manifest, models, catalog, specs, failed, extra)
    if not wanted:
        return 0
    if binary is None:
        for model_id in wanted:
            _log(model_id, "failed", "voxprint_mdl_missing")
        return 0
    command = [str(binary), "download", "--app", app, "--manifest", os.fspath(manifest)]
    for model_id in wanted:
        command.extend(["--model", model_id])
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def _adopt(
    binary: Path | None, app: str, manifest: Path, root: Path, specs: tuple[ModelSpec, ...],
) -> set[str]:
    """One adopt per legacy directory. Returns ids whose adopt failed."""
    failed: set[str] = set()
    groups: dict[Path, list[ModelSpec]] = {}
    order: list[Path] = []
    for spec in specs:
        legacy = weight_dir(root, spec.repo)
        if not legacy.is_dir():
            _log(spec.id, "skipped", "no_legacy")
            continue
        if legacy not in groups:
            groups[legacy] = []
            order.append(legacy)
        groups[legacy].append(spec)
    for legacy in order:
        rows = groups[legacy]
        if binary is None:
            for spec in rows:
                _log(spec.id, "failed", "voxprint_mdl_missing")
                failed.add(spec.id)
            continue
        command = [
            str(binary), "adopt", "--app", app, "--from", os.fspath(legacy),
            "--manifest", os.fspath(manifest),
        ]
        completed = subprocess.run(command, check=False)
        action = "adopted" if completed.returncode == 0 else "failed"
        reason = "" if completed.returncode == 0 else f"exit_{completed.returncode}"
        for spec in rows:
            _log(spec.id, action, reason)
            if completed.returncode != 0:
                failed.add(spec.id)
    return failed


def _missing(
    manifest: Path,
    root: Path,
    catalog: Catalog,
    specs: tuple[ModelSpec, ...],
    failed: set[str],
    extra: tuple[str, ...],
) -> list[str]:
    """Default-plan ids still incomplete, plus requested extras, minus failed adopts."""
    by_id = {spec.id: spec for spec in specs}
    chosen = catalog
    wanted: list[str] = []
    for model_id in [*_planned_ids(manifest, root), *extra]:
        if model_id in failed or model_id in wanted:
            continue
        spec = by_id.get(model_id)
        if spec is None:
            try:
                spec = chosen.by_id(model_id)
            except RegistryError:
                spec = None
        if spec is not None and view_complete(spec, root):
            continue
        wanted.append(model_id)
    return wanted


def _planned_ids(manifest: Path, root: Path) -> list[str]:
    """Ids ``voxprint-mdl download`` would fetch for this machine's tier."""
    parsed = load_manifest(manifest)
    planned = plan_files([parsed], detect_tier(None), models_root=root)
    found: list[str] = []
    for item in planned:
        names = list(item.model_ids) or ([item.model_id] if item.model_id else [])
        for model_id in names:
            if model_id and model_id not in found:
                found.append(model_id)
    return found


def _binary() -> Path | None:
    """The ``voxprint-mdl`` next to this interpreter, or None when it is not installed."""
    name = "voxprint-mdl.exe" if os.name == "nt" else "voxprint-mdl"
    path = Path(sys.executable).with_name(name)
    if path.is_file():
        return path
    return None


def _log(model_id: str, action: str, reason: str = "") -> None:
    safe = _token(model_id)
    suffix = f"id={safe} action={action}"
    if reason:
        suffix += f" reason={_token(reason)}"
    print(f"model adopt {suffix}", flush=True)
    log_detail("model", "adopt", "cpu", suffix)


def _token(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)
    return cleaned[:80] or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
