"""Headless checks for a machine that is about to generate. Nothing is generated."""
from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path

from canvas.app.i18n import text
from canvas.app.identity import version_label
from canvas.app.platform import check_gpu, current_os_check
from canvas.core.formats import OutputFormat, save_image
from canvas.infra.gpu import probe_gpu
from canvas.infra.paths import models_dir
from canvas.infra.runtime import python_supports_tensorrt
from canvas.models.registry import ModelSpec, load_catalog
from canvas.models.snapshots import SnapshotError, snapshot_ready, snapshot_status
from canvas.models.verified import VerifiedCache
from canvas.safety.adapters import FairFaceAge, NsfwViT, Qwen3GuardGen, SiglipZeroShot, YuNetFaces
from canvas.safety.gate import SafetyError, issue
from canvas.safety.image import RawImage


def selftest_report(
    root: Path | None = None, progress: Callable[[str], None] | None = None, *, deep: bool = False,
) -> dict[str, object]:
    """GPU, pinned files, model load, and the three output encoders.

    This check only reads the machine. It does not download models or wheels.
    ``deep`` hashes every file. Otherwise a file whose path, size, and time
    match the shared manifest is not read again. ``progress`` is accepted for
    callers and is not printed to stdout.
    """
    del progress
    folder = models_dir() if root is None else root
    os_result = current_os_check()
    gpu = check_gpu(probe_gpu())
    encoders = _encoders()
    catalog = load_catalog()
    cache = VerifiedCache.open(deep)
    try:
        models = [_model_row(spec, folder, cache) for spec in catalog.models if spec.files]
        classifiers = _classifiers(catalog, folder, cache)
    finally:
        cache.flush()
    ok = bool(
        os_result.ok
        and gpu.ok
        and all(encoders.values())
        and all(bool(row["ok"]) for row in models)
        and all(bool(row["ok"]) for row in classifiers)
    )
    notes: list[str] = []
    if not python_supports_tensorrt():
        notes.append(text("splash.tensorrt_skip", "en"))
    return {
        "ok": ok,
        "version": version_label(),
        "os": {"ok": os_result.ok, "message": os_result.message},
        "gpu": {
            "ok": gpu.ok,
            "message": gpu.message,
            "tier": "" if gpu.tier is None else gpu.tier.value,
        },
        "encoders": encoders,
        "models": models,
        "classifiers": classifiers,
        "notes": notes,
        "deep": deep,
    }


def selftest_json(root: Path | None = None, *, deep: bool = False) -> str:
    """The self-test report as one JSON document."""
    return json.dumps(selftest_report(root, deep=deep), indent=2)


def _model_row(spec: ModelSpec, root: Path, cache: VerifiedCache) -> dict[str, object]:
    state = snapshot_status(spec, root, cache)
    row: dict[str, object] = {"id": spec.id, "ok": True, "detail": ""}
    if spec.mirror_status:
        row["mirror_status"] = spec.mirror_status
    if state == "ready":
        return row
    if state == "not_downloaded":
        row["detail"] = "not_downloaded"
        return row
    row["ok"] = False
    row["detail"] = state
    return row


def _classifiers(catalog, root: Path, cache: VerifiedCache) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    binding = (
        ("prompt-guard", Qwen3GuardGen),
        ("zero-shot", SiglipZeroShot),
        ("face-detector", YuNetFaces),
        ("age", FairFaceAge),
        ("nsfw", NsfwViT),
    )
    for role, cls in binding:
        found = catalog.by_role(role)
        if len(found) != 1:
            rows.append({"id": role, "ok": False, "detail": "missing"})
            continue
        spec = found[0]
        state = snapshot_status(spec, root, cache)
        if state == "not_downloaded":
            rows.append({"id": spec.id, "ok": True, "detail": "not_downloaded"})
            continue
        if state != "ready":
            rows.append({"id": spec.id, "ok": False, "detail": state})
            continue
        try:
            folder = snapshot_ready(spec, root, cache)
        except SnapshotError:
            rows.append({"id": spec.id, "ok": False, "detail": "link_failed"})
            continue
        if folder is None:
            rows.append({"id": spec.id, "ok": False, "detail": "hash_mismatch"})
            continue
        adapter = cls(folder)
        try:
            adapter.load()
        except (SafetyError, ImportError, OSError) as exc:
            rows.append({"id": spec.id, "ok": False, "detail": exc.__class__.__name__})
            continue
        rows.append({"id": spec.id, "ok": True, "detail": ""})
    return rows


def _encoders() -> dict[str, bool]:
    image = issue(RawImage(8, 8, bytes([40, 80, 120]) * 64))
    results: dict[str, bool] = {}
    for fmt in (OutputFormat.AVIF, OutputFormat.JXL, OutputFormat.WEBP):
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / f"probe.{fmt.value}"
                save_image(image, path, fmt)
                results[fmt.value] = path.is_file() and path.stat().st_size > 16
        except (OSError, ValueError):
            results[fmt.value] = False
    return results
