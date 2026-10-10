"""Headless checks for a machine that is about to generate. Nothing is generated."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from canvas.app.identity import version_label
from canvas.app.platform import check_gpu, current_os_check
from canvas.core.formats import OutputFormat, save_image
from canvas.infra.gpu import probe_gpu
from canvas.infra.paths import models_dir
from canvas.models.registry import ModelSpec, load_catalog
from canvas.models.snapshots import snapshot_ready
from canvas.safety.adapters import FairFaceAge, NsfwViT, Qwen3GuardGen, SiglipZeroShot, YuNetFaces
from canvas.safety.gate import SafetyError, issue
from canvas.safety.image import RawImage


def selftest_report(root: Path | None = None) -> dict[str, object]:
    """GPU, pinned files, classifier load, and the three output encoders."""
    folder = root if root is not None else models_dir()
    os_result = current_os_check()
    gpu = check_gpu(probe_gpu())
    encoders = _encoders()
    catalog = load_catalog()
    models = [_model_row(spec, folder) for spec in catalog.models if spec.files]
    classifiers = _classifiers(catalog, folder)
    ok = bool(
        os_result.ok
        and gpu.ok
        and all(encoders.values())
        and all(bool(row["ok"]) for row in models)
        and all(bool(row["ok"]) for row in classifiers)
    )
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
    }


def selftest_json(root: Path | None = None) -> str:
    """The self-test report as one JSON document."""
    return json.dumps(selftest_report(root), indent=2)


def _model_row(spec: ModelSpec, root: Path) -> dict[str, object]:
    ready = snapshot_ready(spec, root) is not None
    detail = "" if ready else "missing"
    if spec.mirror_status:
        detail = spec.mirror_status if not ready else ""
    return {"id": spec.id, "ok": ready, "detail": detail}


def _classifiers(catalog, root: Path) -> list[dict[str, object]]:
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
        folder = snapshot_ready(spec, root)
        if folder is None:
            rows.append({"id": spec.id, "ok": False, "detail": "missing"})
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
