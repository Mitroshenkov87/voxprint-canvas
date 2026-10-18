"""Build ``mdl.json`` for VOXPrint MDL from ``canvas/models/catalog.json``.

The catalog is the one source of truth. The installers pass ``mdl.json`` to
``python -m voxprint_common.mdl download``, so it must list every model the
default pack needs. Regenerate it after a catalog change and commit both:

    python -m canvas.models.mdl_manifest          # write mdl.json
    python -m canvas.models.mdl_manifest --check  # exit 1 when it is stale

URLs are the ones Canvas uses: the primary source (our BF16 repo or the suite mirror)
first, then upstream.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from canvas.models.provision import OPTIONAL_IDS, default_specs
from canvas.models.registry import Catalog, ModelSpec, load_catalog
from canvas.models.snapshots import _urls

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "mdl.json"

# Decision (Aleksandr, 2026-10-18): our BF16 Z-Image-Turbo transformer lives in its own
# repo (Mitroshenkov87/voxprint-z-image-turbo-bf16); the suite mirror holds the author's
# original files. The primary rows point at the BF16 repo in catalog.json. The author's
# FP32 transformer is the optional ``<id>-upstream`` fallback (mirror copy, then upstream);
# the runtime loads it with torch_dtype=bfloat16, so it is cast at load time.


def _primary_urls(spec: ModelSpec, name: str) -> list[str]:
    urls: list[str] = []
    for url in _urls(spec, name, spec.revision, fallback=False):
        if not url.startswith("https://"):
            raise ValueError(f"{spec.id}/{name}: MDL needs an https address, got {url}")
        if url not in urls:
            urls.append(url)
    return urls


def _entry(model_id: str, version: str, tier: str, files: list[dict[str, object]], *,
           optional: bool, licence: str, source: str) -> dict[str, object]:
    return {
        "id": model_id,
        "version": version,
        "vram_tier": tier,
        "optional": optional,
        "licence": licence,
        "source": source,
        "files": files,
    }


def _tier(spec: ModelSpec) -> str:
    return spec.tier if spec.tier in ("16gb", "24gb", "8gb") else "any"


def build(catalog: Catalog | None = None) -> dict[str, object]:
    """The ``mdl.json`` document for this catalog."""
    catalog = load_catalog() if catalog is None else catalog
    defaults = {spec.id for spec in default_specs(catalog)}
    raw = json.loads((ROOT / "canvas" / "models" / "catalog.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in raw["models"]}
    models: list[dict[str, object]] = []
    for spec in catalog.models:
        if not spec.files:
            continue
        row = rows[spec.id]
        files: list[dict[str, object]] = []
        for name, meta in row["files"].items():
            urls = _primary_urls(spec, name)
            if not urls:
                raise ValueError(f"{spec.id}/{name} has no download address")
            files.append({"path": name, "size": int(meta["size"]), "sha256": meta["sha256"], "mirrors": urls})
        optional = spec.id in OPTIONAL_IDS or spec.id not in defaults
        models.append(_entry(spec.id, spec.revision, _tier(spec), files,
                             optional=optional, licence=spec.licence, source="primary"))
        fallback = row.get("fallback")
        if fallback:
            revision = str(fallback["revision"])
            extra: list[dict[str, object]] = []
            for name, meta in fallback["files"].items():
                urls = list(_urls(spec, name, revision, fallback=True))
                if not urls:
                    raise ValueError(f"{spec.id}/{name} has no fallback address")
                extra.append({"path": name, "size": int(meta["size"]), "sha256": meta["sha256"], "mirrors": urls})
            models.append(_entry(f"{spec.id}-upstream", revision, _tier(spec), extra,
                                 optional=True, licence=spec.licence, source="fallback"))
    return {
        "schema": 1,
        "app": "canvas",
        "generated_from": "canvas/models/catalog.json",
        "models": models,
    }


def render(catalog: Catalog | None = None) -> str:
    """``mdl.json`` text, stable across runs."""
    return json.dumps(build(catalog), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write ``mdl.json``, or with ``--check`` report whether it matches the catalog."""
    parser = argparse.ArgumentParser(prog="mdl_manifest")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = MANIFEST.read_text(encoding="utf-8") if MANIFEST.is_file() else ""
        if current != text:
            print("mdl.json does not match canvas/models/catalog.json. Run python -m canvas.models.mdl_manifest.",
                  file=sys.stderr)
            return 1
        return 0
    MANIFEST.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
