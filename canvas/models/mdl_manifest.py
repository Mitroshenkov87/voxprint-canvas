"""Build ``mdl.json`` for VOXPrint MDL from ``canvas/models/catalog.json``.

The catalog is the one source of truth. The installers pass ``mdl.json`` to
``python -m voxprint_common.mdl download``, so it must list every model the
default pack needs. Regenerate it after a catalog change and commit both:

    python -m canvas.models.mdl_manifest          # write mdl.json
    python -m canvas.models.mdl_manifest --check  # exit 1 when it is stale

URLs are the ones Canvas already uses (suite mirror first, then upstream).
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

# Files the catalog serves from the package ("package:<name>") are served to MDL
# from this repository at the commit that added them. The digest still decides.
PACKAGE_URL = "https://raw.githubusercontent.com/Mitroshenkov87/voxprint-canvas/{commit}/canvas/models/{name}"
PACKAGE_COMMITS = {"turbo_transformer_index.json": "159df97ad0d3d0f0324a747d739fe69061b4bbf7"}

# TODO(Aleksandr, 2026-10-18): our BF16 Z-Image-Turbo transformer moves to a separate
# voxprint repo (Model Forge is creating it); the mirror will then hold the author's
# original files. When Model Forge sends the links, change BF16_TRANSFORMER below to
# (new repo URL, revision) and update the transformer sha256/size rows in catalog.json,
# then regenerate. The author's original stays the fallback (``<id>-upstream``).
BF16_TRANSFORMER = ("https://huggingface.co/Mitroshenkov87/voxprint-mirror-z-image-turbo", "588702b873e82c328ab82c76742bf67843411f37")
_BF16_PREFIX = "zimage-turbo-transformer-bf16"


def _primary_urls(spec: ModelSpec, name: str, mirror_name: str) -> list[str]:
    urls: list[str] = []
    if mirror_name.startswith(_BF16_PREFIX):
        repo, revision = BF16_TRANSFORMER
        urls.append(f"{repo.rstrip('/')}/resolve/{revision}/{mirror_name}")
    for url in _urls(spec, name, spec.revision, fallback=False):
        if url.startswith("package:"):
            package = url.split(":", 1)[1]
            url = PACKAGE_URL.format(commit=PACKAGE_COMMITS[package], name=package)
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
            urls = _primary_urls(spec, name, str(meta.get("mirror") or ""))
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
