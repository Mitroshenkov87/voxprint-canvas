"""Load and check the shared model registry."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from canvas.models.pins import DIFFUSERS_PIN
from canvas.safety.constants import FORBIDDEN_MODEL_MARKERS

_CATALOG = Path(__file__).with_name("catalog.json")
_ALLOWED_LICENCES = {"apache-2.0", "mit"}
_SOURCE_KINDS = ("local", "upstream", "huggingface", "huggingface-mirror", "kaggle", "torrent")
_GUIDANCE_KINDS = {"", "guidance_scale", "true_cfg_scale"}


class RegistryError(ValueError):
    """The registry is missing a model or lists one this app must not ship."""


@dataclass(frozen=True)
class Source:
    kind: str
    uri: str


@dataclass(frozen=True)
class ModelSpec:
    id: str
    role: str
    tier: str
    repo: str
    licence: str
    runtime: str
    status: str
    pipeline_classes: tuple[str, ...]
    quant: str
    quant_backend: str
    steps: int | None
    guidance_scale: float | None
    text_encoder: str
    text_encoder_offload: str
    variant: str
    vram_hint_gb: float
    min_vram_gb: float
    device: str
    files: dict[str, str]
    sources: tuple[Source, ...]
    text_base: str
    guidance_kind: str

    def pinned(self) -> bool:
        return bool(self.files) and all(len(value) == 64 for value in self.files.values())


@dataclass(frozen=True)
class Catalog:
    diffusers: str
    models: tuple[ModelSpec, ...]
    default_text_base: str | None = None
    text_bases: tuple[str, ...] = ()

    def by_id(self, model_id: str) -> ModelSpec:
        for spec in self.models:
            if spec.id == model_id:
                return spec
        raise RegistryError(f"unknown model {model_id}")

    def by_role(self, role: str) -> tuple[ModelSpec, ...]:
        return tuple(spec for spec in self.models if spec.role == role)


def _sha_map(files: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, meta in files.items():
        if isinstance(meta, dict):
            out[str(name)] = str(meta.get("sha256") or "")
        else:
            out[str(name)] = str(meta or "")
    return out


def _spec(row: dict[str, Any]) -> ModelSpec:
    guidance = row.get("guidance_scale")
    steps = row.get("steps")
    sources = tuple(Source(str(item["kind"]), str(item.get("uri") or "")) for item in row.get("sources", []))
    return ModelSpec(
        id=str(row["id"]),
        role=str(row["role"]),
        tier=str(row.get("tier") or ""),
        repo=str(row["repo"]),
        licence=str(row["licence"]),
        runtime=str(row.get("runtime") or ""),
        status=str(row.get("status") or ""),
        pipeline_classes=tuple(str(item) for item in row.get("pipeline_classes", [])),
        quant=str(row.get("quant") or ""),
        quant_backend=str(row.get("quant_backend") or ""),
        steps=None if steps is None else int(steps),
        guidance_scale=None if guidance is None else float(guidance),
        text_encoder=str(row.get("text_encoder") or ""),
        text_encoder_offload=str(row.get("text_encoder_offload") or ""),
        variant=str(row.get("variant") or ""),
        vram_hint_gb=float(row.get("vram_hint_gb") or 0),
        min_vram_gb=float(row.get("min_vram_gb") or 0),
        device=str(row.get("device") or ""),
        files=_sha_map(dict(row.get("files") or {})),
        sources=sources,
        text_base=str(row.get("text_base") or ""),
        guidance_kind=str(row.get("guidance_kind") or ""),
    )


def load_catalog(path: Path | None = None) -> Catalog:
    """Read the registry and refuse a forbidden or non-permissive entry."""
    raw = json.loads((path or _CATALOG).read_text(encoding="utf-8"))
    diffusers = str(raw.get("diffusers") or "")
    if diffusers != DIFFUSERS_PIN:
        raise RegistryError("diffusers pin does not match the app")
    models = tuple(_spec(row) for row in raw["models"])
    for spec in models:
        marker = spec.repo.casefold()
        if any(item in marker or item in spec.id.casefold() for item in FORBIDDEN_MODEL_MARKERS):
            raise RegistryError("a listed model is not allowed")
        if spec.licence.casefold() not in _ALLOWED_LICENCES:
            raise RegistryError("a listed model is not Apache-2.0 or MIT")
        if spec.guidance_kind not in _GUIDANCE_KINDS:
            raise RegistryError("unknown guidance kind")
        kinds = [source.kind for source in spec.sources]
        if any(kind not in _SOURCE_KINDS for kind in kinds):
            raise RegistryError("unknown model source")
    text_bases = tuple(str(item) for item in raw.get("text_bases") or [])
    default = raw.get("default_text_base")
    default_text_base = None if default is None else str(default)
    if default_text_base is not None and default_text_base not in text_bases:
        raise RegistryError("the default text model is not in the registry")
    for spec in models:
        if spec.text_base and spec.text_base not in text_bases:
            raise RegistryError("a text model is not listed")
    for family in text_bases:
        tiers = {spec.tier for spec in models if spec.text_base == family}
        if "16gb" not in tiers or "24gb" not in tiers:
            raise RegistryError("a text model is missing a tier")
    return Catalog(diffusers, models, default_text_base, text_bases)


def source_order(spec: ModelSpec) -> tuple[str, ...]:
    """Source kinds in the order the downloader tries them. Empty URIs are skipped later."""
    return tuple(source.kind for source in spec.sources)
