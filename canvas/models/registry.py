"""Load and check the shared model registry."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from canvas.models.pins import DIFFUSERS_PIN
from canvas.safety.constants import FORBIDDEN_MODEL_MARKERS

_CATALOG = Path(__file__).with_name("catalog.json")
_ALLOWED_LICENCES = {"apache-2.0", "mit"}
_SOURCE_KINDS = ("local", "upstream", "huggingface", "huggingface-mirror", "kaggle", "torrent")
_GUIDANCE_KINDS = {"", "guidance_scale", "true_cfg_scale"}
_QUALITY_GATES = {"", "pending", "passed"}


class RegistryError(ValueError):
    """The registry is missing a model or lists one this app must not ship."""


@dataclass(frozen=True)
class Source:
    """One place a weight file may be fetched from."""

    kind: str
    uri: str


@dataclass(frozen=True)
class ModelSpec:
    """One registry row: repo, tier, quant, and source order."""

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
    quality_gate: str = ""
    revision: str = ""
    mirror_revision: str = ""
    mirror_status: str = ""
    origins: dict[str, tuple[str, str]] = field(default_factory=dict)
    fallback_files: dict[str, str] = field(default_factory=dict)
    fallback_revision: str = ""
    nbytes: int = 0

    def pinned(self) -> bool:
        """True when every listed file has a 64-character SHA-256."""
        return bool(self.files) and all(len(value) == 64 for value in self.files.values())


@dataclass(frozen=True)
class FrameSize:
    """Width and height of one output frame."""

    width: int
    height: int


@dataclass(frozen=True)
class SizeTable:
    """Native and ~2K frames for one model family. Both axes are multiples of ``multiple``."""

    multiple: int
    native: dict[str, FrameSize]
    twok: dict[str, FrameSize]


@dataclass(frozen=True)
class Catalog:
    """The loaded registry."""

    diffusers: str
    models: tuple[ModelSpec, ...]
    default_text_base: str | None = None
    text_bases: tuple[str, ...] = ()
    sizes: dict[str, SizeTable] | None = None

    def by_id(self, model_id: str) -> ModelSpec:
        """The model with this id."""
        for spec in self.models:
            if spec.id == model_id:
                return spec
        raise RegistryError(f"unknown model {model_id}")

    def by_role(self, role: str) -> tuple[ModelSpec, ...]:
        """Every model with this role."""
        return tuple(spec for spec in self.models if spec.role == role)


def _nbytes(files: dict[str, Any]) -> int:
    total = 0
    for meta in files.values():
        if isinstance(meta, dict) and meta.get("size"):
            total += int(meta["size"])
    return total


def _sha_map(files: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, meta in files.items():
        if isinstance(meta, dict):
            out[str(name)] = str(meta.get("sha256") or "")
        else:
            out[str(name)] = str(meta or "")
    return out


def _origins(files: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Remote path on the mirror and on the upstream repo. ``-`` means that source has no copy."""
    out: dict[str, tuple[str, str]] = {}
    for name, meta in files.items():
        if isinstance(meta, dict):
            mirror = str(meta.get("mirror", name))
            upstream = str(meta.get("upstream", name))
        else:
            mirror, upstream = str(name), str(name)
        out[str(name)] = (mirror, upstream)
    return out


def _spec(row: dict[str, Any]) -> ModelSpec:
    guidance = row.get("guidance_scale")
    steps = row.get("steps")
    sources = tuple(Source(str(item["kind"]), str(item.get("uri") or "")) for item in row.get("sources", []))
    files = dict(row.get("files") or {})
    fallback_raw = row.get("fallback")
    fallback: dict[str, Any] = fallback_raw if isinstance(fallback_raw, dict) else {}
    fallback_files = dict(fallback.get("files") or {})
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
        files=_sha_map(files),
        sources=sources,
        text_base=str(row.get("text_base") or ""),
        guidance_kind=str(row.get("guidance_kind") or ""),
        quality_gate=str(row.get("quality_gate") or ""),
        revision=str(row.get("revision") or ""),
        mirror_revision=str(row.get("mirror_revision") or ""),
        mirror_status=str(row.get("mirror_status") or ""),
        origins=_origins(files),
        fallback_files=_sha_map(fallback_files),
        fallback_revision=str(fallback.get("revision") or ""),
        nbytes=_nbytes(files),
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
        if spec.quality_gate not in _QUALITY_GATES:
            raise RegistryError("unknown quality gate")
        if "fp8" in spec.quant.casefold():
            if _is_z_image(spec):
                raise RegistryError("Z-Image is BF16 only")
            if spec.quality_gate == "":
                raise RegistryError("an fp8 model needs a quality gate")
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
    sizes = _sizes(raw.get("sizes"), text_bases)
    return Catalog(diffusers, models, default_text_base, text_bases, sizes)


_ASPECTS = ("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21")
_FOUR_K_LONG = 3840


def _frame(value: object, multiple: int, aspect: str) -> FrameSize:
    if not isinstance(value, list) or len(value) != 2:
        raise RegistryError(f"{aspect} needs a width and a height")
    width, height = int(value[0]), int(value[1])
    if width < 1 or height < 1 or width % multiple or height % multiple:
        raise RegistryError(f"{aspect} is not a multiple of {multiple}")
    return FrameSize(width, height)


def _band(raw: object, multiple: int, kind: str) -> dict[str, FrameSize]:
    if not isinstance(raw, dict):
        raise RegistryError(f"the {kind} size table is missing")
    frames = {aspect: _frame(raw.get(aspect), multiple, aspect) for aspect in _ASPECTS}
    if kind == "twok":
        for aspect, frame in frames.items():
            long_side = max(frame.width, frame.height)
            if long_side >= _FOUR_K_LONG or _FOUR_K_LONG % long_side:
                raise RegistryError(f"{aspect} does not upscale to 4K")
            factor = _FOUR_K_LONG // long_side
            if factor not in (2, 4):
                raise RegistryError(f"{aspect} does not upscale to 4K")
    return frames


def _sizes(raw: object, families: tuple[str, ...]) -> dict[str, SizeTable]:
    if not isinstance(raw, dict):
        raise RegistryError("the size table is missing")
    for family in families:
        if not isinstance(raw.get(family), dict):
            raise RegistryError(f"{family} has no sizes")
    tables: dict[str, SizeTable] = {}
    for family, row in raw.items():
        if not isinstance(row, dict):
            raise RegistryError(f"{family} has no sizes")
        multiple = int(row.get("multiple") or 0)
        if multiple not in (16, 64):
            raise RegistryError(f"{family} needs a 16 or 64 size step")
        tables[str(family)] = SizeTable(
            multiple, _band(row.get("native"), multiple, "native"), _band(row.get("twok"), multiple, "twok"),
        )
    return tables


def _is_z_image(spec: ModelSpec) -> bool:
    blob = f"{spec.id} {spec.repo} {spec.text_base}".casefold()
    return "z-image" in blob


def source_order(spec: ModelSpec) -> tuple[str, ...]:
    """Source kinds in the order the downloader tries them. Empty URIs are skipped later."""
    return tuple(source.kind for source in spec.sources)
