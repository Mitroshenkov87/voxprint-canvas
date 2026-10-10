"""Opt-in third-party weights.

A download starts only when the user asks for it, and only from the upstream
address on the entry. Mirror sources are refused. The content checks stay on
for every weight, including these.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from canvas.models.downloader import DownloadError
from canvas.models.pin import PinError, require_pin, verify_bytes
from canvas.models.registry import RegistryError, Source
from canvas.safety.gate import AgeGate
from canvas.safety.image import ScreenedInput

_CATALOG = Path(__file__).with_name("third_party.json")
_MIRROR_KINDS = frozenset({"huggingface-mirror", "kaggle", "torrent"})
_UPSTREAM_LICENCE = "Apache-2.0 (as stated upstream)"
_STATED_LICENCES = {_UPSTREAM_LICENCE.casefold(), "apache-2.0", "mit"}


@dataclass(frozen=True)
class Exclusion:
    kind: str
    url: str
    reason: str


@dataclass(frozen=True)
class Addon:
    id: str
    kind: str
    repo: str
    revision: str
    author: str
    licence: str
    opt_in: bool
    gate: str
    files: dict[str, str]
    sizes: dict[str, int]
    sources: tuple[Source, ...]


@dataclass(frozen=True)
class AddonCatalog:
    exclusions: tuple[Exclusion, ...]
    addons: tuple[Addon, ...]

    def by_id(self, addon_id: str) -> Addon:
        for addon in self.addons:
            if addon.id == addon_id:
                return addon
        raise RegistryError(f"unknown add-on {addon_id}")


def _normalise(url: str) -> str:
    text = url.strip().casefold()
    for prefix in ("https://", "http://"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.startswith("www."):
        text = text[4:]
    return text.rstrip("/")


def _excluded(blob: str, exclusions: tuple[Exclusion, ...]) -> bool:
    folded = blob.casefold()
    for item in exclusions:
        needle = _normalise(item.url)
        if needle and needle in folded:
            return True
    return False


def _addon_blob(row: dict[str, object], sources: tuple[Source, ...]) -> str:
    parts = [str(row.get("id") or ""), str(row.get("repo") or ""), str(row.get("author") or "")]
    parts.extend(source.uri for source in sources)
    return " ".join(parts)


def _check_sources(sources: tuple[Source, ...]) -> None:
    if any(source.kind in _MIRROR_KINDS for source in sources):
        raise RegistryError("a third-party add-on cannot use a mirror")
    if not sources or any(source.kind != "upstream" or not source.uri for source in sources):
        raise RegistryError("a third-party add-on is downloaded only from upstream")


def _sources(row: dict[str, object]) -> tuple[Source, ...]:
    raw = row.get("sources")
    if not isinstance(raw, list):
        return ()
    built: list[Source] = []
    for item in raw:
        if isinstance(item, dict):
            built.append(Source(str(item.get("kind") or ""), str(item.get("uri") or "")))
    return tuple(built)


def _files(row: dict[str, object]) -> tuple[dict[str, str], dict[str, int]]:
    raw = row.get("files")
    files: dict[str, str] = {}
    sizes: dict[str, int] = {}
    if not isinstance(raw, dict):
        return files, sizes
    for name, meta in raw.items():
        if isinstance(meta, dict):
            digest = str(meta.get("sha256") or "")
            sizes[str(name)] = int(meta.get("size") or 0)
        else:
            digest = str(meta or "")
        try:
            files[str(name)] = require_pin(digest)
        except PinError as exc:
            raise RegistryError("a third-party file is unpinned") from exc
    return files, sizes


def _parse_addon(row: dict[str, object], exclusions: tuple[Exclusion, ...]) -> Addon:
    sources = _sources(row)
    _check_sources(sources)
    if _excluded(_addon_blob(row, sources), exclusions):
        raise RegistryError("a listed add-on is excluded")
    kind = str(row.get("kind") or "")
    if kind != "model":
        raise RegistryError("a third-party add-on has an unknown kind")
    if row.get("opt_in") is not True:
        raise RegistryError("a third-party add-on must stay opt-in")
    if str(row.get("gate") or "") != "always":
        raise RegistryError("the content checks stay on")
    licence = str(row.get("licence") or "")
    if licence.casefold() not in _STATED_LICENCES:
        raise RegistryError("a third-party add-on has an unlisted licence")
    revision = str(row.get("revision") or "")
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise RegistryError("a third-party add-on revision is not a full hash")
    files, sizes = _files(row)
    if not files:
        raise RegistryError("a third-party add-on has no files")
    author = str(row.get("author") or "")
    repo = str(row.get("repo") or "")
    if not author or not repo:
        raise RegistryError("a third-party add-on is missing its author")
    return Addon(
        id=str(row["id"]),
        kind=kind,
        repo=repo,
        revision=revision,
        author=author,
        licence=licence,
        opt_in=True,
        gate="always",
        files=files,
        sizes=sizes,
        sources=sources,
    )


def load_addons(path: Path | None = None) -> AddonCatalog:
    """Read the curated list. An excluded URL or a mirror source does not load."""
    raw = json.loads((path or _CATALOG).read_text(encoding="utf-8"))
    exclusions = tuple(
        Exclusion(str(item.get("kind") or "model"), str(item["url"]), str(item.get("reason") or ""))
        for item in raw.get("exclusions") or []
        if isinstance(item, dict) and item.get("url")
    )
    if not exclusions:
        raise RegistryError("the exclusion list is empty")
    addons = tuple(_parse_addon(row, exclusions) for row in raw.get("addons") or [] if isinstance(row, dict))
    return AddonCatalog(exclusions, addons)


def release_addon_request(gate: AgeGate, prompt: str, mode: str = "text") -> ScreenedInput:
    """Screen one add-on request. There is no argument that turns the checks off."""
    return gate.release_request(prompt, mode, ())


def download_addon(addon: Addon, filename: str, dest_dir: Path, transport, *, user_initiated: bool = False) -> Path:
    """Fetch one pinned file from upstream after the user asks. Mirrors are not contacted."""
    if not user_initiated:
        raise DownloadError("not-requested")
    if addon.gate != "always":
        raise RegistryError("the content checks stay on")
    try:
        _check_sources(addon.sources)
    except RegistryError as exc:
        raise DownloadError("mirror") from exc
    catalog = load_addons()
    blob = " ".join((addon.repo, addon.id, *(source.uri for source in addon.sources)))
    if _excluded(blob, catalog.exclusions):
        raise DownloadError("excluded")
    expected = addon.files.get(filename, "")
    try:
        require_pin(expected)
    except PinError:
        raise DownloadError("unpinned") from None
    upstream = addon.sources[0]
    file_source = Source("upstream", f"{upstream.uri.rstrip('/')}/{filename}")
    try:
        data = transport(file_source)
        verify_bytes(data, expected)
    except (OSError, PinError, DownloadError) as exc:
        raise DownloadError(str(exc)) from exc
    folder = dest_dir / addon.repo.replace("/", "--")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
