"""Opt-in Chroma, one exclusion, and no mirror source on a third-party entry."""
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from canvas.models.addons import download_addon, load_addons, release_addon_request
from canvas.models.downloader import DownloadError
from canvas.models.registry import RegistryError, Source
from tests.fakes import World

REVISION = "0e0c60ece1e82b17cb7f77342d765ba5024c40c0"
CHECKPOINT = "d446d9695d08276f61e53653e025289dd96f7a489c27982a1fa54ecabc06642a"
EXCLUSION = "https://civarchive.com/models/2299623"


def _rewrite(tmp_path: Path, mutate) -> Path:
    raw = json.loads(Path("canvas/models/third_party.json").read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "third_party.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_chroma_is_curated_and_the_exclusion_list_has_one_entry():
    catalog = load_addons()
    assert [item.url for item in catalog.exclusions] == [EXCLUSION]
    assert catalog.exclusions[0].kind == "model"
    assert catalog.exclusions[0].reason == "advertises expanded ages"
    blob = " ".join(item.url.casefold() for item in catalog.exclusions)
    assert "chroma" not in blob and "lodestones" not in blob
    chroma = catalog.by_id("chroma1-hd")
    assert chroma.kind == "model"
    assert chroma.repo == "lodestones/Chroma1-HD"
    assert chroma.author == "lodestones"
    assert chroma.revision == REVISION and chroma.revision.startswith("0e0c60ec")
    assert chroma.licence == "Apache-2.0 (as stated upstream)"
    assert chroma.opt_in is True and chroma.gate == "always"
    assert [source.kind for source in chroma.sources] == ["upstream"]
    assert chroma.sources[0].uri == f"https://huggingface.co/lodestones/Chroma1-HD/resolve/{REVISION}"
    assert chroma.files["Chroma1-HD.safetensors"] == CHECKPOINT
    assert chroma.sizes["Chroma1-HD.safetensors"] == 17800038288
    assert len(chroma.files) == 6


@pytest.mark.parametrize("kind", ["huggingface-mirror", "kaggle", "torrent"])
def test_no_mirror_source_can_be_attached_to_a_third_party_entry(tmp_path: Path, kind: str):
    def attach(raw: dict) -> None:
        raw["addons"][0]["sources"].append({"kind": kind, "uri": "https://example.invalid/mirror"})

    with pytest.raises(RegistryError, match="mirror"):
        load_addons(_rewrite(tmp_path, attach))
    chroma = load_addons().by_id("chroma1-hd")
    bad = replace(chroma, sources=chroma.sources + (Source(kind, "https://example.invalid/mirror"),))
    called: list[str] = []

    def transport(source: Source) -> bytes:
        called.append(source.kind)
        return b""

    with pytest.raises(DownloadError) as caught:
        download_addon(bad, "Chroma1-HD.safetensors", tmp_path, transport, user_initiated=True)
    assert caught.value.detail == "mirror"
    assert called == []


def test_an_excluded_url_cannot_be_added(tmp_path: Path):
    digest = "ab" * 32

    def add(raw: dict) -> None:
        raw["addons"].append(
            {
                "id": "blocked-lora",
                "kind": "model",
                "repo": "example/weights",
                "revision": "a" * 40,
                "author": "example",
                "licence": "Apache-2.0 (as stated upstream)",
                "opt_in": True,
                "gate": "always",
                "files": {"weights.safetensors": {"sha256": digest, "size": 1}},
                "sources": [{"kind": "upstream", "uri": EXCLUSION}],
            }
        )

    with pytest.raises(RegistryError, match="excluded"):
        load_addons(_rewrite(tmp_path, add))


def test_the_gate_cannot_be_turned_off(tmp_path: Path):
    with pytest.raises(RegistryError, match="content checks"):
        load_addons(_rewrite(tmp_path, lambda raw: raw["addons"][0].__setitem__("gate", "off")))
    world = World()
    params = inspect.signature(release_addon_request).parameters
    assert "skip" not in params and "bypass" not in params
    screened = release_addon_request(world.gate(), "a red boat")
    assert "guard" in world.calls
    assert screened.prompt == "a red boat"


def test_download_is_user_initiated_and_upstream_only(tmp_path: Path):
    chroma = load_addons().by_id("chroma1-hd")
    called: list[str] = []

    def transport(source: Source) -> bytes:
        called.append(source.uri)
        return b""

    with pytest.raises(DownloadError) as caught:
        download_addon(chroma, "Chroma1-HD.safetensors", tmp_path, transport)
    assert caught.value.detail == "not-requested"
    assert called == []
    payload = b"checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    pinned = replace(chroma, files={"Chroma1-HD.safetensors": digest}, sizes={})

    def fetch(source: Source) -> bytes:
        called.append(source.uri)
        return payload

    path = download_addon(pinned, "Chroma1-HD.safetensors", tmp_path, fetch, user_initiated=True)
    assert path.read_bytes() == payload
    assert called == [f"https://huggingface.co/lodestones/Chroma1-HD/resolve/{REVISION}/Chroma1-HD.safetensors"]
    assert "hf-mirror" not in called[0]
