"""Chroma is opt-in. One CivitAI model stays excluded. Mirrors cannot be attached."""
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from canvas.models.addons import download_addon, load_addons, refuse_excluded, release_addon_request
from canvas.models.downloader import DownloadError
from canvas.models.excluded_weights import EXCLUDED_MODEL_IDS, EXCLUDED_SHA256
from canvas.models.registry import RegistryError
from canvas.safety.constants import APPARENT_AGE_UNCERTAIN_UNDER, SEXUAL_SCORE_BLOCK_AT
from canvas.safety.classifiers import Face
from canvas.safety.gate import GenerationBlocked, SafetyUnavailable
from canvas.safety.inputs import ImageRole, PendingInput
from tests.fakes import World, rgb

REVISION = "0e0c60ece1e82b17cb7f77342d765ba5024c40c0"
CHROMA_SHA = "d446d9695d08276f61e53653e025289dd96f7a489c27982a1fa54ecabc06642a"
EXCLUDED_URL = "https://civarchive.com/models/2299623"


def _entry(**overrides) -> dict:
    payload = b"lora-bytes"
    row = {
        "id": "example-lora",
        "kind": "lora",
        "base": "Z-Image-Turbo",
        "upstream": "https://huggingface.co/example/lora/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/weights.safetensors",
        "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "licence": "MIT (as stated upstream)",
        "author": "example",
    }
    row.update(overrides)
    return row


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "third_party.json"
    path.write_text(json.dumps({"addons": rows}), encoding="utf-8")
    return path


def test_chroma_is_curated_and_one_civitai_model_is_excluded():
    catalog = load_addons()
    assert [addon.id for addon in catalog.addons] == ["chroma1-hd"]
    chroma = catalog.by_id("chroma1-hd")
    assert chroma.kind == "model" and chroma.base == ""
    assert chroma.author == "lodestones"
    assert chroma.licence == "Apache-2.0 (as stated upstream)"
    assert chroma.revision == REVISION and chroma.revision.startswith("0e0c60ec")
    assert chroma.upstream == f"https://huggingface.co/lodestones/Chroma1-HD/resolve/{REVISION}"
    assert chroma.files["Chroma1-HD.safetensors"] == CHROMA_SHA
    assert chroma.sha256 == CHROMA_SHA
    assert len(chroma.files) == 6
    assert EXCLUDED_MODEL_IDS == frozenset({"2299623"})
    assert CHROMA_SHA not in EXCLUDED_SHA256
    assert "03930710409f6cf19c2cf543d76f522feab9d7197a9a6f98e854561912fabf7e" in EXCLUDED_SHA256
    assert "3b13d8cd7cc3fcfe86725b0ad01ee5acf7f5cefeedd747a557d321520af2e01f" in EXCLUDED_SHA256
    with pytest.raises(RegistryError, match="excluded"):
        refuse_excluded(EXCLUDED_URL, "ab" * 32)
    with pytest.raises(RegistryError, match="excluded"):
        refuse_excluded("https://civitai.com/models/2299623", "ab" * 32)
    with pytest.raises(RegistryError, match="excluded"):
        refuse_excluded("https://example.com/weights.safetensors", next(iter(EXCLUDED_SHA256)))
    refuse_excluded(chroma.upstream, chroma.sha256)


@pytest.mark.parametrize("kind", ["huggingface-mirror", "kaggle", "torrent"])
def test_a_third_party_entry_can_never_get_a_mirror_source(tmp_path: Path, kind: str):
    row = _entry()
    row["sources"] = [{"kind": kind, "uri": "https://example.invalid/mirror"}]
    with pytest.raises(RegistryError, match="mirror"):
        load_addons(_write(tmp_path, [row]))
    chroma = json.loads(Path("canvas/models/third_party.json").read_text(encoding="utf-8"))
    chroma["addons"][0]["sources"] = [{"kind": kind, "uri": "https://example.invalid/mirror"}]
    with pytest.raises(RegistryError, match="mirror"):
        load_addons(_write(tmp_path, chroma["addons"]))
    with pytest.raises(RegistryError, match="excluded"):
        load_addons(_write(tmp_path, [_entry(upstream=EXCLUDED_URL, sha256="cd" * 32)]))


def test_download_is_opt_in_upstream_and_lands_in_the_third_party_folder(tmp_path: Path):
    payload = b"lora-bytes"
    addon = load_addons(_write(tmp_path, [_entry()])).by_id("example-lora")
    called: list[str] = []

    def transport(url: str) -> bytes:
        called.append(url)
        return payload

    with pytest.raises(DownloadError) as caught:
        download_addon(addon, tmp_path, transport, user_initiated=True)
    assert caught.value.detail == "not-requested"
    assert called == []
    path = download_addon(addon, tmp_path, transport, enabled=True, user_initiated=True)
    assert called == [addon.upstream]
    assert path.read_bytes() == payload
    assert path.parent.parent == tmp_path / "third-party"
    assert "hf-mirror" not in called[0]
    assert not (tmp_path / "weights.safetensors").exists()


def test_a_third_party_lora_cannot_bypass_or_reconfigure_the_gate(tmp_path: Path):
    row = _entry()
    row["gate"] = "off"
    row["apparent_age_under"] = 1
    row["sexual_score_block_at"] = 0.99
    with pytest.raises(RegistryError, match="content checks"):
        load_addons(_write(tmp_path, [row]))
    addon = load_addons(_write(tmp_path, [_entry()])).by_id("example-lora")
    assert addon.kind == "lora"
    assert not hasattr(addon, "gate")
    assert APPARENT_AGE_UNCERTAIN_UNDER == 25
    assert SEXUAL_SCORE_BLOCK_AT == 0.50
    params = inspect.signature(release_addon_request).parameters
    assert "skip" not in params and "bypass" not in params
    world = World()
    world.safety = "safe"
    world.categories = ()
    with pytest.raises(GenerationBlocked):
        release_addon_request(world.gate(), "nude child", "text")
    world.safety = "safe"
    world.categories = ()
    world.text_scores = {label: 0.0 for label in world.text_scores}
    world.nsfw = 0.95
    world.faces = [Face(0, 0, 1, 1)]
    world.ages = [8.0]
    with pytest.raises(GenerationBlocked):
        release_addon_request(world.gate(), "a red boat", "text", (PendingInput(ImageRole.STYLE, rgb()),))
    world.nsfw = 0.0
    world.faces = []
    world.ages = []
    world.error = "guard"
    with pytest.raises(SafetyUnavailable):
        release_addon_request(world.gate(), "a red boat", "text")
    screened = World().gate().release_request("a red boat", "text", ())
    assert screened.prompt == "a red boat"
