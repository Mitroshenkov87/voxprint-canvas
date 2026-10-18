"""tools/check_model_index.py, offline: a fake index built from mdl.json itself."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

from voxprint_common.mdl.ed25519 import public_key, sign
from voxprint_common.mdl.index import verify_index

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_model_index", ROOT / "tools" / "check_model_index.py")
assert spec and spec.loader
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

MANIFEST = json.loads((ROOT / "mdl.json").read_text(encoding="utf-8"))
SEED = bytes(range(32))
SERIAL = 9


def _world(manifest):
    ours = {e["id"]: e for e in manifest["models"]}
    descriptors, rows = {}, []
    for index_id, ids in tool.CANVAS_IDS.items():
        files = {f["path"]: {"path": f["path"], "size": f["size"], "sha256": f["sha256"]}
                 for model_id in ids for f in ours[model_id]["files"]}
        prebuilt = []
        if f"{ids[0]}-upstream" in ours:
            prebuilt = [{"role": "fallback", "files": [
                {"path": f["path"], "size": f["size"], "sha256": f["sha256"]}
                for f in ours[f"{ids[0]}-upstream"]["files"] if f["path"].startswith("transformer/")]}]
        blob = json.dumps({"files": list(files.values()), "prebuilt": prebuilt}).encode()
        descriptors[index_id] = blob
        rows.append({"id": index_id, "repo": "r", "revision": "v", "sha256": hashlib.sha256(blob).hexdigest()})
    index = json.dumps({"serial": SERIAL, "models": rows}).encode()
    return index, descriptors


def _run(manifest, *, index=None, descriptors=None, signature=None, min_serial=SERIAL):
    built_index, built_desc = _world(MANIFEST)
    index = built_index if index is None else index
    descriptors = built_desc if descriptors is None else descriptors
    signature = sign(index, SEED) if signature is None else signature
    files = {"index.json": index, "index.json.sig": signature}
    keys = (public_key(SEED),)
    return tool.check(
        lambda path: files[path], lambda row: descriptors[row["id"]], manifest,
        verify=lambda payload, sig: verify_index(payload, sig, keys), min_serial=min_serial,
    )


def test_the_committed_manifest_matches_a_matching_index():
    assert _run(MANIFEST) == []


def test_a_changed_file_hash_is_reported():
    changed = copy.deepcopy(MANIFEST)
    entry = next(e for e in changed["models"] if e["id"] == "nsfw-image-detection")
    entry["files"][0]["sha256"] = "0" * 64
    problems = _run(changed)
    assert any(p.startswith("nsfw-image-detection/") for p in problems)


def test_an_unsigned_or_forged_index_is_refused():
    assert _run(MANIFEST, signature=b"")[0].startswith("index.json signature")
    assert _run(MANIFEST, signature=sign(b"other", SEED))[0].startswith("index.json signature")


def test_the_bundled_forge_key_refuses_a_foreign_signature():
    index, _ = _world(MANIFEST)
    assert not tool._verify_bundled(index, sign(index, SEED))


def test_an_older_serial_is_refused():
    problems = _run(MANIFEST, min_serial=SERIAL + 1)
    assert problems == [f"index serial is {SERIAL}, older than the bundled serial {SERIAL + 1}"]


def test_a_descriptor_that_does_not_match_the_index_is_refused():
    _, descriptors = _world(MANIFEST)
    descriptors["yunet-2023mar"] = b'{"files": []}'
    problems = _run(MANIFEST, descriptors=descriptors)
    assert problems == ["yunet-2023mar: descriptor does not match the index sha256"]


def test_the_fp32_fallback_stays_in_the_manifest():
    ids = {e["id"] for e in MANIFEST["models"]}
    assert {"z-image-turbo-bf16-16gb-upstream", "z-image-turbo-bf16-upstream"} <= ids
