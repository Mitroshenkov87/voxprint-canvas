"""tools/check_model_index.py, offline: a fake index built from mdl.json itself."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_model_index", ROOT / "tools" / "check_model_index.py")
assert spec and spec.loader
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

MANIFEST = json.loads((ROOT / "mdl.json").read_text(encoding="utf-8"))


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
    index = json.dumps({"serial": tool.INDEX_SERIAL, "models": rows}).encode()
    return index, descriptors


def _run(manifest, *, index=None, descriptors=None, pin=None, monkeypatch):
    built_index, built_desc = _world(MANIFEST)
    index = built_index if index is None else index
    descriptors = built_desc if descriptors is None else descriptors
    monkeypatch.setattr(tool, "INDEX_SHA256", pin or hashlib.sha256(index).hexdigest())
    return tool.check(lambda _p: index, lambda row: descriptors[row["id"]], manifest)


def test_the_committed_manifest_matches_a_matching_index(monkeypatch):
    assert _run(MANIFEST, monkeypatch=monkeypatch) == []


def test_a_changed_file_hash_is_reported(monkeypatch):
    changed = copy.deepcopy(MANIFEST)
    entry = next(e for e in changed["models"] if e["id"] == "nsfw-image-detection")
    entry["files"][0]["sha256"] = "0" * 64
    problems = _run(changed, monkeypatch=monkeypatch)
    assert any(p.startswith("nsfw-image-detection/") for p in problems)


def test_an_index_that_is_not_the_pinned_one_is_refused(monkeypatch):
    assert _run(MANIFEST, pin="f" * 64, monkeypatch=monkeypatch)[0].startswith("index.json")


def test_a_descriptor_that_does_not_match_the_index_is_refused(monkeypatch):
    _, descriptors = _world(MANIFEST)
    descriptors["yunet-2023mar"] = b'{"files": []}'
    problems = _run(MANIFEST, descriptors=descriptors, monkeypatch=monkeypatch)
    assert problems == ["yunet-2023mar: descriptor does not match the index sha256"]


def test_the_fp32_fallback_stays_in_the_manifest():
    ids = {e["id"] for e in MANIFEST["models"]}
    assert {"z-image-turbo-bf16-16gb-upstream", "z-image-turbo-bf16-upstream"} <= ids
