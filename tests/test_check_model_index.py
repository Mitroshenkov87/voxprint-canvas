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


def _files(entry, predicate=None):
    return {
        f["path"]: {"path": f["path"], "size": f["size"], "sha256": f["sha256"]}
        for f in entry["files"]
        if predicate is None or predicate(f["path"])
    }


def _row(index_id, files, prebuilt=None, engine="diffusers"):
    # Signed descriptors declare runtime.engine "diffusers": the index schema cannot express transformers yet.
    blob = json.dumps({
        "files": list(files.values()), "prebuilt": prebuilt or [], "runtime": {"engine": engine},
    }).encode()
    return blob, {"id": index_id, "repo": "r", "revision": "v", "sha256": hashlib.sha256(blob).hexdigest()}


def _world(manifest):
    ours = {e["id"]: e for e in manifest["models"]}
    descriptors, rows = {}, []
    for index_id, ids in tool.CANVAS_IDS.items():
        files = {}
        for model_id in ids:
            files.update(_files(ours[model_id]))
        prebuilt = []
        if f"{ids[0]}-upstream" in ours:
            prebuilt = [{"role": "fallback", "files": list(_files(
                ours[f"{ids[0]}-upstream"], tool._transformer_path,
            ).values())}]
        blob, row = _row(index_id, files, prebuilt)
        descriptors[index_id] = blob
        rows.append(row)
    # Shared FP32 default: TE/VAE/tokenizer/scheduler plus upstream transformers.
    shared = {}
    pack0 = next(iter(tool.TURBO_PACKS))
    shared.update(_files(ours[pack0], tool._shared_path))
    if f"{pack0}-upstream" in ours:
        shared.update(_files(ours[f"{pack0}-upstream"], tool._transformer_path))
    blob, row = _row(tool.SHARED_INDEX_ID, shared)
    descriptors[tool.SHARED_INDEX_ID] = blob
    rows.append(row)
    for pack_id, index_id in tool.TURBO_PACKS.items():
        blob, row = _row(index_id, _files(ours[pack_id], tool._transformer_path))
        descriptors[index_id] = blob
        rows.append(row)
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


def test_a_bf16_transformer_mismatch_is_reported_against_its_own_row():
    changed = copy.deepcopy(MANIFEST)
    entry = next(e for e in changed["models"] if e["id"] == "z-image-turbo-bf16-16gb")
    item = next(f for f in entry["files"] if f["path"].startswith("transformer/") and f["path"].endswith(".safetensors"))
    item["sha256"] = "0" * 64
    problems = _run(changed)
    assert any(
        p.startswith("z-image-turbo-bf16-16gb/") and p.endswith("differs from z-image-turbo-bf16-16gb")
        for p in problems
    )
    assert not any("differs from z-image-turbo" == p.rsplit(" ", 1)[-1] and "/transformer/" in p for p in problems)


def test_a_shared_text_encoder_mismatch_is_reported_against_z_image_turbo():
    changed = copy.deepcopy(MANIFEST)
    entry = next(e for e in changed["models"] if e["id"] == "z-image-turbo-bf16")
    item = next(f for f in entry["files"] if f["path"].startswith("text_encoder/"))
    item["sha256"] = "0" * 64
    problems = _run(changed)
    assert any(
        p.startswith("z-image-turbo-bf16/text_encoder/") and p.endswith("differs from z-image-turbo")
        for p in problems
    )


def test_an_upstream_transformer_is_checked_against_z_image_turbo():
    changed = copy.deepcopy(MANIFEST)
    entry = next(e for e in changed["models"] if e["id"] == "z-image-turbo-bf16-16gb-upstream")
    item = next(f for f in entry["files"] if f["path"].startswith("transformer/") and f["path"].endswith(".safetensors"))
    item["sha256"] = "0" * 64
    problems = _run(changed)
    assert any(
        p.startswith("z-image-turbo-bf16-16gb-upstream/") and p.endswith("differs from z-image-turbo")
        for p in problems
    )


def _with_engine(index_id, engine):
    index, descriptors = _world(MANIFEST)
    descriptors = dict(descriptors)
    document = json.loads(descriptors[index_id])
    document["runtime"] = {"engine": engine}
    blob = json.dumps(document).encode()
    rows = json.loads(index)
    for row in rows["models"]:
        if row["id"] == index_id:
            row["sha256"] = hashlib.sha256(blob).hexdigest()
    descriptors[index_id] = blob
    return json.dumps(rows).encode(), descriptors


def test_the_writer_row_is_checked_against_the_signed_descriptor():
    assert tool.CANVAS_IDS["qwen3-4b-writer"] == ("qwen3-4b-writer",)
    changed = copy.deepcopy(MANIFEST)
    entry = next(e for e in changed["models"] if e["id"] == "qwen3-4b-writer")
    entry["files"][0]["sha256"] = "0" * 64
    entry["files"][1]["size"] += 1
    problems = _run(changed)
    assert len([p for p in problems if p.startswith("qwen3-4b-writer/")]) == 2
    missing = copy.deepcopy(MANIFEST)
    missing["models"] = [e for e in missing["models"] if e["id"] != "qwen3-4b-writer"]
    assert "qwen3-4b-writer: missing from mdl.json" in _run(missing)


def test_a_missing_writer_row_in_the_index_is_reported():
    index, descriptors = _world(MANIFEST)
    rows = json.loads(index)
    rows["models"] = [row for row in rows["models"] if row["id"] != "qwen3-4b-writer"]
    problems = _run(MANIFEST, index=json.dumps(rows).encode(), descriptors=descriptors)
    assert "qwen3-4b-writer: not in the index" in problems


def test_the_writer_may_declare_diffusers_because_the_schema_cannot_express_transformers():
    # Canvas loads the writer with transformers (QwenRewriter) whatever the descriptor says.
    assert tool.ENGINE_ALLOWED == {"qwen3-4b-writer": frozenset({"diffusers", "transformers"})}
    assert "transformers" in (ROOT / "canvas" / "pipeline" / "enhance.py").read_text(encoding="utf-8")
    for engine in ("diffusers", "transformers"):
        index, descriptors = _with_engine("qwen3-4b-writer", engine)
        assert _run(MANIFEST, index=index, descriptors=descriptors) == []


def test_the_writer_may_not_declare_any_other_engine():
    index, descriptors = _with_engine("qwen3-4b-writer", "onnx")
    assert _run(MANIFEST, index=index, descriptors=descriptors) == [
        "qwen3-4b-writer: descriptor runtime.engine is 'onnx'"
    ]


def test_no_other_id_may_declare_a_different_engine():
    for index_id in ("mirror-qwen3guard-gen-0.6b", "yunet-2023mar", "z-image-turbo", "z-image-turbo-bf16"):
        index, descriptors = _with_engine(index_id, "transformers")
        problems = _run(MANIFEST, index=index, descriptors=descriptors)
        assert f"{index_id}: descriptor runtime.engine is 'transformers'" in problems, index_id


def test_a_descriptor_without_a_runtime_block_is_reported():
    index, descriptors = _world(MANIFEST)
    document = json.loads(descriptors["yunet-2023mar"])
    del document["runtime"]
    blob = json.dumps(document).encode()
    rows = json.loads(index)
    for row in rows["models"]:
        if row["id"] == "yunet-2023mar":
            row["sha256"] = hashlib.sha256(blob).hexdigest()
    descriptors["yunet-2023mar"] = blob
    assert _run(MANIFEST, index=json.dumps(rows).encode(), descriptors=descriptors) == [
        "yunet-2023mar: descriptor runtime.engine is None"
    ]


def test_the_default_floor_is_serial_13_or_the_bundled_serial_whichever_is_higher(monkeypatch):
    assert tool.CANVAS_MIN_SERIAL == 13
    index, descriptors = _world(MANIFEST)
    files = {"index.json": index, "index.json.sig": sign(index, SEED)}
    keys = (public_key(SEED),)

    def run():
        return tool.check(
            lambda path: files[path], lambda row: descriptors[row["id"]], MANIFEST,
            verify=lambda payload, sig: verify_index(payload, sig, keys),
        )

    # The fake index is serial 9: refused by the default floor even if the bundled copy were older.
    monkeypatch.setattr(tool, "_bundled_serial", lambda: 5)
    assert run() == [f"index serial is {SERIAL}, older than the bundled serial 13"]
    monkeypatch.setattr(tool, "_bundled_serial", lambda: 20)
    assert run() == [f"index serial is {SERIAL}, older than the bundled serial 20"]
