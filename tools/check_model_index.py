"""Check mdl.json against the signed VOXPrint model index (Model Forge).

The index is fetched with its detached ``index.json.sig`` and accepted only when
the Ed25519 signature verifies against the Model Forge key bundled in
voxprint-common 0.7.0 and its serial is not older than the index bundled there
(no rollback). Each descriptor is then checked against the signed index digest,
and every file in mdl.json must match a signed descriptor by path, size and SHA-256.
mdl.json keeps those explicit hashes. In the FP32 ``-upstream`` fallback rows, the
transformer files are matched against the ``z-image-turbo`` descriptor (or its
``prebuilt`` fallback). The BF16 packs ``z-image-turbo-bf16`` and
``z-image-turbo-bf16-16gb`` map transformer files to their own index rows; the
shared text encoder, VAE, tokenizer, scheduler and ``model_index.json`` are
checked against the ``z-image-turbo`` row (the FP32 default, pin ``f332072a``).

The 0.7.0 bundled index no longer lists ``moss-image-compressed``. Canvas
must not look that id up as a required pin, must not invent a hash for it,
and must stay absent-safe if a residual reference remains. Canvas treats
Z-Image-Turbo as the default generator; MOSS is an optional download. This
checker still only verifies the ids in ``CANVAS_IDS`` and does not fetch
Z-Image or MOSS weights.

    python tools/check_model_index.py          # network: huggingface.co
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from voxprint_common.mdl.index import load_index, parse_index, verify_index_file

ROOT = Path(__file__).resolve().parents[1]
INDEX_REPO = "Mitroshenkov87/voxprint-model-index"
INDEX_REVISION = "main"

# index id -> mdl.json ids that must match it
CANVAS_IDS: dict[str, tuple[str, ...]] = {
    "mirror-z-image-turbo-controlnet-union-2.1": (
        "z-image-turbo-fun-controlnet-union-bf16-lite",
        "z-image-turbo-fun-controlnet-union-bf16",
    ),
    "mirror-qwen3guard-gen-0.6b": ("qwen3guard-gen-0.6b",),
    "mirror-siglip2-so400m-patch14-384": ("siglip2-so400m-patch14-384",),
    "mirror-fairface-age-image-detection": ("fairface-age",),
    "mirror-nsfw-image-detection": ("nsfw-image-detection",),
    "yunet-2023mar": ("yunet-face",),
}

# Canvas BF16 pack id -> signed index row that covers transformer/* only.
TURBO_PACKS: dict[str, str] = {
    "z-image-turbo-bf16-16gb": "z-image-turbo-bf16-16gb",
    "z-image-turbo-bf16": "z-image-turbo-bf16",
}
SHARED_INDEX_ID = "z-image-turbo"
_SHARED_PREFIXES = ("text_encoder/", "tokenizer/", "vae/", "scheduler/")


def _shared_path(path: str) -> bool:
    """True for TE, VAE, tokenizer, scheduler, and the pipeline index."""
    return path == "model_index.json" or path.startswith(_SHARED_PREFIXES)


def _transformer_path(path: str) -> bool:
    return path.startswith("transformer/")


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "voxprint-canvas-ci"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return bytes(response.read())


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dataset(path: str, revision: str = INDEX_REVISION) -> bytes:
    return _get(f"https://huggingface.co/datasets/{INDEX_REPO}/resolve/{revision}/{path}")


def _descriptor(row: dict[str, str]) -> bytes:
    repo, revision = row["repo"], row["revision"]
    if repo == f"datasets/{INDEX_REPO}":
        return _dataset(f"descriptors/{row['id']}/voxprint-model.json", revision)
    return _get(f"https://huggingface.co/{repo}/resolve/{revision}/voxprint-model.json")


def _verify_bundled(payload: bytes, signature: bytes) -> bool:
    """Verify with the Model Forge key bundled in voxprint-common."""
    with tempfile.TemporaryDirectory() as folder:
        index = Path(folder) / "index.json"
        index.write_bytes(payload)
        (Path(folder) / "index.json.sig").write_bytes(signature)
        return verify_index_file(index)


def _bundled_serial() -> int:
    return load_index(remote=False).serial


def check(
    fetch_index: Callable[[str], bytes] = _dataset,
    fetch_descriptor: Callable[[dict[str, str]], bytes] = _descriptor,
    manifest: dict | None = None,
    *,
    verify: Callable[[bytes, bytes], bool] = _verify_bundled,
    min_serial: int | None = None,
) -> list[str]:
    """Return the problems found; empty when mdl.json matches the signed index."""
    problems: list[str] = []
    raw = fetch_index("index.json")
    try:
        signature = fetch_index("index.json.sig")
    except OSError:
        signature = b""
    if not signature or not verify(raw, signature):
        return ["index.json signature is missing or does not verify with the Model Forge key"]
    index = json.loads(raw.decode("utf-8"))
    parse_index(index)
    floor = _bundled_serial() if min_serial is None else min_serial
    serial = index.get("serial")
    if not isinstance(serial, int) or serial < floor:
        problems.append(f"index serial is {serial}, older than the bundled serial {floor}")
    rows = {row["id"]: row for row in index["models"]}
    manifest = manifest or json.loads((ROOT / "mdl.json").read_text(encoding="utf-8"))
    ours = {entry["id"]: entry for entry in manifest["models"]}
    for index_id, model_ids in CANVAS_IDS.items():
        table = _signed_table(index_id, rows, fetch_descriptor, problems)
        if table is None:
            continue
        files, fallback = table
        for model_id in model_ids:
            # The -upstream row: the author's transformer comes from the prebuilt fallback,
            # the other files are the same bytes as the default set.
            _match_entry(ours, model_id, files, problems, index_id)
            _match_entry(
                ours,
                f"{model_id}-upstream",
                {**files, **fallback},
                problems,
                index_id,
                required=False,
            )
    shared = _signed_table(SHARED_INDEX_ID, rows, fetch_descriptor, problems)
    shared_files = {**shared[0], **shared[1]} if shared is not None else {}
    pack_tables: dict[str, dict[str, dict]] = {}
    for pack_id, index_id in TURBO_PACKS.items():
        table = _signed_table(index_id, rows, fetch_descriptor, problems)
        if table is None:
            continue
        pack_tables[pack_id] = table[0]
    if shared is not None:
        for pack_id, index_id in TURBO_PACKS.items():
            pack_files = pack_tables.get(pack_id)
            if pack_files is None:
                continue
            _match_turbo_pack(ours, pack_id, pack_files, shared_files, problems, index_id)
    return problems


def _signed_table(
    index_id: str,
    rows: dict[str, dict],
    fetch_descriptor: Callable[[dict[str, str]], bytes],
    problems: list[str],
) -> tuple[dict[str, dict], dict[str, dict]] | None:
    """Descriptor file tables for ``index_id``, or None after recording a problem."""
    row = rows.get(index_id)
    if row is None:
        problems.append(f"{index_id}: not in the index")
        return None
    blob = fetch_descriptor(row)
    if _sha(blob) != row["sha256"]:
        problems.append(f"{index_id}: descriptor does not match the index sha256")
        return None
    descriptor = json.loads(blob.decode("utf-8"))
    files = {f["path"]: f for f in descriptor["files"]}
    fallback = {
        f["path"]: f
        for p in descriptor.get("prebuilt", [])
        if p.get("role") == "fallback"
        for f in p["files"]
    }
    return files, fallback


def _match_entry(
    ours: dict,
    entry_id: str,
    table: dict[str, dict],
    problems: list[str],
    index_id: str,
    *,
    required: bool = True,
) -> None:
    entry = ours.get(entry_id)
    if entry is None:
        if required:
            problems.append(f"{entry_id}: missing from mdl.json")
        return
    for item in entry["files"]:
        _match_file(entry_id, item, table, index_id, problems)


def _match_turbo_pack(
    ours: dict,
    pack_id: str,
    pack_files: dict[str, dict],
    shared_files: dict[str, dict],
    problems: list[str],
    pack_index_id: str,
) -> None:
    """Check one BF16 pack: transformer/* on its row, the rest on z-image-turbo."""
    _match_split(ours, pack_id, pack_files, shared_files, problems, pack_index_id, required=True)
    # FP32 fallback transformers live on the shared z-image-turbo row.
    _match_split(
        ours,
        f"{pack_id}-upstream",
        shared_files,
        shared_files,
        problems,
        SHARED_INDEX_ID,
        required=False,
    )


def _match_split(
    ours: dict,
    entry_id: str,
    transformer_table: dict[str, dict],
    shared_table: dict[str, dict],
    problems: list[str],
    transformer_index_id: str,
    *,
    required: bool,
) -> None:
    entry = ours.get(entry_id)
    if entry is None:
        if required:
            problems.append(f"{entry_id}: missing from mdl.json")
        return
    for item in entry["files"]:
        path = item["path"]
        if _transformer_path(path):
            _match_file(entry_id, item, transformer_table, transformer_index_id, problems)
        elif _shared_path(path):
            _match_file(entry_id, item, shared_table, SHARED_INDEX_ID, problems)
        else:
            # Every other path must still match some signed row.
            if path in transformer_table:
                _match_file(entry_id, item, transformer_table, transformer_index_id, problems)
            elif path in shared_table:
                _match_file(entry_id, item, shared_table, SHARED_INDEX_ID, problems)
            else:
                problems.append(f"{entry_id}/{path}: not in descriptor {transformer_index_id} or {SHARED_INDEX_ID}")


def _match_file(
    entry_id: str,
    item: dict,
    table: dict[str, dict],
    index_id: str,
    problems: list[str],
) -> None:
    want = table.get(item["path"])
    if want is None:
        problems.append(f"{entry_id}/{item['path']}: not in descriptor {index_id}")
    elif (want["sha256"], int(want["size"])) != (item["sha256"], int(item["size"])):
        problems.append(f"{entry_id}/{item['path']}: sha256 or size differs from {index_id}")


def main() -> int:
    problems = check()
    for line in problems:
        print(line, file=sys.stderr)
    if not problems:
        print("mdl.json matches the signed model index")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
