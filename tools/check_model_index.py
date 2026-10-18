"""Check mdl.json against the signed VOXPrint model index (Model Forge).

The index is fetched with its detached ``index.json.sig`` and accepted only when
the Ed25519 signature verifies against the Model Forge key bundled in
voxprint-common 0.3.1 and its serial is not older than the index bundled there
(no rollback). Each descriptor is then checked against the signed index digest,
and every file in mdl.json must match its descriptor by path, size and SHA-256.
mdl.json keeps those explicit hashes. In the FP32 ``-upstream`` fallback rows, the
transformer files are matched against the descriptor's ``prebuilt`` fallback.

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
    "mirror-z-image-turbo": ("z-image-turbo-bf16-16gb", "z-image-turbo-bf16"),
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
        row = rows.get(index_id)
        if row is None:
            problems.append(f"{index_id}: not in the index")
            continue
        blob = fetch_descriptor(row)
        if _sha(blob) != row["sha256"]:
            problems.append(f"{index_id}: descriptor does not match the index sha256")
            continue
        descriptor = json.loads(blob.decode("utf-8"))
        files = {f["path"]: f for f in descriptor["files"]}
        fallback = {f["path"]: f for p in descriptor.get("prebuilt", []) if p.get("role") == "fallback"
                    for f in p["files"]}
        for model_id in model_ids:
            # The -upstream row: the author's transformer comes from the prebuilt fallback,
            # the other files are the same bytes as the default set.
            for entry_id, table in ((model_id, files), (f"{model_id}-upstream", {**files, **fallback})):
                entry = ours.get(entry_id)
                if entry is None:
                    if entry_id == model_id:
                        problems.append(f"{model_id}: missing from mdl.json")
                    continue
                for item in entry["files"]:
                    want = table.get(item["path"])
                    if want is None:
                        problems.append(f"{entry_id}/{item['path']}: not in descriptor {index_id}")
                    elif (want["sha256"], int(want["size"])) != (item["sha256"], int(item["size"])):
                        problems.append(f"{entry_id}/{item['path']}: sha256 or size differs from {index_id}")
    return problems


def main() -> int:
    problems = check()
    for line in problems:
        print(line, file=sys.stderr)
    if not problems:
        print("mdl.json matches the signed model index")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
