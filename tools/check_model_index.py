"""Check mdl.json against the signed VOXPrint model index (Model Forge).

mdl.json keeps explicit per-file hashes because the voxprint-common 0.2.4/0.2.5
MDL client cannot reference index ids: it reads only a bundled list-shaped
index, verifies no signature, and ignores descriptor ``prebuilt`` fallbacks.
This check pins one index commit and its digest, verifies each descriptor
against the index, and requires every file in mdl.json to match the descriptor
by path, size and SHA-256. In the FP32 ``-upstream`` fallback rows, the
transformer files are matched against the descriptor's ``prebuilt`` fallback.

    python tools/check_model_index.py          # network: huggingface.co
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_REPO = "Mitroshenkov87/voxprint-model-index"
INDEX_COMMIT = "e648da65869590749494569aeed77ea06cf75281"  # serial 9
INDEX_SHA256 = "104742f9d8cc7c8f693f963dbd871471a3e57fda3d585d52b105b3855d49f45f"
INDEX_SERIAL = 9

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


def _dataset(path: str) -> bytes:
    return _get(f"https://huggingface.co/datasets/{INDEX_REPO}/resolve/{INDEX_COMMIT}/{path}")


def _descriptor(row: dict[str, str]) -> bytes:
    repo, revision = row["repo"], row["revision"]
    if repo == f"datasets/{INDEX_REPO}":
        return _dataset(f"descriptors/{row['id']}/voxprint-model.json")
    return _get(f"https://huggingface.co/{repo}/resolve/{revision}/voxprint-model.json")


def check(fetch_index=_dataset, fetch_descriptor=_descriptor, manifest: dict | None = None) -> list[str]:
    """Return the problems found; empty when mdl.json matches the index."""
    problems: list[str] = []
    raw = fetch_index("index.json")
    if _sha(raw) != INDEX_SHA256:
        return [f"index.json at {INDEX_COMMIT[:8]} does not hash to the pinned digest"]
    index = json.loads(raw.decode("utf-8"))
    if index.get("serial") != INDEX_SERIAL:
        problems.append(f"index serial is {index.get('serial')}, expected {INDEX_SERIAL}")
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
        print(f"mdl.json matches model index serial {INDEX_SERIAL} ({INDEX_COMMIT[:8]})")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
