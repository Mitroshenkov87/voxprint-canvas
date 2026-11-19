#!/usr/bin/env python3
"""Sign a LoRA so Canvas may skip the per-frame output age check for it.

Forge runs this after its own adult-verification pass on a LoRA's training
set. It writes ``<weights>.voxprint-verify.json`` next to the weights file,
signed with this install's shared key (``canvas.lora_verify.key_path()``).
Canvas trusts the record only when the signature checks out *and* the recorded
sha256 matches the exact weights file it loads, so a renamed, edited, or merged
LoRA has no valid record and keeps the full checks.

This is the reference signer. It reuses ``canvas.lora_verify`` so the signing
scheme stays in one place; the ``SCHEMA`` section of the design note documents
the record format for a from-scratch reimplementation.

Usage:
    forge_sign_lora.py WEIGHTS.safetensors \
        --project-id zhenya-belaya-20261114-2202 \
        --dataset-image-count 42 \
        --age-gate-version 0.1.0-rc.45 \
        [--record-id forge-...] [--verified-at ISO8601] [--result pass]

Exit codes: 0 wrote a record, 2 bad arguments or the weights file is missing.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Allow running straight from a checkout without an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from canvas.lora_verify import (  # noqa: E402
    ALGORITHM,
    SCHEMA_ID,
    file_sha256,
    key_path,
    record_path_for,
    sign_record,
    verify_lora,
)


def build_record(args: argparse.Namespace, weights: Path) -> dict:
    """Assemble the record dict, including the signature over the signed fields."""
    record = {
        "schema": SCHEMA_ID,
        "record_id": args.record_id or f"forge-{uuid.uuid4()}",
        "lora_file": weights.name,
        "lora_sha256": file_sha256(weights),
        "project_id": args.project_id,
        "dataset_image_count": int(args.dataset_image_count),
        "age_gate_version": args.age_gate_version,
        "verified": True,
        "result": args.result,
        "verified_at": args.verified_at or datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "algorithm": ALGORITHM,
    }
    record["signature"] = sign_record(record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sign a LoRA verification record for Canvas.")
    parser.add_argument("weights", type=Path, help="Path to the LoRA .safetensors file.")
    parser.add_argument("--project-id", required=True, help="Forge project id for the training run.")
    parser.add_argument("--dataset-image-count", type=int, required=True, help="Number of verified training images.")
    parser.add_argument("--age-gate-version", required=True, help="Age-gate version that verified the set.")
    parser.add_argument("--record-id", default="", help="Record id; a forge-<uuid> is generated when omitted.")
    parser.add_argument("--verified-at", default="", help="ISO 8601 timestamp; now when omitted.")
    parser.add_argument("--result", default="pass", help="Verification result; only 'pass' is trusted by Canvas.")
    args = parser.parse_args(argv)

    weights = args.weights
    if not weights.is_file():
        print(f"weights file not found: {weights}", file=sys.stderr)
        return 2

    record = build_record(args, weights)
    out_path = record_path_for(weights)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    check = verify_lora(weights)
    if not check.verified:
        print(f"wrote {out_path} but it did not verify: {check.reason}", file=sys.stderr)
        return 2
    print(f"wrote {out_path.name} (key {key_path()}, record {record['record_id']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
