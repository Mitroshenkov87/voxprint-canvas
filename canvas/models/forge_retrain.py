"""Ask Model Forge to retrain a LoRA onto another base.

The request is a JSON file in the Forge inbox. Nothing is sent over the network.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

MOSS_IMAGE_SFT = "moss-image-sft"


def forge_inbox() -> Path:
    """The directory Forge reads retrain requests from.

    ``VOXPRINT_FORGE_INBOX`` overrides the default ``~/.voxprint/forge/inbox``.
    """
    raw = os.environ.get("VOXPRINT_FORGE_INBOX", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".voxprint" / "forge" / "inbox"


def request_retrain(
    lora_path: Path | str, trained_base: str, target_base: str = MOSS_IMAGE_SFT,
) -> Path:
    """Write one retrain request and return its path.

    The file is written under a temporary name in the inbox, then replaced
    into place, so Forge never sees a half-written request.

    Args:
        lora_path: The LoRA weight file to retrain.
        trained_base: The base that file was trained on.
        target_base: The base Forge should train. MOSS-Image-SFT when omitted.
    """
    folder = forge_inbox()
    folder.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "kind": "retrain",
        "lora": str(lora_path),
        "trained_base": str(trained_base),
        "target_base": str(target_base),
        "created": created,
    }
    destination = folder / f"retrain-{uuid.uuid4().hex}.json"
    temporary = folder / f".{destination.name}.tmp"
    temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination
