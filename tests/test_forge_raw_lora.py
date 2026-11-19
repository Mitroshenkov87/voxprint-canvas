"""Raw Forge / PEFT safetensors LoRAs must open; family routing stays unchanged."""
from __future__ import annotations

import json
from pathlib import Path

from canvas.models.user_lora import (
    FAMILY_TURBO,
    FAMILY_ZIMAGE,
    THIRD_PARTY_KIND,
    _peft_key,
    detect_base,
    lora_accepts_family,
    open_weights,
)
from tests.lora_files import write_safetensors

# A real Z-Image transformer module, so fits_base would accept the stem.
_STEM = "layers.1.attention.to_q"
_TRAINER = {
    f"{_STEM}.lora_A.weight": b"\x01\x02\x03\x04",
    f"{_STEM}.lora_B.weight": b"\x05\x06\x07\x08",
}
_PEFT = {
    f"base_model.model.{_STEM}.lora_A.default.weight": b"\x01\x02\x03\x04",
    f"base_model.model.{_STEM}.lora_B.default.weight": b"\x05\x06\x07\x08",
}


def _write_mixed_metadata(path: Path, tensors: dict[str, bytes], metadata: dict[str, object]) -> None:
    """Like write_safetensors, but __metadata__ may contain non-string values."""
    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    parts: list[bytes] = []
    for name in sorted(tensors):
        raw = tensors[name]
        header[name] = {"dtype": "U8", "shape": [len(raw)], "data_offsets": [offset, offset + len(raw)]}
        parts.append(raw)
        offset += len(raw)
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(payload).to_bytes(8, "little") + payload + b"".join(parts))


def test_peft_key_strips_base_model_and_default() -> None:
    assert _peft_key(f"base_model.model.{_STEM}.lora_A.default.weight") == f"{_STEM}.lora_A.weight"
    assert _peft_key(f"base_model.model.{_STEM}.lora_B.default.weight") == f"{_STEM}.lora_B.weight"
    assert _peft_key(f"{_STEM}.lora_A.weight") == f"{_STEM}.lora_A.weight"


def test_a_raw_forge_checkpoint_without_kind_opens(tmp_path: Path) -> None:
    """Trainer files have voxprint.lora.base and PEFT names, but no kind (V16 step-000240)."""
    path = tmp_path / "step-000240.safetensors"
    write_safetensors(path, _TRAINER, {"voxprint.lora.base": "z-image"})
    opened = open_weights(path)
    assert opened.base in {FAMILY_ZIMAGE, "z-image"}
    assert opened.kind == THIRD_PARTY_KIND
    assert detect_base(path) == FAMILY_ZIMAGE
    assert lora_accepts_family(opened, FAMILY_TURBO)


def test_a_peft_prefixed_lora_with_non_string_metadata_opens(tmp_path: Path) -> None:
    """PEFT keys plus a non-string header value must be ignored, not rejected."""
    path = tmp_path / "adapter.safetensors"
    _write_mixed_metadata(
        path,
        _PEFT,
        {
            "voxprint.lora.base": "z-image",
            "voxprint.lora.compatible_bases": '["z-image","z-image-turbo"]',
            "step": 240,
        },
    )
    opened = open_weights(path)
    assert opened.base in {FAMILY_ZIMAGE, "z-image"}
    assert lora_accepts_family(opened, FAMILY_TURBO)
    assert lora_accepts_family(opened, FAMILY_ZIMAGE)
