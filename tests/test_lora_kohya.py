"""Kohya ``lora_unet_*`` LoRAs and LoRAs made outside Model Forge, in import and on the command line."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from canvas.models.lora_library import import_lora
from canvas.models.user_lora import LoraRejected, _safetensors, prepare_cli_lora
from tests.fakes import World

STEM = "layers.0.attention.to_q"
KOHYA = "lora_unet_layers_0_attention_to_q"


def _write(path: Path, tensors: dict[str, tuple[str, list[int], bytes]], metadata: dict[str, str] | None = None) -> Path:
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    parts = []
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        parts.append(raw)
        offset += len(raw)
    payload = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(payload).to_bytes(8, "little") + payload + b"".join(parts))
    return path


def _kohya(path: Path, *, prefix: str = KOHYA, alpha: float | None = 2.0, dtype: str = "F32") -> Path:
    numpy_type = {"F32": "<f4", "F16": "<f2"}[dtype]
    down = np.ones((4, 8), dtype=numpy_type).tobytes()
    up = np.full((8, 4), 0.5, dtype=numpy_type).tobytes()
    tensors = {
        f"{prefix}.lora_down.weight": (dtype, [4, 8], down),
        f"{prefix}.lora_up.weight": (dtype, [8, 4], up),
    }
    if alpha is not None:
        tensors[f"{prefix}.alpha"] = ("F32", [], np.array(alpha, dtype="<f4").tobytes())
    return _write(path, tensors)


@pytest.fixture
def home(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    return tmp_path


def _loader_tensors(path: Path) -> dict[str, np.ndarray]:
    header, blob = _safetensors(path)
    out = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, stop = entry["data_offsets"]
        out[name] = np.frombuffer(blob[start:stop], dtype={"F32": "<f4", "F16": "<f2"}[entry["dtype"]])
    return out


def test_a_kohya_lora_is_mapped_and_its_alpha_is_folded_into_lora_b(home: Path):
    lora = prepare_cli_lora(_kohya(home / "k" / "style_k.safetensors"))
    tensors = _loader_tensors(lora.weights)
    assert set(tensors) == {f"{STEM}.lora_A.weight", f"{STEM}.lora_B.weight"}
    assert np.allclose(tensors[f"{STEM}.lora_B.weight"], 0.5 * 2.0 / 4)
    assert np.allclose(tensors[f"{STEM}.lora_A.weight"], 1.0)
    assert lora.kind == "unknown" and lora.trigger == "" and lora.strength_range == (0.0, 2.0)


def test_a_half_precision_kohya_lora_keeps_its_dtype(home: Path):
    lora = prepare_cli_lora(_kohya(home / "k" / "h.safetensors", dtype="F16", alpha=8.0))
    assert np.allclose(_loader_tensors(lora.weights)[f"{STEM}.lora_B.weight"].astype(np.float32), 1.0)


def test_a_kohya_lora_without_alpha_is_used_as_is(home: Path):
    lora = prepare_cli_lora(_kohya(home / "k" / "n.safetensors", alpha=None))
    assert np.allclose(_loader_tensors(lora.weights)[f"{STEM}.lora_B.weight"], 0.5)


def test_an_unmappable_kohya_key_is_a_base_mismatch(home: Path):
    with pytest.raises(LoraRejected) as caught:
        prepare_cli_lora(_kohya(home / "k" / "x.safetensors", prefix="lora_unet_down_blocks_0_attn_to_q"))
    assert caught.value.code == "lora.base_mismatch"
    text_encoder = _kohya(home / "k" / "t.safetensors", prefix="lora_te1_text_model_encoder_layers_0_mlp_fc1")
    with pytest.raises(LoraRejected) as caught:
        prepare_cli_lora(text_encoder)
    assert caught.value.code == "lora.base_mismatch"


def test_the_cli_takes_a_third_party_peft_lora_that_fits_the_base(home: Path):
    tensors = {
        f"transformer.{STEM}.lora_A.weight": ("F32", [4, 8], np.zeros((4, 8), "<f4").tobytes()),
        f"transformer.{STEM}.lora_B.weight": ("F32", [8, 4], np.zeros((8, 4), "<f4").tobytes()),
    }
    lora = prepare_cli_lora(_write(home / "p" / "peft.safetensors", tensors))
    assert lora.kind == "unknown" and lora.id == "peft"
    other = {
        "unet.down.lora_A.weight": ("F32", [1], b"\x00" * 4),
        "unet.down.lora_B.weight": ("F32", [1], b"\x00" * 4),
    }
    with pytest.raises(LoraRejected) as caught:
        prepare_cli_lora(_write(home / "p" / "other.safetensors", other))
    assert caught.value.code == "lora.base_mismatch"
    broken = home / "p" / "broken.safetensors"
    broken.write_bytes(b"\x05\x00\x00\x00\x00\x00\x00\x00{bad}")
    with pytest.raises(LoraRejected) as caught:
        prepare_cli_lora(broken)
    assert caught.value.code == "lora.invalid"


def test_the_cli_refuses_a_third_party_lora_naming_another_base(home: Path):
    path = _kohya(home / "k" / "b.safetensors")
    header, blob = _safetensors(path)
    header["__metadata__"] = {"voxprint.lora.base": "sdxl"}
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(len(payload).to_bytes(8, "little") + payload + blob)
    with pytest.raises(LoraRejected) as caught:
        prepare_cli_lora(path)
    assert caught.value.code == "lora.base_mismatch"


def test_gui_import_takes_a_kohya_lora(home: Path):
    entry = import_lora(
        _kohya(home / "k" / "Paper Cut.safetensors"), World().gate(),
        base_ready=lambda: True, confirm=lambda _manifest: True,
    )
    assert entry is not None and entry.manifest.kind == "unknown" and entry.manifest.name == "Paper Cut"
    lora = entry.as_lora()
    assert set(_loader_tensors(lora.weights)) == {f"{STEM}.lora_A.weight", f"{STEM}.lora_B.weight"}


def test_gui_import_refuses_an_unmappable_kohya_lora(home: Path):
    with pytest.raises(LoraRejected) as caught:
        import_lora(_kohya(home / "k" / "x.safetensors", prefix="lora_unet_mid_block_attn"), World().gate(),
                    base_ready=lambda: True, confirm=lambda _manifest: True)
    assert caught.value.code == "lora.base_mismatch"
