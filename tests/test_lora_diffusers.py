"""A Forge LoRA on a real diffusers Z-Image transformer, scaled down to a few layers."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("peft")
safetensors_torch = pytest.importorskip("safetensors.torch")

from canvas.models.user_lora import (  # noqa: E402
    CODE_INVALID,
    LoraRejected,
    UserLora,
    apply_lora,
    fits_layers,
)

# The module names a Model Forge LoRA carries for each block, as in a real export.
_FORGE_TARGETS = (
    "attention.to_q",
    "attention.to_k",
    "attention.to_v",
    "attention.to_out.0",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
)
_BLOCKS = ("layers.0", "context_refiner.0", "noise_refiner.0")
_RANK = 4


def _tiny() -> object:
    torch.manual_seed(0)
    return diffusers.ZImageTransformer2DModel(
        dim=64,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        cap_feat_dim=32,
        axes_dims=[8, 12, 12],
        axes_lens=[64, 32, 32],
    ).eval()


def _forge_file(path: Path, model: object, *, wrong_in: bool = False) -> Path:
    """Transformer-relative ``lora_A`` / ``lora_B`` names with no ``transformer.`` prefix."""
    modules = dict(model.named_modules())
    tensors = {}
    for block in _BLOCKS:
        for target in _FORGE_TARGETS:
            layer = modules[f"{block}.{target}"]
            width = layer.in_features + (8 if wrong_in else 0)
            tensors[f"{block}.{target}.lora_A.weight"] = torch.randn(_RANK, width) * 0.5
            tensors[f"{block}.{target}.lora_B.weight"] = torch.randn(layer.out_features, _RANK) * 0.5
    safetensors_torch.save_file(tensors, str(path), metadata={"voxprint.lora.base": "z-image"})
    return path


def _lora(weights: Path) -> UserLora:
    return UserLora("garnis-test", "person", weights.parent, weights, "garnis", 1.0, (0.0, 2.0), False)


def _attached(model: object) -> list[str]:
    return [name for name, module in model.named_modules() if "canvas-lora" in (getattr(module, "lora_A", None) or {})]


def test_the_default_diffusers_prefix_drops_every_forge_key(tmp_path: Path) -> None:
    # Why rc.35 rendered nothing: the plain call filters keys by ``transformer.``.
    model = _tiny()
    weights = _forge_file(tmp_path / "garnis.safetensors", model)
    model.load_lora_adapter(str(weights), adapter_name="probe")
    assert not any(hasattr(module, "lora_A") for module in model.modules())


def test_a_forge_lora_reaches_every_targeted_layer_and_changes_the_output(tmp_path: Path) -> None:
    model = _tiny()
    weights = _forge_file(tmp_path / "garnis.safetensors", model)
    layer = dict(model.named_modules())["layers.0.attention.to_q"]
    probe = torch.randn(3, layer.in_features)
    with torch.no_grad():
        before = layer(probe).clone()
    applied = apply_lora(model, _lora(weights), 1.0)
    assert applied == 1.0
    expected = {f"{block}.{target}" for block in _BLOCKS for target in _FORGE_TARGETS}
    assert set(_attached(model)) == expected
    with torch.no_grad():
        after = dict(model.named_modules())["layers.0.attention.to_q"](probe)
    assert not torch.allclose(before, after)


def test_a_bad_shape_lora_is_rejected_before_loading_and_leaves_no_adapter(tmp_path: Path) -> None:
    model = _tiny()
    weights = _forge_file(tmp_path / "bad.safetensors", model, wrong_in=True)
    with pytest.raises(LoraRejected) as caught:
        apply_lora(model, _lora(weights), 1.0)
    assert caught.value.code == CODE_INVALID
    assert _attached(model) == []


@pytest.mark.real_shapes
def test_a_bad_shape_lora_is_rejected_against_the_shipped_base_table(tmp_path: Path) -> None:
    # The tiny model is 64 wide; Z-Image-Turbo is 3840 wide, so the shipped table refuses it.
    weights = _forge_file(tmp_path / "small.safetensors", _tiny())
    with pytest.raises(LoraRejected) as caught:
        fits_layers(weights)
    assert caught.value.code == CODE_INVALID


@pytest.mark.real_shapes
def test_the_real_forge_export_fits_the_shipped_base_table() -> None:
    sample = Path("/workspace/vm-results/model-forge/4090-precheck/rc12/lora/garnis_ai_rc12_quick_step120.safetensors")
    if not sample.is_file():
        pytest.skip("the 4090 Forge export is not on this machine")
    fits_layers(sample)
