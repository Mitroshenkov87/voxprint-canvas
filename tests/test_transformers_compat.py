"""CPU load path for the encoders used by generation and the content gate.

The default test job does not install torch. The library-compat job sets
VOXPRINT_REQUIRE_GEN and installs the pinned stack, so a missing import fails
that job instead of skipping.
"""
import json
import os
from pathlib import Path

import pytest

REQUIRE = os.environ.get("VOXPRINT_REQUIRE_GEN") == "1"

if REQUIRE:
    import diffusers
    import torch
    import transformers
else:
    diffusers = pytest.importorskip("diffusers")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

from diffusers.modular_pipelines.z_image.encoders import get_qwen_prompt_embeds
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    Qwen2Tokenizer,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLVisionConfig,
    Qwen3Config,
    Qwen3Model,
    Siglip2Config,
    Siglip2Model,
    Siglip2TextConfig,
    Siglip2VisionConfig,
    ViTConfig,
)

from canvas.safety.adapters import _pooled_features


def _version(value: str) -> tuple[int, int, int]:
    numbers = [int(part) for part in value.split("+", 1)[0].split(".")[:3]]
    while len(numbers) < 3:
        numbers.append(0)
    return numbers[0], numbers[1], numbers[2]


def _qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=32,
        head_dim=8,
    )


def _vl_text_config() -> Qwen2_5_VLTextConfig:
    return Qwen2_5_VLTextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        rope_parameters={"rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
    )


def test_installed_stack_meets_the_pin():
    assert (5, 14, 0) <= _version(transformers.__version__) < (6, 0, 0)
    assert _version(diffusers.__version__) == (0, 41, 0)
    assert diffusers.ZImagePipeline.__name__ == "ZImagePipeline"
    assert diffusers.ZImageControlNetPipeline.__name__ == "ZImageControlNetPipeline"
    assert callable(get_qwen_prompt_embeds)


def test_qwen3_text_encoder_and_guard_forward_on_cpu(tmp_path: Path):
    config = _qwen3_config()
    config.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path)
    encoder = AutoModel.from_config(loaded).eval()
    assert isinstance(encoder, Qwen3Model)
    token_ids = torch.randint(0, 128, (1, 4))
    encoded = encoder(input_ids=token_ids, attention_mask=torch.ones_like(token_ids), output_hidden_states=True)
    assert tuple(encoded.last_hidden_state.shape) == (1, 4, 32)
    assert tuple(encoded.hidden_states[-2].shape) == (1, 4, 32)
    guard = AutoModelForCausalLM.from_config(loaded).eval()
    assert guard.__class__.__name__ == "Qwen3ForCausalLM"
    assert tuple(guard(input_ids=token_ids).logits.shape) == (1, 4, 128)


def test_qwen2_tokenizer_loads_a_local_vocab(tmp_path: Path):
    vocab = {"<|endoftext|>": 0, "a": 1, "b": 2, "ab": 3}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tmp_path / "merges.txt").write_text("#version: 0.2\na b\n", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"}),
        encoding="utf-8",
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(tmp_path)
    assert tokenizer.encode("ab") == [3]


def test_qwen25_vl_encoder_forwards_on_cpu(tmp_path: Path):
    text = _vl_text_config()
    text_model = Qwen2_5_VLTextModel(text).eval()
    token_ids = torch.randint(0, 128, (1, 4))
    assert tuple(text_model(input_ids=token_ids).last_hidden_state.shape) == (1, 4, 32)
    vision = Qwen2_5_VLVisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=32,
        fullatt_block_indexes=(),
        window_size=4,
        tokens_per_second=1,
    )
    config = Qwen2_5_VLConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=8,
        vision_end_token_id=9,
    )
    config.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path)
    model = Qwen2_5_VLForConditionalGeneration(loaded).eval()
    hidden = model(input_ids=token_ids, output_hidden_states=True).hidden_states[-1]
    assert tuple(hidden.shape) == (1, 4, 32)


def test_siglip2_features_are_pooled_before_normalising(tmp_path: Path):
    text = Siglip2TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=16,
        projection_size=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    vision = Siglip2VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_channels=3,
        patch_size=4,
        num_patches=4,
    )
    config = Siglip2Config(text_config=text, vision_config=vision)
    config.save_pretrained(tmp_path)
    model = Siglip2Model(AutoConfig.from_pretrained(tmp_path)).eval()
    text_out = model.get_text_features(input_ids=torch.randint(0, 64, (2, 4)))
    text_embeds = _pooled_features(text_out)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    assert tuple(text_embeds.shape) == (2, 32)
    image_out = model.get_image_features(
        pixel_values=torch.rand(1, 4, 3 * 4 * 4),
        spatial_shapes=torch.tensor([[2, 2]]),
    )
    image_embeds = _pooled_features(image_out)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    assert tuple(image_embeds.shape) == (1, 32)


def test_image_classifiers_forward_on_cpu():
    pixels = torch.rand(1, 3, 16, 16)
    shared = dict(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=16,
        patch_size=16,
        num_channels=3,
    )
    content = ViTConfig(
        **shared,
        num_labels=2,
        id2label={0: "normal", 1: "nsfw"},
        label2id={"normal": 0, "nsfw": 1},
    )
    content_model = AutoModelForImageClassification.from_config(content).eval()
    assert tuple(content_model(pixel_values=pixels).logits.shape) == (1, 2)
    age = ViTConfig(**shared, num_labels=9)
    age_model = AutoModelForImageClassification.from_config(age).eval()
    assert tuple(age_model(pixel_values=pixels).logits.shape) == (1, 9)
