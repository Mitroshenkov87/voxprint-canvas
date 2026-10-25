"""CPU rewrite with the shipped Qwen3-4B text encoder.

The default test job does not download the encoder. The enhance-weights job
sets ``VOXPRINT_REQUIRE_ENHANCE=1`` after those files are on disk.
"""
from __future__ import annotations

import json
import os

import pytest

REQUIRE = os.environ.get("VOXPRINT_REQUIRE_ENHANCE") == "1"

if not REQUIRE:
    pytest.importorskip("torch")

import torch

from canvas.infra.paths import models_dir
from canvas.models.registry import load_catalog
from canvas.models.snapshots import weight_dir
from canvas.pipeline.enhance import QwenRewriter, causal_lm_ready


@pytest.mark.skipif(not REQUIRE, reason="set VOXPRINT_REQUIRE_ENHANCE=1 to load the text encoder")
def test_shipped_encoder_rewrites_a_tiny_prompt_on_cpu():
    spec = load_catalog().by_id("z-image-turbo-bf16-16gb")
    folder = weight_dir(models_dir(), spec.repo)
    encoder = folder / "text_encoder"
    config = json.loads((encoder / "config.json").read_text(encoding="utf-8"))
    index = json.loads((encoder / "model.safetensors.index.json").read_text(encoding="utf-8"))
    names = index["weight_map"]
    assert causal_lm_ready(config, names)
    assert config.get("tie_word_embeddings") is True
    assert not any("lm_head" in name for name in names)
    assert any(name.endswith("embed_tokens.weight") for name in names)
    device = None if torch.cuda.is_available() else "cpu"
    rewriter = QwenRewriter(encoder, folder / "tokenizer", max_new_tokens=24, device=device)
    text = rewriter.rewrite("a red cup")
    assert rewriter.closed is True
    assert rewriter._model is None
    assert text.strip()
