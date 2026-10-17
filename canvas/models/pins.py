"""Pinned runtime and the diffusers pipeline classes for each mode.

The names and the diffusers pin match the registry. Inference loads these
classes when a snapshot is already on disk.
"""
from __future__ import annotations

DIFFUSERS_PIN = "0.41.0"

PIPELINE_TEXT = "ZImagePipeline"
PIPELINE_IMG2IMG = "ZImageImg2ImgPipeline"
PIPELINE_INPAINT = "ZImageInpaintPipeline"
PIPELINE_CONTROL = "ZImageControlNetPipeline"
PIPELINE_CONTROL_INPAINT = "ZImageControlNetInpaintPipeline"
PIPELINE_QWEN = "QwenImagePipeline"

REPO_TURBO = "Tongyi-MAI/Z-Image-Turbo"
REPO_BASE = "Tongyi-MAI/Z-Image"
REPO_CONTROL = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1"
REPO_QWEN = "Qwen/Qwen-Image-2512"
REPO_GUARD = "Qwen/Qwen3Guard-Gen-0.6B"
REPO_SIGLIP = "google/siglip2-so400m-patch14-384"
REPO_YUNET = "opencv/face_detection_yunet"
REPO_AGE = "dima806/fairface_age_image_detection"
REPO_NSFW = "Falconsai/nsfw_image_detection"

TEXT_ENCODER = "Qwen/Qwen3-4B"
TURBO_STEPS = 8
TURBO_GUIDANCE = 0.0
CREATIVE_STEPS = 50
QWEN_STEPS = 50
QWEN_TRUE_CFG = 4.0
