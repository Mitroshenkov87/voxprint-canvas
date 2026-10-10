"""Pinned runtime and the diffusers pipeline classes for each mode.

Inference itself is not wired yet. The names and the diffusers pin are fixed
so the loader stub cannot drift from the registry.
"""
from __future__ import annotations

DIFFUSERS_PIN = "0.41.0"

PIPELINE_TEXT = "ZImagePipeline"
PIPELINE_IMG2IMG = "ZImageImg2ImgPipeline"
PIPELINE_INPAINT = "ZImageInpaintPipeline"
PIPELINE_CONTROL = "ZImageControlNetPipeline"
PIPELINE_CONTROL_INPAINT = "ZImageControlNetInpaintPipeline"
PIPELINE_EDIT = "QwenImageEditPlusPipeline"
PIPELINE_QWEN = "QwenImagePipeline"

REPO_TURBO = "Tongyi-MAI/Z-Image-Turbo"
REPO_BASE = "Tongyi-MAI/Z-Image"
REPO_CONTROL = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1"
REPO_EDIT = "Qwen/Qwen-Image-Edit-2511"
REPO_QWEN = "Qwen/Qwen-Image-2512"
REPO_GUARD = "Qwen/Qwen3Guard-Gen-0.6B"
REPO_SIGLIP = "google/siglip2-so400m-patch14-384"
REPO_YUNET = "opencv/face_detection_yunet"
REPO_AGE = "dima806/fairface_age_image_detection"
REPO_NSFW = "Falconsai/nsfw_image_detection"

TEXT_ENCODER = "Qwen/Qwen3-4B"
ENCODER_VL = "Qwen/Qwen2.5-VL-7B-Instruct"
TURBO_STEPS = 8
TURBO_GUIDANCE = 0.0
CREATIVE_STEPS = 50
QWEN_STEPS = 50
QWEN_TRUE_CFG = 4.0
