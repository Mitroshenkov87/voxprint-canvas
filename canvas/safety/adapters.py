"""Classifier adapters."""
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from canvas.models.pins import REPO_AGE, REPO_GUARD, REPO_NSFW, REPO_SIGLIP, REPO_YUNET
from canvas.safety.classifiers import Face, GuardVerdict
from canvas.safety.gate import SafetyUnavailable
from canvas.safety.image import RawImage

FAIRFACE_AGE_LOW = {
    "0-2": 0.0,
    "3-9": 3.0,
    "10-19": 10.0,
    "20-29": 20.0,
    "30-39": 30.0,
    "40-49": 40.0,
    "50-59": 50.0,
    "60-69": 60.0,
    "70+": 70.0,
    "more than 70": 70.0,
}

FAIRFACE_AT_MOST_15 = {
    "0-2": 1.0,
    "3-9": 1.0,
    "10-19": 0.6,
    "20-29": 0.0,
    "30-39": 0.0,
    "40-49": 0.0,
    "50-59": 0.0,
    "60-69": 0.0,
    "70+": 0.0,
    "more than 70": 0.0,
}

FAIRFACE_16_17 = {
    "0-2": 0.0,
    "3-9": 0.0,
    "10-19": 0.2,
    "20-29": 0.0,
    "30-39": 0.0,
    "40-49": 0.0,
    "50-59": 0.0,
    "60-69": 0.0,
    "70+": 0.0,
    "more than 70": 0.0,
}

_PROBABILITY_SUM_LOW = 0.99
_PROBABILITY_SUM_HIGH = 1.01


def apparent_age_from_bucket(label: str) -> float:
    """Apparent age from bucket."""
    try:
        return FAIRFACE_AGE_LOW[label]
    except KeyError as exc:
        raise SafetyUnavailable("classifier_error") from exc


def _distribution_weight(probs: Mapping[str, float], table: Mapping[str, float]) -> float:
    """Distribution weight."""
    if not probs:
        raise SafetyUnavailable("classifier_error")
    total = 0.0
    weighted = 0.0
    for label, value in probs.items():
        try:
            weight = table[str(label)]
        except KeyError as exc:
            raise SafetyUnavailable("classifier_error") from exc
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SafetyUnavailable("classifier_error")
        number = float(value)
        if math.isnan(number) or math.isinf(number) or number < 0.0:
            raise SafetyUnavailable("classifier_error")
        total += number
        weighted += weight * number
    if total < _PROBABILITY_SUM_LOW or total > _PROBABILITY_SUM_HIGH:
        raise SafetyUnavailable("classifier_error")
    return weighted


def age_bands(probs: Mapping[str, float]) -> tuple[float, float]:
    """Age bands."""
    return (_distribution_weight(probs, FAIRFACE_AT_MOST_15), _distribution_weight(probs, FAIRFACE_16_17))


def fairface_age_bands(id2label: Mapping[int | str, str], values: Sequence[float]) -> tuple[float, float]:
    """Fairface age bands."""
    mapped: dict[str, float] = {}
    for index, value in enumerate(values):
        if index in id2label:
            label = id2label[index]
        elif str(index) in id2label:
            label = id2label[str(index)]
        else:
            raise SafetyUnavailable("classifier_error")
        name = str(label)
        if name in mapped:
            raise SafetyUnavailable("classifier_error")
        mapped[name] = value
    return age_bands(mapped)


def parse_qwen3guard(text: str) -> GuardVerdict:
    """Parse qwen3guard."""
    safety = ""
    categories: list[str] = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        name = key.strip().casefold()
        body = value.strip()
        if name == "safety":
            safety = body.casefold()
        elif name == "categories" and body and body.casefold() != "none":
            categories = [part.strip() for part in body.split(",") if part.strip()]
    if safety not in ("safe", "controversial", "unsafe"):
        raise SafetyUnavailable("classifier_error")
    return GuardVerdict(safety, tuple(categories))


def _need_folder(folder: Path | None) -> Path:
    if folder is None or not folder.is_dir():
        raise SafetyUnavailable("model_not_ready")
    return folder


def _rgb(image: RawImage) -> Image.Image:
    return Image.frombytes("RGB", (image.width, image.height), image.to_bytes())


def _pooled_features(out: Any) -> Any:
    """Tensor from ``get_*_features``."""
    if hasattr(out, "norm"):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    raise SafetyUnavailable("classifier_error")


class Qwen3GuardGen:
    """Qwen3GuardGen."""

    model_id = REPO_GUARD

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        """Load."""
        folder = _need_folder(self.folder)
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            folder, torch_dtype=torch.float32, local_files_only=True,
        )

    def assess(self, text: str) -> GuardVerdict:
        """Assess."""
        self.load()
        import torch
        tokenizer = self._tokenizer
        model = self._model
        messages = [{"role": "user", "content": text}]
        encoded = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        with torch.no_grad():
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=128)
        reply = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return parse_qwen3guard(reply)


class SiglipZeroShot:
    """SiglipZeroShot."""

    model_id = REPO_SIGLIP

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """Load."""
        folder = _need_folder(self.folder)
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor
        self._processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        self._model = AutoModel.from_pretrained(folder, local_files_only=True)

    def score_text(self, text: str, labels: Sequence[str]) -> dict[str, float]:
        """Score text."""
        return self._score(text=text, image=None, labels=labels)

    def score_image(self, image: RawImage, labels: Sequence[str]) -> dict[str, float]:
        """Score image."""
        return self._score(text=None, image=image, labels=labels)

    def _score(self, *, text: str | None, image: RawImage | None, labels: Sequence[str]) -> dict[str, float]:
        self.load()
        import torch
        processor = self._processor
        model = self._model
        label_list = list(labels)
        text_inputs = processor(text=label_list, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            text_embeds = _pooled_features(model.get_text_features(**text_inputs))
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            if image is not None:
                image_inputs = processor(images=_rgb(image), return_tensors="pt")
                image_embeds = _pooled_features(model.get_image_features(**image_inputs))
            else:
                image_inputs = processor(text=[text or ""], padding="max_length", return_tensors="pt")
                image_embeds = _pooled_features(model.get_text_features(**image_inputs))
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            scale = model.logit_scale.exp()
            bias = getattr(model, "logit_bias", None)
            logits = image_embeds @ text_embeds.T * scale
            if bias is not None:
                logits = logits + bias
            probs = torch.sigmoid(logits)[0]
        return {label: float(probs[index]) for index, label in enumerate(label_list)}


class YuNetFaces:
    """YuNetFaces."""

    model_id = REPO_YUNET

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder
        self._path: Path | None = None

    def load(self) -> None:
        """Load."""
        folder = _need_folder(self.folder)
        if self._path is not None:
            return
        matches = list(folder.glob("*.onnx"))
        if len(matches) != 1:
            raise SafetyUnavailable("model_not_ready")
        self._path = matches[0]

    def detect(self, image: RawImage) -> Sequence[Face]:
        """Detect."""
        self.load()
        import cv2
        pixels = np.frombuffer(image.to_bytes(), dtype=np.uint8).reshape(image.height, image.width, 3)
        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        detector = cv2.FaceDetectorYN.create(str(self._path), "", (image.width, image.height), 0.9, 0.3, 5000)
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        detector.setInputSize((image.width, image.height))
        _retval, faces = detector.detect(bgr)
        if faces is None:
            return ()
        boxes: list[Face] = []
        for row in faces:
            x, y, width, height = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            if width > 0 and height > 0:
                boxes.append(Face(x, y, width, height))
        return boxes


class FairFaceAge:
    """FairFaceAge."""

    model_id = REPO_AGE

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """Load."""
        folder = _need_folder(self.folder)
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self._processor = AutoImageProcessor.from_pretrained(folder, local_files_only=True)
        self._model = AutoModelForImageClassification.from_pretrained(
            folder, local_files_only=True,
        )

    def estimate(self, image: RawImage, faces: Sequence[Face]) -> Sequence[tuple[float, float]]:
        """Estimate."""
        self.load()
        import torch
        picture = _rgb(image)
        bands: list[tuple[float, float]] = []
        for face in faces:
            crop = picture.crop((face.x, face.y, face.x + face.width, face.y + face.height))
            inputs = self._processor(images=crop, return_tensors="pt")
            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
            values = [float(probs[index]) for index in range(int(probs.shape[-1]))]
            bands.append(fairface_age_bands(self._model.config.id2label, values))
        return bands


class NsfwViT:
    """NsfwViT."""

    model_id = REPO_NSFW

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """Load."""
        folder = _need_folder(self.folder)
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self._processor = AutoImageProcessor.from_pretrained(folder, local_files_only=True)
        self._model = AutoModelForImageClassification.from_pretrained(
            folder, local_files_only=True,
        )

    def sexual_score(self, image: RawImage) -> float:
        """Sexual score."""
        self.load()
        import torch
        inputs = self._processor(images=_rgb(image), return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        labels = {int(key): str(value).casefold() for key, value in self._model.config.id2label.items()}
        for index, name in labels.items():
            if name == "nsfw":
                return float(probs[index])
        raise SafetyUnavailable("classifier_error")
