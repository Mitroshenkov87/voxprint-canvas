"""Classifier adapters.

The classes know which registry model they bind to. Calling an adapter before
its weights are on disk refuses the request. NudeNet is not an adapter.

torch, transformers and opencv are imported inside the load methods. That is
not a circular import. Those packages are optional so the unit tests can import
this module without them.
"""
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from canvas.models.pins import REPO_AGE, REPO_GUARD, REPO_NSFW, REPO_SIGLIP, REPO_YUNET
from canvas.safety.classifiers import Face, GuardVerdict
from canvas.safety.gate import SafetyUnavailable
from canvas.safety.image import RawImage

# Lower edge of each FairFace age bucket. The image gate does not use this floor.
# 20-29 maps to 20, which is how a twenty-year bucket used to be treated as under 25.
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

# Share of each bucket that is under 18. 10-19 covers ten years and eight of them
# are under 18. 20-29 and every older bucket contribute 0, so they cannot leak.
FAIRFACE_UNDER_18 = {
    "0-2": 1.0,
    "3-9": 1.0,
    "10-19": 0.8,
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
    """Map a FairFace bucket to its lower edge in years."""
    try:
        return FAIRFACE_AGE_LOW[label]
    except KeyError as exc:
        raise SafetyUnavailable("classifier_error") from exc


def probability_under_18(probs: Mapping[str, float]) -> float:
    """P(age < 18) from a full FairFace softmax.

    An unknown label, a non-numeric weight, or a distribution that does not sum
    to one refuses the request.
    """
    if not probs:
        raise SafetyUnavailable("classifier_error")
    total = 0.0
    under = 0.0
    for label, value in probs.items():
        try:
            weight = FAIRFACE_UNDER_18[str(label)]
        except KeyError as exc:
            raise SafetyUnavailable("classifier_error") from exc
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SafetyUnavailable("classifier_error")
        number = float(value)
        if math.isnan(number) or math.isinf(number) or number < 0.0:
            raise SafetyUnavailable("classifier_error")
        total += number
        under += weight * number
    if total < _PROBABILITY_SUM_LOW or total > _PROBABILITY_SUM_HIGH:
        raise SafetyUnavailable("classifier_error")
    return under


def fairface_under_18(id2label: Mapping[int | str, str], values: Sequence[float]) -> float:
    """P(age < 18) for one face. ``values`` aligns with class ids ``0 .. n-1``."""
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
    return probability_under_18(mapped)


def parse_qwen3guard(text: str) -> GuardVerdict:
    """Parse a Qwen3Guard-Gen reply (``Safety:`` and ``Categories:`` lines)."""
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
    """Tensor from ``get_*_features``.

    Transformers 5 returns ``BaseModelOutputWithPooling``. Older releases return a tensor, which has ``norm``.
    """
    if hasattr(out, "norm"):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    raise SafetyUnavailable("classifier_error")


class Qwen3GuardGen:
    """``Qwen/Qwen3Guard-Gen-0.6B`` prompt classifier."""

    model_id = REPO_GUARD

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        """Open the pinned snapshot. A missing folder refuses the request."""
        folder = _need_folder(self.folder)
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)  # nosec B615
        self._model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            folder, torch_dtype=torch.float32, local_files_only=True,
        )

    def assess(self, text: str) -> GuardVerdict:
        """Classify one prompt with the loaded guard model."""
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
    """``google/siglip2-so400m-patch14-384`` scores for the fixed labels."""

    model_id = REPO_SIGLIP

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """Open the pinned snapshot. A missing folder refuses the request."""
        folder = _need_folder(self.folder)
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor
        self._processor = AutoProcessor.from_pretrained(folder, local_files_only=True)  # nosec B615
        self._model = AutoModel.from_pretrained(folder, local_files_only=True)  # nosec B615

    def score_text(self, text: str, labels: Sequence[str]) -> dict[str, float]:
        """SigLIP2 text scores for the fixed labels."""
        return self._score(text=text, image=None, labels=labels)

    def score_image(self, image: RawImage, labels: Sequence[str]) -> dict[str, float]:
        """SigLIP2 image scores for the fixed labels."""
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
    """OpenCV YuNet face detector. The weights are MIT."""

    model_id = REPO_YUNET

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder
        self._path: Path | None = None

    def load(self) -> None:
        """Open the pinned ONNX file. A missing folder refuses the request."""
        folder = _need_folder(self.folder)
        if self._path is not None:
            return
        matches = list(folder.glob("*.onnx"))
        if len(matches) != 1:
            raise SafetyUnavailable("model_not_ready")
        self._path = matches[0]

    def detect(self, image: RawImage) -> Sequence[Face]:
        """YuNet boxes. An empty sequence means no face, not a failure."""
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
    """``dima806/fairface_age_image_detection`` apparent-age buckets."""

    model_id = REPO_AGE

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """Open the pinned snapshot. A missing folder refuses the request."""
        folder = _need_folder(self.folder)
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self._processor = AutoImageProcessor.from_pretrained(folder, local_files_only=True)  # nosec B615
        self._model = AutoModelForImageClassification.from_pretrained(  # nosec B615
            folder, local_files_only=True,
        )

    def estimate(self, image: RawImage, faces: Sequence[Face]) -> Sequence[float]:
        """P(age < 18) for each face, from the full FairFace distribution."""
        self.load()
        import torch
        picture = _rgb(image)
        under: list[float] = []
        for face in faces:
            crop = picture.crop((face.x, face.y, face.x + face.width, face.y + face.height))
            inputs = self._processor(images=crop, return_tensors="pt")
            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
            values = [float(probs[index]) for index in range(int(probs.shape[-1]))]
            under.append(fairface_under_18(self._model.config.id2label, values))
        return under


class NsfwViT:
    """``Falconsai/nsfw_image_detection`` (MIT). Not NudeNet."""

    model_id = REPO_NSFW

    def __init__(self, folder: Path | None = None) -> None:
        self.folder = folder
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """Open the pinned snapshot. A missing folder refuses the request."""
        folder = _need_folder(self.folder)
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self._processor = AutoImageProcessor.from_pretrained(folder, local_files_only=True)  # nosec B615
        self._model = AutoModelForImageClassification.from_pretrained(  # nosec B615
            folder, local_files_only=True,
        )

    def sexual_score(self, image: RawImage) -> float:
        """Probability that the image is sexual, from 0 to 1."""
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
