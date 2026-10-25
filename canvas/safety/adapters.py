"""Classifier adapters."""
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from canvas.models.pins import REPO_AGE, REPO_AGE_FINE, REPO_GUARD, REPO_NSFW, REPO_SIGLIP, REPO_YUNET
from canvas.safety.classifiers import Face, GuardVerdict
from canvas.safety.gate import SafetyUnavailable
from canvas.safety.image import RawImage

# Device placement only. The same models get the same inputs, thresholds and
# order on either device. A check that hits a device error runs again on the CPU;
# any other error still stops the request.
GENERATOR_RESERVE_BYTES = 12 * 1024**3
GATE_HEADROOM_BYTES = 768 * 1024**2
_gate_cuda_bytes = 0


def _module_bytes(model: Any) -> int:
    total = 0
    for getter in ("parameters", "buffers"):
        items = getattr(model, getter, None)
        if callable(items):
            for tensor in items():
                total += int(tensor.numel()) * int(tensor.element_size())
    return total


def gate_device(need_bytes: int) -> str:
    """``cuda`` when the weights fit in free VRAM, else ``cpu``.

    The generator reserve applies only while the generator already holds VRAM.
    An empty card does not keep that reserve.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        if not torch.cuda.is_available():
            return "cpu"
        free, _total = torch.cuda.mem_get_info()
        generator_now = max(0, int(torch.cuda.memory_reserved()) - _gate_cuda_bytes)
    except (RuntimeError, AssertionError):
        return "cpu"
    if generator_now == 0:
        still_needed = 0
    else:
        still_needed = max(0, GENERATOR_RESERVE_BYTES - generator_now)
    if int(free) - still_needed - GATE_HEADROOM_BYTES >= int(need_bytes):
        return "cuda"
    return "cpu"


def _device_error(exc: BaseException) -> bool:
    """CUDA out-of-memory or another device failure."""
    if type(exc).__name__ in {"OutOfMemoryError", "AcceleratorError"}:
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return "cuda" in message or "out of memory" in message or "device" in message
    return False


def _inputs_to(inputs: Any, device: str) -> Any:
    if device == "cpu":
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) for key, value in inputs.items()}


class _Placed:
    """Puts one loaded model on the GPU when it fits, and falls back to the CPU."""

    device: str = "cpu"
    _model: Any = None
    _cuda_bytes: int = 0

    def _place(self, model: Any) -> Any:
        global _gate_cuda_bytes
        need = _module_bytes(model)
        forced = getattr(self, "_forced", None)
        device = gate_device(need) if forced is None else forced
        if device == "cpu":
            self.device = "cpu"
            return model
        try:
            moved = model.to(device)
        except Exception as exc:
            if not _device_error(exc):
                raise
            self.device = "cpu"
            _empty_cuda_cache()
            return model.to("cpu")
        self.device = device
        self._cuda_bytes = need
        _gate_cuda_bytes += need
        return moved

    def _to_cpu(self) -> None:
        global _gate_cuda_bytes
        if self._model is not None:
            self._model = self._model.to("cpu")
        _gate_cuda_bytes = max(0, _gate_cuda_bytes - self._cuda_bytes)
        self._cuda_bytes = 0
        self.device = "cpu"
        _empty_cuda_cache()

    def release(self) -> None:
        """Drop loaded weights so the picture model can use the GPU. The next check loads them again."""
        self._to_cpu()
        self._model = None

    def _checked(self, run: Any) -> Any:
        """Run one check. A device error moves the model to the CPU and runs the same check again."""
        try:
            return run()
        except Exception as exc:
            if self.device == "cpu" or not _device_error(exc):
                raise
        self._to_cpu()
        return run()


def _empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        return


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


class Qwen3GuardGen(_Placed):
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
        self._model = self._place(AutoModelForCausalLM.from_pretrained(
            folder, torch_dtype=torch.float32, local_files_only=True,
        ))

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

        def run() -> Any:
            with torch.no_grad():
                return self._model.generate(
                    input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device),
                    max_new_tokens=128,
                ).cpu()

        del model
        output = self._checked(run)
        reply = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return parse_qwen3guard(reply)


class SiglipZeroShot(_Placed):
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
        self._model = self._place(AutoModel.from_pretrained(folder, local_files_only=True))

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
        label_list = list(labels)

        def run() -> Any:
            return self._siglip_probs(torch, processor, text, image, label_list)

        probs = self._checked(run)
        return {label: float(probs[index]) for index, label in enumerate(label_list)}

    def _siglip_probs(
        self, torch: Any, processor: Any, text: str | None, image: RawImage | None, label_list: list[str],
    ) -> Any:
        model = self._model
        device = self.device
        text_inputs = _inputs_to(processor(text=label_list, padding="max_length", return_tensors="pt"), device)
        with torch.no_grad():
            text_embeds = _pooled_features(model.get_text_features(**text_inputs))
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            if image is not None:
                image_inputs = _inputs_to(processor(images=_rgb(image), return_tensors="pt"), device)
                image_embeds = _pooled_features(model.get_image_features(**image_inputs))
            else:
                image_inputs = _inputs_to(
                    processor(text=[text or ""], padding="max_length", return_tensors="pt"), device,
                )
                image_embeds = _pooled_features(model.get_text_features(**image_inputs))
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            scale = model.logit_scale.exp()
            bias = getattr(model, "logit_bias", None)
            logits = image_embeds @ text_embeds.T * scale
            if bias is not None:
                logits = logits + bias
            return torch.sigmoid(logits)[0].cpu()


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


class FairFaceAge(_Placed):
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
        self._model = self._place(AutoModelForImageClassification.from_pretrained(
            folder, local_files_only=True,
        ))

    def estimate(self, image: RawImage, faces: Sequence[Face]) -> Sequence[tuple[float, float]]:
        """Estimate."""
        self.load()
        import torch
        picture = _rgb(image)
        bands: list[tuple[float, float]] = []
        for face in faces:
            crop = picture.crop((face.x, face.y, face.x + face.width, face.y + face.height))
            inputs = self._processor(images=crop, return_tensors="pt")

            def run(inputs: Any = inputs) -> Any:
                with torch.no_grad():
                    logits = self._model(**_inputs_to(inputs, self.device)).logits
                    return torch.softmax(logits, dim=-1)[0].cpu()

            probs = self._checked(run)
            values = [float(probs[index]) for index in range(int(probs.shape[-1]))]
            bands.append(fairface_age_bands(self._model.config.id2label, values))
        return bands


_REGRESSOR_CONFIG = {
    "model_name": "mivolo_d1_384",
    "in_chans": 6,
    "num_classes": 3,
    "input_size": 384,
    "with_persons_model": True,
    "only_age": False,
}
_REGRESSOR_MEAN = (0.485, 0.456, 0.406)
_REGRESSOR_STD = (0.229, 0.224, 0.225)
_REGRESSOR_WEIGHTS = "model.safetensors"
_REGRESSOR_PREFIX = "mivolo.model."


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SafetyUnavailable("classifier_error")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise SafetyUnavailable("classifier_error")
    return number


def regressor_scale(config: Mapping[str, Any]) -> tuple[float, float, float]:
    """Regressor scale."""
    for key, expected in _REGRESSOR_CONFIG.items():
        if config.get(key) != expected:
            raise SafetyUnavailable("model_not_ready")
    low = _finite(config.get("min_age"))
    high = _finite(config.get("max_age"))
    mid = _finite(config.get("avg_age"))
    if not low < mid < high:
        raise SafetyUnavailable("model_not_ready")
    return (low, high, mid)


def letterbox(pixels: np.ndarray, size: int) -> np.ndarray:
    """Letterbox."""
    import cv2
    height, width = pixels.shape[:2]
    if height <= 0 or width <= 0:
        raise SafetyUnavailable("classifier_error")
    if height == size and width == size:
        return pixels
    ratio = min(size / height, size / width)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    if (new_w, new_h) != (width, height):
        pixels = cv2.resize(pixels, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    return cv2.copyMakeBorder(pixels, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def face_crop(pixels: np.ndarray, face: Face) -> np.ndarray:
    """Face crop."""
    height, width = pixels.shape[:2]
    x1, y1 = max(face.x, 0), max(face.y, 0)
    x2, y2 = min(face.x + face.width, width), min(face.y + face.height, height)
    if x2 <= x1 or y2 <= y1:
        raise SafetyUnavailable("classifier_error")
    return pixels[y1:y2, x1:x2]


def regressor_ages(raw: Sequence[object], scale: tuple[float, float, float], count: int) -> list[float]:
    """Regressor ages."""
    if len(raw) != count:
        raise SafetyUnavailable("classifier_error")
    low, high, mid = scale
    return [_finite(value) * (high - low) + mid for value in raw]


class FaceAgeRegressor(_Placed):
    """FaceAgeRegressor."""

    model_id = REPO_AGE_FINE

    def __init__(self, folder: Path | None = None, device: str = "auto") -> None:
        self.folder = folder
        self.device = "cpu"
        self._forced = None if device == "auto" else device
        self._model: Any = None
        self._scale: tuple[float, float, float] | None = None

    def load(self) -> None:
        """Load."""
        folder = _need_folder(self.folder)
        if self._model is not None:
            return
        import json

        weights = folder / _REGRESSOR_WEIGHTS
        config_path = folder / "config.json"
        if not weights.is_file() or not config_path.is_file():
            raise SafetyUnavailable("model_not_ready")
        try:
            import torch
            from safetensors.torch import load_file

            from canvas.safety.estimator_net import EstimatorNet
        except ImportError as exc:
            raise SafetyUnavailable("model_not_ready") from exc
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SafetyUnavailable("model_not_ready") from exc
        if not isinstance(config, dict):
            raise SafetyUnavailable("model_not_ready")
        scale = regressor_scale(config)
        state = load_file(str(weights))
        if not state or not all(key.startswith(_REGRESSOR_PREFIX) for key in state):
            raise SafetyUnavailable("model_not_ready")
        model = EstimatorNet(img_size=int(config["input_size"]), num_classes=int(config["num_classes"]))
        stripped = {key[len(_REGRESSOR_PREFIX):]: value.float() for key, value in state.items()}
        try:
            model.load_state_dict(stripped, strict=True)
        except RuntimeError as exc:
            raise SafetyUnavailable("model_not_ready") from exc
        self._model = self._place(model.to(dtype=torch.float32).eval())
        self._scale = scale

    def _tensor(self, pixels: np.ndarray, size: int) -> Any:
        import torch
        mean = np.asarray(_REGRESSOR_MEAN, dtype=np.float32)
        std = np.asarray(_REGRESSOR_STD, dtype=np.float32)
        boxed = letterbox(pixels, size).astype(np.float32) / 255.0
        normal = ((boxed - mean) / std).astype(np.float32).transpose((2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(normal))

    def estimate(self, image: RawImage, faces: Sequence[Face]) -> Sequence[float]:
        """Estimate."""
        self.load()
        if not faces:
            return []
        import torch
        model = self._model
        scale = self._scale
        if scale is None:
            raise SafetyUnavailable("model_not_ready")
        size = int(_REGRESSOR_CONFIG["input_size"])  # type: ignore[call-overload]
        pixels = np.frombuffer(image.to_bytes(), dtype=np.uint8).reshape(image.height, image.width, 3)
        face_batch = torch.stack([self._tensor(face_crop(pixels, face), size) for face in faces])
        empty = torch.zeros((3, size, size), dtype=torch.float32)
        mean = torch.tensor(_REGRESSOR_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(_REGRESSOR_STD, dtype=torch.float32).view(3, 1, 1)
        body = ((empty - mean) / std).unsqueeze(0).expand(len(faces), -1, -1, -1)
        batch = torch.cat((face_batch, body), dim=1)
        del model

        def run() -> Any:
            with torch.no_grad():
                return self._model(batch.to(device=self.device, dtype=torch.float32)).float().cpu()

        output = self._checked(run)
        if output.ndim != 2 or int(output.shape[0]) != len(faces) or int(output.shape[1]) != 3:
            raise SafetyUnavailable("classifier_error")
        return regressor_ages([float(value) for value in output[:, 2]], scale, len(faces))


class NsfwViT(_Placed):
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
        self._model = self._place(AutoModelForImageClassification.from_pretrained(
            folder, local_files_only=True,
        ))

    def sexual_score(self, image: RawImage) -> float:
        """Sexual score."""
        self.load()
        import torch
        inputs = self._processor(images=_rgb(image), return_tensors="pt")

        def run() -> Any:
            with torch.no_grad():
                logits = self._model(**_inputs_to(inputs, self.device)).logits
                return torch.softmax(logits, dim=-1)[0].cpu()

        probs = self._checked(run)
        labels = {int(key): str(value).casefold() for key, value in self._model.config.id2label.items()}
        for index, name in labels.items():
            if name == "nsfw":
                return float(probs[index])
        raise SafetyUnavailable("classifier_error")
