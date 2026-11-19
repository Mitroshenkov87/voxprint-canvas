"""YuNet face detector required before any bake-off GPU stage.

The age check scores detected faces only. A missing install or a model whose
bytes do not match the pin is a preflight error. There is no whole-image fallback.
"""

from __future__ import annotations

import hashlib
import os
from importlib.metadata import distributions
from pathlib import Path

# cv2 is optional at import time so a missing install can be reported as a preflight
# error. Importing it here would abort the process before that message is printed.
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

OPENCV_HEADLESS = "opencv-python-headless"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
FACE_DETECTOR_ID = "yunet"
_INPUT = (320, 320)


def default_model_path() -> Path:
    """Pinned YuNet file, or ``BAKEOFF_YUNET`` when that path is set."""
    override = os.environ.get("BAKEOFF_YUNET", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / "models" / YUNET_FILE


def installed_distributions() -> set[str]:
    """Lower-case names of installed distributions."""
    names: set[str] = set()
    for dist in distributions():
        name = dist.metadata.get("Name")
        if name:
            names.add(name.casefold())
    return names


def face_preflight(model: Path | None = None, installed: set[str] | None = None) -> list[str]:
    """Errors that must stop the run before a GPU stage. Empty means the detector can load."""
    errors: list[str] = []
    names = installed_distributions() if installed is None else {name.casefold() for name in installed}
    if OPENCV_HEADLESS not in names or cv2 is None:
        errors.append(
            "opencv-python-headless is not installed. "
            "Install it before any GPU stage. Whole-image age scoring is not allowed."
        )
    path = default_model_path() if model is None else Path(model)
    if not path.is_file():
        errors.append(
            f"YuNet model missing: {path}. "
            f"Expected {YUNET_FILE} sha256 {YUNET_SHA256}. Source: {YUNET_URL}"
        )
        return errors
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != YUNET_SHA256:
        errors.append(f"YuNet sha256 {digest} does not match pin {YUNET_SHA256} ({path})")
        return errors
    if errors:
        return errors
    detector = _create(path)
    if detector is None:
        errors.append(f"YuNet model failed to load: {path}")
    return errors


def load_detector(model: Path | None = None):
    """Load YuNet after a successful preflight."""
    errors = face_preflight(model)
    if errors:
        raise RuntimeError("face detector preflight failed:\n" + "\n".join(errors))
    detector = _create(default_model_path() if model is None else Path(model))
    if detector is None:
        raise RuntimeError("YuNet model failed to load")
    return detector


def yunet_boxes(detector, image) -> list[tuple[int, int, int, int]]:
    """Face boxes from one RGB image. An empty list means no face was detected."""
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python-headless is not installed")
    rgb = image.convert("RGB")
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    detector.setInputSize((width, height))
    found = detector.detect(bgr)
    if isinstance(found, tuple):
        found = found[1]
    if found is None:
        return []
    boxes: list[tuple[int, int, int, int]] = []
    for row in found:
        x, y, w, h = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))
    return boxes


def face_crops(image, boxes: list[tuple[int, int, int, int]], margin: float = 0.25) -> list:
    """Crops around detected faces. No boxes means no crops, including the whole image."""
    crops = []
    width = height = None
    size = getattr(image, "size", None)
    if size:
        width, height = size
    for x, y, w, h in boxes:
        pad = int(margin * max(w, h))
        left, top = max(0, x - pad), max(0, y - pad)
        right, bottom = x + w + pad, y + h + pad
        if width is not None and height is not None:
            right, bottom = min(width, right), min(height, bottom)
        crops.append(image.crop((left, top, right, bottom)))
    return crops


def _create(path: Path):
    if cv2 is None or not hasattr(cv2, "FaceDetectorYN"):
        return None
    logging = getattr(getattr(cv2, "utils", None), "logging", None)
    if logging is not None:
        logging.setLogLevel(logging.LOG_LEVEL_ERROR)
    return cv2.FaceDetectorYN.create(str(path), "", _INPUT, 0.9, 0.3, 5000)
