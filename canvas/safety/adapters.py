"""Classifier adapters.

The classes know which registry model they bind to. Calling an adapter before
its weights are wired refuses the request. NudeNet is not an adapter.
"""
from typing import Sequence

from canvas.models.pins import REPO_AGE, REPO_GUARD, REPO_NSFW, REPO_SIGLIP, REPO_YUNET
from canvas.safety.classifiers import Face, GuardVerdict
from canvas.safety.gate import SafetyUnavailable
from canvas.safety.image import RawImage

# Lower edge of each FairFace age bucket. 20-29 is under 25, so it stays uncertain.
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
}


def apparent_age_from_bucket(label: str) -> float:
    """Map a FairFace bucket to a conservative age in years."""
    try:
        return FAIRFACE_AGE_LOW[label]
    except KeyError as exc:
        raise SafetyUnavailable("classifier_error") from exc


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


class _Unwired:
    def _refuse(self) -> None:
        raise SafetyUnavailable("model_not_ready")


class Qwen3GuardGen(_Unwired):
    """``Qwen/Qwen3Guard-Gen-0.6B`` prompt classifier."""

    model_id = REPO_GUARD

    def assess(self, text: str) -> GuardVerdict:
        """Not wired yet."""
        del text
        self._refuse()
        raise SafetyUnavailable("model_not_ready")


class SiglipZeroShot(_Unwired):
    """``google/siglip2-so400m-patch14-384`` scores for the fixed labels."""

    model_id = REPO_SIGLIP

    def score_text(self, text: str, labels: Sequence[str]) -> dict[str, float]:
        """Not wired yet."""
        del text, labels
        self._refuse()
        return {}

    def score_image(self, image: RawImage, labels: Sequence[str]) -> dict[str, float]:
        """Not wired yet."""
        del image, labels
        self._refuse()
        return {}


class YuNetFaces(_Unwired):
    """OpenCV YuNet face detector. The weights are MIT."""

    model_id = REPO_YUNET

    def detect(self, image: RawImage) -> Sequence[Face]:
        """Not wired yet."""
        del image
        self._refuse()
        return ()


class FairFaceAge(_Unwired):
    """``dima806/fairface_age_image_detection`` apparent-age buckets."""

    model_id = REPO_AGE

    def estimate(self, image: RawImage, faces: Sequence[Face]) -> Sequence[float]:
        """Not wired yet."""
        del image, faces
        self._refuse()
        return ()


class NsfwViT(_Unwired):
    """``Falconsai/nsfw_image_detection`` (MIT). Not NudeNet."""

    model_id = REPO_NSFW

    def sexual_score(self, image: RawImage) -> float:
        """Not wired yet."""
        del image
        self._refuse()
        return 0.0
