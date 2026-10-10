"""Protocols for the safety adapters. Tests supply fakes. Production loads the stubs."""
from dataclasses import dataclass
from typing import Protocol, Sequence

from canvas.safety.image import RawImage


@dataclass(frozen=True)
class GuardVerdict:
    """Parsed Qwen3Guard-Gen judgement. ``safety`` is safe, controversial or unsafe."""

    safety: str
    categories: tuple[str, ...]


@dataclass(frozen=True)
class Face:
    x: int
    y: int
    width: int
    height: int


class PromptGuard(Protocol):
    def assess(self, text: str) -> GuardVerdict:
        """Classify one prompt. Raising refuses generation."""


class ZeroShot(Protocol):
    def score_text(self, text: str, labels: Sequence[str]) -> dict[str, float]:
        """SigLIP2 text scores for the fixed labels."""

    def score_image(self, image: RawImage, labels: Sequence[str]) -> dict[str, float]:
        """SigLIP2 image scores for the fixed labels."""


class FaceDetector(Protocol):
    def detect(self, image: RawImage) -> Sequence[Face]:
        """YuNet boxes. An empty sequence means no face, not a failure."""


class AgeEstimator(Protocol):
    def estimate(self, image: RawImage, faces: Sequence[Face]) -> Sequence[float]:
        """Apparent age in years, one float per face, lower-bound conservative."""


class NsfwDetector(Protocol):
    def sexual_score(self, image: RawImage) -> float:
        """Probability that the image is sexual, from 0 to 1."""
