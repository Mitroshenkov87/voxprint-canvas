"""Classifiers."""
from dataclasses import dataclass
from typing import Protocol, Sequence

from canvas.safety.image import RawImage


@dataclass(frozen=True)
class GuardVerdict:
    """GuardVerdict."""

    safety: str
    categories: tuple[str, ...]


@dataclass(frozen=True)
class Face:
    """Face."""

    x: int
    y: int
    width: int
    height: int


class PromptGuard(Protocol):
    """Reads one prompt."""

    def assess(self, text: str) -> GuardVerdict:
        """Assess."""


class ZeroShot(Protocol):
    """ZeroShot."""

    def score_text(self, text: str, labels: Sequence[str]) -> dict[str, float]:
        """Score text."""

    def score_image(self, image: RawImage, labels: Sequence[str]) -> dict[str, float]:
        """Score image."""


class FaceDetector(Protocol):
    """FaceDetector."""

    def detect(self, image: RawImage) -> Sequence[Face]:
        """Detect."""


class AgeEstimator(Protocol):
    """AgeEstimator."""

    def estimate(self, image: RawImage, faces: Sequence[Face]) -> Sequence[tuple[float, float]]:
        """Estimate."""


class AgeRegressor(Protocol):
    """AgeRegressor."""

    def estimate(self, image: RawImage, faces: Sequence[Face]) -> Sequence[float]:
        """Estimate."""


class NsfwDetector(Protocol):
    """NsfwDetector."""

    def sexual_score(self, image: RawImage) -> float:
        """Sexual score."""
