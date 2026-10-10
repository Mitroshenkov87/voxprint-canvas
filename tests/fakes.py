"""In-memory classifiers and a small RGB image. No weights are loaded."""
from __future__ import annotations

from canvas.safety.classifiers import Face, GuardVerdict
from canvas.safety.constants import IMAGE_LABELS, PROMPT_LABELS
from canvas.safety.gate import AgeGate, SafetyStack
from canvas.safety.image import RawImage


def rgb(width: int = 2, height: int = 2, color: tuple[int, int, int] = (10, 20, 30)) -> RawImage:
    pixel = bytes(color)
    return RawImage(width, height, pixel * (width * height))


class World:
    """Mutable stand-in for the five classifiers."""

    def __init__(self) -> None:
        self.safety = "safe"
        self.categories: tuple[str, ...] = ()
        self.text_scores = {label: 0.0 for label in PROMPT_LABELS}
        self.image_scores = {label: 0.0 for label in IMAGE_LABELS}
        self.faces: list[Face] = []
        self.ages: list[float] = []
        self.nsfw = 0.0
        self.error = ""
        self.calls: list[str] = []

    def stack(self) -> SafetyStack:
        return SafetyStack(_Guard(self), _Zero(self), _Faces(self), _Ages(self), _Nsfw(self))

    def gate(self) -> AgeGate:
        return AgeGate(self.stack())


class _Guard:
    def __init__(self, world: World) -> None:
        self.world = world

    def assess(self, text: str) -> GuardVerdict:
        del text
        self.world.calls.append("guard")
        if self.world.error == "guard":
            raise RuntimeError("guard failed")
        return GuardVerdict(self.world.safety, self.world.categories)


class _Zero:
    def __init__(self, world: World) -> None:
        self.world = world

    def score_text(self, text: str, labels: object) -> dict[str, float]:
        del text
        self.world.calls.append("text")
        if self.world.error == "zero-text":
            raise RuntimeError("text scores failed")
        return {str(label): self.world.text_scores[str(label)] for label in labels}  # type: ignore[union-attr]

    def score_image(self, image: RawImage, labels: object) -> dict[str, float]:
        del image
        self.world.calls.append("image-score")
        if self.world.error == "zero-image":
            raise RuntimeError("image scores failed")
        return {str(label): self.world.image_scores[str(label)] for label in labels}  # type: ignore[union-attr]


class _Faces:
    def __init__(self, world: World) -> None:
        self.world = world

    def detect(self, image: RawImage) -> list[Face]:
        del image
        self.world.calls.append("faces")
        if self.world.error == "faces":
            raise RuntimeError("faces failed")
        return list(self.world.faces)


class _Ages:
    def __init__(self, world: World) -> None:
        self.world = world

    def estimate(self, image: RawImage, faces: object) -> list[float]:
        del image, faces
        self.world.calls.append("ages")
        if self.world.error == "ages":
            raise RuntimeError("ages failed")
        return list(self.world.ages)


class _Nsfw:
    def __init__(self, world: World) -> None:
        self.world = world

    def sexual_score(self, image: RawImage) -> float:
        del image
        self.world.calls.append("nsfw")
        if self.world.error == "nsfw":
            raise RuntimeError("score failed")
        return self.world.nsfw
