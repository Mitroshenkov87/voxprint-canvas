"""Content checks before a generator runs and before an output image can leave.

Input images are screened before a ``ScreenedInput`` is issued. Output images are
screened before a ``SafeImage`` is issued. A missing model, a bad hash, a load
error, a timeout or any other classifier error refuses the request. Nothing in
this module reads settings, environment variables or flags.
"""
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Sequence

from canvas.safety.classifiers import AgeEstimator, FaceDetector, NsfwDetector, PromptGuard, ZeroShot
from canvas.safety.constants import (
    APPARENT_AGE_UNCERTAIN_UNDER,
    BLOCKED_MESSAGE,
    CLASSIFIER_TIMEOUT_S,
    GUARD_SEXUAL_CATEGORY,
    IMAGE_ADULT_SEXUAL,
    IMAGE_LABELS,
    IMAGE_MINOR,
    IMAGE_MINOR_SEXUAL,
    IMAGE_TIMEOUT_S,
    PROMPT_LABELS,
    PROMPT_MINOR,
    PROMPT_MINOR_SEXUAL,
    PROMPT_SEXUAL,
    SEXUAL_SCORE_BLOCK_AT,
    UNAVAILABLE_MESSAGE,
    ZERO_SHOT_BLOCK_AT,
)
from canvas.safety.denylist import scan
from canvas.safety.image import RawImage, SafeImage, ScreenedFrame, ScreenedInput, _INPUT_TOKEN, _OUTPUT_TOKEN
from canvas.safety.inputs import PendingInput, role_name

log = logging.getLogger("voxprint.canvas.safety")


class SafetyError(Exception):
    """The request stopped. The message is safe to show."""

    user_message = UNAVAILABLE_MESSAGE

    def __init__(self, code: str) -> None:
        super().__init__(self.user_message)
        self.code = code

    def __str__(self) -> str:
        return self.user_message


class GenerationBlocked(SafetyError):
    """The request cannot be completed."""

    user_message = BLOCKED_MESSAGE


class SafetyUnavailable(SafetyError):
    """A check could not finish, so the request stops."""

    user_message = UNAVAILABLE_MESSAGE


@dataclass(frozen=True)
class SafetyStack:
    """What the gate needs before it can run."""

    guard: PromptGuard
    zeroshot: ZeroShot
    faces: FaceDetector
    ages: AgeEstimator
    nsfw: NsfwDetector


def issue(raw: RawImage) -> SafeImage:
    """Copy released pixels into a SafeImage. Only ``review_image`` may call this."""
    return SafeImage(_OUTPUT_TOKEN, raw.width, raw.height, raw.mode, raw.to_bytes())


def issue_input(prompt: str, mode: str, frames: tuple[tuple[str, int, int, bytes], ...]) -> ScreenedInput:
    """Copy screened input images. Only ``release_request`` may call this."""
    built = tuple(ScreenedFrame(role, width, height, data) for role, width, height, data in frames)
    return ScreenedInput(_INPUT_TOKEN, prompt, mode, built)


def _wait(fn, timeout: float):
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout as exc:
            raise SafetyUnavailable("timeout") from exc
        except SafetyError:
            raise
        except Exception as exc:
            raise SafetyUnavailable("classifier_error") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _score(scores: dict[str, float], label: str) -> float:
    try:
        value = float(scores[label])
    except (KeyError, TypeError, ValueError) as exc:
        raise SafetyUnavailable("classifier_error") from exc
    if math.isnan(value) or math.isinf(value):
        raise SafetyUnavailable("classifier_error")
    return value


def _at_least(value: float) -> bool:
    return value >= ZERO_SHOT_BLOCK_AT


def _guard_sexual(safety: str, categories: Sequence[str]) -> bool:
    if safety not in ("unsafe", "controversial"):
        return False
    for category in categories:
        if GUARD_SEXUAL_CATEGORY in category.casefold():
            return True
    return False


class AgeGate:
    """Decides whether a request may continue. There is no off switch."""

    def __init__(self, stack: SafetyStack) -> None:
        self._stack = stack

    def review_prompt(self, text: str) -> None:
        """Decide whether a prompt may continue."""
        signals = scan(text)
        verdict = _wait(lambda: self._stack.guard.assess(text), timeout=CLASSIFIER_TIMEOUT_S)
        if verdict.safety not in ("safe", "controversial", "unsafe"):
            raise SafetyUnavailable("classifier_error")
        scores = _wait(lambda: self._stack.zeroshot.score_text(text, PROMPT_LABELS), timeout=CLASSIFIER_TIMEOUT_S)
        minor_sexual = _at_least(_score(scores, PROMPT_MINOR_SEXUAL))
        sexual = signals.sexual or _guard_sexual(verdict.safety, verdict.categories) or _at_least(_score(scores, PROMPT_SEXUAL))
        minor = signals.minor or minor_sexual or _at_least(_score(scores, PROMPT_MINOR))
        if minor_sexual or (sexual and minor):
            log.info("content check refused a prompt")
            raise GenerationBlocked("prompt")

    def release_request(self, prompt: str, mode: str, inputs: Sequence[PendingInput] = ()) -> ScreenedInput:
        """Accept a prompt and its images, or stop the request."""
        staged: list[bytearray] = []
        try:
            self.review_prompt(prompt)
            packed: list[tuple[str, int, int, bytes]] = []
            for item in inputs:
                name = role_name(item.role)
                self._check_image(item.image, "input")
                staged.append(bytearray(item.image.to_bytes()))
                packed.append((name, item.image.width, item.image.height, bytes(staged[-1])))
            return issue_input(prompt, mode, tuple(packed))
        finally:
            for buf in staged:
                for index in range(len(buf)):
                    buf[index] = 0
                buf.clear()
            for item in inputs:
                item.image.wipe()

    def review_image(self, raw: RawImage) -> SafeImage:
        """Decide whether a finished picture may be kept."""
        try:
            self._check_image(raw, "image")
            return issue(raw)
        finally:
            raw.wipe()

    def _check_image(self, raw: RawImage, code: str) -> None:
        """Run the image checks in a fixed order.

        Faces and ages are read together and must agree in count. A young face
        combined with a sexual score, or a score that already joins those two,
        raises ``GenerationBlocked``. A missing, non-numeric, or timed-out result
        raises ``SafetyUnavailable``. The caller wipes the picture either way.
        """
        faces = _wait(lambda: self._stack.faces.detect(raw), timeout=IMAGE_TIMEOUT_S)
        ages = _wait(lambda: self._stack.ages.estimate(raw, faces), timeout=IMAGE_TIMEOUT_S)
        if len(tuple(ages)) != len(tuple(faces)):
            raise SafetyUnavailable("classifier_error")
        for age in ages:
            if isinstance(age, bool) or not isinstance(age, (int, float)) or math.isnan(float(age)):
                raise SafetyUnavailable("classifier_error")
        nsfw = _wait(lambda: self._stack.nsfw.sexual_score(raw), timeout=IMAGE_TIMEOUT_S)
        if isinstance(nsfw, bool) or not isinstance(nsfw, (int, float)) or math.isnan(float(nsfw)):
            raise SafetyUnavailable("classifier_error")
        scores = _wait(lambda: self._stack.zeroshot.score_image(raw, IMAGE_LABELS), timeout=IMAGE_TIMEOUT_S)
        minor_sexual = _at_least(_score(scores, IMAGE_MINOR_SEXUAL))
        sexual = float(nsfw) >= SEXUAL_SCORE_BLOCK_AT or _at_least(_score(scores, IMAGE_ADULT_SEXUAL)) or minor_sexual
        young = any(float(age) < APPARENT_AGE_UNCERTAIN_UNDER for age in ages) or _at_least(_score(scores, IMAGE_MINOR))
        if minor_sexual or (sexual and young):
            log.info("content check refused an image")
            raise GenerationBlocked(code)
