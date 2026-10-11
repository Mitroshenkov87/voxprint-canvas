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
from typing import Literal, Sequence, assert_never

from canvas.app.i18n import text
from canvas.safety.classifiers import AgeEstimator, FaceDetector, NsfwDetector, PromptGuard, ZeroShot
from canvas.safety.constants import (
    ADOLESCENT_16_17_AT,
    ADOLESCENT_16_17_AT_AGE_18,
    CLASSIFIER_TIMEOUT_S,
    CLEAR_CHILD_AT,
    GUARD_SEXUAL_CATEGORY,
    IMAGE_ADULT_SEXUAL,
    IMAGE_LABELS,
    IMAGE_MINOR,
    IMAGE_MINOR_SEXUAL,
    IMAGE_TIMEOUT_S,
    SEXUAL_SCORE_BLOCK_AT,
    ZERO_SHOT_BLOCK_AT,
)
from canvas.safety.denylist import scan
from canvas.safety.image import RawImage, SafeImage, ScreenedFrame, ScreenedInput, _INPUT_TOKEN, _OUTPUT_TOKEN
from canvas.safety.inputs import PendingInput, role_name

log = logging.getLogger("voxprint.canvas.safety")


class SafetyError(Exception):
    """The request stopped. The message is safe to show."""

    message_key = "gate.unavailable"

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code

    @property
    def user_message(self) -> str:
        """The catalog line for this refusal, in the process language."""
        return text(self.message_key)

    def __str__(self) -> str:
        return self.user_message


class GenerationBlocked(SafetyError):
    """The request cannot be completed."""

    message_key = "gate.blocked"


class SafetyUnavailable(SafetyError):
    """A check could not finish, so the request stops."""

    message_key = "gate.unavailable"


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


def adult_age_support(text: str) -> Literal["none", "adult", "strong"]:
    """How a stated age can support an ambiguous face.

    ``strong`` means every stated age is 20 or more. ``adult`` means every stated
    age is 18 or 19, or a mix of those with older ages. No stated age, or any
    stated age under 18, is ``none``. This never changes the child band.

    The enhancer imports this module, so the age reader stays local.
    """
    from canvas.pipeline.enhance import stated_ages

    ages = stated_ages(text)
    if not ages or any(age < 18 for age in ages):
        return "none"
    if all(age >= 20 for age in ages):
        return "strong"
    return "adult"


def _band_pair(item: object) -> tuple[float, float]:
    if isinstance(item, tuple) and len(item) == 2:
        pair = item
    else:
        raise SafetyUnavailable("classifier_error")
    numbers: list[float] = []
    for value in pair:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SafetyUnavailable("classifier_error")
        number = float(value)
        if math.isnan(number) or math.isinf(number) or number < 0.0 or number > 1.0:
            raise SafetyUnavailable("classifier_error")
        numbers.append(number)
    return (numbers[0], numbers[1])


def image_block_reason(
    nsfw: float,
    bands: Sequence[object],
    scores: dict[str, float],
    *,
    support: Literal["none", "adult", "strong"] = "none",
) -> Literal["allow", "minor-sexual", "clear-child", "adolescent"]:
    """Why a picture would stop, or ``allow``.

    Adult sexual content passes. A minor-sexual image score blocks on its own.
    P(age <= 15) at the child line blocks with sexual content, and a stated age
    cannot remove that. P(16 or 17) blocks only at its own higher line. A stated
    age of 18 or 19 raises that line, and a stated age of 20 or more removes it.
    Years 18 and older are not part of either band.
    """
    parsed = tuple(_band_pair(item) for item in bands)
    minor_sexual = _at_least(_score(scores, IMAGE_MINOR_SEXUAL))
    adult_sexual = _at_least(_score(scores, IMAGE_ADULT_SEXUAL))
    _score(scores, IMAGE_MINOR)
    sexual = float(nsfw) >= SEXUAL_SCORE_BLOCK_AT or adult_sexual or minor_sexual
    clear = any(at_most_15 >= CLEAR_CHILD_AT for at_most_15, _band in parsed)
    match support:
        case "strong":
            adolescent = False
        case "adult":
            adolescent = any(band >= ADOLESCENT_16_17_AT_AGE_18 for _at_most_15, band in parsed)
        case "none":
            adolescent = any(band >= ADOLESCENT_16_17_AT for _at_most_15, band in parsed)
        case _ as unexpected:
            assert_never(unexpected)
    if minor_sexual:
        return "minor-sexual"
    if sexual and clear:
        return "clear-child"
    if sexual and adolescent:
        return "adolescent"
    return "allow"


# Qwen3Guard has one sexual category for adult sexual content and for non-consent.
# The category name itself does not name a minor. Only a category whose name names
# a minor or non-consent blocks on its own. Adult sexual content stays a signal.
_GUARD_RESTRICTED = (
    "minor",
    "child",
    "underage",
    "csam",
    "pedo",
    "non-consent",
    "nonconsent",
    "non-consensual",
    "nonconsensual",
    "rape",
)


def _guard_sexual(safety: str, categories: Sequence[str]) -> bool:
    if safety not in ("unsafe", "controversial"):
        return False
    for category in categories:
        if GUARD_SEXUAL_CATEGORY in category.casefold():
            return True
    return False


def _guard_restricted(categories: Sequence[str]) -> bool:
    """True when a guard category names a minor or non-consent, not plain adult sexual content."""
    for category in categories:
        folded = category.casefold()
        if any(marker in folded for marker in _GUARD_RESTRICTED):
            return True
    return False


def refusal_stage(text: str, safety: str, categories: Sequence[str]) -> str:
    """Why a prompt would stop, or ``allow``.

    ``lexicon-minor-sexual`` and ``lexicon-minor`` come from the term list.
    ``guard-restricted`` is a Qwen3Guard category that names a minor or non-consent.
    Unsafe or controversial plus the plain sexual category is not a block.
    """
    signals = scan(text)
    if signals.minor_sexual:
        return "lexicon-minor-sexual"
    if _guard_restricted(categories):
        return "guard-restricted"
    sexual = signals.sexual or _guard_sexual(safety, categories)
    if sexual and signals.minor:
        return "lexicon-minor"
    return "allow"


class AgeGate:
    """Decides whether a request may continue. There is no off switch."""

    def __init__(self, stack: SafetyStack) -> None:
        self._stack = stack

    def review_prompt(self, text: str) -> None:
        """Decide whether a prompt may continue."""
        verdict = _wait(lambda: self._stack.guard.assess(text), timeout=CLASSIFIER_TIMEOUT_S)
        if verdict.safety not in ("safe", "controversial", "unsafe"):
            raise SafetyUnavailable("classifier_error")
        stage = refusal_stage(text, verdict.safety, verdict.categories)
        if stage != "allow":
            log.info("content check refused a prompt (%s)", stage)
            raise GenerationBlocked("prompt")

    def release_request(self, prompt: str, mode: str, inputs: Sequence[PendingInput] = ()) -> ScreenedInput:
        """Accept a prompt and its images, or stop the request."""
        staged: list[bytearray] = []
        try:
            self.review_prompt(prompt)
            packed: list[tuple[str, int, int, bytes]] = []
            for item in inputs:
                name = role_name(item.role)
                self._check_image(item.image, "input", prompt)
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

    def review_image(self, raw: RawImage, prompt: str = "") -> SafeImage:
        """Decide whether a finished picture may be kept."""
        try:
            self._check_image(raw, "image", prompt)
            return issue(raw)
        finally:
            raw.wipe()

    def _check_image(self, raw: RawImage, code: str, prompt: str = "") -> None:
        """Run the image checks in a fixed order.

        Faces and the two age bands are read together and must agree in count.
        Sexual content blocks on a clear child band, or on the higher 16-17 band
        unless a stated adult age supports that band. A missing, non-numeric,
        out-of-range, or timed-out result raises ``SafetyUnavailable``. The
        caller wipes the picture either way.
        """
        faces = _wait(lambda: self._stack.faces.detect(raw), timeout=IMAGE_TIMEOUT_S)
        ages = _wait(lambda: self._stack.ages.estimate(raw, faces), timeout=IMAGE_TIMEOUT_S)
        if len(tuple(ages)) != len(tuple(faces)):
            raise SafetyUnavailable("classifier_error")
        nsfw = _wait(lambda: self._stack.nsfw.sexual_score(raw), timeout=IMAGE_TIMEOUT_S)
        if isinstance(nsfw, bool) or not isinstance(nsfw, (int, float)) or math.isnan(float(nsfw)) or math.isinf(float(nsfw)):
            raise SafetyUnavailable("classifier_error")
        scores = _wait(lambda: self._stack.zeroshot.score_image(raw, IMAGE_LABELS), timeout=IMAGE_TIMEOUT_S)
        reason = image_block_reason(float(nsfw), ages, scores, support=adult_age_support(prompt))
        match reason:
            case "allow":
                return
            case "minor-sexual" | "clear-child" | "adolescent":
                log.info("content check refused an image (%s)", reason)
                raise GenerationBlocked(code)
            case _ as unexpected:
                assert_never(unexpected)
