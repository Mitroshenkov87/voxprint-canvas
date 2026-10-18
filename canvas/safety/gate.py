"""Content checks before a generator runs and before an output image can leave.

Input images are screened before a ``ScreenedInput`` is issued. Output images are
screened before a ``SafeImage`` is issued. A missing model, a bad hash, a load
error, a timeout or any other classifier error refuses the request. Nothing in
this module reads settings, environment variables or flags.

Product rule (2026-10-18). The prompt check is the primary filter: it runs on the
final prompt string a generator receives, on every route, because a generator
accepts only a ``ScreenedInput``. Explicit or nude intent together with a child,
teen or minor term, a stated age under 18, or a school or student cue blocks. A
clothed school or student scene passes. The output check is a lighter net: in a
sexual frame it drops a clear child or adolescent reading at the existing
thresholds, a minor-sexual image score, a visible school uniform or classroom, or
an anime, manga, cartoon or chibi character. Explicit intent with an anime or
stylised-character cue also blocks the prompt, whatever age is stated.
A youngest-adult or babyface reading with no other marker passes.

Nothing here logs or stores a decision, a score, a prompt or any pixel data.
"""
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
    IMAGE_ANIME,
    IMAGE_CHIBI,
    IMAGE_LABELS,
    IMAGE_MINOR,
    IMAGE_MINOR_SEXUAL,
    IMAGE_CLASSROOM,
    IMAGE_SCHOOL_UNIFORM,
    IMAGE_TIMEOUT_S,
    SEXUAL_SCORE_BLOCK_AT,
    ZERO_SHOT_BLOCK_AT,
)
from canvas.safety.denylist import scan
from canvas.safety.image import RawImage, SafeImage, ScreenedFrame, ScreenedInput, _INPUT_TOKEN, _OUTPUT_TOKEN
from canvas.safety.inputs import PendingInput, role_name

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


class YouthContextBlocked(GenerationBlocked):
    """Explicit content was combined with a school, student or young-age cue."""

    message_key = "gate.blocked_youth"


class StylisedBlocked(GenerationBlocked):
    """Explicit content was combined with an anime or other stylised character."""

    message_key = "gate.blocked_stylised"


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
) -> Literal["allow", "minor-sexual", "clear-child", "adolescent", "minor-body", "school-context", "stylised"]:
    """Score an image."""
    parsed = tuple(_band_pair(item) for item in bands)
    minor_sexual = _at_least(_score(scores, IMAGE_MINOR_SEXUAL))
    adult_sexual = _at_least(_score(scores, IMAGE_ADULT_SEXUAL))
    minor = _at_least(_score(scores, IMAGE_MINOR))
    uniform = _at_least(_score(scores, IMAGE_SCHOOL_UNIFORM))
    classroom = _at_least(_score(scores, IMAGE_CLASSROOM))
    stylised = _at_least(_score(scores, IMAGE_ANIME)) or _at_least(_score(scores, IMAGE_CHIBI))
    sexual = float(nsfw) >= SEXUAL_SCORE_BLOCK_AT or adult_sexual or minor_sexual
    clear = any(at_most_15 >= CLEAR_CHILD_AT for at_most_15, _band in parsed)
    match support:
        case "strong":
            band = False
            adolescent = False
        case "adult":
            band = any(value >= ADOLESCENT_16_17_AT_AGE_18 for _at_most_15, value in parsed)
            adolescent = band
        case "none":
            band = any(value >= ADOLESCENT_16_17_AT for _at_most_15, value in parsed)
            adolescent = band
        case _ as unexpected:
            assert_never(unexpected)
    if minor_sexual:
        return "minor-sexual"
    if sexual and clear:
        return "clear-child"
    if sexual and adolescent:
        return "adolescent"
    if sexual and minor and not parsed:
        return "minor-body"
    if sexual and (uniform or (classroom and (minor or band or clear))):
        return "school-context"
    if sexual and stylised:
        return "stylised"
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
    ``lexicon-youth-context`` is explicit intent (term list or the Qwen3Guard
    sexual category) together with a school or student cue.
    ``lexicon-stylised`` is explicit intent together with an anime or other
    stylised-character cue; a stated adult age does not change it. Unsafe or
    controversial plus the plain sexual category alone is not a block.
    """
    signals = scan(text)
    if signals.minor_sexual:
        return "lexicon-minor-sexual"
    if _guard_restricted(categories):
        return "guard-restricted"
    sexual = signals.sexual or _guard_sexual(safety, categories)
    if sexual and signals.minor:
        return "lexicon-minor"
    if sexual and signals.youth_context:
        return "lexicon-youth-context"
    if sexual and signals.stylised:
        return "lexicon-stylised"
    return "allow"


class AgeGate:
    """Decides whether a request may continue. There is no off switch."""

    def __init__(self, stack: SafetyStack) -> None:
        self._stack = stack

    def review_prompt(self, text: str) -> None:
        """Decide whether a prompt may continue."""
        self._screen_prompt(text)

    def _screen_prompt(self, text: str) -> bool:
        """Screen a prompt."""
        verdict = _wait(lambda: self._stack.guard.assess(text), timeout=CLASSIFIER_TIMEOUT_S)
        if verdict.safety not in ("safe", "controversial", "unsafe"):
            raise SafetyUnavailable("classifier_error")
        stage = refusal_stage(text, verdict.safety, verdict.categories)
        if stage == "lexicon-youth-context":
            raise YouthContextBlocked("prompt")
        if stage == "lexicon-stylised":
            raise StylisedBlocked("prompt")
        if stage != "allow":
            raise GenerationBlocked("prompt")
        return scan(text).sexual or _guard_sexual(verdict.safety, verdict.categories)

    def review_prompts(self, texts: Sequence[str]) -> None:
        """Screen every text of a multi-part request, such as an idea, a scenario and each shot line.

        One blocked text stops the whole request. Each shot still goes through
        ``release_request`` with its final prompt before it is generated.
        """
        for item in texts:
            self.review_prompt(item)

    def release_request(
        self, prompt: str, mode: str, inputs: Sequence[PendingInput] = (), *, explicit: bool = False,
    ) -> ScreenedInput:
        """Screen a request."""
        staged: list[bytearray] = []
        try:
            full = self._screen_prompt(prompt) or explicit
            packed: list[tuple[str, int, int, bytes]] = []
            for item in inputs:
                name = role_name(item.role)
                self._check_image(item.image, "input", prompt, full=full)
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

    def _check_image(self, raw: RawImage, code: str, prompt: str = "", *, full: bool = False) -> None:
        """Screen an image."""
        try:
            score = _wait(lambda: self._stack.nsfw.sexual_score(raw), timeout=IMAGE_TIMEOUT_S)
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise SafetyUnavailable("classifier_error")
            nsfw = float(score)
            if math.isnan(nsfw) or math.isinf(nsfw):
                raise SafetyUnavailable("classifier_error")
        except SafetyUnavailable:
            nsfw = 1.0
        if nsfw < SEXUAL_SCORE_BLOCK_AT and not full:
            return
        scores = _wait(lambda: self._stack.zeroshot.score_image(raw, IMAGE_LABELS), timeout=IMAGE_TIMEOUT_S)
        faces = tuple(_wait(lambda: self._stack.faces.detect(raw), timeout=IMAGE_TIMEOUT_S))
        ages: tuple[object, ...] = ()
        if faces:
            ages = tuple(_wait(lambda: self._stack.ages.estimate(raw, faces), timeout=IMAGE_TIMEOUT_S))
            if len(ages) != len(faces):
                raise SafetyUnavailable("classifier_error")
        reason = image_block_reason(nsfw, ages, scores, support=adult_age_support(prompt))
        match reason:
            case "allow":
                return
            case "minor-sexual" | "clear-child" | "adolescent" | "minor-body" | "school-context" | "stylised":
                raise GenerationBlocked(code)
            case _ as unexpected:
                assert_never(unexpected)
