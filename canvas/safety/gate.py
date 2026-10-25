"""Gate."""
import math
import threading
from dataclasses import dataclass
from typing import Literal, Sequence, assert_never

from canvas.app.i18n import text
from canvas.safety.classifiers import AgeEstimator, AgeRegressor, FaceDetector, NsfwDetector, PromptGuard, ZeroShot
from canvas.safety.constants import (
    ADOLESCENT_16_17_AT,
    ADOLESCENT_16_17_AT_AGE_18,
    CLASSIFIER_TIMEOUT_S,
    CLEAR_CHILD_AT,
    ESTIMATE_CONFIRM_BELOW,
    ESTIMATE_DROP_BELOW,
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
    MODEL_LOAD_TIMEOUT_S,
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
        """User message."""
        return text(self.message_key)

    def __str__(self) -> str:
        return self.user_message


class GenerationBlocked(SafetyError):
    """The request cannot be completed."""

    message_key = "gate.blocked"


class YouthContextBlocked(GenerationBlocked):
    """YouthContextBlocked."""

    message_key = "gate.blocked_youth"


class StylisedBlocked(GenerationBlocked):
    """StylisedBlocked."""

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
    fine: AgeRegressor


def issue(raw: RawImage) -> SafeImage:
    """Issue."""
    return SafeImage(_OUTPUT_TOKEN, raw.width, raw.height, raw.mode, raw.to_bytes())


def issue_input(prompt: str, mode: str, frames: tuple[tuple[str, int, int, bytes], ...]) -> ScreenedInput:
    """Issue input."""
    built = tuple(ScreenedFrame(role, width, height, data) for role, width, height, data in frames)
    return ScreenedInput(_INPUT_TOKEN, prompt, mode, built)


def _wait(fn, timeout: float):
    """Run ``fn`` on a daemon thread and stop waiting when ``timeout`` elapses."""
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            outcome["value"] = fn()
        except SafetyError as exc:
            outcome["error"] = exc
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run, name="canvas-check", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise SafetyUnavailable("timeout")
    error = outcome.get("error")
    if isinstance(error, SafetyError):
        raise error
    if isinstance(error, Exception):
        raise SafetyUnavailable("classifier_error") from error
    return outcome.get("value")


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
    """Adult age support."""
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


def _estimates(items: Sequence[object] | None, count: int) -> tuple[float, ...] | None:
    if items is None:
        return None
    values = tuple(items)
    if len(values) != count:
        raise SafetyUnavailable("classifier_error")
    numbers: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SafetyUnavailable("classifier_error")
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            raise SafetyUnavailable("classifier_error")
        numbers.append(number)
    return tuple(numbers)


def image_block_reason(
    nsfw: float,
    bands: Sequence[object],
    scores: dict[str, float],
    *,
    support: Literal["none", "adult", "strong"] = "none",
    fine: Sequence[object] | None = None,
) -> Literal[
    "allow", "minor-sexual", "clear-child", "low-estimate", "adolescent", "minor-body", "school-context", "stylised",
]:
    """Image block reason."""
    parsed = tuple(_band_pair(item) for item in bands)
    estimates = _estimates(fine, len(parsed))
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
            marks = tuple(value >= ADOLESCENT_16_17_AT_AGE_18 for _at_most_15, value in parsed)
            band = any(marks)
            if estimates is None:
                adolescent = band
            else:
                adolescent = any(mark and age < ESTIMATE_CONFIRM_BELOW for mark, age in zip(marks, estimates))
        case "none":
            marks = tuple(value >= ADOLESCENT_16_17_AT for _at_most_15, value in parsed)
            band = any(marks)
            if estimates is None:
                adolescent = band
            else:
                adolescent = any(mark and age < ESTIMATE_CONFIRM_BELOW for mark, age in zip(marks, estimates))
        case _ as unexpected:
            assert_never(unexpected)
    if minor_sexual:
        return "minor-sexual"
    if sexual and clear:
        return "clear-child"
    if sexual and estimates is not None and any(age < ESTIMATE_DROP_BELOW for age in estimates):
        return "low-estimate"
    if sexual and adolescent:
        return "adolescent"
    if sexual and minor and not parsed:
        return "minor-body"
    if sexual and (uniform or (classroom and (minor or band or clear))):
        return "school-context"
    if sexual and stylised:
        return "stylised"
    return "allow"


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
    """Guard restricted."""
    for category in categories:
        folded = category.casefold()
        if any(marker in folded for marker in _GUARD_RESTRICTED):
            return True
    return False


def refusal_stage(text: str, safety: str, categories: Sequence[str]) -> str:
    """Refusal stage."""
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


_LOAD_ORDER = ("guard", "nsfw", "zeroshot", "faces", "ages", "fine")


class AgeGate:
    """Decides whether a request may continue. There is no off switch."""

    def __init__(self, stack: SafetyStack) -> None:
        self._stack = stack
        self._load_cv = threading.Condition()
        self._load_state = "idle"
        self._load_code = "classifier_error"

    def warm(self) -> None:
        """Load each component once."""
        self._ensure_loaded()

    def release_weights(self) -> None:
        """Drop GPU weights. The next use loads them again.

        Weights that are already on the CPU stay loaded.
        """
        with self._load_cv:
            if self._load_state == "loading":
                while self._load_state == "loading":
                    self._load_cv.wait()
            if self._load_state != "ready":
                return
            on_gpu = False
            for name in _LOAD_ORDER:
                component = getattr(self._stack, name, None)
                if getattr(component, "device", None) == "cuda":
                    on_gpu = True
                    break
            if not on_gpu:
                return
            for name in _LOAD_ORDER:
                component = getattr(self._stack, name, None)
                if getattr(component, "device", None) != "cuda":
                    continue
                release = getattr(component, "release", None)
                if callable(release):
                    release()
            self._load_state = "idle"

    def _ensure_loaded(self) -> None:
        """One load for this gate. Later callers wait for it."""
        with self._load_cv:
            if self._load_state == "ready":
                return
            if self._load_state == "failed":
                raise SafetyUnavailable(self._load_code)
            if self._load_state == "loading":
                while self._load_state == "loading":
                    self._load_cv.wait()
                if self._load_state == "ready":
                    return
                raise SafetyUnavailable(self._load_code)
            self._load_state = "loading"
        failed = True
        code = "classifier_error"
        try:
            self._load_components()
            failed = False
        except SafetyUnavailable as exc:
            code = exc.code
            raise
        finally:
            with self._load_cv:
                if failed:
                    self._load_state = "failed"
                    self._load_code = code
                else:
                    self._load_state = "ready"
                self._load_cv.notify_all()

    def _load_components(self) -> None:
        """Load each component under its own bound."""
        for name in _LOAD_ORDER:
            component = getattr(self._stack, name)
            load = getattr(component, "load", None)
            if not callable(load):
                continue
            _wait(load, timeout=MODEL_LOAD_TIMEOUT_S)

    def review_prompt(self, text: str) -> None:
        """Decide whether a prompt may continue."""
        self._screen_prompt(text)

    def _screen_prompt(self, text: str) -> bool:
        """Screen a prompt."""
        self._ensure_loaded()
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
        """Review prompts."""
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
        self._ensure_loaded()
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
        fine: tuple[object, ...] = ()
        if faces:
            ages = tuple(_wait(lambda: self._stack.ages.estimate(raw, faces), timeout=IMAGE_TIMEOUT_S))
            if len(ages) != len(faces):
                raise SafetyUnavailable("classifier_error")
            fine = tuple(_wait(lambda: self._stack.fine.estimate(raw, faces), timeout=IMAGE_TIMEOUT_S))
            if len(fine) != len(faces):
                raise SafetyUnavailable("classifier_error")
        reason = image_block_reason(nsfw, ages, scores, support=adult_age_support(prompt), fine=fine)
        match reason:
            case "allow":
                return
            case (
                "minor-sexual" | "clear-child" | "low-estimate" | "adolescent" | "minor-body" | "school-context"
                | "stylised"
            ):
                raise GenerationBlocked(code)
            case _ as unexpected:
                assert_never(unexpected)
