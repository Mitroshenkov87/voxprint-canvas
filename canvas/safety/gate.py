"""Gate."""
import contextlib
import hashlib
import math
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Sequence, assert_never

from canvas.app.i18n import text
from canvas.infra.stage_timer import beat_while, log_detail
from canvas.infra.vram_trace import on_cuda
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


def release_generated(raw: RawImage) -> SafeImage:
    """Release a picture Canvas generated. The single exit for every generate path.

    Canvas screens the request, not the result: the prompt check and the
    input-image checks ran in :meth:`AgeGate.release_request` before the
    generator was called, and a request they refuse never reaches it. No
    classifier runs on generated pixels (owner decision 2026-11-16). The raw
    buffer is wiped, as :meth:`AgeGate.review_image` does.
    """
    try:
        return issue(raw)
    finally:
        raw.wipe()


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


def input_minor_reason(
    bands: Sequence[object],
    scores: dict[str, float],
    *,
    support: Literal["none", "adult", "strong"] = "none",
    fine: Sequence[object] | None = None,
) -> Literal["allow", "minor-sexual", "clear-child", "low-estimate", "adolescent", "minor-body"]:
    """The minor criteria of :func:`image_block_reason` for an uploaded photo in an 18+ edit.

    The edit is 18+, so the photo is treated as sexual whatever its NSFW score.
    The thresholds are the same. School and stylised cues alone do not refuse.
    """
    reason = image_block_reason(1.0, bands, scores, support=support, fine=fine)
    match reason:
        case "school-context" | "stylised" | "allow":
            return "allow"
        case "minor-sexual" | "clear-child" | "low-estimate" | "adolescent" | "minor-body":
            return reason
        case _ as unexpected:
            assert_never(unexpected)


# Guard categories that describe a minor. They count as the under-18 side of the rule.
_GUARD_MINOR = ("minor", "child", "underage", "csam", "pedo")
# Guard categories that already are a sexualised minor (both sides at once).
_GUARD_MINOR_SEXUAL = ("csam", "pedo", "child sexual", "child-sexual", "child_sexual", "minor sexual",
                       "sexual minor", "minors sexual")


def _guard_sexual(safety: str, categories: Sequence[str]) -> bool:
    if safety not in ("unsafe", "controversial"):
        return False
    for category in categories:
        if GUARD_SEXUAL_CATEGORY in category.casefold():
            return True
    return False


def _guard_marked(categories: Sequence[str], markers: tuple[str, ...]) -> bool:
    for category in categories:
        folded = category.casefold()
        if any(marker in folded for marker in markers):
            return True
    return False


def adult_content(text: str, safety: str, categories: Sequence[str], *, explicit: bool = False) -> bool:
    """The 18+ side: the 18+ switch is on, the text hints at 18+ content, or the guard says sexual."""
    return explicit or scan(text).sexual or _guard_sexual(safety, categories)


def refusal_stage(text: str, safety: str, categories: Sequence[str], *, explicit: bool = False) -> str:
    """Refuse only a prompt that describes someone under 18 AND hints at 18+ content.

    Owner rule (2026-11-16): the only refusal is a minor in an 18+ context.
    Self-evidently sexualised-minor terms are both sides at once.
    """
    signals = scan(text)
    if signals.minor_sexual:
        return "lexicon-minor-sexual"
    if _guard_marked(categories, _GUARD_MINOR_SEXUAL):
        return "guard-minor-sexual"
    if not adult_content(text, safety, categories, explicit=explicit):
        return "allow"
    if signals.minor:
        return "lexicon-minor"
    if _guard_marked(categories, _GUARD_MINOR):
        return "guard-minor"
    return "allow"


_LOAD_ORDER = ("guard", "nsfw", "zeroshot", "faces", "ages", "fine")


def _component_on_gpu(component: object) -> bool:
    """True when the component's flag or its tensors say the weights are on the GPU."""
    if component is None:
        return False
    if getattr(component, "device", None) == "cuda":
        return True
    model = getattr(component, "_model", None)
    if model is None:
        return False
    return on_cuda(model)


def input_run(gate: object) -> contextlib.AbstractContextManager[None]:
    """The gate's run scope, or a no-op for a gate without one (test fakes)."""
    scope = getattr(gate, "input_run", None)
    return scope() if callable(scope) else contextlib.nullcontext()


class AgeGate:
    """Decides whether a request may continue. There is no off switch."""

    def __init__(self, stack: SafetyStack) -> None:
        self._stack = stack
        self._load_cv = threading.Condition()
        self._load_state = "idle"
        self._load_code = "classifier_error"
        self._loaded: set[str] = set()
        self._run_lock = threading.Lock()
        self._run_depth = 0
        self._verdicts: dict[tuple[str, str], str] = {}

    @contextlib.contextmanager
    def input_run(self) -> Iterator[None]:
        """One run (a --count/--seeds/--prompts run, a series, a batch): an uploaded photo is age-checked once.

        Verdicts are keyed by the photo's sha256 and the adult-age support of
        the prompt, and live until the outermost run ends. A checker error is
        never stored.
        """
        with self._run_lock:
            self._run_depth += 1
        try:
            yield
        finally:
            with self._run_lock:
                self._run_depth -= 1
                if self._run_depth == 0:
                    self._verdicts.clear()

    def warm(self) -> None:
        """Load each component once. For input-image screening and tools; generation uses ``warm_prompt``."""
        self._ensure_loaded()

    def warm_prompt(self) -> None:
        """Load only the prompt guard. Generation needs nothing else.

        The image models load on first use, inside ``_check_image``, when an
        uploaded image is screened. A guard that cannot load raises
        ``SafetyUnavailable``, as ``warm`` does.
        """
        self._ensure_some(("guard",))

    def release_weights(self) -> None:
        """Drop GPU weights. The next use loads them again.

        Weights that are already on the CPU stay loaded. A component whose
        device flag says CPU while its tensors are still on the GPU is parked
        too: the flag is not the allocation.
        """
        with self._load_cv:
            if self._load_state == "loading":
                while self._load_state == "loading":
                    self._load_cv.wait()
            targets: list[object] = []
            for name in _LOAD_ORDER:
                component = getattr(self._stack, name, None)
                if _component_on_gpu(component):
                    targets.append(component)
            if not targets:
                return
            for component in targets:
                release = getattr(component, "release", None)
                if callable(release):
                    release()
            self._loaded.clear()
            if self._load_state == "ready":
                self._load_state = "idle"

    def _ensure_loaded(self) -> None:
        """Load every component once. Later callers wait for it."""
        self._ensure_some(_LOAD_ORDER)

    def _ensure_some(self, names: tuple[str, ...]) -> None:
        """Load ``names`` that are not already in RAM. A prompt does not load the image models."""
        with self._load_cv:
            if self._load_state == "failed":
                raise SafetyUnavailable(self._load_code)
            while self._load_state == "loading":
                self._load_cv.wait()
            if self._load_state == "failed":
                raise SafetyUnavailable(self._load_code)
            pending = tuple(name for name in names if name not in self._loaded)
            if not pending:
                return
            self._load_state = "loading"
        failed = True
        code = "classifier_error"
        try:
            with beat_while("warmup"):
                self._load_components(pending)
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
                    self._loaded.update(pending)
                    if set(_LOAD_ORDER) <= self._loaded:
                        self._load_state = "ready"
                    else:
                        self._load_state = "idle"
                self._load_cv.notify_all()

    def _load_components(self, names: tuple[str, ...] = _LOAD_ORDER) -> None:
        """Load each component under its own bound.

        The wall time is the check's load gap: the forward and the weight
        copy are logged on their own. This line is the rest of ``load``.
        """
        for name in names:
            component = getattr(self._stack, name)
            load = getattr(component, "load", None)
            if not callable(load):
                continue
            started = time.perf_counter()
            _wait(load, timeout=MODEL_LOAD_TIMEOUT_S)
            elapsed = max(0, int((time.perf_counter() - started) * 1000))
            flag = getattr(component, "device", "cpu")
            kind = "cuda" if "cuda" in str(flag) else "cpu"
            log_detail("check_load", "model", kind, f"name={name} ms={elapsed}")

    def review_prompt(self, text: str) -> None:
        """Decide whether a prompt may continue."""
        self._screen_prompt(text)

    def _screen_prompt(self, text: str, *, explicit: bool = False) -> bool:
        """Screen a prompt. True when the request is 18+ (the same detector the photo check uses)."""
        self._ensure_some(("guard",))
        verdict = _wait(lambda: self._stack.guard.assess(text), timeout=CLASSIFIER_TIMEOUT_S)
        if verdict.safety not in ("safe", "controversial", "unsafe"):
            raise SafetyUnavailable("classifier_error")
        categories = tuple(verdict.categories)
        if not all(isinstance(category, str) for category in categories):
            raise SafetyUnavailable("classifier_error")
        if refusal_stage(text, verdict.safety, categories, explicit=explicit) != "allow":
            raise GenerationBlocked("prompt")
        return adult_content(text, verdict.safety, categories, explicit=explicit)

    def review_prompts(self, texts: Sequence[str]) -> None:
        """Review prompts."""
        for item in texts:
            self.review_prompt(item)

    def release_request(
        self, prompt: str, mode: str, inputs: Sequence[PendingInput] = (), *, explicit: bool = False,
    ) -> ScreenedInput:
        """Screen a request: the prompt, then each uploaded photo of an 18+ edit.

        A photo is age-checked only when the edit is 18+ (prompt hints, guard
        sexual category, or the 18+ switch). A request that is not 18+ loads
        and runs no image model.
        """
        staged: list[bytearray] = []
        try:
            adult = self._screen_prompt(prompt, explicit=explicit)
            packed: list[tuple[str, int, int, bytes]] = []
            local: dict[tuple[str, str], str] = {}
            for item in inputs:
                name = role_name(item.role)
                data = item.image.to_bytes()
                if adult:
                    self._check_input(item.image, data, prompt, local)
                staged.append(bytearray(data))
                packed.append((name, item.image.width, item.image.height, bytes(staged[-1])))
            return issue_input(prompt, mode, tuple(packed))
        finally:
            for buf in staged:
                for index in range(len(buf)):
                    buf[index] = 0
                buf.clear()
            for item in inputs:
                item.image.wipe()

    def _check_input(self, raw: RawImage, data: bytes, prompt: str, local: dict[tuple[str, str], str]) -> None:
        """Age-check an uploaded photo of an 18+ edit once per run. Refuse only a minor."""
        support = adult_age_support(prompt)
        key = (hashlib.sha256(data).hexdigest(), support)
        with self._run_lock:
            shared = self._verdicts if self._run_depth > 0 else local
            reason = shared.get(key)
        if reason is None:
            try:
                reason = self._input_reason(raw, support)
            except SafetyError:
                raise
            except Exception as exc:
                raise SafetyUnavailable("classifier_error") from exc
            with self._run_lock:
                (self._verdicts if self._run_depth > 0 else local)[key] = reason
        if reason != "allow":
            raise GenerationBlocked("input")

    def _input_reason(self, raw: RawImage, support: Literal["none", "adult", "strong"]) -> str:
        """Run the age models on an uploaded photo. Any checker failure raises ``SafetyUnavailable``."""
        self._ensure_some(("guard", "zeroshot", "faces", "ages", "fine"))
        scores = _wait(lambda: self._stack.zeroshot.score_image(raw, IMAGE_LABELS), timeout=IMAGE_TIMEOUT_S)
        if not isinstance(scores, dict):
            raise SafetyUnavailable("classifier_error")
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
        return input_minor_reason(ages, scores, support=support, fine=fine)

    def review_image(self, raw: RawImage, prompt: str = "") -> SafeImage:
        """Run the image checks on a picture, then release it.

        No generate path calls this: generated pictures leave through
        :func:`release_generated`. Kept for evaluation tools and tests.
        """
        try:
            self._check_image(raw, "image", prompt)
            return issue(raw)
        finally:
            raw.wipe()

    def _check_image(self, raw: RawImage, code: str, prompt: str = "", *, full: bool = False) -> None:
        """Screen an image."""
        self._ensure_some(_LOAD_ORDER)
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
