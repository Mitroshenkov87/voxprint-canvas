"""Run one shell job through the existing pipeline and the same content gate."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, TypeVar

from PIL import Image

from canvas.app.i18n import text
from canvas.infra.enlarge import EnlargeError
from canvas.infra.errors import note_crash
from canvas.infra.gpu_lock import GpuLockTimeout, lock_path, release, try_acquire
from canvas.infra.paths import models_dir
from canvas.models.loader import ModelNotReady, ZImageGenerator, plan_specs
from canvas.models.user_lora import UserLora
from canvas.models.mdl_registry import selectable
from canvas.models.modes import Tier
from canvas.models.plan import BaseNotSelected, ModeUnavailable, build_plan
from canvas.models.provision import DownloadDeclined
from canvas.models.registry import load_catalog
from canvas.models.runtime import InferenceUnavailable, prepare_series_embeds
from canvas.models.sizes import frame_size
from canvas.models.snapshots import SnapshotError, snapshot_ready
from canvas.models.verified import flush_process_cache, process_cache
from canvas.pipeline.bf16_schedule import VramRefused
from canvas.pipeline.device_guard import DeviceRefused
from canvas.pipeline.adult_direction import improve_adult
from canvas.pipeline.enhance import SYSTEM_PROMPT as ENHANCE_SYSTEM
from canvas.pipeline.enhance import open_rewriter, resolve_prompt, rewrite_text
from canvas.pipeline.gpu_release import release_gate_weights
from canvas.pipeline.engine import run_generator, run_sampler
from canvas.pipeline.pipeline import CoolingDown, EnlargeMissing, JobError, Pipeline
from canvas.safety.gate import AgeGate, SafetyError, SafetyUnavailable, issue_input
from canvas.safety.image import RawImage, SafeImage
from canvas.safety.inputs import PendingInput, role_name
from canvas.ui.shell_layout import PictureJob

_T = TypeVar("_T")


class RunStopped(Exception):
    """The picture stopped with a sentence the window can show."""

    def __init__(self, message: str, *, unavailable: bool = False, code: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.unavailable = unavailable
        self.code = code


class ShellRunner:
    """Turns one :class:`PictureJob` into pixels. The window reviews them again."""

    def __init__(self, *, gate: AgeGate, tier: Tier, enlarge: object) -> None:
        """Keep the gate the window already holds.

        Args:
            gate: The same instance passed to the screen.
            tier: The tier chosen at startup.
        adult: Expert adult content is on. Adds the adult photo direction; a reply
            with a youth cue or an age under 18 is refused, never rephrased.
            enlarge: Enlarge client. 4K uses it.
        """
        self.gate = gate
        self.tier = tier
        self.enlarge = enlarge
        self.img2img_strength: float | None = None
        self.bound_lora: UserLora | None = None
        self.lora_strength: float | None = None
        # A Photoshoot series sets this for the cook pass. A single picture leaves it off.
        self.cook = False
        self._sampled = False

    def precompute(self, prompts: tuple[str, ...]) -> None:
        """Encode every series prompt with the text encoder alone, then unload it.

        Cook loads the transformer and the VAE and reuses these embeddings.
        """
        catalog = load_catalog()
        screened = issue_input(prompts[0] if prompts else "", "text", ())
        try:
            plan = build_plan(catalog, screened, self.tier)
            spec = plan_specs(catalog, plan)[0]
            folder = snapshot_ready(spec, models_dir(), process_cache())
            if folder is None:
                raise ModelNotReady(plan)
            prepare_series_embeds(
                folder, self.tier, prompts, lora=self.bound_lora, lora_strength=self.lora_strength,
            )
        finally:
            flush_process_cache()
            screened.wipe()

    def __call__(self, job: PictureJob, inputs: tuple[PendingInput, ...]) -> RawImage:
        """Generate one picture and return unreviewed pixels.

        Args:
            job: Prompt, mode, and the expert values.
            inputs: Fresh images. The pipeline screens them and wipes them.

        Raises:
            RunStopped: The pipeline refused the picture.
        """
        def draw() -> RawImage:
            if self.cook:
                cooked = self._cook(job, inputs)
                if cooked is None:
                    raise InferenceUnavailable("the generator returned no picture")
                return RawImage(cooked.width, cooked.height, cooked.to_bytes())
            reviewed = self._through_pipeline(job, inputs)
            return RawImage(reviewed.width, reviewed.height, reviewed.data)

        return self._guarded(draw)

    def sample(self, job: PictureJob, inputs: tuple[PendingInput, ...]) -> bool:
        """Sample one series frame and keep its latent. ``__call__`` with the same job decodes it.

        Only the cook pass samples ahead. False means nothing was kept, and the
        caller draws the frame whole with ``__call__``.

        Raises:
            RunStopped: The pipeline refused the frame.
        """
        if not self.cook:
            return False
        return self._guarded(lambda: self._sample(job, inputs))

    def _sample(self, job: PictureJob, inputs: tuple[PendingInput, ...]) -> bool:
        self._cook(job, inputs, sample=True)
        return self._sampled

    def _guarded(self, draw: Callable[[], _T]) -> _T:
        """Run ``draw`` and turn every refusal into ``RunStopped``."""
        try:
            try:
                return draw()
            except DeviceRefused as exc:
                _halt(str(exc), exc, exc.code, unavailable=True)
            except SafetyUnavailable as exc:
                _halt(exc.user_message, exc, f"check.{exc.code}", unavailable=True)
            except GpuLockTimeout as exc:
                _halt(text("startup.lock"), exc, "gpu_busy", unavailable=True)
            except SafetyError as exc:
                _halt(exc.user_message, exc, exc.code, unavailable=False)
            except EnlargeMissing as exc:
                _halt(str(exc), exc, "enlarge_unavailable", unavailable=True)
            except JobError as exc:
                _halt(str(exc), exc, "job", unavailable=False)
            except ModeUnavailable as exc:
                _halt(str(exc), exc, "mode", unavailable=False)
            except CoolingDown as exc:
                _halt(str(exc), exc, "cooling", unavailable=False)
            except BaseNotSelected as exc:
                _halt(str(exc), exc, "base", unavailable=False)
            except EnlargeError as exc:
                _halt(str(exc), exc, "enlarge_unavailable", unavailable=True)
            except (
                ModelNotReady,
                InferenceUnavailable,
                SnapshotError,
                DownloadDeclined,
                VramRefused,
                OSError,
            ) as exc:
                _halt(str(exc), exc, "internal", unavailable=True)
            except Exception as exc:
                _halt(text("gate.unavailable"), exc, "internal", unavailable=True)
        finally:
            release(lock_path())

    def _through_pipeline(self, job: PictureJob, inputs: tuple[PendingInput, ...]) -> SafeImage:
        """One picture through the gate, including its input and output checks."""
        catalog = load_catalog()
        generator = self._generator(job)
        pipeline = Pipeline(
            gate=self.gate,
            generator=generator,
            tier=self.tier,
            use_gpu_lock=True,
            catalog=catalog,
            enlarge=self.enlarge,
        )
        return pipeline.generate(
            job.prompt, job.mode, inputs, aspect=job.aspect, output=job.output, explicit=job.adult_content,
        )

    def _cook(self, job: PictureJob, inputs: tuple[PendingInput, ...], *, sample: bool = False) -> RawImage | None:
        """Draw one series frame, or only sample it. The output check runs after the series is decoded."""
        packed = tuple(
            (role_name(item.role), item.image.width, item.image.height, item.image.to_bytes())
            for item in inputs
        )
        screened = issue_input(job.prompt, job.mode.value, packed)
        path = lock_path()
        try:
            holder = try_acquire("canvas-generate", 600, path)
            if holder is not None:
                raise GpuLockTimeout("the GPU is busy")
            generator = self._generator(job)
            frame = frame_size(load_catalog(), "z-image-turbo", "1:1")
            try:
                if sample:
                    self._sampled = run_sampler(generator, screened, width=frame.width, height=frame.height)
                    return None
                return run_generator(generator, screened, width=frame.width, height=frame.height)
            finally:
                release(path)
        finally:
            screened.wipe()
            for item in inputs:
                item.image.wipe()

    def _generator(self, job: PictureJob) -> ZImageGenerator:
        catalog = load_catalog()
        generator = ZImageGenerator(catalog, self.tier)
        generator.seed = job.seed
        generator.lora_path = job.lora_path
        generator.lora_strength = job.lora_weight
        generator.bound_lora = self.bound_lora
        generator.negative_prompt = job.negative
        if self.img2img_strength is not None:
            generator.img2img_strength = self.img2img_strength
        generator.models = models_dir()
        _offer(generator, "steps", job.steps)
        _offer(generator, "guidance", job.guidance)
        _offer(generator, "sampler", job.sampler)
        return generator


def improve_prompt(
    *,
    gate: AgeGate,
    prompt: str,
    tier: Tier,
    adult: bool = False,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Rewrite ``prompt`` with the same gate. Missing models leave the words as they are.

    Args:
        gate: The screen's content gate.
        prompt: The current field.
        tier: The tier chosen at startup.

    Raises:
        SafetyError: The gate refused the field. The window keeps the old words.
    """
    gate.review_prompt(prompt)
    release_gate_weights(gate)
    if not selectable():
        return prompt
    catalog = load_catalog()
    rewriter = open_rewriter(catalog, tier, models_dir(), progress or _silent, stage="enhancer")
    if adult:
        rewritten = improve_adult(rewriter, prompt, ENHANCE_SYSTEM)
    else:
        rewritten = rewrite_text(rewriter, prompt)
    return resolve_prompt(gate, prompt, rewritten).prompt


def load_rgb(path: Path) -> RawImage:
    """Read one RGB image from disk.

    Args:
        path: PNG, JPEG, WebP, or AVIF file.

    Raises:
        OSError: The file cannot be read as an image.
    """
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return RawImage(rgb.width, rgb.height, rgb.tobytes())
    except (OSError, ValueError) as exc:
        raise OSError(str(exc)) from exc


def _halt(message: str, exc: BaseException, code: str, *, unavailable: bool) -> NoReturn:
    """Stop one picture. An unavailable stop is written to the crash log once."""
    if unavailable:
        note_crash(code or "internal", exc)
    raise RunStopped(message, unavailable=unavailable, code=code) from exc


def _offer(generator: object, name: str, value: object) -> None:
    if hasattr(generator, name):
        setattr(generator, name, value)


def _silent(message: str) -> None:
    del message
