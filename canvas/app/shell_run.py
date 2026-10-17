"""Run one shell job through the existing pipeline and the same content gate."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from canvas.infra.enlarge import EnlargeError
from canvas.infra.paths import models_dir
from canvas.models.loader import ModelNotReady, ZImageGenerator
from canvas.models.mdl_registry import selectable
from canvas.models.modes import Tier
from canvas.models.plan import BaseNotSelected, ModeUnavailable
from canvas.models.provision import DownloadDeclined
from canvas.models.registry import load_catalog
from canvas.models.runtime import InferenceUnavailable
from canvas.models.snapshots import SnapshotError
from canvas.pipeline.bf16_schedule import VramRefused
from canvas.pipeline.enhance import open_rewriter, resolve_prompt, rewrite_text
from canvas.pipeline.pipeline import CoolingDown, EnlargeMissing, JobError, Pipeline
from canvas.safety.gate import AgeGate, SafetyError, SafetyUnavailable
from canvas.safety.image import RawImage
from canvas.safety.inputs import PendingInput
from canvas.ui.shell_layout import PictureJob


class RunStopped(Exception):
    """The picture stopped with a sentence the window can show."""

    def __init__(self, message: str, *, unavailable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.unavailable = unavailable


class ShellRunner:
    """Turns one :class:`PictureJob` into pixels. The window reviews them again."""

    def __init__(self, *, gate: AgeGate, tier: Tier, enlarge: object) -> None:
        """Keep the gate the window already holds.

        Args:
            gate: The same instance passed to the screen.
            tier: The tier chosen at startup.
            enlarge: Enlarge client. 4K uses it.
        """
        self.gate = gate
        self.tier = tier
        self.enlarge = enlarge

    def __call__(self, job: PictureJob, inputs: tuple[PendingInput, ...]) -> RawImage:
        """Generate one picture and return unreviewed pixels.

        Args:
            job: Prompt, mode, and the expert values.
            inputs: Fresh images. The pipeline screens them and wipes them.

        Raises:
            RunStopped: The pipeline refused the picture.
        """
        catalog = load_catalog()
        generator = ZImageGenerator(catalog, self.tier)
        generator.seed = job.seed
        generator.models = models_dir()
        _offer(generator, "steps", job.steps)
        _offer(generator, "guidance", job.guidance)
        _offer(generator, "sampler", job.sampler)
        pipeline = Pipeline(
            gate=self.gate,
            generator=generator,
            tier=self.tier,
            use_gpu_lock=True,
            catalog=catalog,
            enlarge=self.enlarge,
        )
        try:
            safe = pipeline.generate(job.prompt, job.mode, inputs, aspect=job.aspect, output=job.output)
        except SafetyUnavailable as exc:
            raise RunStopped(exc.user_message, unavailable=True) from exc
        except SafetyError as exc:
            raise RunStopped(exc.user_message) from exc
        except EnlargeMissing as exc:
            raise RunStopped(str(exc), unavailable=True) from exc
        except (JobError, ModeUnavailable, CoolingDown, BaseNotSelected) as exc:
            raise RunStopped(str(exc)) from exc
        except (
            ModelNotReady,
            InferenceUnavailable,
            SnapshotError,
            DownloadDeclined,
            VramRefused,
            EnlargeError,
            OSError,
        ) as exc:
            raise RunStopped(str(exc), unavailable=True) from exc
        return RawImage(safe.width, safe.height, safe.data)


def improve_prompt(*, gate: AgeGate, prompt: str, tier: Tier) -> str:
    """Rewrite ``prompt`` with the same gate. Missing models leave the words as they are.

    Args:
        gate: The screen's content gate.
        prompt: The current field.
        tier: The tier chosen at startup.

    Raises:
        SafetyError: The gate refused the field. The window keeps the old words.
    """
    gate.review_prompt(prompt)
    if not selectable():
        return prompt
    catalog = load_catalog()
    rewritten = rewrite_text(open_rewriter(catalog, tier, models_dir(), _silent), prompt)
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


def _offer(generator: object, name: str, value: object) -> None:
    if hasattr(generator, name):
        setattr(generator, name, value)


def _silent(message: str) -> None:
    del message
