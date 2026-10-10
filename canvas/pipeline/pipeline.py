"""Run one job: screen inputs, generate inside the worker boundary, screen the output."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

from canvas.core.formats import DEFAULT_FORMAT, OutputFormat, save_image
from canvas.infra.gpu_lock import gpu_job
from canvas.infra.thermal import Guard
from canvas.models.loader import ModelNotReady
from canvas.models.modes import Mode, Tier, required_roles, visible_modes
from canvas.models.plan import ModeUnavailable
from canvas.pipeline.engine import run_generator
from canvas.safety.gate import AgeGate, SafetyError
from canvas.safety.image import SafeImage
from canvas.safety.inputs import PendingInput


class JobError(Exception):
    """The request is missing an image the chosen mode needs."""

    def __init__(self) -> None:
        super().__init__("This mode needs another image.")


class CoolingDown(Exception):
    """The thermal guard asked the job to wait."""

    def __init__(self) -> None:
        super().__init__("The GPU is cooling down. Try again in a little while.")


class Pipeline:
    """In-memory boundary by default so tests can supply fakes. ``process`` spawns the worker."""

    def __init__(self, gate: AgeGate, generator: object, tier: Tier, *, thermal: Guard | None = None,
                 use_gpu_lock: bool = False) -> None:
        self.gate = gate
        self.generator = generator
        self.tier = tier
        self.thermal = thermal
        self.use_gpu_lock = use_gpu_lock
        self.paused = False

    def generate(self, prompt: str, mode: Mode, inputs: Sequence[PendingInput] = ()) -> SafeImage:
        if mode not in visible_modes(self.tier):
            raise ModeUnavailable()
        needed = required_roles(mode)
        have = {item.role for item in inputs}
        if any(role not in have for role in needed):
            raise JobError()
        with self._lock():
            if self.thermal is not None and self.thermal.should_pause():
                self.paused = True
                raise CoolingDown()
            self.paused = False
            screened = self.gate.release_request(prompt, mode.value, inputs)
            try:
                raw = run_generator(self.generator, screened)
                image = self.gate.review_image(raw)
                if type(image) is not SafeImage:
                    raise TypeError("the checks must release the image")
                return image
            except ModelNotReady as exc:
                raise SafetyError("model_not_ready") from exc
            finally:
                screened.wipe()

    def export(self, prompt: str, mode: Mode, inputs: Sequence[PendingInput], path, fmt: OutputFormat = DEFAULT_FORMAT) -> None:
        """Generate and save. A refusal writes nothing."""
        image = self.generate(prompt, mode, inputs)
        save_image(image, path, fmt)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        if not self.use_gpu_lock:
            yield
            return
        with gpu_job("canvas-generate"):
            yield
