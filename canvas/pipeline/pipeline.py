"""Run one job: screen inputs, generate inside the worker boundary, screen the output."""
from __future__ import annotations

import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from canvas.app.info import ENLARGE_HINT
from canvas.core.exchange import read_picture, write_png
from canvas.core.formats import DEFAULT_FORMAT, OutputFormat, save_image
from canvas.infra.enlarge import EnlargeError, declared_vram_gb
from canvas.infra.gpu_lock import GpuLockTimeout, lock_path, release, try_acquire
from canvas.infra.thermal import Guard
from canvas.models.loader import ModelNotReady
from canvas.models.modes import Mode, Tier, required_roles, visible_modes
from canvas.models.plan import BaseNotSelected, ModeUnavailable
from canvas.models.registry import Catalog
from canvas.models.sizes import FOUR_K_LONG, frame_size, family_for, upscale_factor
from canvas.pipeline.engine import run_generator
from canvas.pipeline.gpu_release import free_vram_gb, release_cuda_cache
from canvas.pipeline.vram_probe import PeakVramProbe
from canvas.safety.gate import AgeGate, SafetyError
from canvas.safety.image import RawImage, SafeImage
from canvas.safety.inputs import PendingInput


class JobError(Exception):
    """The request is missing an image the chosen mode needs."""

    def __init__(self) -> None:
        super().__init__("This mode needs another image.")


class CoolingDown(Exception):
    """The thermal guard asked the job to wait."""

    def __init__(self) -> None:
        super().__init__("The GPU is cooling down. Try again in a little while.")


class EnlargeMissing(Exception):
    """4K was requested and Enlarge is not available."""

    def __init__(self) -> None:
        super().__init__(ENLARGE_HINT)


class _Hold:
    """The suite GPU lock. It is released while Enlarge runs, then taken again."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.path = lock_path()
        self.held = False
        self._lent = False

    def __enter__(self) -> _Hold:
        if not self.enabled:
            return self
        holder = try_acquire("canvas-generate", 600, self.path)
        if holder is not None:
            raise GpuLockTimeout("the GPU is busy")
        self.held = True
        return self

    def __exit__(self, *exc: object) -> None:
        if self.held:
            release(self.path)
            self.held = False

    def lend(self) -> None:
        """Release the lock so Enlarge can use the GPU."""
        if not self.held:
            self._lent = False
            return
        release(self.path)
        self.held = False
        self._lent = True

    def reclaim(self) -> None:
        """Take the lock back after Enlarge returns."""
        if not self._lent:
            return
        self._lent = False
        holder = try_acquire("canvas-generate", 600, self.path)
        if holder is None:
            self.held = True
            return
        raise GpuLockTimeout("the GPU is busy")


class Pipeline:
    """In-memory boundary by default so tests can supply fakes. ``process`` spawns the worker."""

    def __init__(self, gate: AgeGate, generator: object, tier: Tier, *, thermal: Guard | None = None,
                 use_gpu_lock: bool = False, catalog: Catalog | None = None, enlarge: object | None = None) -> None:
        self.gate = gate
        self.generator = generator
        self.tier = tier
        self.thermal = thermal
        self.use_gpu_lock = use_gpu_lock
        self.catalog = catalog
        self.enlarge = enlarge
        self.paused = False

    def generate(self, prompt: str, mode: Mode, inputs: Sequence[PendingInput] = (), *,
                 aspect: str = "1:1", output: str = "native", probe: PeakVramProbe | None = None) -> SafeImage:
        """Run one request and return a picture the window may show.

        The suite GPU lock is held while the picture is made. Native output uses
        the model's frame for the aspect. ``2k`` generates the model's ~2K frame
        and does not upscale. 4K output generates the model's ~2K frame, unloads
        the generator, text encoder, and VAE, and asks Voxprint AI Enlarge to
        make the long side 3840 only when free GPU memory is above Enlarge's
        declared need. The lock is released for that spawn and taken again
        afterwards. The enlarged picture is released before it is
        returned. A stage that does not fit the memory budget stops the job.
        """
        if output not in {"native", "2k", "4k"}:
            raise ValueError("output is native, 2k, or 4k")
        if mode not in visible_modes(self.tier):
            raise ModeUnavailable()
        needed = required_roles(mode)
        have = {item.role for item in inputs}
        if any(role not in have for role in needed):
            raise JobError()
        with self._hold() as hold:
            if self.thermal is not None and self.thermal.should_pause():
                self.paused = True
                raise CoolingDown()
            self.paused = False
            screened = self.gate.release_request(prompt, mode.value, inputs)
            try:
                width, height = self._frame(mode, aspect, output in {"2k", "4k"})
                if output == "4k" and not _enlarge_ready(self.enlarge):
                    raise EnlargeMissing()
                if probe is not None:
                    probe.mark("generate-start")
                attach = getattr(self.generator, "attach_probe", None)
                if attach is not None:
                    attach(probe)
                raw = run_generator(self.generator, screened, width=width, height=height)
                image = self._reviewed(raw, prompt)
                if probe is not None:
                    probe.mark("generate-end")
                if output != "4k":
                    return image
                return self._upscale(image, probe, prompt, hold)
            except ModelNotReady as exc:
                raise SafetyError("model_not_ready") from exc
            finally:
                screened.wipe()

    def export(self, prompt: str, mode: Mode, inputs: Sequence[PendingInput], path, fmt: OutputFormat = DEFAULT_FORMAT, *,
               aspect: str = "1:1", output: str = "native", probe: PeakVramProbe | None = None) -> None:
        """Generate and save. A refusal writes nothing."""
        image = self.generate(prompt, mode, inputs, aspect=aspect, output=output, probe=probe)
        save_image(image, path, fmt)

    def _frame(self, mode: Mode, aspect: str, twok: bool) -> tuple[int | None, int | None]:
        catalog = getattr(self.generator, "catalog", None) or self.catalog
        if catalog is None:
            if twok:
                raise BaseNotSelected()
            return None, None
        try:
            family = family_for(mode, catalog)
        except BaseNotSelected:
            if twok:
                raise
            return None, None
        frame = frame_size(catalog, family, aspect, twok=twok)
        if twok:
            upscale_factor(frame)
        return frame.width, frame.height

    def _upscale(self, image: SafeImage, probe: PeakVramProbe | None, prompt: str, hold: _Hold) -> SafeImage:
        self._release_models()
        if probe is not None:
            probe.mark("unload")
        need = _enlarge_need_gb(self.enlarge)
        free = free_vram_gb()
        if free is not None and free <= need:
            raise EnlargeError("oom", "Enlarge needs more free GPU memory.", False)
        with tempfile.TemporaryDirectory(prefix="voxprint-canvas-") as folder:
            root = Path(folder)
            source = root / "native.png"
            dest = root / "upscaled.avif"
            write_png(source, image.width, image.height, image.data)
            if probe is not None:
                probe.mark("upscale-start")
            enlarge = getattr(self.enlarge, "enlarge")
            hold.lend()
            failed = True
            try:
                result = enlarge(source, output=dest, format_name="avif", long_side=FOUR_K_LONG)
                failed = False
            finally:
                try:
                    hold.reclaim()
                except GpuLockTimeout:
                    if failed:
                        raise
            out_path = Path(result.output)
            upscaled = read_picture(out_path)
            if probe is not None:
                probe.mark("upscale-end")
            reviewed = self._reviewed(upscaled, prompt)
            out_path.unlink(missing_ok=True)
            return reviewed

    def _release_models(self) -> None:
        """Drop the generator, text encoder, and VAE, then return CUDA memory."""
        generator = self.generator
        for name in ("pipe", "pipeline", "text_encoder", "text_encoder_2", "vae", "transformer", "controlnet"):
            if getattr(generator, name, None) is not None:
                setattr(generator, name, None)
        unload = getattr(generator, "unload", None)
        if unload is not None:
            unload()
        release_cuda_cache()

    def _reviewed(self, raw: RawImage, prompt: str) -> SafeImage:
        image = self.gate.review_image(raw, prompt)
        if type(image) is not SafeImage:
            raise TypeError("the checks must release the image")
        return image

    @contextmanager
    def _hold(self) -> Iterator[_Hold]:
        hold = _Hold(self.use_gpu_lock)
        with hold:
            yield hold


def _enlarge_need_gb(enlarge: object | None) -> float:
    caps = getattr(enlarge, "capabilities", None)
    body = caps() if callable(caps) else None
    return declared_vram_gb(body if isinstance(body, dict) else None)


def _enlarge_ready(enlarge: object | None) -> bool:
    if enlarge is None:
        return False
    ready = getattr(enlarge, "ready", None)
    if ready is not None:
        return bool(ready())
    installed = getattr(enlarge, "installed", None)
    if installed is None:
        return True
    return bool(installed())
