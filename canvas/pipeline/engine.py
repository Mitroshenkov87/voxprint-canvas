"""The only calls into a generator. The argument must already be a ScreenedInput."""
from __future__ import annotations

from canvas.safety.image import RawImage, ScreenedInput


def run_generator(generator: object, screened: ScreenedInput, *, width: int | None = None, height: int | None = None) -> RawImage:
    """Run one generator. A raw upload or a plain string cannot get this far."""
    if type(screened) is not ScreenedInput:
        raise TypeError("the generator only accepts a ScreenedInput")
    generate = getattr(generator, "generate")
    args: tuple[object, ...] = (screened,) if width is None or height is None else (screened, width, height)
    image = generate(*args)
    if type(image) is not RawImage:
        raise TypeError("the generator must return an unreviewed image")
    return image


def run_sampler(generator: object, screened: ScreenedInput, *, width: int, height: int) -> bool:
    """Sample one frame and keep its latent. ``run_generator`` with the same input returns its pixels.

    False when the generator has no ``sample``. The caller then draws the frame whole.
    """
    if type(screened) is not ScreenedInput:
        raise TypeError("the generator only accepts a ScreenedInput")
    sample = getattr(generator, "sample", None)
    if not callable(sample):
        return False
    sample(screened, width, height)
    return True
