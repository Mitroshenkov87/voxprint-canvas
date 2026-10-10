"""The only call into a generator. The argument must already be a ScreenedInput."""
from __future__ import annotations

from canvas.safety.image import RawImage, ScreenedInput


def run_generator(generator: object, screened: ScreenedInput) -> RawImage:
    """Run one generator. A raw upload or a plain string cannot get this far."""
    if type(screened) is not ScreenedInput:
        raise TypeError("the generator only accepts a ScreenedInput")
    generate = getattr(generator, "generate")
    image = generate(screened)
    if type(image) is not RawImage:
        raise TypeError("the generator must return an unreviewed image")
    return image
