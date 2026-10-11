"""The 4K handoff is a lossless PNG, 16-bit when the buffer is."""
import struct
from pathlib import Path

import imagecodecs
import numpy as np
from PIL import Image

from canvas.core.exchange import read_picture, write_png


def _bit_depth(path: Path) -> int:
    blob = path.read_bytes()
    return struct.unpack(">IIBBBBB", blob[16:29])[2]


def test_eight_bit_handoff_is_a_lossless_png(tmp_path):
    data = bytes((1, 2, 3)) * 8
    path = tmp_path / "native.png"
    write_png(path, 4, 2, data)
    assert _bit_depth(path) == 8
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.convert("RGB").tobytes() == data
    assert read_picture(path).to_bytes() == data


def test_sixteen_bit_handoff_keeps_the_samples(tmp_path):
    samples = np.zeros((2, 4, 3), dtype="<u2")
    samples[0, 0] = (1000, 2000, 3000)
    path = tmp_path / "native.png"
    write_png(path, 4, 2, samples.tobytes())
    assert _bit_depth(path) == 16
    pixels = imagecodecs.png_decode(path.read_bytes())
    assert pixels.dtype == np.uint16
    assert tuple(int(value) for value in pixels[0, 0]) == (1000, 2000, 3000)
