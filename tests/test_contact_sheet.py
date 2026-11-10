"""The photoshoot overview must show real thumbnails, not colour noise."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from canvas.pipeline.series import _read_frame, _write_sheet


def _pattern(width: int, height: int) -> Image.Image:
    """A distinctive 8-bit RGB image so a mis-decoded thumbnail cannot match by luck."""
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    assert pixels is not None
    for y in range(height):
        for x in range(width):
            pixels[x, y] = ((x * 17) % 256, (y * 13) % 256, (x + y) % 256)
    pixels[0, 0] = (255, 0, 0)
    pixels[width - 1, 0] = (0, 255, 0)
    pixels[0, height - 1] = (0, 0, 255)
    pixels[width - 1, height - 1] = (255, 255, 0)
    return image


def test_read_frame_unfilters_a_png(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    source = _pattern(64, 48)
    source.save(path, format="PNG")  # PIL writes Sub/Up/Paeth filters
    width, height, rgb = _read_frame(path)
    assert (width, height) == source.size
    assert rgb == source.tobytes()
    # The old reader skipped the filter byte but never unfiltered: that buffer diverges.
    raw_skip = _unfiltered_skip(path)
    assert raw_skip != rgb


def test_overview_thumbnails_match_the_frames(tmp_path: Path) -> None:
    frames: list[Path] = []
    sources: list[Image.Image] = []
    for index in range(3):
        path = tmp_path / f"0{index + 1}.png"
        image = _pattern(128, 128)
        # Shift the pattern so each frame is distinct.
        image = image.point(lambda value, offset=index * 40: (value + offset) % 256)
        image.save(path, format="PNG")
        frames.append(path)
        sources.append(image)
    overview = tmp_path / "overview.png"
    _write_sheet(overview, frames)
    sheet = Image.open(overview).convert("RGB")
    cell, gap = 64, 8
    assert sheet.size == (len(frames) * cell + (len(frames) + 1) * gap, cell + 2 * gap)
    for index, source in enumerate(sources):
        left = gap + index * (cell + gap)
        for y in range(cell):
            for x in range(0, cell, 16):
                sx = min(source.width - 1, x * source.width // cell)
                sy = min(source.height - 1, y * source.height // cell)
                assert sheet.getpixel((left + x, gap + y)) == source.getpixel((sx, sy))


def _unfiltered_skip(path: Path) -> bytes:
    """Reproduce the previous contact-sheet bug: drop the filter byte, do not unfilter."""
    import struct
    import zlib

    raw = path.read_bytes()
    pos = 8
    width = height = 0
    compressed = b""
    while pos + 8 <= len(raw):
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        tag = raw[pos + 4:pos + 8]
        data = raw[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
        elif tag == b"IDAT":
            compressed += data
        elif tag == b"IEND":
            break
    inflated = zlib.decompress(compressed)
    rows = []
    stride = width * 3
    cursor = 0
    for _ in range(height):
        cursor += 1
        rows.append(inflated[cursor:cursor + stride])
        cursor += stride
    return b"".join(rows)
