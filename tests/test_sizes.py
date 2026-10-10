"""Registry frames for each aspect, including the ~2K source of a 4K upscale."""
import pytest

from canvas.models.modes import Mode, Tier, visible_modes
from canvas.models.registry import load_catalog
from canvas.models.sizes import (
    FOUR_K_LONG,
    UI_ASPECTS,
    closest_aspect,
    family_for,
    frame_size,
    upscale_factor,
)

QWEN_NATIVE = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
    "21:9": (2016, 864),
    "9:21": (864, 2016),
}

ZIMAGE_NATIVE = {
    "1:1": (1024, 1024),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
    "21:9": (1344, 576),
    "9:21": (576, 1344),
}


@pytest.mark.parametrize("family,expected", [("qwen-image-2512", QWEN_NATIVE), ("z-image-turbo", ZIMAGE_NATIVE)])
@pytest.mark.parametrize("aspect", [*UI_ASPECTS, "9:21"])
def test_native_and_twok_frames(family, expected, aspect):
    catalog = load_catalog()
    table = catalog.sizes[family]
    native = frame_size(catalog, family, aspect)
    twok = frame_size(catalog, family, aspect, twok=True)
    assert (native.width, native.height) == expected[aspect]
    assert native.width % table.multiple == 0 and native.height % table.multiple == 0
    assert twok.width % table.multiple == 0 and twok.height % table.multiple == 0
    assert max(native.width, native.height) < FOUR_K_LONG
    assert max(twok.width, twok.height) < FOUR_K_LONG
    factor = upscale_factor(twok)
    assert factor in (2, 4)
    assert max(twok.width, twok.height) * factor == FOUR_K_LONG
    assert min(twok.width, twok.height) * factor % table.multiple == 0


def test_closest_aspect_and_which_model_owns_the_frame():
    catalog = load_catalog()
    assert closest_aspect(1664, 928) == "16:9"
    assert closest_aspect(928, 1664) == "9:16"
    assert closest_aspect(1024, 1024) == "1:1"
    assert closest_aspect(864, 2016) == "9:16"
    assert family_for(Mode.RESTYLE, catalog) == "z-image-turbo"
    assert family_for(Mode.INSTRUCTION, catalog) == "qwen-image-2512"
    assert Mode.RESTYLE in visible_modes(Tier.GB16)
    qwen = catalog.sizes["qwen-image-2512"].native["16:9"]
    turbo = catalog.sizes["z-image-turbo"].native["16:9"]
    assert (qwen.width, qwen.height) != (turbo.width, turbo.height)
