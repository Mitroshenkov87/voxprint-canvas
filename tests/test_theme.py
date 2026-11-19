"""Palette contrast stays at WCAG AA for the suite pairs."""
from canvas.ui.theme import (
    CARD_GLASS,
    CONTRAST_PAIRS,
    ROOT_GLASS,
    TEXT,
    TEXT_FAINT,
    TEXT_MUTED,
    build_style,
)


def _lum(hex_color: str) -> float:
    rgb = [int(hex_color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4 for channel in rgb]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground: str, background: str) -> float:
    hi, lo = sorted((_lum(foreground), _lum(background)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _over_white(rgba: str) -> str:
    red, green, blue, alpha = [int(value) for value in rgba[5:-1].split(",")]
    mixed = tuple(round(channel * alpha / 255 + 255 * (1 - alpha / 255)) for channel in (red, green, blue))
    return "#%02x%02x%02x" % mixed


def test_contrast_pairs_and_glass_over_white():
    for foreground, background in CONTRAST_PAIRS:
        ratio = _contrast(foreground, background)
        assert ratio >= 4.5, (foreground, background, ratio)
    for glass in (ROOT_GLASS, CARD_GLASS):
        worst = _over_white(glass)
        for foreground in (TEXT, TEXT_MUTED, TEXT_FAINT):
            ratio = _contrast(foreground, worst)
            assert ratio >= 4.5, (foreground, glass, worst)


def test_style_modes():
    glass = build_style(True)
    plain = build_style(False)
    assert ROOT_GLASS in glass and ROOT_GLASS not in plain
    assert "rgba(16,16,22,167)" in build_style(True, more=True)
