"""Canvas on the shared image screen.

The surface is ``canvas``. Presets from the shared file stay, and the style
row writes a Canvas look. Start from, variation, change, and file type live
in Picture setup so the centre result stays large. Instruction editing is
not a row.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from voxprint_common.paths import state_dir
from voxprint_common.ui.presets import PresetFile, PresetGroup, PresetOption
from voxprint_common.ui.settings import Choice, SettingSpec, SettingType, Settings
from voxprint_common.ui.simple_image import SimpleImageLayout as SimpleLayout
from voxprint_common.ui.variants import AppSurface, settings_for, state_names, surface_for, variant_presets

from canvas.app.shell_gate import ContentCheck
from canvas.core.formats import OutputFormat
from canvas.models.modes import Mode
from canvas.safety.gate import AgeGate
from canvas.ui.styles import LOOKS, look_for, style_label


class PictureJob(NamedTuple):
    """One picture the screen is about to run. The pipeline reads these fields."""

    prompt: str
    negative: str
    mode: Mode
    aspect: str
    output: str
    fmt: OutputFormat
    seed: int
    steps: int
    guidance: float
    sampler: str
    copies: int


@dataclass
class CanvasShell:
    """The headless screen, the descriptors, and the gate adapter."""

    layout: SimpleLayout
    settings: Settings
    gate: ContentCheck
    state_dir: Path
    surface: AppSurface


def canvas_state_dir() -> Path:
    """Return ``<home>/state/canvas`` and create it when it is missing."""
    path = state_dir() / "canvas"
    path.mkdir(parents=True, exist_ok=True)
    return path


def open_layout(state_dir_path: Path, *, gate: AgeGate, enlarge_installed: bool = False) -> CanvasShell:
    """Build the image screen for Canvas.

    ``gate`` is required. ``None`` and any stand-in are refused before the
    shared screen is constructed.

    Args:
        state_dir_path: Directory for ``image-settings.json`` and history.
        gate: The process content gate.
        enlarge_installed: True when the 4K chip can be selected.

    Raises:
        TypeError: ``gate`` is missing or is not the Canvas content gate.
    """
    if not isinstance(gate, AgeGate):
        raise TypeError("the screen needs the always-on content gate")
    surface = surface_for("canvas")
    settings = _settings()
    presets = _presets(settings, enlarge_installed=enlarge_installed)
    names = state_names(surface)
    state_dir_path.mkdir(parents=True, exist_ok=True)
    check = ContentCheck(gate=gate)
    layout = SimpleLayout(
        presets=presets,
        settings=settings,
        state_dir=state_dir_path,
        gate=check,
        state_name=names[0],
        history_name=names[1],
    )
    return CanvasShell(layout, settings, check, state_dir_path, surface)


def job_from(layout: SimpleLayout) -> PictureJob:
    """Read the live chips and expert values into one job.

    Args:
        layout: A screen opened by :func:`open_layout`.
    """
    return PictureJob(
        prompt=composed_prompt(layout),
        negative=composed_negative(layout),
        mode=picture_mode(layout),
        aspect=str(layout.presets.get("size", "1:1")),
        output=output_name(layout),
        fmt=_format(layout.get("canvas.format")),
        seed=int(layout.seed),
        steps=_whole(layout.get("shell.steps")),
        guidance=_decimal(layout.get("shell.guidance")),
        sampler=str(layout.get("shell.sampler")),
        copies=layout.batch_size(),
    )


def picture_mode(layout: SimpleLayout) -> Mode:
    """The mode the chips describe. Words use variation. A picture uses Change.

    Args:
        layout: A screen opened by :func:`open_layout`.

    Raises:
        ValueError: A chip value is not one this screen writes.
    """
    source = layout.get("canvas.source")
    if source == "words":
        if layout.get("canvas.variation") == "varied":
            return Mode.TEXT_CREATIVE
        if layout.get("canvas.variation") == "quick":
            return Mode.TEXT
        raise ValueError("unknown variation")
    if source != "picture":
        raise ValueError("unknown source")
    change = layout.get("canvas.change")
    if change == "restyle":
        return Mode.RESTYLE
    if change == "inpaint":
        return Mode.INPAINT
    if change == "outpaint":
        return Mode.OUTPAINT
    if change == "pose":
        return Mode.POSE
    if change == "depth":
        return Mode.DEPTH
    if change == "edge":
        return Mode.EDGE
    raise ValueError("unknown change")


def composed_prompt(layout: SimpleLayout) -> str:
    """User words, the look suffix, and the audience line when it is on.

    Args:
        layout: A screen opened by :func:`open_layout`.
    """
    look = look_for(str(layout.get("canvas.style")))
    policy = layout.prompt_policy()
    parts = [part for part in (policy.template, layout.prompt.strip(), look.suffix) if part]
    return " ".join(parts)


def composed_negative(layout: SimpleLayout) -> str:
    """The live negative plus the audience negative when adult content is off.

    Args:
        layout: A screen opened by :func:`open_layout`.
    """
    live = str(layout.get("shell.negative") or "")
    extra = layout.prompt_policy().negatives
    return ", ".join(part for part in (live, extra) if part)


def output_name(layout: SimpleLayout) -> str:
    """``native``, ``2k``, or ``4k``. Exact size wins until it says to follow the chip.

    Args:
        layout: A screen opened by :func:`open_layout`.
    """
    band = str(layout.get("canvas.output_size"))
    if band == "2k":
        return "2k"
    if band == "4k":
        return "4k"
    if band == "native":
        return "native"
    if band == "follow":
        return "4k" if layout.get("shell.output") == "4k" else "native"
    raise ValueError("unknown output size")


def _settings() -> Settings:
    settings = settings_for(surface_for("canvas"))
    settings.add(_enum("canvas.style", "photo", tuple((look.style_id, style_label(look.style_id)) for look in LOOKS)))
    settings.add(_enum("canvas.source", "words", (("words", "canvas.source.words"), ("picture", "canvas.source.picture"))))
    settings.add(
        _enum(
            "canvas.variation",
            "quick",
            (("quick", "canvas.variation.quick"), ("varied", "canvas.variation.varied")),
        )
    )
    settings.add(
        _enum(
            "canvas.change",
            "restyle",
            (
                ("restyle", "canvas.change.restyle"),
                ("inpaint", "canvas.change.inpaint"),
                ("outpaint", "canvas.change.outpaint"),
                ("pose", "canvas.change.pose"),
                ("depth", "canvas.change.depth"),
                ("edge", "canvas.change.edge"),
            ),
        )
    )
    settings.add(
        _enum(
            "canvas.format",
            "avif",
            (("avif", "canvas.format.avif"), ("jxl", "canvas.format.jxl"), ("webp", "canvas.format.webp")),
        )
    )
    settings.add(
        _enum(
            "canvas.output_size",
            "follow",
            (
                ("follow", "canvas.output_size.follow"),
                ("native", "canvas.output_size.native"),
                ("2k", "canvas.output_size.2k"),
                ("4k", "canvas.output_size.4k"),
            ),
            expert=True,
        )
    )
    return settings


def _enum(
    setting_id: str,
    default: str,
    choices: tuple[tuple[str, str], ...],
    *,
    expert: bool = False,
) -> SettingSpec:
    return SettingSpec(
        setting_id,
        setting_id,
        f"{setting_id}.help",
        SettingType.ENUM,
        default,
        expert=expert,
        group="image" if expert else "",
        choices=tuple(Choice(value, label) for value, label in choices),
    )


def _presets(settings: Settings, *, enlarge_installed: bool) -> PresetFile:
    document = variant_presets(surface_for("canvas"), settings, enlarge_installed=enlarge_installed)
    groups = [_style_group() if group.group_id == "style" else group for group in document.groups]
    return PresetFile(tuple(groups))


def _style_group() -> PresetGroup:
    options = tuple(
        PresetOption(
            look.style_id,
            style_label(look.style_id),
            (("canvas.style", look.style_id), ("shell.negative", look.negative)),
        )
        for look in LOOKS
    )
    return PresetGroup("style", "simple.style", options, "photo")


def _whole(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected a whole number")
    return value


def _decimal(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("expected a number")
    return float(value)


def _format(value: object) -> OutputFormat:
    if value == "avif":
        return OutputFormat.AVIF
    if value == "jxl":
        return OutputFormat.JXL
    if value == "webp":
        return OutputFormat.WEBP
    raise ValueError("unknown file type")
