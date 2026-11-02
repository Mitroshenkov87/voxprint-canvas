"""Canvas on the shared image screen.

The surface is ``canvas``. Presets from the shared file stay, and the style
row writes a Canvas look. Start from, variation, change, and file type live
in Picture setup so the centre result stays large. Instruction editing is
not a row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Mapping
from typing import NamedTuple

from voxprint_common.paths import state_dir
from voxprint_common.ui.presets import PresetFile, PresetGroup, PresetOption
from voxprint_common.ui.settings import Choice, SettingSpec, SettingType, Settings
from voxprint_common.ui.safety import adult_prompt_policy
from voxprint_common.ui.simple_image import SimpleImageLayout as SimpleLayout
from voxprint_common.ui.variants import AppSurface, settings_for, state_names, surface_for, variant_presets

from typing import TYPE_CHECKING

from canvas.core.formats import OutputFormat
from canvas.models.modes import Mode
from canvas.models.pins import TURBO_GUIDANCE, TURBO_STEPS
from canvas.ui.styles import LOOKS, look_for, style_label
from canvas.ui.worker_checks import WorkerChecks

if TYPE_CHECKING:
    from canvas.app.shell_gate import ContentCheck
    from canvas.safety.gate import AgeGate


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
    adult_content: bool = False
    lora_path: str = ""
    lora_weight: float = 0.7


@dataclass
class CanvasShell:
    """The headless screen, the descriptors, and the gate adapter."""

    layout: SimpleLayout
    settings: Settings
    gate: ContentCheck | WorkerChecks
    state_dir: Path
    surface: AppSurface


def canvas_state_dir() -> Path:
    """Return ``<home>/state/canvas`` and create it when it is missing."""
    path = state_dir() / "canvas"
    path.mkdir(parents=True, exist_ok=True)
    return path


def open_layout(state_dir_path: Path, *, gate: AgeGate, enlarge_installed: bool = False) -> CanvasShell:
    """Build the image screen around an in-process gate (headless tools and tests).

    ``gate`` is required. ``None`` and any stand-in are refused before the
    shared screen is constructed. The window uses :func:`open_worker_layout`.

    Args:
        state_dir_path: Directory for ``image-settings.json`` and history.
        gate: The process content gate.
        enlarge_installed: True when the 4K chip can be selected.

    Raises:
        TypeError: ``gate`` is missing or is not the Canvas content gate.
    """
    from canvas.app.shell_gate import ContentCheck
    from canvas.safety.gate import AgeGate

    if not isinstance(gate, AgeGate):
        raise TypeError("the screen needs the always-on content gate")
    return _open(state_dir_path, ContentCheck(gate=gate), enlarge_installed)


def open_worker_layout(state_dir_path: Path, *, checks: WorkerChecks, enlarge_installed: bool = False) -> CanvasShell:
    """Build the image screen for the window. Checks run in the worker.

    Args:
        state_dir_path: Directory for ``image-settings.json`` and history.
        checks: Answers the screen's review hooks from worker replies.
        enlarge_installed: True when the 4K chip can be selected.

    Raises:
        TypeError: ``checks`` is not a :class:`WorkerChecks`.
    """
    if not isinstance(checks, WorkerChecks):
        raise TypeError("the window screen answers its checks from the worker")
    return _open(state_dir_path, checks, enlarge_installed)


def _open(state_dir_path: Path, check: ContentCheck | WorkerChecks, enlarge_installed: bool) -> CanvasShell:
    surface = surface_for("canvas")
    settings = _settings()
    presets = _presets(settings, enlarge_installed=enlarge_installed)
    names = state_names(surface)
    state_dir_path.mkdir(parents=True, exist_ok=True)
    migrate_old_defaults(state_dir_path / names[0])
    layout = SimpleLayout(
        presets=presets,
        settings=settings,
        state_dir=state_dir_path,
        gate=check,
        source_picture=True,
        mask=True,
        state_name=names[0],
        history_name=names[1],
    )
    return CanvasShell(layout, settings, check, state_dir_path, surface)


# Expert defaults saved by rc.30 and earlier.
_OLD_DEFAULTS = {"shell.steps": 20, "shell.guidance": 7.0}


def migrate_old_defaults(path: Path) -> bool:
    """Move saved steps and guidance that still equal the old defaults to the Turbo defaults.

    Values the user changed are kept. Returns True when the file was rewritten.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    values = payload.get("values") if isinstance(payload, dict) else None
    if not isinstance(values, dict):
        return False
    for key, old in _OLD_DEFAULTS.items():
        found = values.get(key)
        if isinstance(found, bool) or not isinstance(found, (int, float)) or float(found) != float(old):
            return False
    values["shell.steps"] = int(TURBO_STEPS)
    values["shell.guidance"] = max(1.0, float(TURBO_GUIDANCE))
    overrides = payload.get("overrides")
    if isinstance(overrides, dict):
        for group, keys in overrides.items():
            if isinstance(keys, list):
                overrides[group] = [key for key in keys if key not in _OLD_DEFAULTS]
    try:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


def job_from(layout: SimpleLayout, live: Mapping[str, object] | None = None) -> PictureJob:
    """Read the live chips and expert values into one job.

    Args:
        layout: A screen opened by :func:`open_layout`.
        live: The live settings Yitro passes to the generator. Its steps,
            guidance, and sampler win over the stored values when present.
    """
    values = dict(live or {})

    def pick(key: str) -> object:
        return values[key] if values.get(key) not in (None, "") else layout.get(key)

    return PictureJob(
        prompt=composed_prompt(layout),
        negative=composed_negative(layout),
        mode=picture_mode(layout),
        aspect=str(layout.presets.get("size", "1:1")),
        output=output_name(layout),
        fmt=_format(layout.get("canvas.format")),
        seed=int(layout.seed),
        steps=_whole(pick("shell.steps")),
        guidance=_decimal(pick("shell.guidance")),
        sampler=str(pick("shell.sampler")),
        copies=layout.batch_size(),
        adult_content=layout.prompt_policy() == adult_prompt_policy(allowed=True),
        lora_path=str(pick("shell.addons") or ""),
        lora_weight=_decimal(pick("shell.lora_weight")),
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
    _turbo_defaults(settings)
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
            "canvas.export",
            "png",
            (
                ("png", "canvas.export.png"),
                ("jpg", "canvas.export.jpg"),
                ("webp", "canvas.export.webp"),
                ("avif", "canvas.export.avif"),
            ),
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


def _turbo_defaults(settings: Settings) -> None:
    """Start Expert mode on the Turbo steps and guidance. The field minimum bounds the guidance."""
    for setting_id, wanted in (("shell.steps", TURBO_STEPS), ("shell.guidance", TURBO_GUIDANCE)):
        spec = settings.get(setting_id)
        if spec is None:
            continue
        value = max(float(wanted), spec.minimum or 0.0)
        settings._specs[setting_id] = replace(spec, default=int(value) if spec.type is SettingType.INT else value)


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
    groups = []
    for group in document.groups:
        if group.group_id == "style":
            group = _style_group()
        elif group.group_id == "quality":
            group = _turbo_quality(group)
        groups.append(group)
    return PresetFile(tuple(groups))


def _turbo_quality(group: PresetGroup) -> PresetGroup:
    """The default quality chip writes the Turbo steps and guidance."""
    turbo = {"shell.steps": TURBO_STEPS, "shell.guidance": max(1.0, TURBO_GUIDANCE)}
    options = tuple(
        replace(option, values=tuple((key, turbo.get(key, value)) for key, value in option.values))
        if option.option_id == group.default
        else option
        for option in group.options
    )
    return replace(group, options=options)


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
