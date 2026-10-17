"""The shared screen refuses to open without the real content gate."""
from __future__ import annotations

from pathlib import Path

import pytest

from canvas.safety.constants import IMAGE_MINOR_SEXUAL
from canvas.safety.gate import GenerationBlocked
from canvas.safety.inputs import ImageRole, PendingInput
from canvas.ui.shell_layout import open_layout
from canvas.ui.styles import LOOKS
from tests.fakes import rgb
from tests.fakes import World
from voxprint_common.ui.settings import SettingType
from voxprint_common.ui.simple_image import Phase


class _Allow:
    def review_prompt(self, prompt: str) -> None:
        del prompt
        return None

    def review_result(self, prompt: str, item_id: str) -> None:
        del prompt, item_id
        return None


def test_constructing_the_layout_without_a_gate_fails(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        open_layout(tmp_path)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        open_layout(tmp_path, gate=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        open_layout(tmp_path, gate=_Allow())  # type: ignore[arg-type]


def test_adult_content_and_expert_values_leave_the_gate_blocking(tmp_path: Path) -> None:
    world = World()
    gate = world.gate()
    shell = open_layout(tmp_path, gate=gate)
    assert shell.gate.age_gate is gate
    before = shell.layout.prompt_policy()
    assert before.template
    for spec in shell.settings.visible(expert=True):
        if not spec.expert:
            continue
        if spec.type is SettingType.BOOL:
            shell.layout.set_value(spec.setting_id, True)
            continue
        if spec.type is SettingType.INT:
            shell.layout.set_value(spec.setting_id, int(spec.default) + 1)
            continue
        if spec.type is SettingType.FLOAT:
            shell.layout.set_value(spec.setting_id, float(spec.default) + 0.5)
            continue
        if spec.type is SettingType.ENUM:
            other = next(choice.value for choice in spec.choices if choice.value != spec.default)
            shell.layout.set_value(spec.setting_id, other)
            continue
        if spec.type is SettingType.TEXT or spec.type is SettingType.PATH:
            shell.layout.set_value(spec.setting_id, "changed")
            continue
        raise AssertionError(spec.type)
    assert shell.layout.get("image.adult_content") is True
    assert shell.layout.prompt_policy().template == ""
    for prompt in ("nude child", "loli"):
        shell.layout.set_prompt(prompt)
        assert shell.layout.generate() is False
        assert shell.layout.phase is Phase.BLOCKED
    world.image_scores[IMAGE_MINOR_SEXUAL] = 0.35
    with pytest.raises(GenerationBlocked):
        gate.release_request("a red boat", "restyle", (PendingInput(ImageRole.INIT, rgb()),))
    world.image_scores[IMAGE_MINOR_SEXUAL] = 0.0
    shell.layout.set_prompt("a red bicycle")
    assert shell.layout.generate() is True
    world.image_scores[IMAGE_MINOR_SEXUAL] = 0.35
    shell.gate.stage("out", rgb())
    assert shell.layout.finish("out") is None
    assert shell.layout.phase is Phase.BLOCKED


def test_a_result_without_pixels_is_refused(tmp_path: Path) -> None:
    shell = open_layout(tmp_path, gate=World().gate())
    shell.layout.set_prompt("a red bicycle")
    assert shell.layout.generate() is True
    assert shell.layout.finish("missing") is None
    assert shell.layout.phase is Phase.BLOCKED


def test_style_chips_write_a_look_and_not_only_a_negative(tmp_path: Path) -> None:
    shell = open_layout(tmp_path, gate=World().gate())
    shell.layout.choose("style", "sketch")
    assert shell.layout.get("canvas.style") == "sketch"
    assert shell.layout.get("shell.negative") == "full color"
    style = next(row for row in shell.layout.rows if row.row_id == "style")
    assert {choice.value for choice in style.choices} == {look.style_id for look in LOOKS}
    assert all(row.row_id != "instruction" for row in shell.layout.rows)
