"""rc.34: old saved defaults move to Turbo, AVIF pictures show, LoRA errors name their cause."""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from canvas.models import runtime
from canvas.ui.shell_layout import migrate_old_defaults


def _state(path: Path, steps: object, guidance: object) -> Path:
    payload = {
        "schema": 1,
        "values": {"shell.steps": steps, "shell.guidance": guidance, "shell.sampler": "euler"},
        "chosen": {"quality": "balanced"},
        "overrides": {"quality": ["shell.steps", "shell.guidance"]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_old_saved_defaults_move_to_turbo(tmp_path: Path) -> None:
    path = _state(tmp_path / "image-settings.json", 20, 7.0)
    assert migrate_old_defaults(path) is True
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["values"]["shell.steps"] == 8
    assert saved["values"]["shell.guidance"] == 0.0
    assert saved["values"]["shell.sampler"] == "euler"
    assert saved["overrides"]["quality"] == []


@pytest.mark.parametrize(("steps", "guidance"), [(30, 7.0), (20, 4.5), (12, 2.0)])
def test_customized_values_are_kept(tmp_path: Path, steps: int, guidance: float) -> None:
    path = _state(tmp_path / "image-settings.json", steps, guidance)
    before = path.read_text(encoding="utf-8")
    assert migrate_old_defaults(path) is False
    assert path.read_text(encoding="utf-8") == before


def test_a_missing_file_is_left_alone(tmp_path: Path) -> None:
    assert migrate_old_defaults(tmp_path / "absent.json") is False


def test_an_avif_picture_loads_into_a_pixmap(tmp_path: Path) -> None:
    pytest.importorskip("PySide6.QtWidgets")
    image_mod = pytest.importorskip("PIL.Image")
    from PIL import features

    if not features.check("avif"):
        pytest.skip("this Pillow has no AVIF")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from canvas.ui.picture_file import load_pixmap

    path = tmp_path / "pic.avif"
    image_mod.new("RGB", (32, 24), (200, 30, 30)).save(path)
    pixmap = load_pixmap(path)
    assert not pixmap.isNull()
    assert (pixmap.width(), pixmap.height()) == (32, 24)
    assert load_pixmap(tmp_path / "missing.avif").isNull()


def test_a_lora_load_error_names_the_real_cause(tmp_path: Path) -> None:
    def fail(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'peft'")

    transformer = types.SimpleNamespace(load_lora_adapter=fail, set_adapters=lambda *a, **k: None)
    lora = types.SimpleNamespace(weights=tmp_path / "a.safetensors", default_strength=1.0)
    original = runtime.apply_lora

    def apply(t, chosen, strength):
        t.load_lora_adapter(str(chosen.weights), adapter_name="user")

    runtime.apply_lora = apply
    try:
        with pytest.raises(runtime.LoraLoadError) as caught:
            runtime._apply_lora_reported(transformer, lora, 1.0)
    finally:
        runtime.apply_lora = original
    message = str(caught.value)
    assert "ModuleNotFoundError" in message and "peft" in message
    assert "fit" not in message.lower()
    assert caught.value.__context__ is None or caught.value.__suppress_context__
