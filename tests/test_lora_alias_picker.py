"""The LoRA pickers show the trained alias, not the library id (rc.45 finding 4)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QComboBox, QRadioButton

from canvas.models.user_lora import UserLora
from canvas.ui import lora_panel, wizards


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _lora(**overrides: object) -> UserLora:
    fields: dict[str, object] = {
        "id": "vxpkswswl-254c5409",
        "kind": "style",
        "folder": Path("/tmp/fake-lora"),
        "weights": Path("/tmp/fake-lora/w.safetensors"),
        "trigger": "vxpkswswl",
        "default_strength": 1.0,
        "strength_range": (0.0, 2.0),
        "adult_content": False,
    }
    fields.update(overrides)
    return UserLora(**fields)  # type: ignore[arg-type]


def _fill(picker: "lora_panel.LoraPicker", loras: list[UserLora], monkeypatch) -> QComboBox:
    monkeypatch.setattr(lora_panel, "available_loras", lambda _root=None: loras)
    picker.reload("", 1.0)
    combo = picker.findChild(QComboBox, "loraPick")
    assert combo is not None
    return combo


def test_the_simple_combo_shows_the_alias_and_keeps_the_id(app, monkeypatch) -> None:
    picker = lora_panel.LoraPicker(lambda _key, _strength: None)
    combo = _fill(picker, [_lora(alias="vox_garnis", name="Garnet Style")], monkeypatch)
    assert combo.itemText(1) == "vox_garnis"
    assert combo.itemData(1) == "vxpkswswl-254c5409"


def test_the_simple_combo_falls_back_to_name_then_id(app, monkeypatch) -> None:
    picker = lora_panel.LoraPicker(lambda _key, _strength: None)
    named = _lora(alias="", name="Garnet Style")
    bare = _lora(alias="", name="", id="vxpab12cd")
    combo = _fill(picker, [named, bare], monkeypatch)
    assert combo.itemText(1) == "Garnet Style"
    assert combo.itemData(1) == "vxpkswswl-254c5409"
    assert combo.itemText(2) == "vxpab12cd"
    assert combo.itemData(2) == "vxpab12cd"


def test_the_addons_list_shows_the_alias_first() -> None:
    assert wizards._lora_label(_lora(alias="vox_garnis", name="Garnet Style")) == "vox_garnis"
    assert wizards._lora_label(_lora(alias="", name="Garnet Style")) == "Garnet Style"
    assert wizards._lora_label(_lora(alias="", name="")) == "vxpkswswl-254c5409"
    adult = wizards._lora_label(_lora(alias="vox_garnis", name="Garnet Style", adult_content=True))
    assert adult.startswith("vox_garnis —")


def test_the_addons_wizard_labels_use_the_alias(app) -> None:
    index = type("Index", (), {"available": (_lora(alias="vox_garnis", name="Garnet Style"),), "skipped": ()})()
    wizard = wizards.AddonsWizard(None, index, "", None, lambda _path, _strength: None)
    button = wizard.findChild(QRadioButton, "loraOption-vxpkswswl-254c5409")
    assert button is not None and button.text() == "vox_garnis"
    wizard.close()
    wizard.deleteLater()
