"""Discovery, listing, strength, and adapter load for Model Forge LoRAs."""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QLabel, QPlainTextEdit, QPushButton, QRadioButton

from canvas.app.i18n import text
from canvas.infra.paths import models_dir
from canvas.models.loader import ZImageGenerator
from canvas.models.modes import Mode, Tier
from canvas.models.registry import load_catalog
from canvas.models.runtime import _place_and_bind, _release_lora
from canvas.models.user_lora import (
    REASON_BASE,
    REASON_FILE,
    REASON_INCOMPLETE,
    REASON_PERSON,
    REASON_PREVIEW,
    REASON_RECORD,
    REASON_SCHEMA,
    REASON_TYPE,
    apply_lora,
    clamp_strength,
    discover,
    resolve_selected,
    unload_lora,
)
from canvas.pipeline.pipeline import Pipeline
from canvas.safety.constants import IMAGE_ADULT_SEXUAL
from canvas.safety.gate import GenerationBlocked, SafetyError
from canvas.safety.image import RawImage
from canvas.ui.shell_layout import job_from, open_layout
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import Face, World
from tests.lora_attr import encode
from tests.lora_files import write_lora
from tests.test_gate_product_rule import ADULT_EXPLICIT, CLOTHED_STUDENT, EXPLICIT, YOUTH
from tools.declared_age_shadow import KEY
from voxprint_common.i18n import set_language

_KEYS = (
    "canvas.addons.none",
    "canvas.addons.empty",
    "canvas.addons.adult_label",
    "canvas.addons.trigger",
    "canvas.addons.skipped",
    "canvas.addons.reason.record",
    "canvas.addons.reason.schema",
    "canvas.addons.reason.type",
    "canvas.addons.reason.base",
    "canvas.addons.reason.file",
    "canvas.addons.reason.person",
    "canvas.addons.reason.incomplete",
    "canvas.addons.reason.preview",
)


class _Transformer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.peft_config: dict[str, str] = {}

    def load_lora_adapter(self, path: str, adapter_name: str = "default", **kwargs: object) -> None:
        del kwargs
        self.calls.append(("load", adapter_name, path))
        self.peft_config = {adapter_name: path}

    def set_adapters(self, names: list[str], weights: list[float] | None = None) -> None:
        self.calls.append(("set", tuple(names), None if weights is None else tuple(weights)))

    def delete_adapters(self, names: list[str]) -> None:
        self.calls.append(("delete", tuple(names)))
        for name in names:
            self.peft_config.pop(name, None)


def _person(root: Path, lora_id: str = "ada-aaaaaaaa") -> Path:
    tensors = {"a.weight": b"\x01\x00\x02", "b.weight": b"\x03\x04"}
    token = encode(19, tensors)
    assert token is not None
    return write_lora(
        root,
        lora_id,
        kind="person",
        tensors=tensors,
        trigger="adatrigger",
        adult_content=True,
        metadata={
            "voxprint.subject.name": "Ada",
            "voxprint.subject.stage_name": "Ada",
            "voxprint.subject.trigger": "adatrigger",
            "voxprint.subject.consent_obtained": "true",
            "voxprint.subject.adult_confirmed": "true",
            "voxprint.subject.provenance_note": "studio",
            "voxprint.subject.release_attached": "false",
            "voxprint.subject.release_sha256": "",
            "voxprint.subject.consent_method": "written",
            "voxprint.lora.base": "z-image-turbo",
            "voxprint.lora.base_revision": "f332072aa78be7aecdf3ee76d5c247082da564a6",
            "voxprint.lora.rank": "16",
            "voxprint.lora.alpha": "16",
            "voxprint.lora.steps": "1000",
            "voxprint.lora.kind": "person",
            "voxprint.lora.trained_on": "2026-01-01",
            "voxprint.lora.adult_content": "true",
            KEY: token,
        },
        extra={"consent_method": "written"},
    )


def test_discovery_lists_matching_kinds_and_skips_bad_records(tmp_path: Path) -> None:
    write_lora(tmp_path, "boat-aaaaaaaa", kind="object", trigger="boattrigger")
    write_lora(tmp_path, "ink-bbbbbbbb", kind="style", adult_content=True, trigger="styletag")
    _person(tmp_path, "ada-cccccccc")
    write_lora(tmp_path, "late-dddddddd", kind="object")
    write_lora(tmp_path, "base-eeeeeeee", extra={"base": {"mdl_id": "moss-image-base", "revision": "abc"}})
    write_lora(tmp_path, "hash-ffffffff", file_hash="0" * 64)
    write_lora(tmp_path, "size-11111111", file_size=1)
    write_lora(tmp_path, "schema-22222222", extra={"schema": "voxprint-model/2"})
    write_lora(tmp_path, "type-33333333", extra={"type": "base"})
    write_lora(tmp_path, "kind-44444444", extra={"kind": "pet"})
    write_lora(tmp_path, "flags-55555555", kind="person", person_flags="false")
    write_lora(tmp_path, "omit-66666666", kind="person", person_flags=None)
    write_lora(tmp_path, "plain-77777777", kind="object", notice=False)
    write_lora(tmp_path, "view-88888888", preview=b"preview", preview_hash="0" * 64)
    index = discover(tmp_path)
    assert [item.id for item in index.available] == ["boat-aaaaaaaa", "late-dddddddd", "ada-cccccccc", "ink-bbbbbbbb"]
    assert index.available[0].adult_content is False
    assert index.available[2].adult_content is True
    assert index.available[2].trigger == "adatrigger"
    assert index.available[3].adult_content is True
    reasons = {item.name: item.reason_key for item in index.skipped}
    assert reasons["base-eeeeeeee"] == REASON_BASE
    assert reasons["hash-ffffffff"] == REASON_FILE
    assert reasons["size-11111111"] == REASON_FILE
    assert reasons["schema-22222222"] == REASON_SCHEMA
    assert reasons["type-33333333"] == REASON_TYPE
    assert reasons["kind-44444444"] == REASON_RECORD
    assert reasons["flags-55555555"] == REASON_PERSON
    assert reasons["omit-66666666"] == REASON_PERSON
    assert reasons["plain-77777777"] == REASON_INCOMPLETE
    assert reasons["view-88888888"] == REASON_PREVIEW
    selected = resolve_selected(index.available[0].folder, tmp_path)
    assert selected is not None and selected.id == "boat-aaaaaaaa"
    assert resolve_selected("", tmp_path) is None
    assert resolve_selected(tmp_path / "user-models" / "lora" / "hash-ffffffff", tmp_path) is None


def test_a_missing_adult_label_stays_false_and_does_not_change_listing(tmp_path: Path) -> None:
    write_lora(tmp_path, "boat-aaaaaaaa", kind="object")
    write_lora(tmp_path, "ink-bbbbbbbb", kind="object", adult_content=False)
    index = discover(tmp_path)
    assert [item.adult_content for item in index.available] == [False, False]
    assert index.skipped == ()


def test_strength_is_clamped_to_the_recorded_range(tmp_path: Path) -> None:
    folder = write_lora(tmp_path, "boat-aaaaaaaa", default_strength=0.8, strength_range=(0.0, 1.5))
    lora = discover(tmp_path).available[0]
    assert lora.folder == folder
    assert clamp_strength(lora, 2.0) == 1.5
    assert clamp_strength(lora, -1) == 0.0
    assert clamp_strength(lora, 0.7) == 0.7
    assert clamp_strength(lora, True) == 0.8  # type: ignore[arg-type]


def test_one_adapter_is_loaded_and_a_deselection_unloads_it(tmp_path: Path) -> None:
    write_lora(tmp_path, "boat-aaaaaaaa", trigger="boattrigger")
    write_lora(tmp_path, "ink-bbbbbbbb", kind="style", trigger="styletag")
    first, second = discover(tmp_path).available
    transformer = _Transformer()
    applied = apply_lora(transformer, first, 2.0)
    assert applied == 1.5
    apply_lora(transformer, second, 0.4)
    unload_lora(transformer)
    assert transformer.calls == [
        ("load", "canvas-lora", str(first.weights)),
        ("set", ("canvas-lora",), (1.5,)),
        ("delete", ("canvas-lora",)),
        ("load", "canvas-lora", str(second.weights)),
        ("set", ("canvas-lora",), (0.4,)),
        ("delete", ("canvas-lora",)),
    ]
    assert transformer.peft_config == {}


def test_render_binds_the_transformer_and_releases_it(tmp_path: Path) -> None:
    write_lora(tmp_path, "boat-aaaaaaaa")
    lora = discover(tmp_path).available[0]
    transformer = _Transformer()

    class _Pipe:
        def __init__(self) -> None:
            self.transformer = transformer

        def to(self, device: str) -> None:
            del device

    pipe = _Pipe()
    _place_and_bind(pipe, Tier.GB24, lora, 2.0)
    assert transformer.peft_config
    assert transformer.calls[-1] == ("set", ("canvas-lora",), (1.5,))
    _release_lora(pipe)
    assert transformer.peft_config == {}
    _place_and_bind(pipe, Tier.GB24, None, None)
    assert transformer.calls[-1][0] == "delete"


def test_addon_strings_exist_in_both_locales() -> None:
    for key in _KEYS:
        english = text(key, "en", name="sample", reason="because")
        russian = text(key, "ru", name="sample", reason="because")
        assert english
        assert russian


def test_the_app_does_not_read_the_attribute_or_import_the_shadow_tool() -> None:
    for path in Path("canvas").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "attr_v" not in source
        assert "declared_age" not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all("declared_age" not in alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert "declared_age" not in node.module


def test_generator_passes_the_selected_lora_and_leaves_the_prompt(monkeypatch, tmp_path: Path) -> None:
    folder = write_lora(tmp_path, "boat-aaaaaaaa", trigger="boattrigger")
    seen: dict[str, object] = {}

    def fake_render(*args: object, **kwargs: object) -> RawImage:
        del args
        seen["lora"] = kwargs.get("lora")
        seen["strength"] = kwargs.get("lora_strength")
        return RawImage(2, 2, b"\x11\x22\x33" * 4)

    monkeypatch.setattr("canvas.models.loader.snapshot_ready", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr("canvas.models.loader.render", fake_render)
    generator = ZImageGenerator(load_catalog(), Tier.GB24)
    generator.lora_path = str(folder)
    generator.lora_strength = 2.0
    screened = World().gate().release_request("a red boat", "text")
    generator.generate(screened, 64, 64)
    lora = seen["lora"]
    assert getattr(lora, "id", "") == "boat-aaaaaaaa"
    assert seen["strength"] == 2.0
    assert screened.prompt == "a red boat"
    generator.lora_path = ""
    generator.generate(screened, 64, 64)
    assert seen["lora"] is None


def _decision(path: str, prompt: str, configure) -> str:
    world = World()
    if configure is not None:
        configure(world)
    generator = ZImageGenerator(load_catalog(), Tier.GB24)
    generator.lora_path = path
    generator.lora_strength = 0.8
    pipeline = Pipeline(world.gate(), generator, Tier.GB16, use_gpu_lock=False)
    try:
        pipeline.generate(prompt, Mode.TEXT)
    except GenerationBlocked as exc:
        return f"{type(exc).__name__}:{exc.code}"
    except SafetyError as exc:
        return f"{type(exc).__name__}:{exc.code}"
    return "keep"


def _image(ages: tuple[float, float]):
    def configure(world: World) -> None:
        world.nsfw = 0.95
        world.faces = [Face(0, 0, 2, 2)]
        world.ages = [ages]
        world.image_scores[IMAGE_ADULT_SEXUAL] = 0.9

    return configure


def test_a_person_lora_with_every_flag_matches_the_gate_fixtures(monkeypatch, tmp_path: Path) -> None:
    folder = _person(tmp_path)
    assert discover(tmp_path).available[0].id == folder.name
    monkeypatch.setattr("canvas.models.loader.snapshot_ready", lambda *args, **kwargs: tmp_path)

    def fake_render(*args: object, **kwargs: object) -> RawImage:
        del args, kwargs
        return RawImage(2, 2, b"\x11\x22\x33" * 4)

    monkeypatch.setattr("canvas.models.loader.render", fake_render)
    prompts = ("a red boat", "nude child", *EXPLICIT, *YOUTH, *CLOTHED_STUDENT, *ADULT_EXPLICIT)
    cases = [(prompt, None) for prompt in prompts]
    cases.append(("a red boat", _image((0.0, 0.17))))
    cases.append(("a red boat", _image((0.0, 0.0))))
    plain = tuple(_decision("", prompt, configure) for prompt, configure in cases)
    loaded = tuple(_decision(str(folder), prompt, configure) for prompt, configure in cases)
    assert loaded == plain
    assert plain[prompts.index("a red boat")] == "keep"
    assert plain[prompts.index("nude child")] != "keep"
    assert loaded[len(prompts)] != "keep"


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    set_language("en")
    yield tmp_path
    set_language("en")


def test_the_addons_view_lists_reasons_and_the_trigger_is_a_chip(app, home: Path) -> None:
    del app
    root = models_dir()
    good = write_lora(root, "boat-aaaaaaaa", kind="object", trigger="boattrigger", default_strength=0.8)
    write_lora(root, "ink-bbbbbbbb", kind="style", adult_content=True, trigger="styletag")
    write_lora(root, "hash-cccccccc", file_hash="0" * 64)
    write_lora(root, "flags-dddddddd", kind="person", person_flags=None, trigger="personword")
    shell = open_layout(home / "state", gate=World().gate())
    window = ShellWindow(shell, tier=Tier.GB16, enlarge=object())
    window.view.set_prompt_text("a red boat")
    wizard = window.addons_wizard()
    names = [radio.objectName() for radio in wizard.findChildren(QRadioButton) if radio.objectName().startswith("lora")]
    assert names == ["loraOption-boat-aaaaaaaa", "loraOption-ink-bbbbbbbb"]
    skipped = [label.text() for label in wizard.findChildren(QLabel, "addonsSkipped")]
    assert any("hash-cccccccc" in line for line in skipped)
    assert any("flags-dddddddd" in line for line in skipped)
    assert text("canvas.addons.reason.file", "en") in " ".join(skipped)
    assert text("canvas.addons.reason.person", "en") in " ".join(skipped)
    adult = wizard.findChild(QRadioButton, "loraOption-ink-bbbbbbbb")
    assert adult is not None and text("canvas.addons.adult_label") in adult.text()
    wizard.findChild(QRadioButton, "loraOption-boat-aaaaaaaa").click()
    spin = wizard.findChild(QDoubleSpinBox, "addonsStrength")
    assert spin.value() == pytest.approx(0.8)
    assert spin.maximum() == pytest.approx(1.5)
    spin.setValue(1.0)
    wizard.findChild(QPushButton, "addonsSave").click()
    assert window.shell.layout.get("shell.addons") == str(good)
    assert job_from(window.shell.layout).lora_path == str(good)
    assert job_from(window.shell.layout).lora_weight == pytest.approx(1.0)
    box = window.findChild(QPlainTextEdit, "promptBox")
    assert box.toPlainText() == "a red boat"
    window.show()
    chip = window.findChild(QPushButton, "loraTrigger")
    assert chip is not None and not chip.isHidden() and chip.isVisible()
    assert chip.text() == "boattrigger"
    chip.click()
    assert box.toPlainText() == "boattrigger a red boat"
    assert window.shell.layout.prompt == box.toPlainText()
    again = window.addons_wizard()
    again_radio = again.findChild(QRadioButton, "loraOption-boat-aaaaaaaa")
    assert again_radio is not None and again_radio.isChecked()
    assert again.findChild(QDoubleSpinBox, "addonsStrength").value() == pytest.approx(1.0)
    again.close()
    window.close()
