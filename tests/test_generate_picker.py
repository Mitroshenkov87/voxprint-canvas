"""The generate-model picker: manifest metadata, the installed list, and the plan hook.

Every fixture is synthetic: tiny manifests in a temp suite folder, blob files of a
few bytes under a temp models folder, no real weights, prompts, or pictures.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from canvas.models.generate_picker import (
    SETTING_KEY,
    GenerateChoice,
    choices,
    implied_family,
    picked_spec,
    preselect,
    split,
)
from canvas.models.modes import Mode, Tier
from canvas.models.plan import PIPELINE_MOSS, PIPELINE_TEXT, build_plan
from canvas.models.registry import load_catalog
from tests.fakes import World, rgb

MDL = Path("canvas/mdl.json")


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    from voxprint_common.i18n import set_language

    set_language("en")
    yield tmp_path
    set_language("en")


@pytest.fixture(autouse=True)
def _close_shell_windows():
    """Close the windows a test showed, so later tests do not paint them."""
    yield
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    instance = QApplication.instance()
    if instance is None:
        return
    from canvas.ui.shell_screen import ShellWindow

    for widget in instance.topLevelWidgets():
        if isinstance(widget, ShellWindow) and widget.isVisible():
            widget.close()

# Task groups recognised by voxprint-common 0.7.1 (docs/mdl-fields.md).
# A missing or unknown group falls under "Other" in Manage models.
KNOWN_GROUPS = frozenset({
    "generate", "edit", "upscale", "restore", "voice", "transcribe",
    "translate", "writer", "safety", "internal",
})

# The exact picker metadata every shipped placeholder row must carry.
EXPECTED = {
    "z-image-turbo-bf16": ("generate", "24gb", "Z-Image Turbo (recommended)", "Fast, high-quality pictures from text; the default."),
    "z-image-turbo-bf16-16gb": ("generate", "16gb", "Z-Image Turbo for 16 GB cards", "The default model, sized to fit 16 GB graphics cards."),
    "z-image-turbo-svdquant-int4": ("generate", "16gb", "Z-Image Turbo, compact 4-bit", "Smaller and lighter; slightly lower detail."),
    "z-image-turbo-svdquant-int4-r256": ("generate", "16gb", "Z-Image Turbo, compact 4-bit (higher detail)", "4-bit build that keeps more fine detail."),
    "z-image-creative-bf16": ("generate", "24gb", "Z-Image Creative", "Slower, more varied and artistic results."),
    "z-image-turbo-fun-controlnet-union-bf16": ("edit", "24gb", "Pose & edge guidance", "Follows a pose, edges or depth from a reference picture."),
    "z-image-turbo-fun-controlnet-union-bf16-lite": ("edit", "16gb", "Pose & edge guidance, lite", "The same guidance, sized for 16 GB cards."),
    "qwen3-4b-writer": ("writer", "any", "Photoshoot writer (Qwen3 4B)", "Writes the shot list for Photoshoot series."),
    "qwen3guard-gen-0.6b": ("safety", "any", "Prompt safety check", "Checks prompts before generating."),
    "siglip2-so400m-patch14-384": ("safety", "any", "Image safety check", "Checks input and output pictures."),
    "yunet-face": ("safety", "any", "Face finder", "Finds faces for the safety and likeness checks."),
    "fairface-age": ("safety", "any", "Age check", "Part of the adults-only check."),
    "nsfw-image-detection": ("safety", "any", "Content check", "Classifies picture content for the 18+ setting."),
}


def _screened(mode: Mode):
    return World().gate().release_request("a red boat", mode.value, [])


def _row(model_id: str, tier: str, group: str | None, files: list, **extra) -> dict:
    row = {"id": model_id, "version": "1", "vram_tier": tier, "files": files,
           "on_demand": True, "optional": True}
    if group is not None:
        row["group"] = group
    row.update(extra)
    return row


def test_the_13_placeholder_rows_carry_the_exact_picker_metadata():
    doc = json.loads(MDL.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in doc["models"]}
    assert len(rows) == 13
    for model_id, (group, tier, label, purpose) in EXPECTED.items():
        row = rows[model_id]
        assert row["vram_tier"] == tier, model_id
        assert row["group"] == group, model_id
        assert row["label"] == label, model_id
        assert row["purpose"] == purpose, model_id
        # Every pre-existing field is kept.
        assert row["status"] == "unavailable" and row["files"] == [], model_id
        assert row["display_name"] and row["version"], model_id
        assert group in KNOWN_GROUPS, model_id


def test_every_canvas_mdl_group_is_recognised() -> None:
    doc = json.loads(MDL.read_text(encoding="utf-8"))
    assert len(doc["models"]) == 13
    for row in doc["models"]:
        assert row.get("group") in KNOWN_GROUPS, row["id"]


def test_common_loader_and_plan_accept_the_optional_fields(tmp_path: Path):
    catalog_mod = pytest.importorskip("voxprint_common.mdl.catalog")
    gpu = pytest.importorskip("voxprint_common.gpu_probe")
    # The shipped placeholder loads fine with the extra fields present.
    manifest = catalog_mod.load_manifest(MDL.resolve())
    assert len(manifest.models) == 13
    # Common 0.7.1 surfaces them on Model. Canvas still reads the JSON itself
    # for the picker, and the loaded models carry the same strings.
    by_id = {model.id: model for model in manifest.models}
    for model_id, (group, _tier, label, purpose) in EXPECTED.items():
        model = by_id[model_id]
        assert model.label == label, model_id
        assert model.purpose == purpose, model_id
        assert model.group == group, model_id
    # An available row with the fields plans like any other row.
    payload = b"toy-weights"
    digest = hashlib.sha256(payload).hexdigest()
    doc = {
        "schema": 1,
        "app": "canvas",
        "runtimes": [],
        "models": [
            _row("toy-generator", "16gb", "generate",
                 [{"path": "toy.bin", "size": len(payload), "sha256": digest, "mirrors": ["https://example.invalid/toy.bin"]}],
                 label="Toy Generator", purpose="A synthetic row."),
        ],
    }
    path = tmp_path / "mdl.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = catalog_mod.load_manifest(path)
    planned = catalog_mod.plan_files(
        [loaded], gpu.VramTier.GB16, models_root=tmp_path / "models", only=("toy-generator",)
    )
    assert [item.model_id for item in planned] == ["toy-generator"]


def test_choices_list_only_installed_generate_models(tmp_path: Path, monkeypatch):
    root, _manifest = _synthetic_install(tmp_path, monkeypatch)
    found = choices()
    # gen-16-notdown (blob missing), edit-control, safe-guard, writer-row and the
    # unavailable placeholder stay out; the rest keep manifest order.
    assert [choice.id for choice in found] == ["gen-16-one", "gen-24-two", "gen-16-moss"]
    one, two, moss = found
    # label is the text, purpose is the tooltip.
    assert one.text == "Generator One" and one.tip == "First compact."
    # No label: display_name wins; no purpose: empty tooltip.
    assert two.text == "Generator Two" and two.tip == ""
    assert moss.text == "MOSS compact"
    # None of the synthetic ids is in the Canvas catalog.
    assert all(not choice.catalogued for choice in found)
    # The row appears once its blob lands, and disappears when the size turns wrong.
    assert "gen-16-notdown" not in {choice.id for choice in choices()}
    (root / "blobs" / _DIGESTS["missing"]).write_bytes(b"missing" * 8)
    assert [choice.id for choice in choices()] == ["gen-16-one", "gen-24-two", "gen-16-moss", "gen-16-notdown"]
    (root / "blobs" / _DIGESTS["missing"]).write_bytes(b"m")
    assert "gen-16-notdown" not in {choice.id for choice in choices()}
    (root / "blobs" / _DIGESTS["moss"]).write_bytes(b"moss")  # wrong size now
    assert [choice.id for choice in choices()] == ["gen-16-one", "gen-24-two"]


def test_a_missing_group_is_inferred_from_the_catalog_role(tmp_path: Path, monkeypatch):
    _synthetic_install(tmp_path, monkeypatch, infer_group_rows=True)
    found = choices()
    ids = {choice.id for choice in found}
    # MOSS is a catalog generator: an installed row without a group field joins,
    # and it is catalogued, so the plan can run it.
    assert "moss-image-sft-bf16" in ids
    (moss,) = [choice for choice in found if choice.id == "moss-image-sft-bf16"]
    assert moss.catalogued and moss.tier == "24gb"
    # Control and safety catalog rows never join, with or without a group field.
    assert "z-image-turbo-fun-controlnet-union-bf16" not in ids
    assert "qwen3guard-gen-0.6b" not in ids


def test_sections_and_preselect_follow_the_detected_tier():
    compact_row = _choice("compact-16", "16gb")
    full_row = _choice("full-24", "24gb")
    both = (compact_row, full_row)
    compact, full = split(both)
    assert compact == (compact_row,) and full == (full_row,)
    # A 16 GB card never auto-defaults to the full-size row; a 24 GB card can
    # still take a compact build once it is stored.
    assert preselect(both, Tier.GB16, "") == "compact-16"
    assert preselect(both, Tier.GB24, "") == "full-24"
    assert preselect(both, Tier.GB24, "compact-16") == "compact-16"
    assert preselect((full_row,), Tier.GB16, "") == "full-24"
    assert preselect((), Tier.GB16, "compact-16") == ""


def _choice(model_id: str, tier: str) -> GenerateChoice:
    return GenerateChoice(id=model_id, text=model_id, tip="", tier=tier, catalogued=False)


def test_build_plan_honors_an_installed_catalogued_pick():
    catalog = load_catalog()
    # A compact build chosen on a 24 GB card plans that exact row.
    plan = build_plan(catalog, _screened(Mode.TEXT), Tier.GB24, model_id="z-image-turbo-bf16-16gb")
    assert plan.model_id == "z-image-turbo-bf16-16gb" and plan.pipeline_class == PIPELINE_TEXT
    assert plan.tier == "24gb"  # the card tier; the row is the picked one
    # A MOSS pick plans the MOSS pipeline.
    moss = build_plan(catalog, _screened(Mode.TEXT), Tier.GB16, model_id="moss-image-sft-bf16")
    assert moss.model_id == "moss-image-sft-bf16" and moss.pipeline_class == PIPELINE_MOSS
    # A creative pick plans the creative row with its own steps.
    creative = build_plan(catalog, _screened(Mode.TEXT), Tier.GB24, model_id="z-image-creative-bf16")
    assert creative.model_id == "z-image-creative-bf16" and creative.steps == 50
    # No pick: the family mapping is unchanged.
    default = build_plan(catalog, _screened(Mode.TEXT), Tier.GB16)
    assert default.model_id == "z-image-turbo-bf16-16gb"


def test_build_plan_keeps_planning_when_the_pick_cannot_run():
    catalog = load_catalog()
    # The compact 4-bit ids are not in the catalog: the plan falls back to the
    # family row instead of breaking.
    plan = build_plan(catalog, _screened(Mode.TEXT), Tier.GB24, model_id="z-image-turbo-svdquant-int4")
    assert plan.model_id == "z-image-turbo-bf16"
    # A non-generator id never routes a picture onto a check model.
    plan = build_plan(catalog, _screened(Mode.TEXT), Tier.GB16, model_id="qwen3guard-gen-0.6b")
    assert plan.model_id == "z-image-turbo-bf16-16gb"
    # A bound LoRA's trained family wins over a pick of another family.
    plan = build_plan(
        catalog, _screened(Mode.TEXT), Tier.GB16, base="moss-image-sft", model_id="z-image-turbo-bf16"
    )
    assert plan.pipeline_class == PIPELINE_MOSS and plan.model_id != "z-image-turbo-bf16"
    # A pick of the LoRA's own family still plans the picked row.
    plan = build_plan(
        catalog, _screened(Mode.TEXT), Tier.GB24, base="z-image-turbo", model_id="z-image-turbo-bf16-16gb"
    )
    assert plan.model_id == "z-image-turbo-bf16-16gb"


def test_picked_spec_and_implied_family():
    catalog = load_catalog()
    assert picked_spec(catalog, "z-image-creative-bf16") is not None
    assert picked_spec(catalog, "z-image-turbo-svdquant-int4") is None  # not catalogued
    assert picked_spec(catalog, "siglip2-so400m-patch14-384") is None  # not a generator
    assert picked_spec(catalog, "") is None
    assert implied_family("z-image-turbo-bf16", catalog) == "z-image-turbo"
    assert implied_family("moss-image-sft-bf16", catalog) == "moss"
    assert implied_family("z-image-creative-bf16", catalog) == ""
    assert implied_family("z-image-turbo-svdquant-int4", catalog) == ""


_DIGESTS: dict[str, str] = {}


def _files(root: Path, tag: str, *, on_disk: bool = True) -> list:
    payload = tag.encode() * 8
    digest = hashlib.sha256(payload).hexdigest()
    _DIGESTS[tag] = digest
    if on_disk:
        (root / "blobs").mkdir(parents=True, exist_ok=True)
        (root / "blobs" / digest).write_bytes(payload)
    return [{"path": f"{tag}.bin", "size": len(payload), "sha256": digest,
             "mirrors": [f"https://example.invalid/{tag}.bin"]}]


def _synthetic_install(tmp_path: Path, monkeypatch, *, infer_group_rows: bool = False):
    """A fake suite install: manifest under the suite root, blobs under the models dir."""
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VOXPRINT_SUITE_ROOT", str(tmp_path / "suite"))
    _DIGESTS.clear()
    root = tmp_path / "home" / "models"
    rows = [
        _row("gen-16-one", "16gb", "generate", _files(root, "one"), label="Generator One", purpose="First compact."),
        _row("gen-24-two", "24gb", "generate", _files(root, "two"), display_name="Generator Two"),
        _row("gen-16-moss", "16gb", "generate", _files(root, "moss"), label="MOSS compact"),
        _row("gen-16-notdown", "16gb", "generate", _files(root, "missing", on_disk=False), label="Not downloaded"),
        _row("edit-control", "24gb", "edit", _files(root, "ctrl"), label="ControlNet"),
        _row("safe-guard", "any", "safety", _files(root, "guard"), label="Guard"),
        _row("writer-row", "any", "writer", _files(root, "writer"), label="Writer"),
        _row("ph-old", "any", "generate", [], label="Placeholder", status="unavailable", reason="placeholder"),
    ]
    if infer_group_rows:
        # No group field: the Canvas catalog role decides. moss-image-sft-bf16 is a
        # generator (joins); the control and the guard rows are not (stay out).
        rows.extend([
            _row("moss-image-sft-bf16", "24gb", None, _files(root, "catalog-moss")),
            _row("z-image-turbo-fun-controlnet-union-bf16", "24gb", None, _files(root, "catalog-control")),
            _row("qwen3guard-gen-0.6b", "any", None, _files(root, "catalog-guard")),
        ])
    app = tmp_path / "suite" / "canvas"
    app.mkdir(parents=True)
    manifest = app / "mdl.json"
    manifest.write_text(json.dumps({"schema": 1, "app": "canvas", "runtimes": [], "models": rows}), encoding="utf-8")
    return root, manifest


@pytest.fixture
def qt_app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _window(app, home: Path, tier: Tier):
    from tests.worker_support import inline_window

    return inline_window(home / "state", tier=tier)


def test_the_picker_shows_sections_badges_and_preselects_the_card_tier(
    qt_app, tmp_path: Path, monkeypatch
):
    from PySide6.QtWidgets import QLabel, QPushButton, QToolButton, QWidget

    _synthetic_install(tmp_path, monkeypatch)
    window = _window(qt_app, tmp_path / "home", Tier.GB16)
    window.show()
    qt_app.processEvents()
    picker = window.findChild(QWidget, "modelPicker")
    assert picker is not None and picker.isVisibleTo(window.view.root)
    # The 16 GB section is open with the first compact row selected; the 24 GB
    # section sits collapsed under the disclosure control.
    assert picker.findChild(QLabel, "modelSection-compact").isVisibleTo(picker)
    assert not picker.findChild(QWidget, "modelRows-full").isVisibleTo(picker)
    compact_rows = picker.findChild(QWidget, "modelRows-compact")
    full_rows = picker.findChild(QWidget, "modelRows-full")
    assert [button.text() for button in compact_rows.findChildren(QPushButton, "modelPick")] == [
        "Generator One", "MOSS compact",
    ]
    assert [badge.text() for badge in compact_rows.findChildren(QLabel, "modelBadge")] == ["16 GB", "16 GB"]
    assert [badge.text() for badge in full_rows.findChildren(QLabel, "modelBadge")] == ["24 GB+"]
    assert picker._group.checkedButton().text() == "Generator One"
    toggle = picker.findChild(QToolButton, "modelPickerOthers")
    assert toggle.text() == "Show models for other cards" and toggle.isVisibleTo(picker)

    # A full-size row is picked with a click: the compact section folds under
    # the toggle and the pick is stored.
    full_rows.findChild(QPushButton, "modelPick").click()
    qt_app.processEvents()
    assert window.shell.layout.get(SETTING_KEY) == "gen-24-two"
    assert not compact_rows.isVisibleTo(picker)
    assert full_rows.isVisibleTo(picker)
    # The disclosure reveals the other section again.
    toggle.click()
    qt_app.processEvents()
    assert compact_rows.isVisibleTo(picker) and toggle.text() == "Hide models for other cards"
    # Words-to-image plans carry the stored pick into the job.
    window.view.set_prompt_text("a red boat")
    from canvas.ui.shell_layout import job_from

    assert job_from(window.shell.layout).model_id == "gen-24-two"
    window.close()


def test_a_24gb_window_preselects_the_full_size_section(qt_app, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QPushButton, QToolButton, QWidget

    _synthetic_install(tmp_path, monkeypatch)
    window = _window(qt_app, tmp_path / "home", Tier.GB24)
    window.show()
    qt_app.processEvents()
    picker = window.findChild(QWidget, "modelPicker")
    assert picker._group.checkedButton().text() == "Generator Two"
    assert picker.findChild(QWidget, "modelRows-full").isVisibleTo(picker)
    assert not picker.findChild(QWidget, "modelRows-compact").isVisibleTo(picker)
    # A compact build may still be chosen on a 24 GB card.
    compact_rows = picker.findChild(QWidget, "modelRows-compact")
    toggle = picker.findChild(QToolButton, "modelPickerOthers")
    toggle.click()
    qt_app.processEvents()
    compact_rows.findChildren(QPushButton, "modelPick")[0].click()
    qt_app.processEvents()
    assert window.shell.layout.get(SETTING_KEY) == "gen-16-one"
    window.close()


def test_the_picker_hides_when_no_generate_model_is_installed(qt_app, home: Path, monkeypatch):
    from PySide6.QtWidgets import QWidget

    # No pinned manifest anywhere: only the placeholder stub, all unavailable.
    from canvas.models import mdl_registry

    monkeypatch.setattr(mdl_registry, "_BUNDLED", home / "missing.json")
    window = _window(qt_app, home, Tier.GB16)
    window.show()
    qt_app.processEvents()
    picker = window.findChild(QWidget, "modelPicker")
    assert picker is not None and not picker.isVisibleTo(window.view.root)
    window.close()


def test_varied_words_use_the_pick_only_when_the_catalog_can_run_it(qt_app, home: Path):
    window = _window(qt_app, home, Tier.GB24)
    layout = window.shell.layout
    layout.set_value("canvas.variation", "varied")
    from canvas.ui.shell_layout import picture_mode

    # No pick yet: "varied" stays the creative mode.
    assert picture_mode(layout) is Mode.TEXT_CREATIVE
    # A catalogued generate pick answers "varied" on its own.
    layout.set_value(SETTING_KEY, "z-image-turbo-bf16")
    assert picture_mode(layout) is Mode.TEXT
    # An installed but uncatalogued pick cannot run, so "varied" is unchanged.
    layout.set_value(SETTING_KEY, "gen-16-one")
    assert picture_mode(layout) is Mode.TEXT_CREATIVE
    window.close()


def test_the_pick_reaches_the_worker_job(qt_app, home: Path, monkeypatch):
    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    seen: list[str] = []

    def runner(job, inputs):
        del inputs
        seen.append(job.model_id)
        return rgb(8, 8)

    from PySide6.QtWidgets import QPushButton

    from tests.worker_support import inline_window, settle

    window = inline_window(home / "state", runner=runner, tier=Tier.GB16)
    window.shell.layout.set_value(SETTING_KEY, "z-image-turbo-bf16-16gb")
    window.view.set_prompt_text("a red boat")
    window.findChild(QPushButton, "generate").click()
    settle(window)
    assert seen == ["z-image-turbo-bf16-16gb"]
    window.close()


def test_the_runner_hands_the_pick_to_the_generator():
    from tests.worker_support import make_session

    session = make_session(tier=Tier.GB16)
    job = _job("moss-image-sft-bf16")
    generator = session.runner._generator(job)
    assert generator.model_id == "moss-image-sft-bf16"
    empty = session.runner._generator(_job(""))
    assert empty.model_id is None


def _job(model_id: str):
    from canvas.core.formats import OutputFormat
    from canvas.ui.shell_layout import PictureJob

    return PictureJob(
        prompt="a red boat", negative="", mode=Mode.TEXT, aspect="1:1", output="native",
        fmt=OutputFormat.WEBP, seed=1, steps=8, guidance=0.0, sampler="euler", copies=1,
        model_id=model_id,
    )


def test_a_catalogued_pick_updates_the_shared_family(qt_app, home: Path, monkeypatch):
    from voxprint_common.suite_settings import model_default, set_model_default

    window = _window(qt_app, home, Tier.GB16)
    set_model_default("z-image-turbo")
    window._pick_model("moss-image-sft-bf16")
    assert model_default() == "moss"
    assert window.shell.layout.get(SETTING_KEY) == "moss-image-sft-bf16"
    # Creative and uncatalogued picks imply no family: the shared value stays.
    window._pick_model("z-image-creative-bf16")
    assert model_default() == "moss"
    window._pick_model("z-image-turbo-bf16")
    assert model_default() == "z-image-turbo"
    window.close()
