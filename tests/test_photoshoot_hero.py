"""Photoshoot style reference: restyle at the shared strength, one series history entry."""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PIL import Image
from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton

from canvas.models import runtime
from canvas.pipeline.series import REFERENCE_MIN, REFERENCE_STRENGTH
from canvas.models.modes import Tier
from canvas.models.plan import PIPELINE_IMG2IMG, PIPELINE_TEXT, CallPlan
from canvas.ui.shell_layout import open_layout
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import World
from tests.test_photoshoot import IDEA, PERSON, _payload, fixed_appearance
from voxprint_common.i18n import set_language


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    set_language("en")
    yield tmp_path
    set_language("en")


class _Pipe:
    calls: list[tuple[str, dict]] = []

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, **kwargs):
        _Pipe.calls.append((self.name, kwargs))
        shade = 40 + 20 * len(_Pipe.calls)
        image = Image.new("RGB", (kwargs["width"], kwargs["height"]), (shade, 90, 120))
        return types.SimpleNamespace(images=[image])


def _stub_diffusers():
    def factory(name):
        return types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _Pipe(name))

    return types.SimpleNamespace(
        ZImagePipeline=factory("text"),
        ZImageImg2ImgPipeline=factory("img2img"),
    )


def _stub_torch():
    class _Generator:
        def __init__(self, device=None):
            self.device = device

        def manual_seed(self, seed):
            self.seed = seed
            return self

    cuda = types.SimpleNamespace(is_available=lambda: False)
    return types.SimpleNamespace(bfloat16="bf16", Generator=_Generator, cuda=cuda)


class _RenderRunner:
    """Screens the inputs through the real gate and calls runtime.render with a stub pipeline."""

    def __init__(self, gate, folder: Path) -> None:
        self.gate = gate
        self.folder = folder
        self.img2img_strength: float | None = None

    def __call__(self, job, inputs):
        screened = self.gate.release_request(job.prompt, job.mode.value, inputs)
        pipeline = PIPELINE_IMG2IMG if inputs else PIPELINE_TEXT
        plan = CallPlan("diffusers", pipeline, ("stub",), "bf16", 8, None, "none", "24gb", job.mode.value)
        return runtime.render(
            plan, self.folder, screened, 64, 64, Tier.GB24,
            seed=job.seed, strength=self.img2img_strength, negative=job.negative,
        )


def test_hero_frame_feeds_img2img_at_045_and_the_series_is_one_history_entry(app, home: Path, tmp_path, monkeypatch):
    del app
    _Pipe.calls = []
    monkeypatch.setattr(runtime, "_import_torch", _stub_torch)
    monkeypatch.setattr(runtime, "_import_diffusers", _stub_diffusers)
    monkeypatch.setattr(runtime, "_place", lambda pipe, tier: None)
    monkeypatch.setattr(runtime, "reset_peak_allocated", lambda: None)
    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    gate = World().gate()
    shell = open_layout(home / "state", gate=gate)
    runner = _RenderRunner(gate, tmp_path)
    window = ShellWindow(shell, tier=Tier.GB16, enlarge=object(), runner=runner)
    window.photoshoot.writer = lambda _prompt: _payload(IDEA)
    window.photoshoot.strength = "high"
    window.view.set_prompt_text(IDEA)
    window.show()
    window.findChild(QCheckBox, "photoshootMode").click()
    window.findChild(QPushButton, "generate").click()
    QApplication.processEvents()

    names = [name for name, _ in _Pipe.calls]
    assert names[0] == "text"
    assert len(names) == 6 and set(names[1:]) == {"img2img"}
    hero_pixels = Image.open(next((home / "state" / "pictures" / "series").iterdir()) / "01.png").convert("RGB")
    for _name, kwargs in _Pipe.calls[1:]:
        assert kwargs["strength"] == pytest.approx(REFERENCE_STRENGTH)
        assert kwargs["strength"] >= REFERENCE_MIN
        assert kwargs["image"].tobytes() == hero_pixels.tobytes()
    window.close()

    layout = window.shell.layout
    assert len(layout.history) == 1
    entry = layout.history[0]
    assert entry.item_id.startswith("series-")
    assert entry.prompt == IDEA
    recorded = json.loads((home / "state" / "pictures" / "series-map.json").read_text(encoding="utf-8"))
    assert list(recorded) == [entry.item_id]
    assert recorded[entry.item_id]["kind"] == "series"
    assert recorded[entry.item_id]["frames"] == [f"{index:02}.png" for index in range(1, 7)]
    assert window._series_folders[entry.item_id] == recorded[entry.item_id]["folder"]


def test_a_person_series_anchors_at_the_default_strength(app, home: Path, tmp_path, monkeypatch):
    del app
    _Pipe.calls = []
    monkeypatch.setattr(runtime, "_import_torch", _stub_torch)
    monkeypatch.setattr(runtime, "_import_diffusers", _stub_diffusers)
    monkeypatch.setattr(runtime, "_place", lambda pipe, tier: None)
    monkeypatch.setattr(runtime, "reset_peak_allocated", lambda: None)
    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    gate = World().gate()
    shell = open_layout(home / "state", gate=gate)
    window = ShellWindow(shell, tier=Tier.GB16, enlarge=object(), runner=_RenderRunner(gate, tmp_path))
    window.photoshoot.writer = lambda _prompt: _payload(PERSON)
    assert window.photoshoot.strength == "medium"
    window.view.set_prompt_text(PERSON)
    window.show()
    window.findChild(QCheckBox, "photoshootMode").click()
    window.findChild(QPushButton, "generate").click()
    QApplication.processEvents()

    look = fixed_appearance(PERSON, int(window.shell.layout.seed))
    names = [name for name, _ in _Pipe.calls]
    assert names[0] == "text"
    assert len(names) == 6 and set(names[1:]) == {"img2img"}
    hero_pixels = Image.open(next((home / "state" / "pictures" / "series").iterdir()) / "01.png").convert("RGB")
    for _name, kwargs in _Pipe.calls:
        assert "A single photograph of" in kwargs["prompt"]
        assert kwargs["prompt"].index(look) < kwargs["prompt"].index(PERSON)
        assert "collage, grid, panels, split screen" in kwargs["negative_prompt"]
    for _name, kwargs in _Pipe.calls[1:]:
        assert kwargs["strength"] == pytest.approx(REFERENCE_STRENGTH)
        assert kwargs["strength"] >= REFERENCE_MIN
        assert kwargs["image"].tobytes() == hero_pixels.tobytes()
        assert look in kwargs["prompt"]
    assert all(look in row.full and row.full.count(look) == 1 for row in window.photoshoot.rows)
    folder = next((home / "state" / "pictures" / "series").iterdir())
    document = json.loads((folder / "series.json").read_text(encoding="utf-8"))
    assert document["appearance"] == look
    assert document["age"] == 25
    window.close()
