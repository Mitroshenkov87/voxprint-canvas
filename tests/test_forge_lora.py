"""A Model Forge LoRA file loads through the existing adapter."""
from __future__ import annotations

import hashlib
import json
import os
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QPushButton

from canvas.app.i18n import text
from canvas.app.main import _parser, main
from canvas.app.photoshoot_cli import model_generate, run_command
from canvas.app.shell_run import ShellRunner
from canvas.models import runtime
from canvas.models.modes import Mode, Tier
from canvas.models.plan import PIPELINE_TEXT, CallPlan
from canvas.models.user_lora import (
    CLI_STRENGTH_RANGE,
    CODE_BASE_MISMATCH,
    CODE_INVALID,
    CODE_SCALE,
    LoraRejected,
    apply_lora,
    clamp_strength,
    cli_scale,
    discover,
    prepare_cli_lora,
    resolve_selected,
)
from canvas.safety.image import RawImage
from canvas.ui.shell_layout import PictureJob, open_layout
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import World
from tests.lora_files import write_lora, write_safetensors
from tests.test_photoshoot import IDEA, _frame, _payload
from tests.test_photoshoot_cli import _ready, _writers
from tests.test_user_lora import _decision
from voxprint_common.i18n import set_language

_FORGE_A = "layers.0.attention.to_q.lora_A.weight"
_FORGE_B = "layers.0.attention.to_q.lora_B.weight"
_LORA_ID = "kettleword-abc12345"


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


def _metadata(base: str, trigger: str = "kettleword") -> dict[str, str]:
    return {
        "voxprint.subject.name": "Kettle",
        "voxprint.subject.stage_name": "",
        "voxprint.subject.trigger": trigger,
        "voxprint.subject.consent_obtained": "false",
        "voxprint.subject.adult_confirmed": "false",
        "voxprint.subject.provenance_note": "",
        "voxprint.subject.release_attached": "false",
        "voxprint.subject.release_sha256": "",
        "voxprint.subject.rights_confirmed": "true",
        "voxprint.lora.base": base,
        "voxprint.lora.base_revision": "04cc4abb7c5069926f75c9bfde9ef43d49423021",
        "voxprint.lora.rank": "16",
        "voxprint.lora.alpha": "16",
        "voxprint.lora.steps": "100",
        "voxprint.lora.kind": "object",
        "voxprint.lora.trained_on": "2026-01-01",
        "voxprint.lora.adult_content": "false",
    }


def _export(
    folder: Path,
    *,
    base: str = "z-image",
    tensors: dict[str, bytes] | None = None,
    sidecar: bool = True,
    sha: str | None = None,
) -> Path:
    """One file in the Forge export directory, with the Forge tensor names."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_LORA_ID}.safetensors"
    weights = tensors if tensors is not None else {_FORGE_A: b"\x01\x02", _FORGE_B: b"\x03\x04"}
    write_safetensors(path, weights, _metadata(base))
    if sidecar:
        raw = path.read_bytes()
        document = {
            "schema": "voxprint-model/1",
            "min_schema": 1,
            "id": _LORA_ID,
            "kind": "object",
            "type": "lora",
            "producer": "voxprint-model-forge 0.1.0",
            "file": path.name,
            "sha256": sha if sha is not None else hashlib.sha256(raw).hexdigest(),
            "size": path.stat().st_size,
            "base": {"mdl_id": base, "revision": "04cc4abb7c5069926f75c9bfde9ef43d49423021"},
            "licence": "Apache-2.0",
            "default_strength": 0.8,
            "strength_range": [0.0, 1.5],
            "tags": [],
            "voxprint.lora.adult_content": False,
        }
        (folder / "voxprint-model.json").write_text(json.dumps(document), encoding="utf-8")
    return path


def test_a_forge_export_keeps_its_keys_name_and_trigger(tmp_path: Path) -> None:
    path = _export(tmp_path / "project" / "export")
    lora = prepare_cli_lora(path)
    assert lora.id == _LORA_ID
    assert lora.trigger == "kettleword"
    assert lora.weights == path
    assert lora.default_strength == 0.7
    assert lora.strength_range == CLI_STRENGTH_RANGE
    assert cli_scale(0) == 0
    assert cli_scale(2) == 2
    transformer = _Transformer()
    assert apply_lora(transformer, lora, 2.0) == 2.0
    assert transformer.calls[0] == ("load", "canvas-lora", str(path))
    selected = resolve_selected(path)
    assert selected is not None and selected.trigger == "kettleword" and selected.id == _LORA_ID


def test_a_down_up_file_is_renamed_to_lora_a_and_lora_b(tmp_path: Path) -> None:
    path = _export(
        tmp_path / "project" / "export",
        tensors={
            "diffusion_model.layers.0.attention.to_q.lora_down.weight": b"\x01\x02",
            "diffusion_model.layers.0.attention.to_q.lora_up.weight": b"\x03\x04",
        },
    )
    lora = prepare_cli_lora(path)
    assert lora.weights != path
    header = json.loads(lora.weights.read_bytes()[8:8 + int.from_bytes(lora.weights.read_bytes()[:8], "little")])
    assert _FORGE_A in header
    assert _FORGE_B in header


def test_the_wrong_base_and_a_bad_file_use_stable_codes(tmp_path: Path) -> None:
    moss = _export(tmp_path / "moss", base="moss-image-base")
    with pytest.raises(LoraRejected) as mismatch:
        prepare_cli_lora(moss)
    assert mismatch.value.code == CODE_BASE_MISMATCH
    bad = tmp_path / "notes.safetensors"
    bad.write_bytes(b"not safetensors")
    with pytest.raises(LoraRejected) as invalid:
        prepare_cli_lora(bad)
    assert invalid.value.code == CODE_INVALID
    short = tmp_path / "short.safetensors"
    short.write_bytes((40).to_bytes(8, "little") + b"{}")
    with pytest.raises(LoraRejected) as truncated:
        prepare_cli_lora(short)
    assert truncated.value.code == CODE_INVALID
    mismatched = _export(tmp_path / "hash", sha="0" * 64)
    with pytest.raises(LoraRejected) as digest:
        prepare_cli_lora(mismatched)
    assert digest.value.code == CODE_INVALID
    with pytest.raises(LoraRejected) as scale:
        cli_scale(2.1)
    assert scale.value.code == CODE_SCALE


def test_a_listed_z_image_folder_uses_the_command_range(tmp_path: Path) -> None:
    folder = write_lora(
        tmp_path,
        "kettle-aaaaaaaa",
        base="z-image",
        trigger="kettleword",
        tensors={_FORGE_A: b"\x01\x02", _FORGE_B: b"\x03\x04"},
        strength_range=(0.0, 1.5),
    )
    listed = discover(tmp_path).available
    assert [item.id for item in listed] == ["kettle-aaaaaaaa"]
    assert clamp_strength(listed[0], 2.0) == 1.5
    opened = prepare_cli_lora(folder)
    assert opened.strength_range == (0.0, 2.0)
    assert apply_lora(_Transformer(), opened, 2.0) == 2.0


def test_commands_report_the_lora_and_reject_it_before_the_gpu(monkeypatch, capsys, tmp_path: Path) -> None:
    path = _export(tmp_path / "project" / "export")
    calls: list[str] = []
    monkeypatch.setattr("canvas.app.main.probe_gpu", lambda: calls.append("gpu"))
    missing = main([
        "generate", "--prompt", "a red boat", "--out", str(tmp_path / "out.avif"),
        "--lora", str(tmp_path / "missing.safetensors"), "--json",
    ])
    body = json.loads(capsys.readouterr().out)
    assert missing == 2
    assert calls == []
    assert body["error"] == CODE_INVALID
    assert body["message"] == text("lora.invalid", "en")
    moss = _export(tmp_path / "moss", base="moss-image-base")
    mismatch = main([
        "photoshoot", "--prompt", IDEA, "--out", str(tmp_path / "series"),
        "--lora", str(moss), "--json",
    ])
    assert mismatch == 2
    assert json.loads(capsys.readouterr().out)["error"] == CODE_BASE_MISMATCH
    assert calls == []
    scaled = main([
        "generate", "--prompt", "a red boat", "--out", str(tmp_path / "out.avif"),
        "--lora", str(path), "--lora-scale", "2.5", "--json",
    ])
    assert scaled == 2
    assert json.loads(capsys.readouterr().out)["error"] == CODE_SCALE
    plain = main([
        "generate", "--prompt", "a red boat", "--out", str(tmp_path / "out.avif"),
        "--lora", str(tmp_path / "missing.safetensors"),
    ])
    assert plain == 2
    assert text("lora.invalid", "en") in capsys.readouterr().err


def test_photoshoot_json_names_the_lora_and_every_frame_keeps_it(monkeypatch, capsys, tmp_path: Path) -> None:
    path = _export(tmp_path / "project" / "export")
    lora = prepare_cli_lora(path)
    seen: list[tuple[str, float, str]] = []

    class _Runner:
        def __init__(self, *, gate: object, tier: object, enlarge: object) -> None:
            del gate, tier, enlarge
            self.img2img_strength: float | None = None
            self.bound_lora = None

        def __call__(self, job: PictureJob, inputs: tuple[object, ...]) -> object:
            del inputs
            bound = "" if self.bound_lora is None else self.bound_lora.id
            seen.append((job.lora_path, job.lora_weight, bound))
            return _frame()

    monkeypatch.setattr("canvas.app.photoshoot_cli.ShellRunner", _Runner)
    gate = World().gate()
    payload, code = run_command(
        idea=IDEA,
        folder=tmp_path / "frames",
        frames=2,
        seed=1,
        style="photo",
        places="same",
        adult=False,
        age_gate=gate,
        rewrite=lambda _prompt: _payload(IDEA, 2),
        generate=model_generate(gate, Tier.GB16, "photo", False, lora, 0.7),
    )
    assert code == 0 and payload["made"] == 2
    assert seen == [(str(lora.weights), 0.7, lora.id), (str(lora.weights), 0.7, lora.id)]
    _ready(monkeypatch)

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        del line, seed, hero, strength
        return _frame()

    _writers(monkeypatch, generate)
    command = main([
        "photoshoot", "--prompt", IDEA, "--frames", "2", "--out", str(tmp_path / "out"),
        "--json", "--lora", str(path), "--lora-scale", "1.2",
    ])
    body = json.loads(capsys.readouterr().out)
    assert command == 0
    assert body["lora"] == _LORA_ID
    assert body["lora_scale"] == 1.2
    assert body["lora_trigger"] == "kettleword"
    parsed = _parser().parse_args(["generate", "--prompt", "a red boat", "--out", "picture.avif"])
    assert parsed.lora == ""
    assert parsed.lora_scale == 0.7


def test_the_runner_passes_the_bound_lora_to_the_generator(monkeypatch, tmp_path: Path) -> None:
    path = _export(tmp_path / "project" / "export")
    lora = prepare_cli_lora(path)
    seen: dict[str, object] = {}

    class _Generator:
        def __init__(self, catalog: object, tier: object) -> None:
            del catalog, tier
            seen["generator"] = self
            self.seed = 0
            self.lora_path = ""
            self.lora_strength = 0.0
            self.bound_lora = None
            self.models = None

    class _Pipeline:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def generate(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            return types.SimpleNamespace(width=2, height=2, data=b"\x11\x22\x33" * 4)

    monkeypatch.setattr("canvas.app.shell_run.ZImageGenerator", _Generator)
    monkeypatch.setattr("canvas.app.shell_run.Pipeline", _Pipeline)
    runner = ShellRunner(gate=World().gate(), tier=Tier.GB16, enlarge=object())
    runner.bound_lora = lora
    job = PictureJob(
        "a red boat", "", Mode.TEXT, "1:1", "native", "avif", 1, 8, 0.0, "euler", 1,
        lora_path=str(lora.weights), lora_weight=0.7,
    )
    runner(job, ())
    generator = seen["generator"]
    assert getattr(generator, "bound_lora", None) is lora
    assert getattr(generator, "lora_path", "") == str(path)
    assert getattr(generator, "lora_strength", None) == 0.7


def test_a_series_keeps_one_lora_and_unloads_it_at_the_end(tmp_path: Path, monkeypatch) -> None:
    path = _export(tmp_path / "project" / "export")
    lora = prepare_cli_lora(path)
    transformer = _Transformer()
    loads = [0]

    class _Pipe:
        def __init__(self) -> None:
            self.transformer = transformer

        def __call__(self, **kwargs: object) -> object:
            image = Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"])), (40, 90, 120))
            return types.SimpleNamespace(images=[image])

    def text_load(*_args: object, **_kwargs: object) -> _Pipe:
        loads[0] += 1
        return _Pipe()

    diffusers = types.SimpleNamespace(
        ZImagePipeline=types.SimpleNamespace(from_pretrained=text_load, from_pipe=lambda pipe: pipe),
    )

    class _Generator:
        def __init__(self, device: object = None) -> None:
            self.device = device

        def manual_seed(self, seed: int) -> _Generator:
            del seed
            return self

    cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    torch = types.SimpleNamespace(bfloat16="bf16", Generator=_Generator, cuda=cuda)
    monkeypatch.setattr(runtime, "_import_torch", lambda: torch)
    monkeypatch.setattr(runtime, "_import_diffusers", lambda: diffusers)
    monkeypatch.setattr(runtime, "_place", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime, "reset_peak_allocated", lambda: None)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: 32 * 1024**3)
    gate = World().gate()
    plan = CallPlan("diffusers", PIPELINE_TEXT, ("stub",), "bf16", 8, None, "none", "16gb", "text")
    try:
        with runtime.pipeline_scope():
            for _index in range(2):
                screened = gate.release_request("a red boat", "text", ())
                runtime.render(
                    plan, tmp_path / "weights", screened, 32, 32, Tier.GB16,
                    seed=1, lora=lora, lora_strength=0.7,
                )
        assert loads[0] == 1
        assert [call[0] for call in transformer.calls] == ["load", "set", "delete"]
        assert transformer.calls[0] == ("load", "canvas-lora", str(path))
        assert transformer.peft_config == {}
    finally:
        runtime.reset_pipeline_cache()


def test_a_forge_file_keeps_the_same_prompt_decisions(monkeypatch, tmp_path: Path) -> None:
    path = _export(tmp_path / "project" / "export")
    assert resolve_selected(path) is not None
    monkeypatch.setattr("canvas.models.loader.snapshot_ready", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(
        "canvas.models.loader.render",
        lambda *_args, **_kwargs: RawImage(2, 2, b"\x11\x22\x33" * 4),
    )
    prompts = ("a red boat", "nude child")
    plain = tuple(_decision("", prompt, None) for prompt in prompts)
    loaded = tuple(_decision(str(path), prompt, None) for prompt in prompts)
    assert loaded == plain
    assert plain[0] == "keep"
    assert plain[1] != "keep"


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


def test_the_trigger_chip_inserts_the_forge_word(app, home: Path, tmp_path: Path) -> None:
    del app
    path = _export(tmp_path / "project" / "export")
    shell = open_layout(home / "state", gate=World().gate())
    window = ShellWindow(shell, tier=Tier.GB16, enlarge=object())
    window.view.set_prompt_text("a red boat")
    window._save_addons(str(path), 0.7)
    window.show()
    box = window.findChild(QPlainTextEdit, "promptBox")
    assert box.toPlainText() == "a red boat"
    chip = window.findChild(QPushButton, "loraTrigger")
    assert chip is not None and chip.isVisible()
    assert chip.text() == "kettleword"
    chip.click()
    assert box.toPlainText() == "kettleword a red boat"
    window.close()
