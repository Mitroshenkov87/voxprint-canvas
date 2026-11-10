"""GUI and CLI must share one LoRA apply path; the adapter has to actually load."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from canvas.app.platform import GpuCheck, OsCheck
from canvas.app.shell_run import ShellRunner
from canvas.core.formats import OutputFormat
from canvas.infra.gpu import GpuInfo
from canvas.models.loader import ZImageGenerator
from canvas.models.lora_library import import_lora, resolve_library
from canvas.models.modes import Mode, Tier
from canvas.models.registry import load_catalog
from canvas.models.runtime import _place_and_bind
from canvas.models.user_lora import (
    CODE_INVALID,
    LoraApplyResult,
    LoraRejected,
    UserLora,
    apply_bound_lora,
    prepare_cli_lora,
)
from canvas.safety.image import RawImage
from canvas.ui.shell_layout import PictureJob
from tests.fakes import World
from tests.voxlora_files import manifest, write_voxlora


@pytest.fixture
def home(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    return tmp_path


class _Layer:
    def __init__(self) -> None:
        self.lora_A: dict[str, object] = {}


class _Transformer:
    def __init__(self, *, matches: bool = True, fuse: bool = False) -> None:
        self.layer = _Layer()
        self.matches = matches
        self.weights: list[float] = []
        self.peft_config: dict[str, object] = {}
        self.fused = 0
        self._fuse = fuse
        self.loaded: list[str] = []

    def named_modules(self):
        yield "", self
        yield "layer", self.layer

    def load_lora_adapter(self, path: str, adapter_name: str, **kwargs: object) -> None:
        del kwargs
        self.loaded.append(path)
        if self.matches:
            self.layer.lora_A[adapter_name] = object()
            self.peft_config[adapter_name] = object()

    def set_adapters(self, names, weights=None, **kwargs: object) -> None:
        del names, kwargs
        self.weights = list(weights or [])

    def delete_adapters(self, names) -> None:
        for name in names:
            self.peft_config.pop(name, None)
            self.layer.lora_A.pop(name, None)

    def fuse_lora(self) -> None:
        if not self._fuse:
            raise TypeError("fuse_lora is not on this object")
        self.fused += 1


def _lora(tmp_path: Path, alias: str = "vox_lamp") -> UserLora:
    return UserLora(
        id="lamp-1",
        kind="object",
        folder=tmp_path,
        weights=tmp_path / "a.safetensors",
        trigger="lamptrig",
        default_strength=0.8,
        strength_range=(0.0, 1.5),
        adult_content=False,
        base="z-image-turbo",
        alias=alias,
    )


def test_apply_bound_lora_sets_the_holder_loads_and_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    transformer = _Transformer(fuse=True)
    holder = SimpleNamespace()
    lora = _lora(tmp_path)
    caplog.set_level(logging.INFO, logger="canvas.models.user_lora")
    result = apply_bound_lora(lora, 1.0, holder=holder, transformer=transformer)
    assert isinstance(result, LoraApplyResult)
    assert result.lora_id == "lamp-1"
    assert result.alias == "vox_lamp"
    assert result.strength == 1.0
    assert result.base == "z-image-turbo"
    assert result.modules_patched == 1
    assert holder.bound_lora is lora
    assert holder.lora_strength == 1.0
    assert holder.lora_path == str(lora.weights)
    assert transformer.weights == [1.0]
    assert transformer.fused == 1
    assert transformer.loaded == [str(lora.weights)]
    assert any(
        "LoRA applied" in record.getMessage()
        and "id=lamp-1" in record.getMessage()
        and "alias=vox_lamp" in record.getMessage()
        and "strength=1.0" in record.getMessage()
        and "base=z-image-turbo" in record.getMessage()
        and "modules=1" in record.getMessage()
        for record in caplog.records
    )


def test_apply_bound_lora_fails_when_no_module_is_patched(tmp_path: Path) -> None:
    transformer = _Transformer(matches=False)
    with pytest.raises(LoraRejected) as caught:
        apply_bound_lora(_lora(tmp_path), 0.8, transformer=transformer)
    assert caught.value.code == CODE_INVALID
    assert transformer.weights == []


def test_place_and_bind_uses_the_shared_apply(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}
    lora = _lora(tmp_path)
    transformer = object()
    pipe = SimpleNamespace(transformer=transformer)

    def fake(applied, strength, *, holder=None, transformer=None):
        seen["lora"] = applied
        seen["strength"] = strength
        seen["holder"] = holder
        seen["transformer"] = transformer
        return LoraApplyResult(applied.id, applied.alias, strength, applied.base, 4)

    monkeypatch.setattr("canvas.models.runtime.apply_bound_lora", fake)
    monkeypatch.setattr("canvas.models.runtime._place", lambda *_args, **_kwargs: None)
    _place_and_bind(pipe, Tier.GB24, lora, 1.0)
    assert seen["lora"] is lora
    assert seen["strength"] == 1.0
    assert seen["holder"] is pipe
    assert seen["transformer"] is transformer


def test_shell_runner_binds_through_the_shared_apply(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[tuple[object, ...]] = []
    real = apply_bound_lora
    lora = _lora(tmp_path)

    def wrapped(applied, strength, *, holder=None, transformer=None):
        seen.append((applied, strength, holder, transformer))
        return real(applied, strength, holder=holder, transformer=transformer)

    monkeypatch.setattr("canvas.app.shell_run.apply_bound_lora", wrapped)
    runner = ShellRunner(gate=World().gate(), tier=Tier.GB24, enlarge=object())
    runner.bound_lora = lora
    runner.lora_strength = 1.0
    job = PictureJob(
        "a red lamp", "", Mode.TEXT, "1:1", "native", OutputFormat.PNG, 7, 8, 0.0, "euler", 1,
        lora_path=str(lora.weights), lora_weight=1.0,
    )
    generator = runner._generator(job)
    assert seen, "the GUI path must call the shared LoRA apply function"
    applied, strength, holder, transformer = seen[0]
    assert applied is lora
    assert strength == 1.0
    assert holder is generator
    assert transformer is None
    assert generator.bound_lora is lora
    assert generator.lora_strength == 1.0


def test_generator_passes_a_bound_lora_to_the_painter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}
    lora = _lora(tmp_path)

    def fake_render(*args: object, **kwargs: object) -> RawImage:
        del args
        seen["lora"] = kwargs.get("lora")
        seen["strength"] = kwargs.get("lora_strength")
        return RawImage(2, 2, b"\x11\x22\x33" * 4)

    monkeypatch.setattr("canvas.models.loader.snapshot_ready", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr("canvas.models.loader.render", fake_render)
    generator = ZImageGenerator(load_catalog(), Tier.GB24)
    generator.base = "z-image-turbo"
    apply_bound_lora(lora, 1.0, holder=generator)
    screened = World().gate().release_request("a red lamp", "text")
    generator.generate(screened, 64, 64)
    assert seen["lora"] is lora
    assert seen["strength"] == 1.0


def test_cli_accepts_the_library_alias_id_and_path(home: Path) -> None:
    package = write_voxlora(home / "in", record=manifest(alias="vox_lamp", kind="object"))
    entry = import_lora(package, World().gate(), base_ready=lambda: True)
    assert entry.manifest.alias == "vox_lamp"
    by_alias = prepare_cli_lora("vox_lamp")
    by_id = prepare_cli_lora(entry.id)
    by_path = prepare_cli_lora(entry.folder)
    assert by_alias.id == entry.id
    assert by_alias.alias == "vox_lamp"
    assert by_id.id == entry.id
    assert by_path.id == entry.id
    assert resolve_library("vox_lamp").id == entry.id
    with pytest.raises(LoraRejected) as caught:
        prepare_cli_lora("vox_missing")
    assert caught.value.code == CODE_INVALID


def _stub_cli_generate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, export: object, generator_cls: type) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr("canvas.app.main.check_gpu", lambda _info: GpuCheck(True, "", Tier.GB24))
    monkeypatch.setattr(
        "canvas.app.main.probe_gpu",
        lambda: GpuInfo("RTX 4090", 23.99, 20, 8, 9, "fake"),
    )
    monkeypatch.setattr("canvas.app.main.install_supporting", lambda *args, **kwargs: None)
    monkeypatch.setattr("canvas.app.main.load_catalog_stack", lambda *args, **kwargs: object())

    class FakeScreened:
        def wipe(self) -> None:
            return None

    class FakeGate:
        def __init__(self, stack: object) -> None:
            return None

        def warm(self) -> None:
            return None

        def release_request(self, *args: object, **kwargs: object) -> FakeScreened:
            return FakeScreened()

    class FakePipeline:
        def __init__(self, gate: object, generator: object, *args: object, **kwargs: object) -> None:
            self.generator = generator
            FakePipeline.last = self

        def export(self, *args: object, **kwargs: object) -> None:
            export(self, *args, **kwargs)

    monkeypatch.setattr("canvas.app.main.AgeGate", FakeGate)
    monkeypatch.setattr("canvas.app.main.build_plan", lambda *args, **kwargs: object())
    monkeypatch.setattr("canvas.app.main.ensure_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr("canvas.app.main.ZImageGenerator", generator_cls)
    monkeypatch.setattr("canvas.app.main.Pipeline", FakePipeline)
    monkeypatch.setattr("canvas.app.main.EnlargeClient", lambda *args, **kwargs: object())


def test_cli_generate_applies_the_alias_and_writes_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    from canvas.app.main import main

    package = write_voxlora(tmp_path / "in", record=manifest(alias="vox_lamp", kind="object"))
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    entry = import_lora(package, World().gate(), base_ready=lambda: True)
    applied: list[tuple[object, ...]] = []
    real = apply_bound_lora
    exported: dict[str, object] = {}

    class FakeGenerator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.seed = 0
            self.models = None

    def wrapped(lora, strength, *, holder=None, transformer=None):
        applied.append((lora, strength, holder, transformer))
        return real(lora, strength, holder=holder, transformer=transformer)

    def export(self, *args: object, **kwargs: object) -> None:
        exported["generator"] = self.generator
        exported["metadata"] = kwargs.get("metadata")
        exported["seed"] = kwargs.get("seed")
        Path(args[3] if len(args) > 3 else tmp_path / "out.png").write_bytes(b"ok")

    monkeypatch.setattr("canvas.app.main.apply_bound_lora", wrapped)
    _stub_cli_generate(monkeypatch, tmp_path, export, FakeGenerator)
    out = tmp_path / "out.png"
    code = main([
        "--save-metadata",
        "generate",
        "--prompt", "vox_lamp, a red lamp on a table",
        "--aspect", "1:1",
        "--out", str(out),
        "--lora", "vox_lamp",
        "--lora-scale", "1.0",
        "--seed", "12",
        "--json",
    ])
    capsys.readouterr()
    assert code == 0
    assert applied, "the CLI path must call the shared LoRA apply function"
    lora, strength, holder, transformer = applied[0]
    assert lora.id == entry.id
    assert lora.alias == "vox_lamp"
    assert strength == 1.0
    assert holder is exported["generator"]
    assert transformer is None
    assert holder.bound_lora is lora
    assert holder.lora_strength == 1.0
    meta = exported["metadata"]
    assert isinstance(meta, dict)
    assert meta.get("seed") == 12
    assert meta.get("prompt")
    assert meta.get("lora_strength") == 1.0
    assert meta.get("loras")
    assert os.environ.get("VOXPRINT_CANVAS_SAVE_METADATA") == "1"


def test_photoshoot_cli_job_records_save_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from canvas.app.photoshoot_cli import model_generate

    monkeypatch.setenv("VOXPRINT_CANVAS_SAVE_METADATA", "1")
    lora = _lora(tmp_path)
    seen: dict[str, object] = {}

    def fake_call(job, inputs):
        seen["job"] = job
        return RawImage(2, 2, b"\x11\x22\x33" * 4)

    # model_generate created a runner in the closure. Reach it by wrapping the generate call
    # after replacing ShellRunner.__call__.
    monkeypatch.setattr(ShellRunner, "__call__", lambda self, job, inputs: fake_call(job, inputs))
    generate = model_generate(World().gate(), Tier.GB24, "photo", False, lora=lora, lora_scale=1.0)
    assert generate.lora is lora
    assert generate.lora_scale == 1.0
    generate("a red lamp on a table", 5, None, None)
    job = seen["job"]
    assert job.save_metadata is True
    assert job.lora_path == str(lora.weights)
    assert job.lora_weight == 1.0
