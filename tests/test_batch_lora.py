"""A LoRA batch loads the base and the adapter once, and a full card stops it cleanly."""
from __future__ import annotations

import json
import os
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image

from canvas.app.batch_cli import run_batch_command
from canvas.app.main import main
from canvas.app.shell_run import RunStopped, ShellRunner
from canvas.infra.gpu_lock import GpuLockTimeout
from canvas.models import runtime
from canvas.models.modes import Tier
from canvas.models.plan import PIPELINE_TEXT, CallPlan
from canvas.models.user_lora import prepare_cli_lora
from canvas.pipeline.bf16_schedule import VramRefused, is_out_of_memory
from canvas.pipeline.batch import BatchEntry
from canvas.safety.image import RawImage
from tests.fakes import World
from tests.lora_files import write_safetensors
from tests.test_batch import _entries, _rewrite
from tests.test_forge_lora import _FORGE_A, _FORGE_B, _Transformer, _export, _metadata
from tests.test_photoshoot import IDEA
from tests.test_photoshoot_cli import _ready

CUP = "a red ceramic cup on a wooden table, studio light"


class OutOfMemoryError(RuntimeError):
    """Stands in for ``torch.cuda.OutOfMemoryError``; the check goes by the class name."""


class _Pipe:
    def __init__(self, transformer: _Transformer) -> None:
        self.transformer = transformer

    def __call__(self, **kwargs: object) -> object:
        image = Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"])), (40, 90, 120))
        return types.SimpleNamespace(images=[image])


def _install(monkeypatch, transformer: _Transformer, *, cuda: bool = False) -> list[int]:
    loads = [0]

    def text_load(*_args: object, **_kwargs: object) -> _Pipe:
        loads[0] += 1
        return _Pipe(transformer)

    diffusers = types.SimpleNamespace(
        ZImagePipeline=types.SimpleNamespace(from_pretrained=text_load, from_pipe=lambda pipe: pipe),
    )

    class _Generator:
        def __init__(self, device: object = None) -> None:
            self.device = device

        def manual_seed(self, seed: int) -> _Generator:
            del seed
            return self

    cuda_api = types.SimpleNamespace(is_available=lambda: cuda, empty_cache=lambda: None)
    torch = types.SimpleNamespace(bfloat16="bf16", Generator=_Generator, cuda=cuda_api)
    monkeypatch.setattr(runtime, "_import_torch", lambda: torch)
    monkeypatch.setattr(runtime, "_import_diffusers", lambda: diffusers)
    monkeypatch.setattr(runtime, "_place", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime, "reset_peak_allocated", lambda: None)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: 32 * 1024**3)
    return loads


def _plan() -> CallPlan:
    return CallPlan("diffusers", PIPELINE_TEXT, ("stub",), "bf16", 8, None, "none", "16gb", "text")


def _draw_with(gate, folder: Path, lora, line: str) -> RawImage:
    screened = gate.release_request(line, "text", ())
    return runtime.render(_plan(), folder, screened, 32, 32, Tier.GB16, seed=1, lora=lora, lora_strength=0.7)


def test_a_lora_batch_loads_the_base_and_the_adapter_once(tmp_path: Path, monkeypatch) -> None:
    lora = prepare_cli_lora(_export(tmp_path / "project" / "export"))
    transformer = _Transformer()
    loads = _install(monkeypatch, transformer)
    gate = World().gate()

    def frame_for(entry: BatchEntry):
        del entry
        return lambda line, seed, hero, strength: _draw_with(gate, tmp_path / "weights", lora, line)

    folder = tmp_path / "out"
    folder.mkdir()
    try:
        body, code = run_batch_command(
            entries=_entries({"prompt": IDEA}, {"shot": CUP, "count": 2}, {"shot": CUP, "seed": 9}),
            folder=folder, style="photo", places="same", adult=False,
            age_gate=gate, rewrite=_rewrite, frame_for=frame_for,
        )
    finally:
        runtime.reset_pipeline_cache()
    assert code == 0 and body["made"] == 4
    assert loads[0] == 1
    assert [call[0] for call in transformer.calls].count("load") == 1


def test_a_full_card_stops_the_whole_batch_with_one_error_per_entry(tmp_path: Path) -> None:
    drawn: list[str] = []

    def frame_for(entry: BatchEntry):
        del entry

        def draw(line: str, seed: int, hero: object | None, strength: float | None) -> RawImage:
            del seed, hero, strength
            drawn.append(line)
            raise RunStopped(str(VramRefused()), unavailable=True, code="vram")

        return draw

    folder = tmp_path / "out"
    folder.mkdir()
    body, code = run_batch_command(
        entries=_entries({"shot": CUP}, {"shot": CUP, "seed": 2}, {"shot": CUP, "seed": 3}),
        folder=folder, style="photo", places="same", adult=False,
        age_gate=World().gate(), rewrite=_rewrite, frame_for=frame_for,
    )
    assert code == 1 and body["ok"] is False
    assert len(drawn) == 1
    rows = body["entries"]
    assert [row["status"] for row in rows] == ["error", "error", "error"]
    assert {row["code"] for row in rows} == {"vram"}
    assert all("does not fit on the GPU" in row["message"] for row in rows)
    assert json.loads((folder / "batch.json").read_text(encoding="utf-8"))["ok"] is False


def test_an_out_of_memory_error_while_binding_the_adapter_is_a_clean_refusal(tmp_path: Path, monkeypatch) -> None:
    lora = prepare_cli_lora(_export(tmp_path / "project" / "export"))
    transformer = _Transformer()

    def full(*_args: object, **_kwargs: object) -> None:
        raise OutOfMemoryError("CUDA out of memory")

    transformer.load_lora_adapter = full  # type: ignore[method-assign]
    loads = _install(monkeypatch, transformer)
    gate = World().gate()
    try:
        with runtime.pipeline_scope():
            with pytest.raises(VramRefused):
                _draw_with(gate, tmp_path / "weights", lora, "a red boat")
            assert runtime._slot is None
    finally:
        runtime.reset_pipeline_cache()
    assert loads[0] == 1


def test_a_lora_load_is_refused_before_the_weights_open_when_the_card_is_full(tmp_path: Path, monkeypatch) -> None:
    lora = prepare_cli_lora(_export(tmp_path / "project" / "export"))
    loads = _install(monkeypatch, _Transformer(), cuda=True)
    monkeypatch.setattr(runtime, "_cuda_is_up", lambda _torch: True)
    monkeypatch.setattr(runtime, "note_image_fit", lambda: False)
    gate = World().gate()
    try:
        with pytest.raises(VramRefused):
            _draw_with(gate, tmp_path / "weights", lora, "a red boat")
    finally:
        runtime.reset_pipeline_cache()
    assert loads[0] == 0


def test_the_runner_reports_an_out_of_memory_error_as_vram() -> None:
    runner = ShellRunner(gate=World().gate(), tier=Tier.GB16, enlarge=object())

    def full() -> None:
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    with pytest.raises(RunStopped) as caught:
        runner._guarded(full)
    assert caught.value.code == "vram" and caught.value.unavailable
    assert is_out_of_memory(OutOfMemoryError("x"))
    assert not is_out_of_memory(RuntimeError("something else"))


def test_a_busy_gpu_writes_an_error_entry_and_exits_1(monkeypatch, tmp_path: Path, capsys) -> None:
    _ready(monkeypatch)

    @contextmanager
    def busy(*_args: object, **_kwargs: object):
        raise GpuLockTimeout("busy")
        yield

    monkeypatch.setattr("canvas.app.main.pinned", busy)
    manifest = tmp_path / "shoot.json"
    manifest.write_text(json.dumps({"entries": [{"prompt": IDEA}, {"shot": CUP}]}), encoding="utf-8")
    code = main(["batch", "--manifest", str(manifest), "--json"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "gpu_busy"
    summary = json.loads((tmp_path / "shoot_batch" / "batch.json").read_text(encoding="utf-8"))
    assert summary["ok"] is False
    assert [row["code"] for row in summary["entries"]] == ["gpu_busy", "gpu_busy"]


def test_an_out_of_memory_error_that_escapes_the_batch_exits_1_without_a_traceback(
    monkeypatch, tmp_path: Path, capsys,
) -> None:
    _ready(monkeypatch)

    def full(*_args: object, **_kwargs: object):
        raise OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr("canvas.app.main.model_rewrite", full)
    manifest = tmp_path / "shoot.json"
    manifest.write_text(json.dumps({"entries": [{"prompt": IDEA}]}), encoding="utf-8")
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    code = main(["batch", "--manifest", str(manifest), "--json"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "vram"
    summary = json.loads((tmp_path / "shoot_batch" / "batch.json").read_text(encoding="utf-8"))
    assert summary["entries"][0]["status"] == "error"


def test_a_raw_forge_file_with_the_older_object_kind_opens(tmp_path: Path) -> None:
    meta = _metadata("z-image")
    meta["voxprint.lora.kind"] = "thing"
    path = tmp_path / "kettle.safetensors"
    write_safetensors(path, {_FORGE_A: b"\x01\x02", _FORGE_B: b"\x03\x04"}, meta)
    opened = prepare_cli_lora(path)
    assert opened.kind == "object" and opened.trigger == "kettleword"
