"""The writer and the text encoder stay on the GPU, or the series stops."""
from __future__ import annotations

import time
import types
from pathlib import Path

import pytest
from PIL import Image

from canvas.app.i18n import text
from canvas.infra.stage_timer import HEARTBEAT_S, StageTimes
from canvas.models import runtime
from canvas.models.modes import Tier
from canvas.pipeline.device_guard import DeviceRefused, place_on_cuda, require_cuda
from canvas.pipeline.series import run_series
from tests.fakes import World
from tests.test_photoshoot import IDEA


class _Device:
    def __init__(self, kind: str) -> None:
        self.type = kind


class _Param:
    def __init__(self, kind: str) -> None:
        self.device = _Device(kind)


class _Weight:
    def __init__(self) -> None:
        self.device = "cpu"
        self.pointer = 11

    def data_ptr(self) -> int:
        return self.pointer

    def to(self, device: str) -> _Weight:
        moved = _Weight()
        moved.device = device
        moved.pointer = self.pointer
        return moved


class _Slot:
    def __init__(self, weight: _Weight) -> None:
        self.data = weight


class _Resident:
    def __init__(self) -> None:
        self.weight = _Slot(_Weight())
        self.called = False

    def parameters(self):
        yield self.weight

    def to(self, device: str) -> _Resident:
        self.called = True
        return self


def test_require_cuda_stops_when_a_parameter_stays_on_cpu() -> None:
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))

    class _Model:
        def __init__(self, kinds: tuple[str, ...]) -> None:
            self._kinds = kinds

        def parameters(self):
            for kind in self._kinds:
                yield _Param(kind)

    with pytest.raises(DeviceRefused) as caught:
        require_cuda(_Model(("cpu",)), "writer.device", torch)
    assert caught.value.code == "device.cpu"
    assert str(caught.value) == text("error.device")
    require_cuda(_Model(()), "writer.device", torch)
    require_cuda(_Model(("cuda",)), "writer.device", torch)
    idle = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    require_cuda(_Model(("cpu",)), "writer.device", idle)


def test_place_on_cuda_moves_parameter_data_without_copying_the_module() -> None:
    module = _Resident()
    placed = place_on_cuda(module)
    assert placed is module
    assert module.weight.data.device == "cuda"
    assert module.called is False


def test_place_on_cuda_uses_module_to_when_there_are_no_parameters() -> None:
    seen: list[str] = []

    class _Empty:
        def parameters(self):
            return iter(())

        def to(self, device: str) -> _Empty:
            seen.append(device)
            return self

    place_on_cuda(_Empty())
    assert seen == ["cuda"]


def test_a_long_stage_logs_start_end_and_a_heartbeat(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr("canvas.infra.stage_timer.HEARTBEAT_S", 0.05)
    monkeypatch.setattr("canvas.infra.stage_timer._device_and_vram", lambda: ("cuda", 128))
    clock = StageTimes()
    with clock.measure("sample"):
        time.sleep(0.18)
    err = capsys.readouterr().err
    assert "stage sample beat device=cuda vram_mib=128" in err
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "stage sample start device=cuda vram_mib=128" in log
    assert "stage sample end device=cuda vram_mib=128" in log
    assert "stage sample beat device=cuda vram_mib=128" in log
    assert HEARTBEAT_S == 30.0


def test_a_short_stage_logs_start_and_end_without_a_heartbeat(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr("canvas.infra.stage_timer._device_and_vram", lambda: ("cpu", 4))
    clock = StageTimes()
    with clock.measure("save"):
        pass
    err = capsys.readouterr().err
    assert "beat" not in err
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "stage save start device=cpu vram_mib=4" in log
    assert "stage save end device=cpu vram_mib=4" in log
    assert "beat" not in log


def test_the_encoder_reference_stays_on_the_gpu(monkeypatch, tmp_path: Path) -> None:
    seen: list[str] = []

    def probe(_folder: Path, _torch: object, storage: str) -> list[list[float]]:
        seen.append(storage)
        return [[1.0, 0.0]]

    monkeypatch.setattr(runtime, "_probe_vectors", probe)
    monkeypatch.setattr(runtime, "_fp8_ready", lambda _torch: True)
    runtime.clear_series_embeds()
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    score = runtime._encoder_cosine(tmp_path / "weights", Tier.GB16, torch)
    assert seen == ["bf16", "fp8"]
    assert score == 1.0
    runtime.clear_series_embeds()


def test_an_encoder_probe_does_not_swallow_a_device_stop(monkeypatch, tmp_path: Path) -> None:
    def probe(_folder: Path, _torch: object, _storage: str) -> list[list[float]]:
        raise DeviceRefused()

    monkeypatch.setattr(runtime, "_probe_vectors", probe)
    monkeypatch.setattr(runtime, "_fp8_ready", lambda _torch: True)
    runtime.clear_series_embeds()
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    with pytest.raises(DeviceRefused):
        runtime._encoder_cosine(tmp_path / "weights", Tier.GB16, torch)


def test_cpu_encoder_storage_stops_on_cuda() -> None:
    encoder = types.SimpleNamespace(history=[], device=_Device("cpu"))

    def to(device: str) -> object:
        encoder.history.append(device)
        encoder.device = _Device(device)
        return encoder

    encoder.to = to
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    with pytest.raises(DeviceRefused):
        runtime._place_encoder(encoder, "cpu", torch, Tier.GB16)
    assert encoder.history == []


def test_a_latent_is_decoded_on_cuda_in_bf16(monkeypatch) -> None:
    class _Latent:
        def __truediv__(self, other: object) -> _Latent:
            calls.append(("div", other))
            return self

        def __add__(self, other: object) -> _Latent:
            calls.append(("add", other))
            return self

    class _Config:
        scaling_factor = 0.3611
        shift_factor = 0.1159
        force_upcast = True

    class _Vae:
        def __init__(self) -> None:
            self.config = _Config()
            self.calls: list[object] = []

        def to(self, *args: object) -> _Vae:
            self.calls.append(args)
            return self

        def enable_tiling(self) -> None:
            self.calls.append(("tile",))

        def decode(self, latents: object) -> types.SimpleNamespace:
            self.calls.append(("decode", latents))
            return types.SimpleNamespace(sample="decoded")

    calls: list[object] = []
    vae = _Vae()
    image = Image.new("RGB", (4, 4), (1, 2, 3))

    def postprocess(decoded: object, output_type: str) -> list[Image.Image]:
        assert decoded == "decoded"
        assert output_type == "pil"
        return [image]

    pipe = types.SimpleNamespace(vae=vae, image_processor=types.SimpleNamespace(postprocess=postprocess))
    monkeypatch.setattr(runtime, "allocated_bytes", lambda: 13 * 1024**3)
    monkeypatch.setattr(runtime, "_free_vram_bytes", lambda: 1024**3)
    torch = types.SimpleNamespace(bfloat16="bf16", cuda=types.SimpleNamespace(is_available=lambda: True))
    runtime._disable_vae_upcast(pipe)
    assert vae.config.force_upcast is False
    assert vae.calls == []
    raw = runtime._image_from_result(pipe, types.SimpleNamespace(images=[_Latent()]), torch)
    assert ("cuda", "bf16") in vae.calls
    assert ("tile",) in vae.calls
    assert ("div", 0.3611) in calls
    assert ("add", 0.1159) in calls
    assert raw.width == 4 and raw.height == 4


def test_a_pil_result_does_not_move_the_vae() -> None:
    class _Vae:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def to(self, *args: object) -> _Vae:
            self.calls.append(args)
            return self

    vae = _Vae()
    pipe = types.SimpleNamespace(vae=vae, image_processor=None)
    image = Image.new("RGB", (2, 2), (4, 5, 6))
    torch = types.SimpleNamespace(bfloat16="bf16", cuda=types.SimpleNamespace(is_available=lambda: True))
    raw = runtime._image_from_result(pipe, types.SimpleNamespace(images=[image]), torch)
    assert vae.calls == []
    assert raw.to_bytes() == image.tobytes()


def test_a_writer_device_stop_ends_the_series(tmp_path: Path) -> None:
    def rewrite(_prompt: str) -> str:
        raise DeviceRefused()

    result = run_series(
        idea=IDEA,
        count=2,
        style="photo",
        place_mode="same",
        seed=3,
        strength=None,
        include_camera=lambda: True,
        rewrite=rewrite,
        age_gate=World().gate(),
        draw_frame=lambda *_args: None,
        cancelled=lambda: False,
        folder=tmp_path / "series",
    )
    assert result.stopped == "error"
    assert result.code == "device.cpu"
    assert result.reason == text("error.device")
    assert result.kept == ()
