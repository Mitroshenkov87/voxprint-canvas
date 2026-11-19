"""The card is pinned before CUDA starts, and cleanup never ends the process."""
import sys
import types

import pytest

from canvas.app import main as app_main
from canvas.infra import gpu_lock, settings
from canvas.pipeline import gpu_release


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setattr(settings, "_cached", settings.Settings())


def test_flag_pins_the_card_before_cuda_starts(monkeypatch):
    seen: list[str | None] = []
    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: seen.append(app_main.os.environ.get("CUDA_VISIBLE_DEVICES")) or False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert app_main.main(["--gpu", "1", "--version"]) == 0
    assert app_main.os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    fake.cuda.is_available()
    assert seen == ["1"]


def test_flag_after_the_command_and_over_the_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    args = app_main._parser().parse_args(["batch", "--manifest", "m.json", "--gpu", "1"])
    app_main._apply_gpu_setting(args)
    assert app_main.os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_toml_gpu_is_used_without_a_flag(monkeypatch):
    monkeypatch.setattr(settings, "_cached", settings.Settings(gpu=1))
    app_main._apply_gpu_setting(app_main._parser().parse_args([]))
    assert app_main.os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_lock_key_follows_the_pinned_card(monkeypatch):
    monkeypatch.setattr(gpu_lock, "_cuda_uuid", lambda: None)
    app_main._apply_gpu_setting(app_main._parser().parse_args(["--gpu", "1"]))
    assert gpu_lock.device_key() == "1"


def test_cache_release_survives_an_error(monkeypatch):
    def boom() -> None:
        raise RuntimeError("CUDA out of memory")

    cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=boom)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    assert gpu_release.release_cuda_cache() is False


def test_memory_error_becomes_a_vram_stop():
    from canvas.app.shell_run import RunStopped, ShellRunner

    def draw() -> None:
        raise MemoryError()

    runner = ShellRunner.__new__(ShellRunner)
    with pytest.raises(RunStopped) as stopped:
        runner._guarded(draw)
    assert stopped.value.code == "vram"
