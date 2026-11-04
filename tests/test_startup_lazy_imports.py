"""The window import set stays light: no numpy, Pillow, imagecodecs or model pipelines.

A meta-path finder fakes PySide6, voxprint_common, imagecodecs and psutil, so
the checks run on machines without the runtime dependencies and touch no GPU or
model files. One hermetic subprocess imports ``canvas.app.main`` and every
``canvas.ui`` module and reports what got loaded; an in-process check drives
``_launch`` with fakes to pin the warm worker hello.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

STUBBED_TOPS = ("PySide6", "voxprint_common", "imagecodecs", "psutil")

# Model-side modules that only the CLI commands and the worker may load.
CLI_ONLY = (
    "canvas.app.batch_cli",
    "canvas.app.photoshoot_bench",
    "canvas.app.photoshoot_cli",
    "canvas.app.selftest",
    "canvas.core.exchange",
    "canvas.models.loader",
    "canvas.models.runtime",
    "canvas.pipeline.adult_direction",
    "canvas.pipeline.bf16_schedule",
    "canvas.pipeline.device_guard",
    "canvas.pipeline.enhance",
    "canvas.pipeline.photoshoot",
    "canvas.pipeline.pipeline",
    "canvas.pipeline.schedule",
    "canvas.pipeline.series",
    "canvas.safety.gate",
)

CHECK_SCRIPT = textwrap.dedent(
    r'''
    import importlib
    import importlib.abc
    import importlib.machinery
    import json
    import pkgutil
    import sys

    STUBBED = ("PySide6", "voxprint_common", "imagecodecs", "psutil")
    CLI_ONLY = set(%s)


    class _DummyMeta(type):
        def __getattr__(cls, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _Dummy

        def __call__(cls, *args, **kwargs):
            return super().__call__()


    class _Dummy(metaclass=_DummyMeta):
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return _Dummy

        def __call__(self, *args, **kwargs):
            return _Dummy()


    class _Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.__getattr__ = lambda name: _Dummy if not name.startswith("__") else None
            module.__dir__ = lambda: []


    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] not in STUBBED:
                return None
            return importlib.machinery.ModuleSpec(name, _Loader(), is_package=True)


    sys.meta_path.insert(0, _Finder())

    import canvas.app.main  # noqa: F401
    import canvas.ui

    for module in pkgutil.iter_modules(canvas.ui.__path__):
        importlib.import_module("canvas.ui." + module.name)

    from canvas.core.formats import OutputFormat, extension, parse_export_format, parse_format

    parsed = {
        "png": parse_format(".png") is OutputFormat.PNG,
        "avif": parse_format("avif") is OutputFormat.AVIF,
        "jpeg": parse_export_format("jpg") is OutputFormat.JPEG,
        "jxl_suffix": extension(OutputFormat.JXL) == ".jxl",
    }
    heavy_roots = ("numpy", "PIL", "imagecodecs", "torch", "diffusers", "transformers", "onnxruntime")
    loaded = set(sys.modules)
    print(json.dumps({
        "heavy": sorted(name for name in loaded if name.split(".")[0] in heavy_roots),
        "cli": sorted(name for name in loaded if name in CLI_ONLY),
        "parsed": parsed,
    }))
    '''
) % (json.dumps(CLI_ONLY),)


class _DummyMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Dummy

    def __call__(cls, *args, **kwargs):
        return super().__call__()


class _Dummy(metaclass=_DummyMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _Dummy

    def __call__(self, *args, **kwargs):
        return _Dummy()


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__getattr__ = lambda name: _Dummy if not name.startswith("__") else None
        module.__dir__ = lambda: []


class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] not in STUBBED_TOPS:
            return None
        return importlib.machinery.ModuleSpec(name, _StubLoader(), is_package=True)


@pytest.fixture(autouse=True)
def _fake_runtime_deps():
    """Stand-ins for the missing runtime deps; removed again when the module ends."""
    finder = _StubFinder()
    sys.meta_path.insert(0, finder)
    before = set(sys.modules)
    yield
    sys.meta_path.remove(finder)
    for name in list(sys.modules):
        if name not in before and (
            name.split(".")[0] in STUBBED_TOPS + ("canvas", "numpy", "PIL") or name == "canvas"
        ):
            del sys.modules[name]


def test_window_import_set_stays_light() -> None:
    """Importing the window pulls no image stack and no CLI-only module."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    done = subprocess.run(
        [sys.executable, "-c", CHECK_SCRIPT],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["heavy"] == [], "loaded: " + " ".join(result["heavy"])
    assert result["cli"] == [], "loaded: " + " ".join(result["cli"])
    assert result["parsed"] == {"png": True, "avif": True, "jpeg": True, "jxl_suffix": True}


def test_launch_starts_worker_hello_before_the_event_loop(monkeypatch, tmp_path) -> None:
    """``_launch`` warms the worker (hello) right after the window shows."""
    from canvas.app import main as app_main
    from canvas.models.modes import Tier

    events: list[str] = []

    class FakeClient:
        def __init__(self, host: object, *, tier: str) -> None:
            self.host = host
            self.tier = tier
            self.adapter = object()

        def start(self) -> None:
            events.append("start")

        def when_ready(self, go, fail) -> None:
            events.append("when_ready")

        def shutdown(self) -> None:
            events.append("shutdown")

    class FakeWindow:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def show(self) -> None:
            events.append("show")

    class FakeApp:
        def processEvents(self) -> None:
            pass

        def exec(self) -> int:
            events.append("exec")
            return 0

    class FakeSplash:
        def set_stage(self, message: str) -> None:
            del message

        def finish(self, window) -> None:
            del window

        def close(self) -> None:
            pass

    monkeypatch.setattr(app_main, "install_defaults", lambda catalog, stage: None)
    monkeypatch.setattr(app_main, "load_catalog", lambda: object())
    monkeypatch.setattr(app_main, "EnlargeClient", lambda: SimpleNamespace(ready=lambda: True))
    monkeypatch.setattr(app_main, "WorkerClient", FakeClient)
    monkeypatch.setattr(app_main, "default_host", lambda: object())
    monkeypatch.setattr(app_main, "ShellWindow", FakeWindow)
    monkeypatch.setattr(app_main, "open_worker_layout", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(app_main, "canvas_state_dir", lambda: tmp_path)
    monkeypatch.setattr(app_main, "install_crash_bar", lambda *args, **kwargs: None)

    code = app_main._launch(Tier.GB16, FakeSplash(), FakeApp())

    assert code == 0
    assert events == ["show", "start", "exec", "shutdown"]


def test_series_constants_are_shared_without_the_pipeline() -> None:
    """The window reads the series limits from a module with no model imports."""
    from canvas.pipeline import photoshoot, series_text

    assert photoshoot.MAX_FRAMES == series_text.MAX_FRAMES == 99
    assert photoshoot.DEFAULT_FRAMES == series_text.DEFAULT_FRAMES == 6
    assert photoshoot.TEST_SERIES_CHARACTER == series_text.TEST_SERIES_CHARACTER
