"""Suite manifest v1: migration, coexistence, the lock, and ref removal."""
import hashlib
import importlib
import json
import multiprocessing
import re
import subprocess
from multiprocessing.synchronize import Event as ProcessEvent
from importlib import metadata
from pathlib import Path

import pytest

from canvas.infra.manifest import (
    FUTURE_IMPORT,
    LOCK_TIMEOUT_S,
    ManifestError,
    SCHEMA,
    SHARED_RUNTIME_SCHEMA,
    load_manifest,
    manifest_lock,
    register_component,
    unregister_app,
)
from canvas.infra.paths import shared_dir
from canvas.infra.runtime import LINUX_PYTHON_URL, TORCH_INDEX, TORCHAO_VERSION, WINDOWS_PYTHON_URL, refill_wheels, restore_tensorrt
from canvas.models.provision import default_specs, is_optional, pack_notice, repair
from canvas.models.registry import load_catalog


def test_shared_runtime_schema_stays_local_until_voxprint_common():
    assert SHARED_RUNTIME_SCHEMA == 1
    assert SCHEMA == SHARED_RUNTIME_SCHEMA
    assert FUTURE_IMPORT == "voxprint_common.runtime"
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(FUTURE_IMPORT)


_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
_SHA = "ab" * 32
_MP = multiprocessing.get_context("spawn")


def _lock_sha(files: dict[str, str]) -> str:
    body = "".join(f"{name} {digest}\n" for name, digest in sorted(files.items()))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _hold_lock(shared: str, started: ProcessEvent, release: ProcessEvent) -> None:
    with manifest_lock(Path(shared)):
        started.set()
        if not release.wait(30):
            raise TimeoutError("the lock holder was not released")


def _register_named(shared: str, app: str) -> None:
    register_component(
        Path(shared),
        "torch-2.11.0-cu130",
        kind="wheels",
        version="2.11.0+cu130",
        path="wheels",
        sha256=_SHA,
        app=app,
    )


def test_old_canvas_shape_migrates_to_schema_1(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    shared = shared_dir()
    files = {"weights.bin": _SHA}
    document = {
        "version": 1,
        "note": "keep-me",
        "resources": {
            "example-pack": {
                "kind": "pack",
                "path": "models/Example--Pack",
                "bytes": 3,
                "files": files,
                "apps": ["canvas"],
                "refcount": 1,
                "source": "",
            },
            "suite": {
                "kind": "suite",
                "path": ".",
                "bytes": 0,
                "files": {},
                "apps": ["canvas"],
                "refcount": 1,
                "source": "marker",
            },
        },
    }
    path = shared / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded = load_manifest(shared)
    pack = loaded["example-pack"]
    assert pack.kind == "model"
    assert pack.refs == ("canvas",)
    assert pack.path == "models/Example--Pack"
    assert pack.sha256 == _lock_sha(files)
    assert pack.version == "0"
    assert _ISO.match(pack.installed_at)
    assert loaded["suite"].kind == "tool"
    assert loaded["suite"].version == "marker"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema"] == 1
    assert raw["note"] == "keep-me"
    assert "version" not in raw
    assert "resources" not in raw
    assert set(raw) == {"schema", "components", "note"}


def test_enlarge_manifest_keeps_its_refs_and_extra_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    shared = shared_dir()
    enlarge_sha = "11" * 32
    document = {
        "schema": 1,
        "owner": "enlarge",
        "components": {
            "python-3.14": {
                "kind": "runtime",
                "version": "3.14.8",
                "path": "runtime/py314-torch2.11-cu130",
                "sha256": enlarge_sha,
                "installed_at": "2026-10-01T00:00:00+00:00",
                "refs": ["enlarge"],
                "vendor": "suite",
            }
        },
    }
    path = shared / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded = load_manifest(shared)
    assert loaded["python-3.14"].refs == ("enlarge",)
    assert json.loads(path.read_text(encoding="utf-8"))["owner"] == "enlarge"

    used = register_component(
        shared, "torch-2.11.0-cu130", kind="wheels", version="2.11.0+cu130",
        path="wheels", sha256=_SHA, app="canvas",
    )
    assert used == "torch-2.11.0-cu130"
    register_component(
        shared, "python-3.14", kind="runtime", version="3.14.8",
        path="runtime/py314-torch2.11-cu130", sha256=enlarge_sha, app="canvas",
    )
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["owner"] == "enlarge"
    runtime = stored["components"]["python-3.14"]
    assert runtime["refs"] == ["enlarge", "canvas"]
    assert runtime["vendor"] == "suite"
    assert runtime["path"] == "runtime/py314-torch2.11-cu130"
    assert runtime["sha256"] == enlarge_sha

    other_sha = "33" * 32
    side = register_component(
        shared, "python-3.14", kind="runtime", version="3.14.8",
        path="runtime/py314-torch2.11-cu130", sha256=other_sha, app="canvas",
    )
    assert side != "python-3.14"
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["components"]["python-3.14"]["sha256"] == enlarge_sha
    assert after["components"]["python-3.14"]["refs"] == ["enlarge", "canvas"]
    assert after["components"][side]["sha256"] == other_sha
    assert after["components"][side]["path"] != "runtime/py314-torch2.11-cu130"
    assert after["components"][side]["refs"] == ["canvas"]


def test_concurrent_writers_keep_every_ref(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    shared = shared_dir()
    names = [f"app{index}" for index in range(8)]
    processes = [
        _MP.Process(target=_register_named, args=(str(shared), name))
        for name in names
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    assert set(load_manifest(shared)["torch-2.11.0-cu130"].refs) == set(names)


def test_lock_times_out_while_another_process_holds_it(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr("canvas.infra.manifest.LOCK_TIMEOUT_S", 0.4)
    assert LOCK_TIMEOUT_S == 30
    shared = shared_dir()
    started = _MP.Event()
    release = _MP.Event()
    holder = _MP.Process(target=_hold_lock, args=(str(shared), started, release))
    holder.start()
    assert started.wait(10)
    try:
        with pytest.raises(ManifestError):
            register_component(
                shared, "torch-2.11.0-cu130", kind="wheels", version="2.11.0+cu130",
                path="wheels", sha256=_SHA, app="canvas",
            )
    finally:
        release.set()
        holder.join(10)
    assert holder.exitcode == 0


def test_write_fsyncs_then_renames_the_temp_file(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    shared = shared_dir()
    synced: list[int] = []

    def spy(fd: int) -> None:
        synced.append(fd)

    monkeypatch.setattr("canvas.infra.manifest.os.fsync", spy)
    register_component(
        shared, "tensorrt-10", kind="runtime", version="10.16.1.11",
        path="tensorrt-10", sha256=_SHA, app="canvas",
    )
    assert synced
    assert not (shared / "manifest.json.tmp").exists()
    raw = json.loads((shared / "manifest.json").read_text(encoding="utf-8"))
    assert list(raw)[:2] == ["schema", "components"]
    assert raw["schema"] == 1


def test_refs_remove_a_component_only_when_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    shared = shared_dir()
    target = shared / "models" / "Example--Pack" / "weights.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"abc")
    register_component(
        shared, "example-pack", kind="model", version="1",
        path="models/Example--Pack", sha256=_SHA, app="canvas",
    )
    register_component(
        shared, "example-pack", kind="model", version="1",
        path="models/Example--Pack", sha256=_SHA, app="enlarge",
    )
    assert load_manifest(shared)["example-pack"].refs == ("canvas", "enlarge")
    assert unregister_app(shared, "canvas") == []
    assert target.is_file()
    assert load_manifest(shared)["example-pack"].refs == ("enlarge",)
    assert unregister_app(shared, "enlarge") == ["example-pack"]
    assert not target.exists()
    assert "example-pack" not in load_manifest(shared)


def test_a_shared_directory_stays_while_another_component_refs_it(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    shared = shared_dir()
    runtime = shared / "runtime" / "py314-torch2.11-cu130"
    marker = runtime / "py.marker"
    marker.parent.mkdir(parents=True)
    marker.write_text("runtime", encoding="utf-8")
    register_component(
        shared, "python-3.14", kind="runtime", version="3.14.8",
        path="runtime/py314-torch2.11-cu130", sha256=_SHA, app="enlarge",
    )
    register_component(
        shared, "torch-2.11.0-cu130", kind="wheels", version="2.11.0+cu130",
        path="runtime/py314-torch2.11-cu130", sha256="44" * 32, app="canvas",
    )
    assert unregister_app(shared, "canvas") == ["torch-2.11.0-cu130"]
    assert marker.is_file()
    assert "python-3.14" in load_manifest(shared)


def test_component_path_cannot_escape_the_shared_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    with pytest.raises(ManifestError):
        register_component(
            shared_dir(), "escape", kind="model", version="1",
            path="../outside", sha256=_SHA, app="canvas",
        )


def test_cpu_onnxruntime_dist_is_not_installed():
    names = {dist.metadata["Name"].lower() for dist in metadata.distributions()}
    assert "onnxruntime" not in names
    try:
        requires = metadata.requires("opencv-python-headless") or []
    except metadata.PackageNotFoundError:
        requires = []
    for requirement in requires:
        name = requirement.split(";")[0].split("[")[0].split()[0].lower().replace("_", "-")
        assert name != "onnxruntime"
    root = Path(__file__).resolve().parents[1]
    for relative in ("pyproject.toml", "constraints.txt", "installer/linux/install.sh", "installer/windows/install.ps1"):
        text = (root / relative).read_text(encoding="utf-8")
        assert "onnxruntime-gpu" in text
        assert re.findall(r"onnxruntime(?!-gpu)", text) == []


def test_optional_pack_notice_includes_size_and_estimate():
    catalog = load_catalog()
    creative = catalog.by_id("z-image-creative-bf16")
    lite = catalog.by_id("z-image-turbo-fun-controlnet-union-bf16-lite")
    assert is_optional(creative) and is_optional(lite)
    turbo = catalog.by_id("z-image-turbo-bf16-16gb")
    assert is_optional(turbo) is False
    assert turbo.nbytes > 1_000_000_000
    notice = pack_notice(lite)
    assert notice.startswith("ControlNet lite is ")
    assert "Estimated time" in notice
    assert "Creative Z-Image" in pack_notice(creative)
    specs = default_specs(catalog)
    ids = {spec.id for spec in specs}
    assert "z-image-turbo-bf16-16gb" in ids
    assert "z-image-creative-bf16" not in ids
    roles = [spec.role for spec in specs]
    assert roles[-1] == "generator"
    assert roles.index("prompt-guard") < roles.index("generator")


def test_repair_refetches_a_broken_pack(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    shared = shared_dir()
    register_component(
        shared, "z-image-turbo-bf16-16gb", kind="model", version="1",
        path="models/Tongyi-MAI--Z-Image-Turbo", sha256=_SHA, app="canvas",
    )
    calls: list[str] = []

    def fake_download(spec, root, progress=None):
        calls.append(spec.id)
        if progress is not None:
            progress("repairing")
        return root

    monkeypatch.setattr("canvas.models.provision.download_snapshot", fake_download)
    notes: list[str] = []
    fixed = repair(load_catalog(), notes.append)
    assert calls == ["z-image-turbo-bf16-16gb"]
    assert fixed == ["z-image-turbo-bf16-16gb"]
    assert any("Repairing" in note for note in notes)


def test_repair_leaves_an_enlarge_component_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    register_component(
        shared_dir(), "z-image-turbo-bf16-16gb", kind="model", version="1",
        path="models/Tongyi-MAI--Z-Image-Turbo", sha256=_SHA, app="enlarge",
    )
    calls: list[str] = []
    monkeypatch.setattr(
        "canvas.models.provision.download_snapshot",
        lambda spec, root, progress=None: calls.append(spec.id),
    )
    assert repair(load_catalog(), lambda _note: None) == []
    assert calls == []


def test_installers_use_the_suite_runtime():
    linux = Path("installer/linux/install.sh").read_text(encoding="utf-8")
    windows = Path("installer/windows/install.ps1").read_text(encoding="utf-8")
    assert LINUX_PYTHON_URL in linux
    assert WINDOWS_PYTHON_URL in windows
    assert "cpython-3.14" in LINUX_PYTHON_URL
    assert "python-3.14" in WINDOWS_PYTHON_URL
    assert TORCH_INDEX == "https://download.pytorch.org/whl/cu130"
    assert TORCH_INDEX in linux
    assert TORCH_INDEX in windows
    assert "runtime/py314-torch2.11-cu130" in linux
    assert "runtime/py314-torch2.11-cu130" in windows
    assert "tensorrt-10" in linux
    assert "tensorrt-10" in windows
    assert "onnxruntime-gpu" in linux
    assert "onnxruntime-gpu" in windows
    assert "$SHARED/wheels" in linux
    assert "Voxprint\\shared" in windows
    assert "Installing the default image pack..." in linux
    assert "Installing the default image pack..." in windows


def test_installers_resolve_numpy_from_pypi_and_pin_cuda_torch():
    linux = Path("installer/linux/install.sh").read_text(encoding="utf-8")
    windows = Path("installer/windows/install.ps1").read_text(encoding="utf-8")
    config_path = Path("canvas/infra/uv.cu130.toml")
    assert config_path.is_file()
    config = config_path.read_text(encoding="utf-8")
    assert 'index-url = "https://pypi.org/simple"' in config
    assert "torch-backend" not in config
    assert 'name = "pytorch-cu130"' in config
    assert "explicit = true" in config
    assert 'torch = { index = "pytorch-cu130" }' in config
    assert 'torchvision = { index = "pytorch-cu130" }' in config
    sources = config.split("[sources]", 1)[1]
    assert "torchao" not in sources
    for script in (linux, windows):
        assert "uv.cu130.toml" in script
        assert "--config-file" in script
        assert "torch==2.11.0+cu130" in script
        assert "torchvision==0.26.0+cu130" in script
        assert "torchao==0.18.0" in script
        assert "TensorRT is skipped on Python 3.14." in script
    assert '--index-url "$TORCH_INDEX"' in linux
    assert "--extra-index-url https://pypi.org/simple" in linux
    assert "onnxruntime-gpu==1.31.0" in linux
    assert '"--index-url", $TorchIndex' in windows
    assert '"--index-url", "https://pypi.org/simple"' in windows
    assert "uv==0.13.0" in linux
    assert "uv==0.13.0" in windows


def test_wheel_repair_pins_torch_to_cuda_and_torchao_to_pypi(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    seen: list[tuple[list[str], object]] = []

    def capture(args, **kwargs):
        seen.append((list(args), kwargs.get("stdout")))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("canvas.infra.runtime.subprocess.run", capture)
    refill_wheels(lambda _note: None)
    torch_args, torch_out = seen[0]
    other_args, other_out = seen[1]
    assert torch_args[torch_args.index("--index-url") + 1] == TORCH_INDEX
    assert torch_args[torch_args.index("--extra-index-url") + 1] == "https://pypi.org/simple"
    assert "torch==2.11.0+cu130" in torch_args
    assert "torchvision==0.26.0+cu130" in torch_args
    assert "torchao" not in " ".join(torch_args)
    assert other_args[other_args.index("--index-url") + 1] == "https://pypi.org/simple"
    assert "--extra-index-url" not in other_args
    assert f"torchao=={TORCHAO_VERSION}" in other_args
    assert "onnxruntime-gpu==1.31.0" in other_args
    assert torch_out is other_out
    assert TORCHAO_VERSION == "0.18.0"
    assert "torchao==0.18.0" in Path("pyproject.toml").read_text(encoding="utf-8")


def test_tensorrt_on_python_314_is_skipped(monkeypatch):
    monkeypatch.setattr("canvas.infra.runtime.python_supports_tensorrt", lambda: False)
    called: list[list[str]] = []
    monkeypatch.setattr("canvas.infra.runtime._pip", lambda args: called.append(args))
    notes: list[str] = []
    restore_tensorrt(notes.append)
    assert called == []
    assert notes == ["TensorRT is skipped on Python 3.14."]
