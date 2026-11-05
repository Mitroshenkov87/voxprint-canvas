"""Suite manifest v1: migration, coexistence, the lock, and ref removal."""
import hashlib
import importlib
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from multiprocessing.synchronize import Event as ProcessEvent
from importlib import metadata
from pathlib import Path

import pytest

from canvas.infra.manifest import (
    FUTURE_IMPORT,
    MANIFEST_NAME,
    LOCK_TIMEOUT_S,
    ManifestError,
    SCHEMA,
    SHARED_RUNTIME_SCHEMA,
    load_manifest,
    manifest_lock,
    register_app,
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


@pytest.mark.parametrize("stored", ["{blob}", "C:/Users/me/AppData/Local/VOXPrint/shared/models/blobs/ab/cd"])
def test_an_mdl_model_with_an_absolute_path_is_read_kept_and_never_deleted(monkeypatch, tmp_path, stored):
    # VOXPrint MDL records each downloaded file by its absolute path (windows-install, PR #42).
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    shared = shared_dir()
    blob = shared / "models" / "blobs" / "ab" / ("cd" * 31)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"weights")
    path = stored.format(blob=blob.as_posix())
    (shared / MANIFEST_NAME).write_text(json.dumps({"schema": SCHEMA, "components": {
        "model-abcdabcdabcd": {"kind": "model", "version": "1", "path": path, "sha256": _SHA,
                                "installed_at": "2026-10-18T04:26:43+00:00", "refs": ["canvas"]},
    }}), encoding="utf-8")
    register_app(shared)
    assert load_manifest(shared)["model-abcdabcdabcd"].path == path
    assert unregister_app(shared, "canvas") == ["model-abcdabcdabcd"]
    assert blob.is_file()


@pytest.mark.parametrize("kind,path", [("tool", "/opt/elsewhere"), ("model", "/x/../etc"), ("runtime", "C:/x")])
def test_other_absolute_or_escaping_paths_are_still_refused(monkeypatch, tmp_path, kind, path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    shared = shared_dir()
    shared.mkdir(parents=True, exist_ok=True)
    (shared / MANIFEST_NAME).write_text(json.dumps({"schema": SCHEMA, "components": {
        "x": {"kind": kind, "version": "1", "path": path, "sha256": _SHA, "installed_at": "", "refs": ["canvas"]},
    }}), encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(shared)


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
    for relative in (
        "pyproject.toml",
        "constraints.txt",
        "constraints-cuda.txt",
        "installer/linux/install.sh",
        "installer/windows/install.ps1",
    ):
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


def test_repair_does_not_download_a_model(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    shared = shared_dir()
    register_component(
        shared, "z-image-turbo-bf16-16gb", kind="model", version="1",
        path="models/Tongyi-MAI--Z-Image-Turbo", sha256=_SHA, app="canvas",
    )
    assert repair(load_catalog(), lambda _note: None) == []
    assert not (shared / "models" / "Tongyi-MAI--Z-Image-Turbo").exists()


def test_shared_roots_are_not_registered_or_deleted(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    shared = shared_dir()
    for root_name in ("models", "runtime", "voices", "cache"):
        with pytest.raises(ManifestError):
            register_component(
                shared, f"root-{root_name}", kind="model", version="1",
                path=root_name, sha256=_SHA, app="canvas",
            )
    runtime = shared / "runtime" / "py314-torch2.11-cu130"
    marker = runtime / "py.marker"
    marker.parent.mkdir(parents=True)
    marker.write_text("runtime", encoding="utf-8")
    register_component(
        shared, "python-3.14", kind="runtime", version="3.14.8",
        path="runtime/py314-torch2.11-cu130", sha256=_SHA, app="canvas",
    )
    assert unregister_app(shared, "canvas") == ["python-3.14"]
    assert marker.is_file()


def test_repair_leaves_an_enlarge_component_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    register_component(
        shared_dir(), "z-image-turbo-bf16-16gb", kind="model", version="1",
        path="models/Tongyi-MAI--Z-Image-Turbo", sha256=_SHA, app="enlarge",
    )
    assert repair(load_catalog(), lambda _note: None) == []
    assert not (shared_dir() / "models" / "Tongyi-MAI--Z-Image-Turbo").exists()


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
    assert "$SHARED/wheels" not in linux
    assert 'WHEELS="$APP/setup-work/wheels"' in linux
    assert 'Join-Path $SuiteHome "shared"' in windows
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
    assert "[sources]" not in config
    assert "override-dependencies" not in config
    assert "torchao" not in config
    for script in (linux, windows):
        assert "--index-strategy" not in script
        assert "unsafe-best-match" not in script
        assert "--index-url" in script
        assert "--find-links" in script
        assert "--config-file" not in script
        assert "torch==2.11.0+cu130" in script
        assert "torchvision==0.26.0+cu130" in script
        assert "torchao==0.18.0" in script
        assert "[gen]" in script
        assert "Repairing the suite runtime..." in script
        assert "PyTorch is already installed." in script
        assert "TensorRT is skipped on Python 3.14." in script
    assert "PyTorch wheels are already downloaded." in linux
    assert 'WHEELS="$APP/setup-work/wheels"' in linux
    assert '--find-links "$WHEELS"' in linux
    assert "$SHARED/wheels" not in linux
    assert "UV_NO_CONFIG" in linux
    assert "--no-config" in linux
    assert "--isolated" in linux
    assert "constraints-cuda.txt" in linux
    assert "VIRTUAL_ENV" in linux
    assert "CONDA_" in linux
    assert "HTTP_PROXY" in linux
    assert 'cd "$APP"' in linux
    assert 'APP_PYTHON="$APP/python"' in linux
    assert "refuse_symlink" in linux
    assert "SUDO_USER" in linux
    assert "PYTHONNOUSERSITE" in linux
    assert "handoff_tree" in linux
    assert 'chown -R "$SUDO_USER" "$1"' in linux
    assert '"$1" -s -c' in linux
    assert 'as_user "$VENV/bin/python" -m canvas.infra.layout migrate --install-dir "$APP"' in linux
    assert "VOXPRINT_HOME must be an absolute path." in linux
    assert 'getent passwd "$SUDO_USER"' in linux
    assert 'HOME="$USER_HOME"' in linux
    assert 'APP_DIR="$USER_HOME/.local/share/applications"' in linux
    assert "Moving the existing suite runtime aside:" in linux
    assert 'as_user mv -- "$RUNTIME"' in linux
    assert 'chown -R "$SUDO_USER" "$RUNTIME"' in linux
    assert "The suite runtime is already installed" not in linux
    assert 'as_user ln -sfn "$VENV/bin/voxprint-canvas" "$BIN_DIR/voxprint-canvas"' in linux
    assert 'as_user cp "$desktop_file" "$APP_DIR/voxprint-canvas.desktop"' in linux
    move = linux.split("move_occupied_runtime_aside()", 1)[1].split(
        'if [ "${VOXPRINT_INSTALL_LIB:-}" = "1" ]', 1
    )[0]
    assert "rm " not in move
    assert "--without-pip" in linux
    assert 'gpu_ready "$RUNTIME' not in linux
    assert 'install_gpu "$RUNTIME' not in linux
    assert 'Join-Path $Work "wheels"' in windows
    assert 'Join-Path $env:TEMP ("voxprint-wheels-' not in windows
    assert 'Join-Path $Work ("voxprint-gpu-probe-' in windows
    assert 'Join-Path $env:TEMP ("voxprint-gpu-probe-' not in windows
    assert "UV_PYTHON_INSTALL_DIR" in windows
    assert "UV_PYTHON_BIN_DIR" in windows
    assert "UV_CACHE_DIR" in windows
    assert "PYTHONNOUSERSITE" in windows
    assert "--without-pip" in windows
    assert "Assert-AdminPath" in windows
    assert "function Resolve-ContainedPath" in windows
    assert "GetFinalPathNameByHandleW" in windows
    assert "this link points outside the Python folder" in windows
    assert 'Resolve-ContainedPath $script:Base $PythonDir' in windows
    assert 'Resolve-ContainedPath $executable $PythonDir' in windows
    assert '$env:VOXPRINT_INSTALL_LIB -ne "1"' in windows
    assert "Test-Elevated" in windows
    assert "Test-PathUnder" in windows
    assert "Get-MachineProgramFiles" in windows
    assert "ProgramFilesDir" in windows
    assert "USERPROFILE" in windows
    assert "an elevated install must live under Program Files" in windows
    assert 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion' in windows
    assert "python install 3.14 --reinstall" in windows
    assert "Get-Command py" not in windows
    assert "voxprint-gpu-probe-" in windows
    assert "-c $code" not in windows
    assert "pyvenv.cfg" in windows
    assert "--target" in windows
    assert "Test-SuiteGpuFiles" in windows
    assert "Remove-Item -LiteralPath $Fetch" in windows
    assert "& $RuntimePy" not in windows
    assert "Test-SuiteGpuPackages $RuntimePy" not in windows
    assert "Install-SuiteGpuPackages $RuntimePy" not in windows
    assert "$RuntimePy = $RootPy" not in windows
    assert "venv --upgrade $Runtime" not in windows
    assert "venv --upgrade $Venv" in windows
    assert "Repair-SharedRuntimePackages" in windows
    assert "Install-SuiteGpuPackages $Base $RuntimeSite" not in windows
    assert "if (-not (Test-TorchCudaFiles $RuntimeSite))" in windows
    assert "if (-not $visionOk)" in windows
    assert "if (-not $gpuOk)" in windows
    assert "Repairing torch in the suite runtime..." in windows
    assert "Repairing torchvision in the suite runtime..." in windows
    assert "Repairing ONNX Runtime in the suite runtime..." in windows
    assert '--index-url "$TORCH_INDEX"' in linux
    assert "--extra-index-url https://pypi.org/simple" in linux
    assert "onnxruntime-gpu==1.31.0" in linux
    assert '"--index-url", $TorchIndex' in windows
    assert '"--index-url", "https://pypi.org/simple"' in windows
    assert "uv==0.13.0" in linux
    assert "uv==0.13.0" in windows
    assert "Start-Transcript" in windows
    assert "canvas-install.log" in windows
    assert "WindowsApps" in windows
    assert "pip uninstall" in linux
    assert "pip uninstall" in windows
    assert "Assert-NoReparse" in windows
    assert "ReparsePoint" in windows
    assert "UV_NO_CONFIG" in windows
    assert "--no-config" in windows
    assert "--isolated" in windows
    assert "constraints-cuda.txt" in windows
    assert "VIRTUAL_ENV" in windows
    assert "CONDA_" in windows
    assert "HTTP_PROXY" in windows
    assert "Set-Location -LiteralPath $App" in windows
    ensure = windows.split("function Ensure-SharedRuntime", 1)[1].split("function Invoke-Machine", 1)[0]
    assert ensure.index("Assert-NoReparse $Runtime") < ensure.index("venv --without-pip")
    machine = windows.split("function Invoke-Machine", 1)[1].split("function Invoke-User", 1)[0]
    assert machine.index("Resolve-ContainedPath $script:Base $PythonDir") < machine.index("Assert-AdminPath $script:Base")
    assert "Move-Item -LiteralPath $Runtime" in windows
    assert ".bak-" in windows
    assert "The suite runtime is already installed" not in windows
    assert "Remove-Item -LiteralPath $Runtime" not in windows
    assert 'ValidateSet("All", "Machine", "User", "Cleanup")' in windows
    assert "torch==2.11.0+cu130" in Path("constraints-cuda.txt").read_text(encoding="utf-8")
    assert "torchvision==0.26.0+cu130" in Path("constraints-cuda.txt").read_text(encoding="utf-8")
    assert "onnxruntime-gpu==1.31.0" in Path("constraints-cuda.txt").read_text(encoding="utf-8")
    iss = Path("installer/windows/voxprint-canvas.iss").read_text(encoding="utf-8")
    assert "CurStepChanged" in iss
    assert "ssPostInstall" in iss
    assert "SuppressibleMsgBox" in iss
    assert "ExitProcess" in iss
    assert "[Run]" not in iss
    assert "[UninstallRun]" not in iss
    assert "Started := Exec(" in iss
    post, uninstall = iss.split("procedure CurUninstallStepChanged", 1)
    assert "ExecAsOriginalUser" in post
    assert "-Scope Machine" in post
    assert "-Scope User" in post
    assert "-Scope Cleanup" in post
    assert "function RunSetup" in iss
    step = post.split("procedure CurStepChanged", 1)[1]
    assert step.index("-Scope Cleanup") < step.index("ExitProcess")
    assert "if Code = 0 then" in step
    assert "ExpandConstant('{app}')" in post
    assert "ExpandConstant('{app}\\setup-work\\canvas-install.log')" in post
    assert "ExpandConstant('{localappdata}\\VOXPrint AI Suite\\logs\\canvas-install.log')" in post
    assert "ExecAsOriginalUser" not in uninstall
    assert "-I -m canvas.app.main --unregister-models" in uninstall
    assert "icacls.exe" not in iss
    assert "CurUninstallStepChanged" in iss
    assert "usUninstall" in iss


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


def _source_install_functions(env, script, cwd):
    return subprocess.run(
        ["sh", "-c", script],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="The Linux installer script runs on Linux.")
def test_sudo_install_uses_the_passwd_home_not_root(tmp_path):
    home = tmp_path / "alice"
    home.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "id").write_text("#!/bin/sh\necho 0\n", encoding="utf-8")
    (bindir / "getent").write_text(
        "#!/bin/sh\nprintf '%s\\n' 'alice:x:1000:1000:Alice:" + str(home) + ":/bin/sh'\n",
        encoding="utf-8",
    )
    for path in bindir.iterdir():
        path.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    env["VOXPRINT_INSTALL_LIB"] = "1"
    env["HOME"] = "/root"
    env["SUDO_USER"] = "alice"
    env.pop("VOXPRINT_HOME", None)
    env["XDG_DATA_HOME"] = "/root/.local/share"
    root = Path(__file__).resolve().parents[1]
    script = """
. ./installer/linux/install.sh
user=$(installing_user_home)
USER_HOME=$user
PROFILE_XDG=$(profile_xdg)
data=$(data_root)
printf '%s\\n' "$user" "$data" "$PROFILE_XDG"
"""
    result = _source_install_functions(env, script, root)
    user, data, xdg = result.stdout.splitlines()
    assert user == str(home)
    assert data == str(home / ".local" / "share" / "voxprint-ai-suite")
    assert xdg == str(home / ".local" / "share")

    inside = home / ".config" / "share"
    inside.mkdir(parents=True)
    env["XDG_DATA_HOME"] = str(inside)
    result = _source_install_functions(env, script, root)
    user, data, xdg = result.stdout.splitlines()
    assert user == str(home)
    assert data == str(inside / "voxprint-ai-suite")
    assert xdg == str(inside)

    env["VOXPRINT_HOME"] = "relative/voxprint"
    script_relative = """
. ./installer/linux/install.sh
USER_HOME=$HOME
PROFILE_XDG=$HOME/.local/share
data_root
"""
    rejected = subprocess.run(
        ["sh", "-c", script_relative],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "absolute path" in rejected.stderr
    env.pop("VOXPRINT_HOME", None)
    finished = subprocess.run(
        ["sh", "-c", ". ./installer/linux/install.sh\nUSER_HOME=$HOME\nhandoff_parents relative\nprintf '%s\\n' done\n"],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert finished.stdout.strip() == "done"


@pytest.mark.skipif(sys.platform == "win32", reason="The Linux installer script runs on Linux.")
def test_linux_setup_moves_a_standalone_runtime_aside(tmp_path):
    home = tmp_path / "alice"
    runtime = home / ".local" / "share" / "voxprint" / "shared" / "runtime" / "py314-torch2.11-cu130"
    binary = runtime / "bin" / "python3"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    marker = runtime / "keep-me.txt"
    marker.write_text("user data", encoding="utf-8")
    env = os.environ.copy()
    env["VOXPRINT_INSTALL_LIB"] = "1"
    env["HOME"] = str(home)
    env.pop("SUDO_USER", None)
    root = Path(__file__).resolve().parents[1]
    script = """
. ./installer/linux/install.sh
USER_HOME=$1
SHARED=$2
RUNTIME=$3
move_occupied_runtime_aside
"""
    shared = runtime.parent.parent
    result = subprocess.run(
        ["sh", "-c", script, "sh", str(home), str(shared), str(runtime)],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Moving the existing suite runtime aside:" in result.stdout
    assert not runtime.exists()
    backups = list(runtime.parent.glob(runtime.name + ".bak-*"))
    assert len(backups) == 1
    assert (backups[0] / "keep-me.txt").read_text(encoding="utf-8") == "user data"
    assert (backups[0] / "bin" / "python3").is_file()


def _windows_junction(link: Path, target: Path) -> None:
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows reparse points are checked by the Windows installer.")
@pytest.mark.parametrize("host", ["pwsh.exe", "powershell.exe"])
def test_windows_follows_a_python_junction_only_inside_the_install(tmp_path, host):
    # PowerShell 7 first; Windows PowerShell 5.1 stays covered as the fallback host.
    if shutil.which(host) is None:
        pytest.skip(f"{host} is not installed")
    py_root = tmp_path / "python"
    real = py_root / "real"
    real.mkdir(parents=True)
    real_exe = real / "python.exe"
    real_exe.write_bytes(b"")
    alias = py_root / "cpython-3.14-windows-x86_64-none"
    _windows_junction(alias, real)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "python.exe").write_bytes(b"")
    escape = py_root / "escape"
    _windows_junction(escape, outside)
    linked_root_target = tmp_path / "linked-real"
    (linked_root_target / "real").mkdir(parents=True)
    (linked_root_target / "real" / "python.exe").write_bytes(b"")
    linked_root = tmp_path / "linked-python"
    _windows_junction(linked_root, linked_root_target)
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["VOXPRINT_INSTALL_SCRIPT"] = str(root / "installer" / "windows" / "install.ps1")
    env["PYROOT"] = str(py_root)
    env["INSIDE"] = str(alias / "python.exe")
    env["ESCAPE"] = str(escape / "python.exe")
    env["LINKROOT"] = str(linked_root)
    env["ROOTLINK"] = str(linked_root / "real" / "python.exe")
    script = r"""
$ErrorActionPreference = "Stop"
$env:VOXPRINT_INSTALL_LIB = "1"
. $env:VOXPRINT_INSTALL_SCRIPT
$resolved = Resolve-ContainedPath $env:INSIDE $env:PYROOT
Write-Output ("RESOLVED=" + $resolved)
try {
    Resolve-ContainedPath $env:ESCAPE $env:PYROOT | Out-Null
    Write-Output "ESCAPE_ACCEPTED"
} catch {
    if ($_.Exception.Message -notlike "*outside the Python folder*") { throw }
    Write-Output "ESCAPE_REJECTED"
}
try {
    Resolve-ContainedPath $env:ROOTLINK $env:LINKROOT | Out-Null
    Write-Output "ROOT_ACCEPTED"
} catch {
    if ($_.Exception.Message -notlike "*contains a link*") { throw }
    Write-Output "ROOT_REJECTED"
}
"""
    result = subprocess.run(
        [host, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    resolved = next(line for line in lines if line.startswith("RESOLVED=")).split("=", 1)[1]
    assert resolved.lower() == str(real_exe).lower()
    assert "cpython-3.14-windows-x86_64-none" not in resolved.lower()
    assert "ESCAPE_REJECTED" in lines
    assert "ROOT_REJECTED" in lines
