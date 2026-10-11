"""Shared manifest refcount, repair, and the install-time pack offer."""
import hashlib
from pathlib import Path

import pytest

from canvas.infra.manifest import ManifestError, load_manifest, register_resource, unregister_app
from canvas.infra.paths import shared_dir
from canvas.infra.runtime import LINUX_PYTHON_URL, WINDOWS_PYTHON_URL
from canvas.models.provision import default_specs, is_optional, pack_notice, repair
from canvas.models.registry import load_catalog


def test_refcount_deletes_a_pack_only_at_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    shared = shared_dir()
    target = shared / "models" / "Example--Pack" / "weights.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    files = {"weights.bin": digest}
    register_resource(
        shared, "example-pack", kind="pack", path="models/Example--Pack",
        nbytes=3, files=files, app="canvas",
    )
    register_resource(
        shared, "example-pack", kind="pack", path="models/Example--Pack",
        nbytes=3, files=files, app="other-app",
    )
    stored = load_manifest(shared)["example-pack"]
    assert stored.refcount == 2
    assert stored.apps == ("canvas", "other-app")
    assert unregister_app(shared, "canvas") == []
    assert target.is_file()
    assert load_manifest(shared)["example-pack"].refcount == 1
    assert unregister_app(shared, "other-app") == ["example-pack"]
    assert not target.exists()
    assert "example-pack" not in load_manifest(shared)


def test_resource_path_cannot_escape_the_shared_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    with pytest.raises(ManifestError):
        register_resource(
            shared_dir(), "escape", kind="pack", path="../outside",
            nbytes=0, files={}, app="canvas",
        )


def test_optional_pack_notice_includes_size_and_estimate():
    catalog = load_catalog()
    creative = catalog.by_id("z-image-creative-bf16")
    lite = catalog.by_id("z-image-turbo-fun-controlnet-union-bf16-lite")
    edit = catalog.by_id("qwen-image-edit-2511-fp8")
    assert is_optional(creative) and is_optional(lite) and is_optional(edit)
    turbo = catalog.by_id("z-image-turbo-bf16-16gb")
    assert is_optional(turbo) is False
    assert turbo.nbytes > 1_000_000_000
    notice = pack_notice(lite)
    assert notice.startswith("ControlNet lite is ")
    assert "Estimated time" in notice
    assert "Creative Z-Image" in pack_notice(creative)
    assert "Qwen-Image-Edit-2511" in pack_notice(edit)
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
    register_resource(
        shared, "z-image-turbo-bf16-16gb", kind="pack", path="models/Tongyi-MAI--Z-Image-Turbo",
        nbytes=1, files={"missing.bin": "ab" * 32}, app="canvas",
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


def test_installers_use_the_shared_python_addresses():
    linux = Path("installer/linux/install.sh").read_text(encoding="utf-8")
    windows = Path("installer/windows/install.ps1").read_text(encoding="utf-8")
    assert LINUX_PYTHON_URL in linux
    assert WINDOWS_PYTHON_URL in windows
    assert "$SHARED/wheels" in linux
    assert "Voxprint\\shared" in windows
    assert "Installing the default image pack..." in linux
    assert "Installing the default image pack..." in windows
