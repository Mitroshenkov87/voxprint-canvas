"""Legacy folders are adopted by hash. The loader still does not read them."""
from __future__ import annotations

import hashlib
import inspect
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from canvas.infra.paths import models_dir
from canvas.models.adopt import (
    AdoptResult,
    _local_adopt,
    _mdl_adopt,
    adopt_cli,
    adopt_required,
    mdl_adopt_command,
)
from canvas.models.registry import load_catalog
from canvas.models.snapshots import weight_dir
from canvas.models.views import blob_path


def test_adopt_cli_is_one_line_until_rc29() -> None:
    assert adopt_cli.__doc__ == "TODO: remove in rc.29, use voxprint-mdl adopt."
    body = [line.strip() for line in inspect.getsource(adopt_cli).splitlines() if line.strip()]
    assert body[-1] == "return _local_adopt(spec, legacy, root)"


def test_adopt_required_uses_the_cli_adapter(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("VOXPRINT_HOME", str(home))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    root = tmp_path / "models"
    root.mkdir()
    catalog = load_catalog()
    writer = catalog.by_id("qwen3-4b-writer")
    guard = catalog.by_role("prompt-guard")[0]
    for spec in (writer, guard):
        legacy = weight_dir(root, spec.repo)
        legacy.mkdir(parents=True)
        (legacy / "config.json").write_bytes(b"legacy")
    seen: list[Path] = []

    def fake_cli(spec, legacy: Path, models: Path) -> AdoptResult:
        del models
        seen.append(legacy)
        return AdoptResult(spec.id, files=1, ok=1, fetched=0)

    monkeypatch.setattr("canvas.models.adopt.adopt_cli", fake_cli)
    monkeypatch.setattr("canvas.models.adopt.view_complete", lambda spec, models: spec.id == guard.id)
    adopt_required(root, catalog)
    assert seen == [weight_dir(root, writer.repo)]
    log = (home / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "id=qwen3-4b-writer files=1 ok=1 fetched=0" in log
    assert "ms=" in log
    assert (weight_dir(root, writer.repo) / "config.json").read_bytes() == b"legacy"
    assert (weight_dir(root, guard.repo) / "config.json").read_bytes() == b"legacy"


def test_mdl_adopt_invokes_the_cli_and_leaves_the_folder(tmp_path: Path, monkeypatch) -> None:
    legacy = tmp_path / "Tongyi-MAI--Z-Image-Turbo"
    legacy.mkdir()
    (legacy / "config.json").write_bytes(b"keep")
    spec = load_catalog().by_id("qwen3-4b-writer")
    seen: dict[str, object] = {}

    def fake_run(argv: list[str], check: bool = False, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del check, kwargs
        seen["argv"] = list(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("canvas.models.adopt.subprocess.run", fake_run)
    _mdl_adopt(spec, legacy, tmp_path)
    assert seen["argv"] == mdl_adopt_command(legacy)
    assert seen["argv"] == [
        sys.executable, "-m", "voxprint_common.mdl", "adopt", "--app", "canvas", "--from", os.fspath(legacy),
    ]
    assert (legacy / "config.json").read_bytes() == b"keep"


def test_matching_hashes_are_adopted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    one, two = b"weight-one", b"weight-two"
    files = {
        "a.safetensors": hashlib.sha256(one).hexdigest(),
        "nested/b.safetensors": hashlib.sha256(two).hexdigest(),
    }
    spec = _spec(files)
    root = models_dir()
    legacy = weight_dir(root, spec.repo)
    legacy.mkdir(parents=True)
    (legacy / "a.safetensors").write_bytes(one)
    (legacy / "nested").mkdir()
    (legacy / "nested" / "b.safetensors").write_bytes(two)
    result = _local_adopt(spec, legacy, root)
    assert (result.files, result.ok, result.fetched) == (2, 2, 0)
    assert os.path.samefile(blob_path(root, files["a.safetensors"]), legacy / "a.safetensors")
    assert os.path.samefile(
        blob_path(root, files["nested/b.safetensors"]), legacy / "nested" / "b.safetensors",
    )
    folder = next((root / "views" / "canvas").iterdir())
    assert folder.parent == root / "views" / "canvas"
    assert (folder / "a.safetensors").read_bytes() == one
    assert (folder / "nested" / "b.safetensors").read_bytes() == two
    assert (legacy / "a.safetensors").read_bytes() == one
    assert legacy.is_dir()


def test_a_mismatch_is_fetched_and_the_legacy_folder_stays(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    good, bad, fetched = b"good-weight", b"bad-weight", b"fetched-weight"
    files = {
        "good.safetensors": hashlib.sha256(good).hexdigest(),
        "bad.safetensors": hashlib.sha256(fetched).hexdigest(),
    }
    spec = _spec(files)
    root = models_dir()
    legacy = weight_dir(root, spec.repo)
    legacy.mkdir(parents=True)
    (legacy / "good.safetensors").write_bytes(good)
    (legacy / "bad.safetensors").write_bytes(bad)
    fetched_names: list[str] = []

    def fake_fetch(model, models: Path, names: tuple[str, ...]) -> int:
        del model
        fetched_names.extend(names)
        blob = blob_path(models, files["bad.safetensors"])
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(fetched)
        return len(names)

    monkeypatch.setattr("canvas.models.adopt._fetch", fake_fetch)
    result = _local_adopt(spec, legacy, root)
    assert result.ok == 1
    assert result.fetched == 1
    assert "bad.safetensors" in result.fetched_paths
    assert fetched_names == ["bad.safetensors"]
    assert os.path.samefile(blob_path(root, files["good.safetensors"]), legacy / "good.safetensors")
    assert (legacy / "bad.safetensors").read_bytes() == bad
    assert (legacy / "good.safetensors").read_bytes() == good
    folder = next((root / "views" / "canvas").iterdir())
    assert (folder / "bad.safetensors").read_bytes() == fetched
    assert legacy.is_dir()


def _spec(files: dict[str, str]):
    base = next(spec for spec in load_catalog().models if spec.role == "writer")
    return replace(base, id="adopt-local", repo="Org/Name", files=files, fallback_files={})
