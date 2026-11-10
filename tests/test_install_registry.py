"""Canvas registers itself in installed-apps.json and never falls back to the placeholder stub."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canvas.models import mdl_registry
from voxprint_common.mdl.catalog import discover

PLACEHOLDER = mdl_registry.shipped_manifest()
REAL = mdl_registry.bundled_manifest()


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VOXPRINT_SUITE_ROOT", str(tmp_path / "suite"))
    return tmp_path


def _rows() -> list[dict]:
    return json.loads(mdl_registry.registry_path().read_text(encoding="utf-8"))["apps"]


def _write(rows: list[dict]) -> None:
    path = mdl_registry.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"apps": rows}), encoding="utf-8")


def test_register_creates_the_file_and_is_idempotent(tmp_path: Path) -> None:
    assert mdl_registry.register_install() is True
    assert _rows() == [{"id": "canvas", "path": str(REAL.parent)}]
    before = mdl_registry.registry_path().read_text()
    assert mdl_registry.register_install() is False
    assert mdl_registry.registry_path().read_text() == before
    assert [app.id for app in discover()] == ["canvas"]


def test_register_keeps_other_apps_and_fixes_a_stale_path(tmp_path: Path) -> None:
    other = tmp_path / "enlarge"
    other.mkdir()
    (other / "mdl.json").write_text("{}", encoding="utf-8")
    _write([{"id": "enlarge", "path": str(other)}, {"id": "canvas", "path": str(tmp_path / "gone")}])
    assert mdl_registry.register_install() is True
    assert _rows() == [{"id": "enlarge", "path": str(other)}, {"id": "canvas", "path": str(REAL.parent)}]

    _write([{"id": "enlarge", "path": str(other)}])
    assert mdl_registry.register_install() is True
    assert [row["id"] for row in _rows()] == ["enlarge", "canvas"]


def test_register_leaves_an_unreadable_registry_alone() -> None:
    path = mdl_registry.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    assert mdl_registry.register_install() is False
    assert path.read_text() == "not json"


def test_register_models_command_writes_the_registry(monkeypatch) -> None:
    from canvas.app import main as app_main

    assert app_main.main(["--register-models"]) == 0
    assert [row["id"] for row in _rows()] == ["canvas"]


def test_registry_without_canvas_still_gives_real_pins(tmp_path: Path) -> None:
    other = tmp_path / "enlarge"
    other.mkdir()
    (other / "mdl.json").write_text("{}", encoding="utf-8")
    _write([{"id": "enlarge", "path": str(other)}])
    assert mdl_registry.manifest_path() == REAL
    assert mdl_registry.selectable()


def test_registry_pointing_at_a_placeholder_gives_real_pins(tmp_path: Path) -> None:
    _write([{"id": "canvas", "path": str(PLACEHOLDER.parent)}])
    assert mdl_registry.has_real_pins(PLACEHOLDER) is False
    assert mdl_registry.manifest_path() == REAL
    assert {m.id for m in mdl_registry.selectable()} >= {"z-image-turbo-bf16-16gb"}


def test_a_registered_real_manifest_wins(tmp_path: Path) -> None:
    folder = tmp_path / "app"
    folder.mkdir()
    (folder / "mdl.json").write_text(REAL.read_text(encoding="utf-8"), encoding="utf-8")
    _write([{"id": "canvas", "path": str(folder)}])
    assert mdl_registry.manifest_path() == folder / "mdl.json"


def test_placeholder_is_used_only_without_any_real_pins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mdl_registry, "_BUNDLED", tmp_path / "missing.json")
    assert mdl_registry.manifest_path() == PLACEHOLDER
    assert mdl_registry.selectable() == ()
    assert mdl_registry.register_install() is False
