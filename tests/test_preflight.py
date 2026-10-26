"""Preflight refuses a heavy load before any model is opened."""
from __future__ import annotations

from pathlib import Path

from canvas.models.adopt import gate_specs, pins_for, required_specs
from canvas.models.loader import load_catalog_stack
from canvas.models.registry import load_catalog
from canvas.models.views import blob_path
from canvas.safety.gate import SafetyUnavailable
from canvas.app.preflight import MIN_FREE_DISK_BYTES, MIN_RAM_BYTES, preflight


def test_a_missing_picture_view_fails_before_the_stack_loads(tmp_path: Path, monkeypatch) -> None:
    home = _home(tmp_path, monkeypatch)
    called: list[str] = []
    monkeypatch.setattr(
        "canvas.models.loader.load_safety_stack",
        lambda *args, **kwargs: called.append("heavy") or None,
    )
    try:
        load_catalog_stack(tmp_path)
    except SafetyUnavailable as exc:
        assert exc.code == "missing_model"
    else:
        raise AssertionError("expected a missing view")
    assert called == []
    log = _log(home)
    assert log.splitlines() == [
        line for line in log.splitlines() if "preflight fail reason=view" in line
    ]
    assert len(log.splitlines()) == 1


def test_a_missing_gate_view_is_its_own_reason(tmp_path: Path, monkeypatch) -> None:
    home = _home(tmp_path, monkeypatch)
    root = tmp_path / "models"
    root.mkdir()
    catalog = load_catalog()
    gates = {spec.id for spec in gate_specs(catalog)}
    for spec in required_specs(catalog):
        if spec.id not in gates:
            _plant(root, spec)
    result = preflight(root, catalog, disk_free=_plenty, ram_bytes=_plenty_ram, clock=_still)
    assert result.ok is False
    assert result.reason == "gate"
    assert "preflight fail reason=gate" in _log(home)


def test_low_disk_low_ram_and_the_budget_each_fail(tmp_path: Path, monkeypatch) -> None:
    home = _home(tmp_path, monkeypatch)
    root = tmp_path / "models"
    root.mkdir()
    catalog = load_catalog()
    for spec in required_specs(catalog):
        _plant(root, spec)
    disk = preflight(root, catalog, disk_free=lambda path: 0, ram_bytes=_plenty_ram, clock=_still)
    ram = preflight(root, catalog, disk_free=_plenty, ram_bytes=lambda: 0, clock=_still)
    late = {"n": 0}

    def clock() -> float:
        late["n"] += 1
        return 0.0 if late["n"] == 1 else 10.0

    budget = preflight(root, catalog, disk_free=_plenty, ram_bytes=_plenty_ram, clock=clock)
    assert (disk.ok, disk.reason) == (False, "disk")
    assert (ram.ok, ram.reason) == (False, "ram")
    assert (budget.ok, budget.reason) == (False, "budget")
    text = _log(home)
    assert "preflight fail reason=disk" in text
    assert "preflight fail reason=ram" in text
    assert "preflight fail reason=budget" in text
    assert MIN_FREE_DISK_BYTES == 1 * 1024**3
    assert MIN_RAM_BYTES == 12 * 1024**3


def test_a_complete_machine_is_ok(tmp_path: Path, monkeypatch) -> None:
    home = _home(tmp_path, monkeypatch)
    root = tmp_path / "models"
    root.mkdir()
    catalog = load_catalog()
    for spec in required_specs(catalog):
        _plant(root, spec)
    result = preflight(root, catalog, disk_free=_plenty, ram_bytes=_plenty_ram, clock=_still)
    assert result.ok is True
    assert result.reason == "ok"
    assert "preflight ok reason=ok" in _log(home)


def _home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("VOXPRINT_HOME", str(home))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    return home


def _plant(root: Path, spec) -> None:
    for digest in pins_for(spec).values():
        path = blob_path(root, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"x")


def _plenty(path: Path) -> int:
    del path
    return MIN_FREE_DISK_BYTES


def _plenty_ram() -> int:
    return MIN_RAM_BYTES


def _still() -> float:
    return 0.0


def _log(home: Path) -> str:
    return (home / "logs" / "voxprint.log").read_text(encoding="utf-8")
