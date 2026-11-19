"""Preflight refuses a heavy load before any model is opened."""
from __future__ import annotations

from pathlib import Path

from canvas.app.preflight import MIN_FREE_DISK_BYTES, MIN_RAM_BYTES, preflight
from canvas.models.loader import load_catalog_stack
from canvas.models.registry import load_catalog
from canvas.models.required import WRITER_ID, gate_specs, pins_for, required_specs
from canvas.models.snapshots import weight_dir
from canvas.models.views import blob_path
from canvas.safety.gate import SafetyUnavailable


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


def test_a_legacy_folder_is_not_downloaded_again(tmp_path: Path, monkeypatch) -> None:
    home = _home(tmp_path, monkeypatch)
    root = tmp_path / "models"
    root.mkdir()
    catalog = load_catalog()
    gates = {spec.id for spec in gate_specs(catalog)}
    pictures = [spec for spec in required_specs(catalog) if spec.id not in gates]
    spec = pictures[0]
    assert spec.id == WRITER_ID
    legacy = weight_dir(root, spec.repo)
    legacy.mkdir()
    marker = legacy / "weights.safetensors"
    marker.write_bytes(b"keep")
    calls: list[str] = []
    monkeypatch.setattr("canvas.app.preflight.request_download", calls.append)
    result = preflight(root, catalog, writer=True, disk_free=_plenty, ram_bytes=_plenty_ram, clock=_still)
    assert result.ok is False
    assert result.reason == "view"
    assert calls == []
    assert marker.read_bytes() == b"keep"
    assert "preflight fail reason=view" in _log(home)


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


def test_a_plain_prompt_run_does_not_require_or_queue_the_writer(tmp_path: Path, monkeypatch) -> None:
    home = _home(tmp_path, monkeypatch)
    root = tmp_path / "models"
    root.mkdir()
    catalog = load_catalog()
    for spec in required_specs(catalog):
        if spec.id != WRITER_ID:
            _plant(root, spec)
    calls: list[str] = []
    monkeypatch.setattr("canvas.app.preflight.request_download", calls.append)
    plain = preflight(root, catalog, disk_free=_plenty, ram_bytes=_plenty_ram, clock=_still)
    assert (plain.ok, plain.reason) == (True, "ok")
    assert calls == []
    scenario = preflight(root, catalog, writer=True, disk_free=_plenty, ram_bytes=_plenty_ram, clock=_still)
    assert (scenario.ok, scenario.reason) == (False, "view")
    assert calls == [WRITER_ID]
    assert "preflight fail reason=view" in _log(home)


def test_every_gate_model_is_still_required_without_the_writer(tmp_path: Path, monkeypatch) -> None:
    _home(tmp_path, monkeypatch)
    root = tmp_path / "models"
    root.mkdir()
    catalog = load_catalog()
    gates = gate_specs(catalog)
    assert len(gates) == 6
    for missing in gates:
        for spec in required_specs(catalog):
            if spec.id not in {WRITER_ID, missing.id}:
                _plant(root, spec)
        for digest in pins_for(missing).values():
            blob_path(root, digest).unlink(missing_ok=True)
        # The FP32 fallback never stands in for a gate model.
        monkeypatch.setattr("canvas.app.preflight.fallback_ready", lambda *args, **kwargs: root)
        result = preflight(root, catalog, disk_free=_plenty, ram_bytes=_plenty_ram, clock=_still)
        assert (result.ok, result.reason) == (False, "gate"), missing.id


def test_a_complete_fp32_upstream_view_stands_in_for_the_bf16_generator(tmp_path: Path, monkeypatch) -> None:
    _home(tmp_path, monkeypatch)
    root = tmp_path / "models"
    root.mkdir()
    catalog = load_catalog()
    turbo = catalog.by_id("z-image-turbo-bf16-16gb")
    assert turbo.fallback_files
    for spec in required_specs(catalog):
        if spec.id not in {WRITER_ID, turbo.id}:
            _plant(root, spec)
    monkeypatch.setattr("canvas.app.preflight.request_download", lambda model_id: True)
    # No BF16 blobs and no FP32 view: refused.
    refused = preflight(root, catalog, disk_free=_plenty, ram_bytes=_plenty_ram, clock=_still)
    assert (refused.ok, refused.reason) == (False, "view")
    # The loader's fallback_ready finds a verified FP32 view: accepted.
    asked: list[str] = []

    def ready(spec, where, cache=None):
        asked.append(spec.id)
        assert where == root
        return root / "views" / f"{spec.id}-upstream"

    monkeypatch.setattr("canvas.app.preflight.fallback_ready", ready)
    result = preflight(root, catalog, disk_free=_plenty, ram_bytes=_plenty_ram, clock=_still)
    assert (result.ok, result.reason) == (True, "ok")
    assert turbo.id in asked
    # A fallback that fails its hash check does not pass.
    monkeypatch.setattr("canvas.app.preflight.fallback_ready", lambda *args, **kwargs: None)
    again = preflight(root, catalog, disk_free=_plenty, ram_bytes=_plenty_ram, clock=_still)
    assert (again.ok, again.reason) == (False, "view")


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


def test_a_missing_view_asks_mdl_to_adopt_an_on_disk_copy_first(monkeypatch, tmp_path) -> None:
    """voxprint-common 0.7.3: repair_store links an identical copy (no download) before a fetch is queued."""
    from types import SimpleNamespace

    from canvas.app import preflight as pf

    calls: list[tuple] = []
    spec = SimpleNamespace(id="z-image-turbo-bf16-16gb", pinned=lambda: True, fallback_files={})
    state = {"done": False}

    def fake_repair(manifests, *, only, include_groups):
        calls.append((only, tuple(include_groups)))
        state["done"] = True
        return SimpleNamespace(adopted=19, missing=0, ready=True, repaired_views=1)

    monkeypatch.setattr("voxprint_common.mdl.repair_store.repair_store", fake_repair)
    monkeypatch.setattr(pf, "view_complete", lambda _spec, _root: state["done"])
    assert pf._missing((spec,), tmp_path) is None
    assert calls == [(("z-image-turbo-bf16-16gb",), ())]

    state["done"] = False
    monkeypatch.setattr(
        "voxprint_common.mdl.repair_store.repair_store",
        lambda *a, **k: SimpleNamespace(adopted=0, missing=4, ready=False, repaired_views=0),
    )
    assert pf._missing((spec,), tmp_path) is spec
