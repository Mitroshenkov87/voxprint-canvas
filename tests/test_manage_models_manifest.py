import json
from pathlib import Path

from canvas.models import mdl_layout


def test_on_disk_models_list_for_any_card(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "mdl.json"
    source.write_text(
        json.dumps({"models": [{"id": "a", "vram_tier": "24gb"}, {"id": "b", "vram_tier": "24gb"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mdl_layout, "manifest_path", lambda: source)
    monkeypatch.setattr(mdl_layout, "files_for", lambda model_id, root=None: {} if model_id == "a" else None)
    out = mdl_layout.on_disk_manifest(tmp_path / "out" / "m.json")
    tiers = {row["id"]: row["vram_tier"] for row in json.loads(out.read_text())["models"]}
    assert tiers == {"a": "any", "b": "24gb"}


def test_unreadable_manifest_gives_none(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mdl_layout, "manifest_path", lambda: tmp_path / "missing.json")
    assert mdl_layout.on_disk_manifest(tmp_path / "m.json") is None
