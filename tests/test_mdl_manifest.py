"""mdl.json is generated from the catalog and is never empty."""
import json
from pathlib import Path

import pytest

from canvas.models.mdl_manifest import render
from canvas.models.provision import default_specs
from canvas.models.registry import load_catalog

MANIFEST = Path("mdl.json")
GATE_ROLES = ("prompt-guard", "zero-shot", "face-detector", "age", "nsfw", "age-fine")


def _doc() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_mdl_json_matches_the_catalog():
    assert MANIFEST.read_text(encoding="utf-8") == render(), "run python -m canvas.models.mdl_manifest"


def test_every_gate_model_and_the_default_generator_are_required_downloads():
    doc = _doc()
    assert doc["schema"] == 1 and doc["app"] == "canvas"
    assert doc["models"], "mdl.json must not be empty"
    rows = {row["id"]: row for row in doc["models"]}
    catalog = load_catalog()
    for role in GATE_ROLES:
        (spec,) = catalog.by_role(role)
        row = rows[spec.id]
        assert row["optional"] is False
        assert {f["path"]: f["sha256"] for f in row["files"]} == dict(spec.files)
    for spec in default_specs(catalog):
        assert rows[spec.id]["optional"] is False
        for item in rows[spec.id]["files"]:
            assert len(item["sha256"]) == 64 and item["size"] > 0 and item["mirrors"]
            assert all(url.startswith("https://") for url in item["mirrors"])


def test_pins_licences_and_sources():
    rows = {row["id"]: row for row in _doc()["models"]}
    assert rows["z-image-turbo-bf16-16gb"]["version"] == "f332072aa78be7aecdf3ee76d5c247082da564a6"
    assert rows["qwen3guard-gen-0.6b"]["version"] == "fada3b2f655b89601929198343c94cd2f64d93cc"
    assert rows["siglip2-so400m-patch14-384"]["version"] == "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
    assert rows["fairface-age"]["version"] == "4e02ab8057ea7fd74b1670940995c5dfda3e6ec0"
    assert rows["nsfw-image-detection"]["version"] == "96cb0d0342c7afb80cab76ecc58b265fa44da256"
    assert rows["nsfw-image-detection"]["licence"] == "Apache-2.0"
    yunet = rows["yunet-face"]
    assert yunet["version"] == "47534e27c9851bb1128ccc0102f1145e27f23f98"
    (item,) = yunet["files"]
    assert item["sha256"] == "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    assert all("/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/" in url for url in item["mirrors"])
    assert not any("/raw/main/" in url for row in rows.values() for f in row["files"] for url in f["mirrors"])
    turbo = rows["z-image-turbo-bf16-16gb"]
    assert turbo["source"] == "primary"
    bf16 = "https://huggingface.co/Mitroshenkov87/voxprint-z-image-turbo-bf16/resolve/b6a12e018b150c9790626286daeaa1149d1ae598/"
    mirror = "https://huggingface.co/Mitroshenkov87/voxprint-mirror-z-image-turbo/resolve/75806c7417b435993a2b8b8346dd993dc09a52d5/"
    for item in turbo["files"]:
        if item["path"].startswith("transformer/"):
            assert item["mirrors"][0] == bf16 + item["path"]
        else:
            assert item["mirrors"][0] == mirror + item["path"]
            assert "Tongyi-MAI/Z-Image-Turbo/resolve/f332072" in item["mirrors"][1]
    fallback = rows["z-image-turbo-bf16-16gb-upstream"]
    assert fallback["source"] == "fallback" and fallback["optional"] is True
    for item in fallback["files"]:
        assert item["mirrors"][0] == mirror + item["path"]
        assert "Tongyi-MAI/Z-Image-Turbo/resolve/f332072" in item["mirrors"][1]
    gate_mirrors = {
        "qwen3guard-gen-0.6b": "voxprint-mirror-qwen3guard-gen-0.6b/resolve/1e5fb12633c55f4cb07fd22445839226c30efbeb/",
        "siglip2-so400m-patch14-384": "voxprint-mirror-siglip2-so400m-patch14-384/resolve/9c2b8022209e02798ce8cb10206de07c725caa59/",
        "fairface-age": "voxprint-mirror-fairface-age-image-detection/resolve/321f7b4a5388a9dbb571096e6d98e9de410d2e41/",
        "nsfw-image-detection": "voxprint-mirror-nsfw-image-detection/resolve/5a304d9137e475cc0bd95da9db1d8c0bea854c3b/",
    }
    for model_id, prefix in gate_mirrors.items():
        for item in rows[model_id]["files"]:
            assert item["mirrors"][0] == "https://huggingface.co/Mitroshenkov87/" + prefix + item["path"]
            assert len(item["mirrors"]) == 2
    control = "voxprint-mirror-z-image-turbo-controlnet-union-2.1/resolve/34c99d2bcaa323341a4c0df5f867d049f420333a/"
    for model_id in ("z-image-turbo-fun-controlnet-union-bf16", "z-image-turbo-fun-controlnet-union-bf16-lite"):
        (item,) = rows[model_id]["files"]
        assert item["mirrors"][0] == "https://huggingface.co/Mitroshenkov87/" + control + item["path"]


def test_no_reference_to_the_removed_bf16_mirror_paths():
    for name in ("mdl.json", "canvas/models/catalog.json"):
        body = Path(name).read_text(encoding="utf-8")
        assert "zimage-turbo-transformer-bf16" not in body
        assert "588702b873e82c328ab82c76742bf67843411f37" not in body
        assert "package:" not in body


def test_mdl_accepts_the_manifest_and_plans_the_default_pack():
    catalog_mod = pytest.importorskip("voxprint_common.mdl.catalog")
    gpu = pytest.importorskip("voxprint_common.gpu_probe")
    manifest = catalog_mod.load_manifest(MANIFEST.resolve())
    planned = catalog_mod.plan_files([manifest], gpu.VramTier.GB16, models_root=Path("/nonexistent-models"))
    ids = {item.model_id for item in planned}
    catalog = load_catalog()
    assert {spec.id for spec in default_specs(catalog)} <= ids
