"""Registry rules: permissive licences, no forbidden models, matching diffusers pin."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from canvas.models.loader import _control_file, ensure_weights, load_catalog_stack
from canvas.models.pins import DIFFUSERS_PIN, PIPELINE_CONTROL
from canvas.models.plan import CallPlan
from canvas.models.features import instruction_edit_available
from canvas.models.registry import Catalog, RegistryError, load_catalog
from canvas.safety.gate import SafetyUnavailable


def _rewrite(tmp_path: Path, mutate) -> Path:
    raw = json.loads(Path("canvas/models/catalog.json").read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_catalog_loads_and_rejects_a_bad_pin_or_licence(tmp_path: Path):
    catalog = load_catalog()
    assert catalog.by_id("z-image-turbo-bf16-16gb").steps == 8
    assert catalog.by_id("z-image-turbo-bf16-16gb").quant == "bf16"
    ids = {spec.id for spec in catalog.models}
    assert "z-image-turbo-fp8" not in ids
    assert "z-image-turbo-fun-controlnet-union-fp8" not in ids
    with pytest.raises(RegistryError):
        load_catalog(_rewrite(tmp_path, lambda raw: raw.__setitem__("diffusers", "0.32.0")))
    with pytest.raises(RegistryError):
        load_catalog(_rewrite(tmp_path, lambda raw: raw["models"][0].__setitem__("licence", "GPL-3.0")))
    with pytest.raises(RegistryError):
        load_catalog(_rewrite(tmp_path, lambda raw: raw["models"][0].__setitem__("repo", "notai/nudenet")))
    with pytest.raises(RegistryError):
        load_catalog(_rewrite(tmp_path, lambda raw: raw["models"][0]["sources"].append({"kind": "http", "uri": "https://example.invalid"})))


def test_z_image_fp8_is_rejected_and_edit_fp8_keeps_its_gate(tmp_path: Path):
    catalog = load_catalog()
    ids = {spec.id for spec in catalog.models}
    assert "qwen-image-2512-fp8" not in ids
    assert "qwen-image-2512-int4" not in ids
    assert catalog.by_id("qwen-image-edit-2511-fp8").quality_gate == "passed"

    def as_fp8(raw: dict) -> None:
        raw["models"][0]["quant"] = "fp8-e4m3"
        raw["models"][0]["quality_gate"] = "passed"

    with pytest.raises(RegistryError, match="BF16"):
        load_catalog(_rewrite(tmp_path, as_fp8))

    def drop_gate(raw: dict) -> None:
        for row in raw["models"]:
            if row["id"] == "qwen-image-edit-2511-fp8":
                row["quality_gate"] = ""

    with pytest.raises(RegistryError, match="quality gate"):
        load_catalog(_rewrite(tmp_path, drop_gate))


def test_unpinned_weights_fail_closed(tmp_path: Path):
    spec = load_catalog().by_id("qwen3guard-gen-0.6b")
    with pytest.raises(SafetyUnavailable) as caught:
        ensure_weights(spec, tmp_path)
    assert caught.value.code == "missing_model"
    with pytest.raises(SafetyUnavailable) as caught:
        load_catalog_stack(tmp_path)
    assert caught.value.code == "missing_model"


def test_hash_mismatch_and_a_present_file(tmp_path: Path):
    spec = load_catalog().by_id("qwen3guard-gen-0.6b")
    payload = b"weights"
    digest = hashlib.sha256(payload).hexdigest()
    pinned = replace(spec, files={"weights": digest})
    folder = tmp_path / spec.repo.replace("/", "--")
    folder.mkdir()
    target = folder / "weights"
    target.write_bytes(b"other")
    with pytest.raises(SafetyUnavailable) as caught:
        ensure_weights(pinned, tmp_path)
    assert caught.value.code == "hash_mismatch"
    target.write_bytes(payload)
    ensure_weights(pinned, tmp_path)


def test_release_pins_match_the_frozen_snapshots():
    catalog = load_catalog()
    turbo = catalog.by_id("z-image-turbo-bf16")
    assert turbo.revision == "f332072aa78be7aecdf3ee76d5c247082da564a6"
    assert turbo.mirror_revision == "588702b873e82c328ab82c76742bf67843411f37"
    assert turbo.pinned()
    assert turbo.files["transformer/diffusion_pytorch_model-00001-of-00003.safetensors"] == (
        "d38279715f7d31232fd14602dec2810f91bf2b4ec2d2960ff9501d3d939f7357"
    )
    official = turbo.fallback_files["transformer/diffusion_pytorch_model-00001-of-00003.safetensors"]
    assert official == "95facd593e2549e8252acb571c653d57f7ddb7f1060d4e81712f152555a88804"
    index = turbo.files["transformer/diffusion_pytorch_model.safetensors.index.json"]
    packed = Path("canvas/models/turbo_transformer_index.json").read_bytes()
    assert hashlib.sha256(packed).hexdigest() == index
    creative = catalog.by_id("z-image-creative-bf16")
    assert creative.revision == "04cc4abb7c5069926f75c9bfde9ef43d49423021"
    assert creative.pinned()
    edit = catalog.by_id("qwen-image-edit-2511-fp8")
    assert edit.revision == "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
    assert edit.mirror_revision == "32a23ad2004c32096ae39371bd8332e440b8d1ae"
    assert edit.mirror_status == "published"
    assert edit.pinned()
    yunet = catalog.by_id("yunet-face")
    assert yunet.files["face_detection_yunet_2023mar.onnx"] == (
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    )
    guard = catalog.by_id("qwen3guard-gen-0.6b")
    assert guard.revision == "fada3b2f655b89601929198343c94cd2f64d93cc"
    assert guard.pinned()


def test_instruction_edit_is_unavailable_until_the_compressed_model():
    catalog = load_catalog()
    flag = catalog.features["instruction_edit"]
    assert flag.status == "unavailable"
    assert flag.reason == "model_in_preparation"
    assert instruction_edit_available(catalog) is False


def test_control_file_follows_the_tier(tmp_path: Path):
    catalog = load_catalog()
    payload = b"pack"
    digest = hashlib.sha256(payload).hexdigest()
    lite = catalog.by_id("z-image-turbo-fun-controlnet-union-bf16-lite")
    full = catalog.by_id("z-image-turbo-fun-controlnet-union-bf16")
    lite_name = next(iter(lite.files))
    full_name = next(iter(full.files))
    models = []
    for spec in catalog.models:
        if spec.id == lite.id:
            models.append(replace(spec, files={lite_name: digest}))
        elif spec.id == full.id:
            models.append(replace(spec, files={full_name: digest}))
        else:
            models.append(spec)
    edited = Catalog(catalog.diffusers, tuple(models), catalog.default_text_base, catalog.text_bases, catalog.sizes)
    folder = tmp_path / "alibaba-pai--Z-Image-Turbo-Fun-Controlnet-Union-2.1"
    folder.mkdir()
    (folder / lite_name).write_bytes(payload)
    (folder / full_name).write_bytes(payload)
    plan = CallPlan(
        DIFFUSERS_PIN,
        PIPELINE_CONTROL,
        ("Tongyi-MAI/Z-Image-Turbo", "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1"),
        "bf16",
        8,
        0.0,
        "",
        "24gb",
        "pose",
    )
    found = _control_file(edited, plan, tmp_path)
    assert found is not None and found.name == full_name
    found_lite = _control_file(edited, replace(plan, tier="16gb"), tmp_path)
    assert found_lite is not None and found_lite.name == lite_name
