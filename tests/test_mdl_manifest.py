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
        assert rows[spec.id]["optional"] is spec.optional
        for item in rows[spec.id]["files"]:
            assert len(item["sha256"]) == 64 and item["size"] > 0 and item["mirrors"]
            assert all(url.startswith("https://") for url in item["mirrors"])


def test_exactly_one_turbo_row_and_it_is_the_fp8_pack():
    """One Turbo row, required, every weight the suite mirror under /fp8/.

    No ``-upstream`` rows and no creative row (Aleksandr 2026-11-17: the FP8
    16 GB pack is the only Turbo build, for every tier).
    """
    rows = {row["id"]: row for row in _doc()["models"]}
    generate_rows = [row for row in rows.values() if row["group"] == "generate"]
    assert [row["id"] for row in generate_rows] == ["z-image-turbo-bf16-16gb"]
    turbo = generate_rows[0]
    assert turbo["optional"] is False
    assert turbo["label"] == "Z-Image Turbo"
    assert "z-image-turbo-bf16" not in rows
    assert "z-image-creative-bf16" not in rows
    assert not [model_id for model_id in rows if model_id.endswith("-upstream")]
    for item in turbo["files"]:
        if item["path"].startswith(("transformer/", "text_encoder/")) and item["path"].endswith(".safetensors"):
            assert all("/fp8/" in url for url in item["mirrors"]), item["path"]


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
    mirror = "https://huggingface.co/Mitroshenkov87/voxprint-mirror-z-image-turbo/resolve/111b2548fc0ef370976c9b43d1aed976cb7a2cf2/"
    for item in turbo["files"]:
        if item["path"].startswith(("transformer/", "text_encoder/")) and (
            item["path"].endswith((".safetensors", ".safetensors.index.json"))
        ):
            # FP8 weights and their shard index: the suite mirror copy under fp8/ is the only address.
            assert item["mirrors"] == [mirror + "fp8/" + item["path"], ]
        else:
            # Configs, tokenizer and VAE are unchanged upstream files at the mirror root.
            assert item["mirrors"][0] == mirror + item["path"]
            assert "Tongyi-MAI/Z-Image-Turbo/resolve/f332072" in item["mirrors"][1]
    assert "z-image-turbo-bf16-16gb-upstream" not in rows  # the fallback is not listed
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
    writer = rows["qwen3-4b-writer"]
    assert writer["version"] == "c9bd464550d4078c72af0dd22aa18d0437868ce3"
    mirror = "https://huggingface.co/Mitroshenkov87/voxprint-huihui-qwen3-4b-instruct-2507-abliterated/resolve/6e368340799452e5cba7a1e423120c7b0bc0473d/"
    assert writer["display_name"] == "Huihui Qwen3 4B Instruct 2507 (scenario writer)"
    assert writer["licence"] == "Apache-2.0"
    # Optional since 2026-11-15: plain generation never needs it; Photoshoot asks for it when used.
    assert writer["optional"] is True
    assert "qwen3-4b-writer" not in {spec.id for spec in default_specs(load_catalog())}
    for item in writer["files"]:
        # Mirror only; the upstream commit above is the pin, not a download address.
        assert item["mirrors"] == [mirror + item["path"]]
        assert len(item["sha256"]) == 64
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
    required = {spec.id for spec in default_specs(catalog) if not spec.optional}
    assert required <= ids
    assert "z-image-turbo-bf16-16gb" in ids  # the FP8 pack itself is the required Turbo row
    # The writer is not planned for a default install.
    assert "qwen3-4b-writer" not in ids


def test_turbo_plans_compose_the_fp8_transformer_and_shared_text_stack():
    """Asked for by id, the FP8 pack composes its transformer with the shared TE/VAE/tokenizer."""
    catalog_mod = pytest.importorskip("voxprint_common.mdl.catalog")
    gpu = pytest.importorskip("voxprint_common.gpu_probe")
    manifest = catalog_mod.load_manifest(MANIFEST.resolve())
    cases = (
        (gpu.VramTier.GB16, "z-image-turbo-bf16-16gb"),
        (gpu.VramTier.GB24, "z-image-turbo-bf16-16gb"),
    )
    rows = {row["id"]: row for row in _doc()["models"]}
    for tier, model_id in cases:
        planned = catalog_mod.plan_files(
            [manifest], tier, models_root=Path("/nonexistent-models"), only=(model_id,),
        )
        files = {
            item.name: item
            for item in planned
            if model_id in (item.model_ids or ([item.model_id] if item.model_id else []))
        }
        expected = {item["path"]: item for item in rows[model_id]["files"]}
        assert set(files) == set(expected)
        for path, item in files.items():
            want = expected[path]
            assert item.sha256 == want["sha256"] and item.size == want["size"]
            if path.startswith(("transformer/", "text_encoder/")) and path.endswith(".safetensors"):
                assert "/fp8/" in item.urls[0]
            else:
                assert "voxprint-mirror-z-image-turbo" in item.urls[0]


# The shipped installer catalog (root mdl.json) has 10 rows: the pinned catalog
# models. It has no FP32 ``-upstream`` fallbacks and no uncatalogued SVDQuant ids.
# The packaged canvas/mdl.json placeholder carries those 10 ids plus the
# two SVDQuant stubs (12 rows); SVDQuant cannot be added to the installer
# catalog until it has real pins.
# Task groups recognised by voxprint-common 0.7.1 (docs/mdl-fields.md).
_KNOWN_GROUPS = frozenset({
    "generate", "edit", "upscale", "restore", "voice", "transcribe",
    "translate", "writer", "safety", "internal",
})

SHIPPED_EXPECTED = {
    "z-image-turbo-bf16-16gb": (
        "generate",
        "Z-Image Turbo",
        "Fast, high-quality pictures from text; the default. Transformer and text encoder stored as FP8.",
    ),
    "z-image-turbo-fun-controlnet-union-bf16-lite": (
        "edit",
        "Pose & edge guidance, lite",
        "The same guidance, sized for 16 GB cards.",
    ),
    "z-image-turbo-fun-controlnet-union-bf16": (
        "edit",
        "Pose & edge guidance",
        "Follows a pose, edges or depth from a reference picture.",
    ),
    "qwen3guard-gen-0.6b": (
        "safety",
        "Prompt safety check",
        "Checks prompts before generating.",
    ),
    "siglip2-so400m-patch14-384": (
        "safety",
        "Image safety check",
        "Checks input and output pictures.",
    ),
    "yunet-face": (
        "safety",
        "Face finder",
        "Finds faces for the safety and likeness checks.",
    ),
    "fairface-age": (
        "safety",
        "Age check",
        "Part of the adults-only check.",
    ),
    "mivolo-v2-age": (
        "safety",
        "Fine age check",
        "A closer age estimate used with the adults-only check.",
    ),
    "nsfw-image-detection": (
        "safety",
        "Content check",
        "Classifies picture content for the 18+ setting.",
    ),
    "qwen3-4b-writer": (
        "writer",
        "Photoshoot writer (Qwen3 4B)",
        "Writes the shot list for Photoshoot series.",
    ),
}


def test_every_shipped_mdl_row_has_label_purpose_and_a_known_group():
    doc = _doc()
    rows = {row["id"]: row for row in doc["models"]}
    assert len(rows) == 10
    assert set(rows) == set(SHIPPED_EXPECTED)
    for model_id, (group, label, purpose) in SHIPPED_EXPECTED.items():
        row = rows[model_id]
        assert row["label"] == label, model_id
        assert row["purpose"] == purpose, model_id
        assert row["group"] == group, model_id
        assert group in _KNOWN_GROUPS, model_id


def test_common_loader_accepts_the_shipped_display_fields():
    catalog_mod = pytest.importorskip("voxprint_common.mdl.catalog")
    manifest = catalog_mod.load_manifest(MANIFEST.resolve())
    assert len(manifest.models) == 10
    by_id = {model.id: model for model in manifest.models}
    for model_id, (group, label, purpose) in SHIPPED_EXPECTED.items():
        model = by_id[model_id]
        assert model.label == label, model_id
        assert model.purpose == purpose, model_id
        assert model.group == group, model_id

