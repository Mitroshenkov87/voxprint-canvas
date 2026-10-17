"""Instruction edit: published mirror pins, FP8 loader wiring, and the download."""
import json
from pathlib import Path

import pytest

from canvas.models.features import instruction_edit_available
from canvas.models.edit import (
    ENCODER_FP8_TENSORS,
    TRANSFORMER_SHARDS,
    EditLoadError,
    plan_fp8_assignment,
    vl_checkpoint_key,
)
from canvas.models.modes import Mode, Tier
from canvas.models.pin import PinError, resolved_sha256
from canvas.models.plan import build_plan
from canvas.models.provision import ensure_plan, install_defaults, install_spec
from canvas.models.registry import load_catalog
from canvas.models.snapshots import SnapshotError, _urls, download_snapshot
from canvas.safety.inputs import ImageRole, PendingInput
from tests.fakes import World, rgb

_MIRROR = "32a23ad2004c32096ae39371bd8332e440b8d1ae"
_UPSTREAM = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
_FIXTURE = Path("tests/fixtures/qwen_edit_pins.json")


def test_registry_pins_resolve_from_the_offline_fixture():
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    spec = load_catalog().by_id("qwen-image-edit-2511-fp8")
    assert spec.mirror_status == "published"
    assert spec.mirror_revision == fixture["mirror_revision"] == _MIRROR
    assert spec.revision == fixture["upstream_revision"] == _UPSTREAM
    assert spec.quality_gate == "passed"
    assert set(spec.files) == set(fixture["files"])
    for name, row in fixture["files"].items():
        assert spec.files[name] == resolved_sha256(row["lfs_oid"], row["sidecar"])
        assert spec.origins[name] == (row["mirror"], row["upstream"])
    # The BF16 text-encoder shards are in the mirror and match upstream, and the
    # 24 GB pack uses the FP8 encoder instead. VAE stays in the pack. model_index
    # and the scheduler are not in the mirror, so they stay pinned to upstream.
    for name in fixture["bf16_text_encoder_in_mirror_not_packed"]:
        assert name not in spec.files
    assert "vae/diffusion_pytorch_model.safetensors" in spec.files
    assert spec.origins["model_index.json"] == ("-", "model_index.json")
    assert spec.origins["scheduler/scheduler_config.json"][0] == "-"
    assert spec.nbytes < 35_000_000_000
    with pytest.raises(PinError):
        resolved_sha256("ab" * 32, "cd" * 32)


def test_edit_download_urls_use_the_mirror_revision_and_upstream_pin():
    spec = load_catalog().by_id("qwen-image-edit-2511-fp8")
    shard = TRANSFORMER_SHARDS[0]
    urls = _urls(spec, shard, spec.revision, fallback=False)
    assert urls == (
        "https://huggingface.co/Mitroshenkov87/voxprint-mirror-qwen-image-edit-2511"
        f"/resolve/{_MIRROR}/{shard}",
    )
    index = _urls(spec, "model_index.json", spec.revision, fallback=False)
    assert index == (
        f"https://huggingface.co/Qwen/Qwen-Image-Edit-2511/resolve/{_UPSTREAM}/model_index.json",
    )
    vae = _urls(spec, "vae/diffusion_pytorch_model.safetensors", spec.revision, fallback=False)
    assert vae[0].startswith(
        f"https://huggingface.co/Mitroshenkov87/voxprint-mirror-qwen-image-edit-2511/resolve/{_MIRROR}/"
    )
    assert vae[1].startswith(f"https://huggingface.co/Qwen/Qwen-Image-Edit-2511/resolve/{_UPSTREAM}/")


def test_vl_keys_match_transformers_5_19_and_a_partial_apply_is_refused():
    header = {
        "__metadata__": {"format": "voxprint-fp8-v1"},
        "model.layers.0.mlp.down_proj.weight": {"dtype": "F8_E4M3", "shape": [2, 2]},
        "model.layers.0.mlp.down_proj.weight.scale_weight": {"dtype": "BF16", "shape": []},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [4, 2]},
        "visual.blocks.0.norm1.weight": {"dtype": "BF16", "shape": [2]},
        "lm_head.weight": {"dtype": "F8_E4M3", "shape": [4, 2]},
        "lm_head.weight.scale_weight": {"dtype": "BF16", "shape": []},
    }
    module_keys = {
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.embed_tokens.weight",
        "model.visual.blocks.0.norm1.weight",
        "lm_head.weight",
    }
    plan = plan_fp8_assignment([header], module_keys, vl_checkpoint_key)
    assert plan.fp8_count == 2
    assert plan.assignments["model.layers.0.mlp.down_proj.weight"] == (
        "model.language_model.layers.0.mlp.down_proj.weight"
    )
    assert vl_checkpoint_key("visual.blocks.0.norm1.weight") == "model.visual.blocks.0.norm1.weight"
    with pytest.raises(EditLoadError):
        plan_fp8_assignment([header], module_keys, remap=None)
    assert ENCODER_FP8_TENSORS == 197


def test_edit_loader_keeps_fp8_weights_and_offloads(monkeypatch, tmp_path: Path):
    folder = tmp_path
    (folder / "transformer").mkdir()
    (folder / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (folder / "text_encoder").mkdir()
    (folder / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    installed: list[tuple[str, tuple[str, ...], bool]] = []

    def fake_install(module, paths, remap):
        installed.append((type(module).__name__, tuple(path.name for path in paths), remap is not None))
        if remap is None:
            return 17
        return 197

    class _Pipe:
        def __init__(self) -> None:
            self.offloaded = False
            self.tiled = False

        def enable_model_cpu_offload(self) -> None:
            self.offloaded = True

        def enable_vae_tiling(self) -> None:
            self.tiled = True

        def to(self, device: str):
            raise AssertionError(device)

    seen: dict[str, object] = {}

    class _Transformer:
        @staticmethod
        def from_config(config):
            seen["transformer_config"] = config
            return _Transformer()

    class _Text:
        @staticmethod
        def from_config(config):
            seen["text_config"] = config
            return _Text()

    class _Diffusers:
        QwenImageTransformer2DModel = _Transformer

        class QwenImageEditPlusPipeline:
            @staticmethod
            def from_pretrained(path, **kwargs):
                seen["pretrained"] = kwargs
                seen["folder"] = path
                return _Pipe()

    class _Transformers:
        class AutoConfig:
            @staticmethod
            def from_pretrained(path, local_files_only):
                seen["auto_config"] = (path, local_files_only)
                return {"hidden": 4}

        Qwen2_5_VLForConditionalGeneration = _Text

    class _Accel:
        @staticmethod
        def init_empty_weights():
            class _Ctx:
                def __enter__(self):
                    seen["empty"] = True
                    return None

                def __exit__(self, *exc: object) -> bool:
                    return False

            return _Ctx()

    monkeypatch.setattr("canvas.models.edit._import_diffusers", lambda: _Diffusers)
    monkeypatch.setattr("canvas.models.edit._import_transformers", lambda: _Transformers)
    monkeypatch.setattr("canvas.models.edit._import_accelerate", lambda: _Accel)
    monkeypatch.setattr("canvas.models.edit._import_torch", lambda: object())
    monkeypatch.setattr("canvas.models.edit.install_fp8_files", fake_install)
    from canvas.models.edit import load_edit_pipeline

    pipe = load_edit_pipeline(folder, "bf16")
    assert seen["empty"] is True
    assert seen["transformer_config"] == {}
    assert seen["auto_config"] == (folder / "text_encoder", True)
    pretrained = seen["pretrained"]
    assert isinstance(pretrained, dict)
    assert pretrained["torch_dtype"] == "bf16"
    assert pretrained["local_files_only"] is True
    assert isinstance(pretrained["transformer"], _Transformer)
    assert isinstance(pretrained["text_encoder"], _Text)
    assert pipe.offloaded is True
    assert pipe.tiled is True
    assert installed[0][2] is False
    assert installed[0][1] == TRANSFORMER_SHARDS
    assert installed[1][2] is True
    assert installed[1][1] == ("qwen2.5-vl-7b-text-encoder-fp8.safetensors",)


def test_render_calls_the_edit_loader(monkeypatch, tmp_path: Path):
    from canvas.models.pins import DIFFUSERS_PIN, PIPELINE_EDIT
    from canvas.models.plan import CallPlan
    from canvas.models.runtime import render

    called: dict[str, object] = {}

    class _Image:
        width = 2
        height = 2

        def convert(self, mode: str):
            del mode
            return self

        def tobytes(self) -> bytes:
            return bytes([9, 8, 7]) * 4

    class _Pipe:
        def __call__(self, **kwargs):
            called["kwargs"] = kwargs
            return type("Result", (), {"images": [_Image()]})()

        def to(self, device: str):
            raise AssertionError(device)

    def fake_load(folder, dtype, low_vram=False):
        called["folder"] = folder
        called["dtype"] = dtype
        called["low_vram"] = low_vram
        return _Pipe()

    class _Gen:
        def manual_seed(self, seed: int) -> None:
            called["seed"] = seed

    class _Torch:
        bfloat16 = "bf16"

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        @staticmethod
        def Generator(device: str):
            called["device"] = device
            return _Gen()

    monkeypatch.setattr("canvas.models.runtime._import_torch", lambda: _Torch)
    monkeypatch.setattr("canvas.models.runtime._import_diffusers", lambda: object())
    monkeypatch.setattr("canvas.models.runtime.load_edit_pipeline", fake_load)
    screened = World().gate().release_request(
        "paint the door blue", Mode.INSTRUCTION.value, [PendingInput(ImageRole.INIT, rgb())],
    )
    plan = CallPlan(
        DIFFUSERS_PIN, PIPELINE_EDIT, ("Qwen/Qwen-Image-Edit-2511",), "fp8", 40, 4.0,
        "model-cpu-offload", "24gb", "instruction", "true_cfg_scale", "Qwen/Qwen2.5-VL-7B-Instruct",
    )
    image = render(plan, tmp_path, screened, 1328, 1328, Tier.GB24, seed=3)
    assert called["folder"] == tmp_path
    assert called["dtype"] == "bf16"
    assert called["low_vram"] is False
    kwargs = called["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["true_cfg_scale"] == 4.0
    assert kwargs["num_inference_steps"] == 40
    assert "guidance_scale" not in kwargs
    assert image.width == 2


def test_empty_init_matches_diffusers_0_41_and_transformers_5_19():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    accelerate = pytest.importorskip("accelerate")
    from canvas.models.edit import _empty, _restore_meta_buffers

    config = {
        "patch_size": 2,
        "in_channels": 64,
        "out_channels": 16,
        "num_layers": 1,
        "attention_head_dim": 128,
        "num_attention_heads": 24,
        "joint_attention_dim": 3584,
        "axes_dims_rope": [16, 56, 56],
        "zero_cond_t": True,
    }
    transformer = _empty(accelerate, diffusers.QwenImageTransformer2DModel, config)
    assert next(transformer.parameters()).is_meta
    assert transformer.pos_embed.pos_freqs.is_meta is False
    assert transformer.pos_embed.pos_freqs.device.type == "cpu"
    with accelerate.init_empty_weights(include_buffers=True):
        meta_rope = diffusers.QwenImageTransformer2DModel.from_config(config)
    assert meta_rope.pos_embed.pos_freqs.is_meta
    _restore_meta_buffers(meta_rope, torch)
    assert meta_rope.pos_embed.pos_freqs.is_meta is False
    assert meta_rope.pos_embed.neg_freqs.device.type == "cpu"

    text = transformers.Qwen2_5_VLConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "max_position_embeddings": 32,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 2, 2]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 64,
            "fullatt_block_indexes": [],
        },
    )
    encoder = _empty(accelerate, transformers.Qwen2_5_VLForConditionalGeneration, text, torch.bfloat16)
    names = {name for name, _tensor in encoder.named_parameters()}
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in names
    assert "model.visual.blocks.0.norm1.weight" in names
    assert "lm_head.weight" in names
    assert all(not buf.is_meta for _name, buf in encoder.named_buffers())


def test_fp8_linear_computes_in_bf16():
    torch = pytest.importorskip("torch")
    from canvas.models.edit import _assign

    linear = torch.nn.Linear(4, 4, bias=True)
    source = torch.randn(4, 4, dtype=torch.bfloat16)
    scale = (source.abs().max() / 448).to(dtype=torch.bfloat16).reshape(())
    fp8 = (source / scale).to(torch.float8_e4m3fn)
    bias = linear.bias.detach().to(dtype=torch.bfloat16).clone()
    linear.bias = torch.nn.Parameter(bias, requires_grad=False)
    _assign(linear, "weight", fp8, scale, torch)
    inputs = torch.randn(2, 4, dtype=torch.bfloat16)
    output = linear(inputs)
    reference = torch.nn.functional.linear(inputs, fp8.to(torch.bfloat16) * scale, bias)
    assert linear.weight.dtype == torch.float8_e4m3fn
    assert output.dtype == torch.bfloat16
    assert torch.allclose(output, reference)


@pytest.mark.skipif(not instruction_edit_available(), reason="instruction_edit is model_in_preparation")
def test_instruction_edit_offers_the_pack_and_records_it(monkeypatch, tmp_path: Path):
    from canvas.infra.manifest import load_manifest
    from canvas.infra.paths import shared_dir

    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    notes: list[str] = []

    def fake_download(spec, root, progress=None):
        del progress
        notes.append(spec.id)
        return root

    monkeypatch.setattr("canvas.models.provision.download_snapshot", fake_download)
    catalog = load_catalog()
    screened = World().gate().release_request(
        "paint the door blue", Mode.INSTRUCTION.value, [PendingInput(ImageRole.INIT, rgb())],
    )
    plan = build_plan(catalog, screened, Tier.GB24)
    ensure_plan(catalog, plan, notes.append, confirm=lambda notice: notes.append(notice) or True)
    assert "qwen-image-edit-2511-fp8" in notes
    notice = next(item for item in notes if "Estimated time" in str(item))
    assert str(notice).startswith("Qwen-Image-Edit-2511 is ")
    assert "GB" in str(notice)
    components = load_manifest(shared_dir())
    assert any(
        component.version == "qwen-image-edit-2511-fp8" and component.kind == "model"
        for component in components.values()
    )


@pytest.mark.skipif(instruction_edit_available(), reason="the edit pack downloads when the feature is available")
def test_setup_does_not_download_instruction_edit_or_int4(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    seen: list[str] = []

    def fake_download(spec, root, progress=None):
        del root, progress
        seen.append(spec.id)
        return tmp_path

    monkeypatch.setattr("canvas.models.provision.download_snapshot", fake_download)
    catalog = load_catalog()
    install_defaults(catalog, lambda _message: None)
    install_spec(catalog.by_id("qwen-image-edit-2511-fp8"), lambda _message: None, confirm=None)
    install_spec(catalog.by_id("z-image-turbo-svdquant-int4"), lambda _message: None, confirm=None)
    screened = World().gate().release_request(
        "paint the door blue", Mode.INSTRUCTION.value, [PendingInput(ImageRole.INIT, rgb())],
    )
    ensure_plan(catalog, build_plan(catalog, screened, Tier.GB24), lambda _message: None, confirm=None)
    assert "qwen-image-edit-2511-fp8" not in seen
    assert "z-image-turbo-svdquant-int4" not in seen
    with pytest.raises(SnapshotError) as caught:
        download_snapshot(catalog.by_id("qwen-image-edit-2511-fp8"), tmp_path)
    assert caught.value.detail == "this model is not available for download"
