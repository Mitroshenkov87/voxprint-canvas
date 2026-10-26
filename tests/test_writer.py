"""The scenario writer is a separate pinned model. The generator encoder stays stock."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canvas.app.i18n import text
from canvas.app.photoshoot_cli import model_rewrite
from canvas.app.shell_run import improve_prompt
from canvas.models.modes import Tier
from canvas.models.pins import TEXT_ENCODER
from canvas.models.registry import load_catalog
from canvas.models.verified import process_cache
from voxprint_common.mdl.catalog import load_manifest
from canvas.infra import vram_trace
from canvas.infra.stage_guard import PEAK_LIMIT_BYTES, StageBlocked
from canvas.infra.stage_timer import StageTimes, set_current
from canvas.pipeline.device_guard import DeviceRefused
from canvas.pipeline.enhance import WRITER_ID, QwenRewriter, encoder_names, open_rewriter, turbo_spec, writer_spec
from tests.fakes import World
from tests.test_photoshoot import IDEA


REVISION = "c9bd464550d4078c72af0dd22aa18d0437868ce3"
MIRROR_REVISION = "2c219b659d064d24236fd748c19cc2e579cbe3be"
UPSTREAM = "https://huggingface.co/huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated"
MIRROR = "https://huggingface.co/Mitroshenkov87/voxprint-mirror-huihui-qwen3-4b-instruct-2507-abliterated"


def test_writer_catalog_pins_the_upstream_revision() -> None:
    spec = load_catalog().by_id(WRITER_ID)
    assert spec.repo == "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated"
    assert spec.revision == REVISION
    assert spec.mirror_revision == MIRROR_REVISION
    assert spec.licence == "Apache-2.0"
    assert spec.quant == "bf16"
    assert spec.role == "writer"
    assert spec.pinned()
    assert spec.files["model-00001-of-00002.safetensors"] == (
        "c6bbaf4d73c46bd9c4ef1a32cd65e6524fe9e71aa28ca63865c1eb6a7ad51a98"
    )
    assert spec.files["model-00002-of-00002.safetensors"] == (
        "b248ed21539e354dedbab9d73f06b0b2f633396664121fd0b3e1c17bc560840f"
    )
    assert spec.files["config.json"] == "5443f8ef61f1602830e997f577dc333179cf28240135f13438e750f4fb2fc839"
    manifest = load_manifest(Path(__file__).resolve().parents[1] / "mdl.json")
    writer = next(model for model in manifest.models if model.id == WRITER_ID)
    config = next(item for item in writer.files if item.path == "config.json")
    assert config.sha256 == "5443f8ef61f1602830e997f577dc333179cf28240135f13438e750f4fb2fc839"
    assert config.urls == (
        f"{MIRROR}/resolve/{MIRROR_REVISION}/config.json",
        f"{UPSTREAM}/resolve/{REVISION}/config.json",
    )
    weight = next(item for item in writer.files if item.path == "model-00001-of-00002.safetensors")
    assert weight.sha256 == "c6bbaf4d73c46bd9c4ef1a32cd65e6524fe9e71aa28ca63865c1eb6a7ad51a98"
    assert weight.urls[0].startswith(f"{MIRROR}/resolve/{MIRROR_REVISION}/")
    assert weight.urls[1].startswith(f"{UPSTREAM}/resolve/{REVISION}/")
    turbo = load_catalog().by_id("z-image-turbo-bf16-16gb")
    assert turbo.text_encoder == TEXT_ENCODER == "Qwen/Qwen3-4B"
    assert any(name.startswith("text_encoder/") for name in encoder_names(turbo))
    assert turbo_spec(load_catalog(), Tier.GB16).id == "z-image-turbo-bf16-16gb"


def test_open_rewriter_downloads_the_writer_and_reuses_one_folder(tmp_path: Path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def ready(spec, root, cache=None):
        del spec, root, cache
        return None

    def download(spec, root, progress=None):
        del root
        seen["id"] = spec.id
        folder = tmp_path / "writer"
        folder.mkdir()
        if progress is not None:
            progress("shard")
        return folder

    class Fake:
        def __init__(self, encoder_dir: Path, tokenizer_dir: Path, **kwargs: object) -> None:
            del kwargs
            seen["encoder"] = encoder_dir
            seen["tokenizer"] = tokenizer_dir

    monkeypatch.setattr("canvas.pipeline.enhance.snapshot_ready", ready)
    monkeypatch.setattr("canvas.pipeline.enhance.download_snapshot", download)
    monkeypatch.setattr("canvas.pipeline.enhance.QwenRewriter", Fake)
    notes: list[str] = []
    open_rewriter(load_catalog(), Tier.GB16, tmp_path, notes.append)
    assert seen["id"] == WRITER_ID
    assert seen["encoder"] == seen["tokenizer"] == tmp_path / "writer"
    assert notes[0] == text("writer.download", "en")
    assert writer_spec(load_catalog()).id == WRITER_ID


def test_open_rewriter_reuses_the_process_hash_cache(tmp_path: Path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def ready(spec: object, root: object, cache: object = None) -> Path:
        del spec, root
        seen["cache"] = cache
        return tmp_path

    class Fake:
        def __init__(self, encoder_dir: Path, tokenizer_dir: Path, **kwargs: object) -> None:
            del encoder_dir, tokenizer_dir, kwargs

    monkeypatch.setattr("canvas.pipeline.enhance.snapshot_ready", ready)
    monkeypatch.setattr("canvas.pipeline.enhance.QwenRewriter", Fake)
    open_rewriter(load_catalog(), Tier.GB16, tmp_path)
    assert seen["cache"] is process_cache()


def test_scenario_and_enhance_use_the_writer_id(tmp_path: Path, monkeypatch) -> None:
    ids: list[str] = []

    def ready(spec, root, cache=None):
        del spec, root, cache
        return None

    def download(spec, root, progress=None):
        del root, progress
        ids.append(spec.id)
        folder = tmp_path / spec.id
        folder.mkdir(exist_ok=True)
        return folder

    class Fake:
        def __init__(self, encoder_dir: Path, tokenizer_dir: Path, **kwargs: object) -> None:
            del encoder_dir, tokenizer_dir, kwargs

        def rewrite(self, prompt: str) -> str:
            return prompt

        def complete(self, prompt: str, *, system: str) -> str:
            del prompt, system
            return "{}"

        def close(self) -> None:
            return None

    monkeypatch.setattr("canvas.pipeline.enhance.snapshot_ready", ready)
    monkeypatch.setattr("canvas.pipeline.enhance.download_snapshot", download)
    monkeypatch.setattr("canvas.pipeline.enhance.QwenRewriter", Fake)
    monkeypatch.setattr("canvas.app.shell_run.selectable", lambda: True)
    improve_prompt(gate=World().gate(), prompt=IDEA, tier=Tier.GB16)
    model_rewrite(Tier.GB16, False)("ping")
    assert ids == [WRITER_ID, WRITER_ID]


def _writer_dir(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"], "tie_word_embeddings": True}),
        encoding="utf-8",
    )
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model.safetensors"}}),
        encoding="utf-8",
    )
    return folder


def _arm_writer(
    monkeypatch, devices: list[str], codes: list[str], *, fail: str, cuda: bool = True,
    ram: int = 32 * 1024**3, calls: list[dict[str, object]] | None = None,
    peak: dict[str, int] | None = None,
) -> None:
    monkeypatch.setattr("canvas.models.runtime.system_ram_bytes", lambda: ram)
    class _Tensor:
        shape = (1, 2)

        def to(self, device: str) -> _Tensor:
            del device
            return self

    class _Tokenizer:
        def apply_chat_template(self, messages: object, tokenize: bool = False, add_generation_prompt: bool = True,
                                enable_thinking: bool = False) -> str:
            del messages, tokenize, add_generation_prompt, enable_thinking
            return "ready"

        def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, _Tensor]:
            del text, return_tensors
            return {"input_ids": _Tensor()}

        def decode(self, tokens: object, skip_special_tokens: bool = True) -> str:
            del tokens, skip_special_tokens
            return "1. wide shot | standing | 50 mm lens | loft"

    class _Output:
        def __getitem__(self, key: object) -> list[int]:
            del key
            return [1, 2, 3]

    class _Model:
        def to(self, device: str) -> _Model:
            devices.append(f"to:{device}")
            if fail == "oom" and "cuda" in str(device):
                raise RuntimeError("CUDA out of memory")
            return self

        def generate(self, **kwargs: object) -> _Output:
            if calls is not None:
                calls.append(dict(kwargs))
            return _Output()

    class _Null:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> bool:
            del args
            return False

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return cuda

        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def reset_peak_memory_stats() -> None:
            # The live copy is one model. The historical reserve is what the 14 GB check reads.
            if peak is not None and peak["bytes"] > 8 * 1024**3:
                peak["bytes"] = 8 * 1024**3

    class _Torch:
        bfloat16 = "bf16"
        cuda = _Cuda

        @staticmethod
        def inference_mode() -> _Null:
            return _Null()

    class _Transformers:
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> _Tokenizer:
                del args, kwargs
                return _Tokenizer()

    def causal(_transformers: object, _torch: object, folder: Path, device: str) -> _Model:
        del _transformers, _torch, folder
        devices.append(device)
        if peak is not None:
            peak["bytes"] = 16 * 1024**3
        if fail == "oom" and device != "cpu":
            raise RuntimeError("CUDA out of memory")
        if fail == "other":
            raise RuntimeError("weights failed")
        return _Model()

    monkeypatch.setattr("canvas.pipeline.enhance._generation_libs", lambda: (_Torch(), _Transformers()))
    monkeypatch.setattr("canvas.pipeline.enhance._causal_model", causal)
    monkeypatch.setattr("canvas.pipeline.enhance.note_error", codes.append)


def _run_writer(folder: Path, *, device: str | None = None) -> tuple[str, str]:
    clock = StageTimes()
    previous = set_current(clock)
    try:
        rewriter = QwenRewriter(folder, folder, device=device)
        reply = rewriter.complete("ping", system="sys")
    finally:
        set_current(previous)
    return reply, clock.log_line()


def test_oom_on_cuda_stops_instead_of_generating_on_cpu(tmp_path: Path, monkeypatch) -> None:
    devices: list[str] = []
    codes: list[str] = []
    folder = _writer_dir(tmp_path / "encoder")
    _arm_writer(monkeypatch, devices, codes, fail="oom")
    clock = StageTimes()
    previous = set_current(clock)
    try:
        with pytest.raises(DeviceRefused) as caught:
            QwenRewriter(folder, folder).complete("ping", system="sys")
    finally:
        set_current(previous)
    assert caught.value.code == "device.cpu"
    assert str(caught.value) == text("error.device")
    assert devices == ["cpu", "to:cuda"]
    assert codes == ["writer.oom"]
    line = clock.log_line()
    assert "writer_fallback" not in line
    assert "writer=cpu" not in line
    assert "writer_generate=" not in line


def test_a_cuda_writer_records_its_device_without_a_fallback(tmp_path: Path, monkeypatch) -> None:
    devices: list[str] = []
    codes: list[str] = []
    folder = _writer_dir(tmp_path / "encoder")
    _arm_writer(monkeypatch, devices, codes, fail="")
    _reply, line = _run_writer(folder)
    assert devices == ["cpu", "to:cuda"]
    assert codes == []
    assert "writer=cuda" in line
    assert "writer_load=" in line
    assert "writer_generate=" in line
    assert "writer_tps=tps" in line
    assert "writer_fallback=" not in line


def test_a_kept_writer_is_read_into_ram_once(tmp_path: Path, monkeypatch) -> None:
    devices: list[str] = []
    codes: list[str] = []
    folder = _writer_dir(tmp_path / "encoder")
    _arm_writer(monkeypatch, devices, codes, fail="", ram=96 * 1024**3)
    _reply, line = _run_writer(folder)
    _run_writer(folder)
    assert devices == ["cpu", "to:cuda"]
    assert codes == []
    assert "writer=cuda" in line


def test_a_writer_error_that_is_not_oom_is_raised(tmp_path: Path, monkeypatch) -> None:
    devices: list[str] = []
    codes: list[str] = []
    folder = _writer_dir(tmp_path / "encoder")
    _arm_writer(monkeypatch, devices, codes, fail="other")
    with pytest.raises(RuntimeError, match="weights failed"):
        _run_writer(folder)
    assert codes == []


def test_no_cuda_is_a_visible_cpu_fallback(tmp_path: Path, monkeypatch) -> None:
    devices: list[str] = []
    codes: list[str] = []
    folder = _writer_dir(tmp_path / "encoder")
    _arm_writer(monkeypatch, devices, codes, fail="", cuda=False)
    _reply, line = _run_writer(folder)
    assert devices == ["cpu"]
    assert codes == ["writer.nocuda"]
    assert "writer=cpu" in line
    assert "writer_fallback=nocuda" in line


def test_a_16gb_enhancer_stays_bf16(tmp_path: Path, monkeypatch) -> None:
    devices: list[str] = []
    codes: list[str] = []
    folder = _writer_dir(tmp_path / "encoder")
    _arm_writer(monkeypatch, devices, codes, fail="")
    vram_trace._holders.clear()
    clock = StageTimes()
    previous = set_current(clock)
    rewriter = QwenRewriter(folder, folder, device="cuda", tier=Tier.GB16, stage="enhancer")
    try:
        reply = rewriter.complete("ping", system="sys")
    finally:
        set_current(previous)
        rewriter.close()
        vram_trace._holders.clear()
    assert reply.startswith("1.")
    assert devices[0] == "cpu"
    assert devices.count("to:cuda") == 1
    assert not any(code.startswith("enhancer.bf16") for code in codes)


def test_a_16gb_load_reclaims_the_reserved_peak_before_admit(tmp_path: Path, monkeypatch) -> None:
    devices: list[str] = []
    codes: list[str] = []
    peak = {"bytes": 0}
    seen: list[int] = []
    folder = _writer_dir(tmp_path / "encoder")
    _arm_writer(monkeypatch, devices, codes, fail="", peak=peak)
    monkeypatch.setattr("canvas.infra.stage_guard.on_cuda", lambda _module: True)

    def guard(stage: str, tier: object) -> None:
        del stage, tier
        seen.append(peak["bytes"])
        if peak["bytes"] > PEAK_LIMIT_BYTES:
            raise StageBlocked("vram.peak")

    monkeypatch.setattr("canvas.infra.stage_guard.guard_peak", guard)
    vram_trace._holders.clear()
    clock = StageTimes()
    previous = set_current(clock)
    rewriter = QwenRewriter(folder, folder, device="cuda", tier=Tier.GB16)
    try:
        reply = rewriter.complete("ping", system="sys")
    finally:
        set_current(previous)
        rewriter.close()
        vram_trace._holders.clear()
    assert reply.startswith("1.")
    assert devices[0] == "cpu"
    assert seen
    assert seen[0] <= PEAK_LIMIT_BYTES
    assert codes == []


def test_writer_generate_stays_on_gpu_with_the_kv_cache(tmp_path: Path, monkeypatch) -> None:
    devices: list[str] = []
    codes: list[str] = []
    calls: list[dict[str, object]] = []
    folder = _writer_dir(tmp_path / "encoder")
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    _arm_writer(monkeypatch, devices, codes, fail="", calls=calls)
    reply, _line = _run_writer(folder)
    assert reply.startswith("1.")
    assert calls and calls[0]["use_cache"] is True
    assert "streamer" not in calls[0]
    assert "stopping_criteria" not in calls[0]
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "stage writer tokens" in log
    assert "tokens=" in log and "tok_s=" in log
    assert "dtype=bf16" in log and "kv_cache=on" in log
    assert "prefill_ms=" in log and "decode_tok_s=" in log
    assert "graph=eager" in log
    assert "device=" in log


def test_an_explicit_cpu_writer_stops_when_cuda_is_available(tmp_path: Path, monkeypatch) -> None:
    devices: list[str] = []
    codes: list[str] = []
    folder = _writer_dir(tmp_path / "encoder")
    _arm_writer(monkeypatch, devices, codes, fail="oom")
    with pytest.raises(DeviceRefused) as caught:
        _run_writer(folder, device="cpu")
    assert caught.value.code == "device.cpu"
    assert devices == []
    assert codes == ["writer.device"]
    assert "writer.oom" not in codes
