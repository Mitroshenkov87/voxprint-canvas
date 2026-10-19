"""The scenario writer is a separate pinned model. The generator encoder stays stock."""
from __future__ import annotations

from pathlib import Path

from canvas.app.i18n import text
from canvas.app.photoshoot_cli import model_rewrite
from canvas.app.shell_run import improve_prompt
from canvas.models.modes import Tier
from canvas.models.pins import TEXT_ENCODER
from canvas.models.registry import load_catalog
from canvas.models.snapshots import _urls
from canvas.pipeline.enhance import WRITER_ID, encoder_names, open_rewriter, turbo_spec, writer_spec
from tests.fakes import World
from tests.test_photoshoot import IDEA


REVISION = "c9bd464550d4078c72af0dd22aa18d0437868ce3"
UPSTREAM = "https://huggingface.co/huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated"


def test_writer_catalog_pins_the_upstream_revision() -> None:
    spec = load_catalog().by_id(WRITER_ID)
    assert spec.repo == "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated"
    assert spec.revision == REVISION
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
    urls = _urls(spec, "config.json", spec.revision, fallback=False)
    assert urls == (f"{UPSTREAM}/resolve/{REVISION}/config.json",)
    assert all("voxprint-mirror" not in url for url in urls)
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
