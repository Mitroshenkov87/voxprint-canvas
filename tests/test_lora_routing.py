"""Each LoRA runs on its own base. A base Canvas cannot run is a Forge retrain."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from canvas.app.i18n import text
from voxprint_common.i18n import set_language
from canvas.app.main import main
from canvas.models.forge_retrain import request_retrain
from canvas.models.loader import ZImageGenerator
from canvas.models.modes import Tier
from canvas.models.registry import load_catalog
from canvas.models.user_lora import (
    CODE_BASE_MISMATCH,
    CODE_RETRAIN,
    MOSS_IMAGE_SFT,
    ZIMAGE_TURBO_MDL_ID,
    LoraRejected,
    LoraRoute,
    UserLora,
    base_choice,
    detect_base,
    fits_base,
    route_lora,
)
from canvas.safety.image import RawImage
from canvas.worker_jobs import Session, _generate, _retrain_refusal
from tests.fakes import World
from tests.lora_files import write_lora, write_safetensors

_PAIR = {
    "layers.0.attention.to_q.lora_A.weight": b"\x01\x02",
    "layers.0.attention.to_q.lora_B.weight": b"\x03\x04",
}
_FOREIGN = {
    "down_blocks.0.attn.to_q.lora_A.weight": b"\x01",
    "down_blocks.0.attn.to_q.lora_B.weight": b"\x02",
}
_RETRAIN = (
    "This LoRA was trained on sdxl, which Canvas cannot run. "
    "Retrain it for MOSS-Image in Model Forge (voxprint-forge prepare --base moss)."
)
_CREATED = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _file(folder: Path, name: str, metadata: dict[str, str], tensors: dict[str, bytes] | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    write_safetensors(path, _PAIR if tensors is None else tensors, metadata)
    return path


def _forge(folder: Path, base: str, name: str = "cup.safetensors") -> Path:
    """A Forge weight file. ``prepare_cli_lora`` accepts the kind and the paired matrices."""
    return _file(
        folder,
        name,
        {
            "voxprint.lora.base": base,
            "voxprint.lora.kind": "object",
            "voxprint.subject.trigger": "teapot",
        },
    )


def _entry(path: Path, base: str = "") -> UserLora:
    return UserLora(path.stem, "object", path.parent, path, "teapot", 0.7, (0.0, 2.0), False, base=base)


def _screened():
    return World().gate().release_request("a red teapot on a table", "text")


def _patch_draw(monkeypatch, tmp_path: Path) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr("canvas.models.loader.snapshot_ready", lambda *args, **kwargs: calls.append("snapshot") or tmp_path)
    monkeypatch.setattr(
        "canvas.models.loader.render",
        lambda *args, **kwargs: calls.append("render") or RawImage(2, 2, b"\x11\x22\x33" * 4),
    )
    monkeypatch.setattr(
        "canvas.models.moss.render_moss",
        lambda *args, **kwargs: calls.append("moss") or RawImage(2, 2, b"\x11\x22\x33" * 4),
    )
    monkeypatch.setattr("canvas.models.loader.run_bf16_16gb", lambda **kwargs: calls.append("schedule"))
    monkeypatch.setattr("canvas.models.loader.sample_latent", lambda *args, **kwargs: calls.append("sample"))
    return calls


def test_detect_base_reads_the_header_then_metadata_then_the_filename(tmp_path: Path) -> None:
    moss = _file(tmp_path / "header", "cup.safetensors", {"voxprint.lora.base": "moss-image"})
    assert detect_base(moss) == MOSS_IMAGE_SFT
    assert detect_base(_file(tmp_path / "sft", "cup.safetensors", {"voxprint.lora.base": "moss-image-sft"})) == MOSS_IMAGE_SFT
    assert detect_base(_file(tmp_path / "base", "cup.safetensors", {"voxprint.lora.base": "moss-image-base"})) == MOSS_IMAGE_SFT
    assert detect_base(_file(tmp_path / "turbo", "cup.safetensors", {"voxprint.lora.base": "z-image"})) == ZIMAGE_TURBO_MDL_ID
    assert detect_base(_file(tmp_path / "sdxl", "cup.safetensors", {"voxprint.lora.base": "sdxl"})) == "sdxl"
    # The header wins over a filename tag.
    tagged = _file(tmp_path / "both", "cup.moss.safetensors", {"voxprint.lora.base": "sdxl"})
    assert detect_base(tagged) == "sdxl"
    architecture = _file(tmp_path / "arch", "cup.safetensors", {"modelspec.architecture": "MOSS-Image"})
    assert detect_base(architecture) == MOSS_IMAGE_SFT
    version = _file(tmp_path / "version", "cup.safetensors", {"ss_base_model_version": "ZImageTurbo"})
    assert detect_base(version) == ZIMAGE_TURBO_MDL_ID
    # A metadata family wins over the filename. A string that names neither falls through to it.
    mixed = _file(tmp_path / "mixed", "cup.moss.safetensors", {"modelspec.architecture": "z-image"})
    assert detect_base(mixed) == ZIMAGE_TURBO_MDL_ID
    named = _file(tmp_path / "named", "cup.moss.safetensors", {"modelspec.architecture": "sdxl"})
    assert detect_base(named) == MOSS_IMAGE_SFT
    assert detect_base(_file(tmp_path / "zname", "cup.z-image.safetensors", {})) == ZIMAGE_TURBO_MDL_ID
    assert detect_base(_file(tmp_path / "case", "cup.MOSS.safetensors", {})) == MOSS_IMAGE_SFT
    untagged = _file(tmp_path / "old", "cup.safetensors", {})
    assert detect_base(untagged) == ZIMAGE_TURBO_MDL_ID
    recorded = _entry(_file(tmp_path / "side", "cup.moss.safetensors", {}), "sdxl")
    assert detect_base(recorded) == "sdxl"


def test_route_lora_loads_a_listed_base_and_asks_for_a_retrain_otherwise(tmp_path: Path) -> None:
    bases = (MOSS_IMAGE_SFT, ZIMAGE_TURBO_MDL_ID)
    moss = _entry(_file(tmp_path / "moss", "cup.safetensors", {"voxprint.lora.base": "moss-image"}))
    assert route_lora(moss, bases, MOSS_IMAGE_SFT) == LoraRoute(MOSS_IMAGE_SFT)
    assert base_choice(moss, bases, MOSS_IMAGE_SFT).run_on == MOSS_IMAGE_SFT
    assert base_choice(moss, bases, MOSS_IMAGE_SFT).note_base == ""
    zimage = _entry(_file(tmp_path / "z", "cup.safetensors", {"voxprint.lora.base": "z-image"}))
    assert route_lora(zimage, bases, MOSS_IMAGE_SFT) == LoraRoute(ZIMAGE_TURBO_MDL_ID)
    untagged = _entry(_file(tmp_path / "old", "cup.safetensors", {}))
    assert route_lora(untagged, bases, MOSS_IMAGE_SFT) == LoraRoute(ZIMAGE_TURBO_MDL_ID, "")
    other = _entry(_file(tmp_path / "sdxl", "cup.safetensors", {"voxprint.lora.base": "sdxl"}))
    assert route_lora(other, bases, MOSS_IMAGE_SFT) == LoraRoute("", "sdxl")
    choice = base_choice(other, bases, MOSS_IMAGE_SFT)
    assert choice.run_on == "" and choice.note_base == "sdxl"


def test_fits_base_checks_z_image_module_names_only(tmp_path: Path) -> None:
    moss = _file(tmp_path, "moss.safetensors", {"voxprint.lora.base": "moss-image-sft"}, _FOREIGN)
    fits_base(moss)
    untagged = _file(tmp_path, "old.safetensors", {}, _FOREIGN)
    with pytest.raises(LoraRejected) as caught:
        fits_base(untagged)
    assert caught.value.code == CODE_BASE_MISMATCH


def test_a_moss_lora_selects_the_moss_generator(monkeypatch, tmp_path: Path) -> None:
    calls = _patch_draw(monkeypatch, tmp_path)
    folder = write_lora(tmp_path, "moss-aaaaaaaa", base="moss-image")
    generator = ZImageGenerator(load_catalog(), Tier.GB24)
    generator.lora_path = str(folder)
    generator.generate(_screened(), 64, 64)
    assert generator.base == MOSS_IMAGE_SFT
    assert calls == ["snapshot", "moss"]


def test_a_z_image_lora_selects_z_image_turbo(monkeypatch, tmp_path: Path) -> None:
    calls = _patch_draw(monkeypatch, tmp_path)
    folder = write_lora(tmp_path, "zimg-aaaaaaaa", base="z-image")
    generator = ZImageGenerator(load_catalog(), Tier.GB24)
    generator.lora_path = str(folder)
    generator.generate(_screened(), 64, 64)
    assert generator.base == ZIMAGE_TURBO_MDL_ID
    assert calls == ["snapshot", "render"]


def test_an_sdxl_lora_is_rejected_before_any_loader_runs(monkeypatch, tmp_path: Path) -> None:
    calls = _patch_draw(monkeypatch, tmp_path)
    weights = _forge(tmp_path, "sdxl")
    generator = ZImageGenerator(load_catalog(), Tier.GB24)
    generator.bound_lora = _entry(weights, "sdxl")
    with pytest.raises(LoraRejected) as caught:
        generator.generate(_screened(), 64, 64)
    assert caught.value.code == CODE_RETRAIN
    assert caught.value.details["base"] == "sdxl"
    assert calls == []


def test_the_cli_exits_2_with_a_retrain_error(capsys, tmp_path: Path) -> None:
    weights = _forge(tmp_path, "sdxl")
    out = tmp_path / "out.webp"
    code = main([
        "generate", "--prompt", "a red teapot on a table", "--lora", str(weights), "--out", str(out), "--json",
    ])
    assert code == 2
    body = json.loads(capsys.readouterr().out)
    assert body["ok"] is False
    assert body["error"] == CODE_RETRAIN
    assert body["message"] == _RETRAIN
    assert body["retrain"] == {"from": "sdxl", "to": MOSS_IMAGE_SFT}
    assert not out.exists()


def test_the_cli_prints_the_retrain_sentence(capsys, tmp_path: Path) -> None:
    weights = _forge(tmp_path, "sdxl")
    code = main([
        "generate", "--prompt", "a red teapot on a table", "--lora", str(weights), "--out", str(tmp_path / "out.webp"),
    ])
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == _RETRAIN


def test_request_retrain_writes_one_inbox_file(monkeypatch, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    monkeypatch.setenv("VOXPRINT_FORGE_INBOX", str(inbox))
    weights = tmp_path / "cup.safetensors"
    written = request_retrain(weights, "sdxl")
    assert written.parent == inbox
    assert written.name.startswith("retrain-") and written.suffix == ".json"
    body = json.loads(written.read_text(encoding="utf-8"))
    assert body == {
        "kind": "retrain",
        "lora": str(weights),
        "trained_base": "sdxl",
        "target_base": MOSS_IMAGE_SFT,
        "created": body["created"],
    }
    assert _CREATED.match(body["created"])
    assert list(inbox.glob(".*.tmp")) == []


def _job(lora_path: str) -> dict[str, object]:
    return {
        "prompt": "a red teapot on a table",
        "negative": "",
        "mode": "text",
        "aspect": "1:1",
        "output": "native",
        "fmt": "png",
        "seed": 1,
        "steps": 8,
        "guidance": 0.0,
        "sampler": "euler",
        "copies": 1,
        "lora_path": lora_path,
    }


class _Runner:
    def __init__(self, lora: UserLora | None = None) -> None:
        self.bound_lora = lora
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        raise AssertionError("the job ran")


def test_the_worker_refuses_a_mismatched_lora_before_the_runner(tmp_path: Path) -> None:
    weights = _forge(tmp_path, "sdxl")
    runner = _Runner()
    session = Session(gate=None, check=None, tier=None, runner=runner, improver=lambda **_kwargs: "")
    reply = _generate(object(), session, {"job": _job(str(weights))})
    assert runner.calls == 0
    assert reply["code"] == CODE_RETRAIN
    assert reply["unavailable"] is True
    assert reply["retrain"] == {"from": "sdxl", "to": MOSS_IMAGE_SFT}
    assert "voxprint-forge prepare --base moss" in str(reply["message"])
    matched = _entry(_forge(tmp_path / "z", "z-image-turbo"))
    assert _retrain_refusal("generate", "", _Runner(matched)) is None


def test_the_window_offers_a_retrain_and_does_not_run_the_job(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from canvas.ui.shell_layout import open_layout
    from canvas.ui.shell_screen import ShellWindow

    inbox = tmp_path / "inbox"
    monkeypatch.setenv("VOXPRINT_FORGE_INBOX", str(inbox))
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path / "models"))
    set_language("en")
    app = QApplication.instance() or QApplication([])
    del app
    weights = _forge(tmp_path / "lora", "sdxl")
    shell = open_layout(tmp_path / "state", gate=World().gate())

    class _Worker:
        def __init__(self) -> None:
            self.calls: list[object] = []
            self.state = "ready"

        def submit(self, *args: object, **kwargs: object) -> None:
            self.calls.append(args)

        def when_ready(self, *args: object, **kwargs: object) -> None:
            self.calls.append(args)

        def cancel(self, *args: object) -> None:
            self.calls.append(args)

    worker = _Worker()
    window = ShellWindow(shell, tier=Tier.GB16, enlarge=object(), worker=worker)  # type: ignore[arg-type]
    window.show()
    window.shell.layout.set_value("shell.addons", str(weights))
    assert window._generate() is False
    assert worker.calls == []
    assert "sdxl" in window.shell.layout.reason
    assert "voxprint-forge" in window.shell.layout.reason
    button = window.findChild(QPushButton, "loraRetrain")
    assert button is not None
    assert button.text() == text("canvas.lora.retrain_action")
    button.click()
    body = window.findChild(QLabel, "InfoBarBody")
    assert body is not None
    assert body.text() == text("canvas.lora.retrain_queued")
    written = list(inbox.glob("retrain-*.json"))
    assert len(written) == 1
    saved = json.loads(written[0].read_text(encoding="utf-8"))
    assert saved["kind"] == "retrain"
    assert saved["lora"] == str(weights)
    assert saved["trained_base"] == "sdxl"
    assert saved["target_base"] == MOSS_IMAGE_SFT
    window.close()
