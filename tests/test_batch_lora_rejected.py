"""A LoRA that does not fit ends a batch with exit 2 and lora.invalid, before any picture is drawn."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from canvas.app.main import main
from canvas.app.shell_run import ShellRunner
from canvas.models.user_lora import CODE_INVALID, apply_lora
from tests.test_batch_lora import CUP
from tests.test_forge_lora import _FORGE_A, _FORGE_B, _export, _metadata
from tests.test_lora_unapplied import _Transformer
from tests.test_photoshoot_cli import _ready, _writers


def _matrix_file(path: Path, shapes: dict[str, list[int]]) -> Path:
    """A safetensors file with U8 matrices of the given shapes and the Forge header."""
    header: dict[str, object] = {"__metadata__": _metadata("z-image")}
    offset = 0
    for name in sorted(shapes):
        size = shapes[name][0] * shapes[name][1]
        header[name] = {"dtype": "U8", "shape": shapes[name], "data_offsets": [offset, offset + size]}
        offset += size
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(payload).to_bytes(8, "little") + payload + b"\x01" * offset)
    return path


def _manifest(tmp_path: Path, rows: int) -> Path:
    manifest = tmp_path / "shoot.json"
    entries = [{"shot": CUP, "seed": 100 + row} for row in range(rows)]
    manifest.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return manifest


@pytest.mark.real_shapes
def test_a_lora_whose_matrices_do_not_fit_the_base_exits_2_before_any_model_loads(
    monkeypatch, tmp_path: Path, capsys,
) -> None:
    _ready(monkeypatch)
    drawn: list[str] = []
    _writers(monkeypatch, lambda line, *_args: drawn.append(line))
    # Z-Image-Turbo's to_q is 3840 x 3840; this adapter is 16 x 3000.
    bad = _matrix_file(
        tmp_path / "bad_shape.safetensors",
        {_FORGE_A: [16, 3000], _FORGE_B: [3840, 16]},
    )
    code = main(["batch", "--manifest", str(_manifest(tmp_path, 3)), "--lora", str(bad), "--json"])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["error"] == CODE_INVALID
    assert drawn == []


def test_a_lora_the_loaded_model_refuses_ends_every_row_with_exit_2(
    monkeypatch, tmp_path: Path, capsys,
) -> None:
    # rc.35: the refusal was taken as one dropped frame, every row loaded again, and the batch exited 1.
    _ready(monkeypatch)
    attempts: list[int] = []

    def generate_for(gate, tier, style, adult, lora, scale, negative):
        del style, adult, negative
        runner = ShellRunner(gate=gate, tier=tier, enlarge=object())

        def generate(line: str, seed: int, hero: object, strength: object) -> object:
            del line, seed, hero, strength
            attempts.append(1)
            return runner._guarded(lambda: apply_lora(_Transformer(matches=False), lora, scale))

        return generate

    _writers(monkeypatch, None)
    monkeypatch.setattr("canvas.app.main.model_generate", generate_for)
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    export = _export(tmp_path / "project" / "export", tensors={_FORGE_A: b"\x01\x02", _FORGE_B: b"\x03\x04"})
    code = main(["batch", "--manifest", str(_manifest(tmp_path, 3)), "--lora", str(export), "--json"])
    assert code == 2
    body = json.loads(capsys.readouterr().out)
    assert [row["code"] for row in body["entries"]] == [CODE_INVALID] * 3
    assert len(attempts) == 1
