"""The offline report reads the bound value and does not change the app."""
from __future__ import annotations

import ast
import base64
import json
from pathlib import Path

from PIL import Image

from canvas.safety.classifiers import Face
from canvas.safety.constants import IMAGE_ADULT_SEXUAL
from canvas.safety.gate import AgeGate, SafetyStack
from tests.fakes import World
from tests.lora_attr import encode, pack
from tests.lora_files import write_safetensors
from tools.declared_age_shadow import KEY, build_report, main, read_declared, write_report

_TENSORS = {"a.weight": b"\x01\x00\x02", "b.weight": b"\x03\x04"}
_OTHER = {"a.weight": b"\x09\x00\x02", "b.weight": b"\x03\x04"}


def _file(path: Path, tensors: dict[str, bytes], token: str) -> None:
    write_safetensors(path, tensors, {KEY: token, "voxprint.subject.trigger": "ada"})


def test_a_bound_value_round_trips_and_a_changed_tensor_or_copy_is_empty(tmp_path: Path) -> None:
    token = encode(19, _TENSORS)
    assert token is not None
    path = tmp_path / "ada.safetensors"
    _file(path, _TENSORS, token)
    assert read_declared(path) == 19
    assert read_declared({KEY: token}, _TENSORS) == 19
    for value in (18, 20, 99):
        assert read_declared({KEY: encode(value, _TENSORS)}, _TENSORS) == value
    assert encode(17, _TENSORS) is None
    assert encode(100, _TENSORS) is None
    assert encode(True, _TENSORS) is None
    assert read_declared({KEY: pack(17, _TENSORS)}, _TENSORS) is None
    assert read_declared({KEY: pack(100, _TENSORS)}, _TENSORS) is None
    changed = path.read_bytes()
    flipped = changed[:-1] + bytes([changed[-1] ^ 0xFF])
    path.write_bytes(flipped)
    assert read_declared(path) is None
    copied = tmp_path / "other.safetensors"
    _file(copied, _OTHER, token)
    assert read_declared(copied) is None
    blob = bytearray(base64.b64decode(token))
    blob[-1] ^= 0xFF
    assert read_declared({KEY: base64.b64encode(bytes(blob)).decode("ascii")}, _TENSORS) is None
    assert read_declared(tmp_path) is None
    assert read_declared(tmp_path / "missing.safetensors") is None
    assert read_declared(object()) is None
    assert read_declared({}) is None
    assert read_declared({KEY: "!!!"}, _TENSORS) is None
    assert read_declared({KEY: ""}, _TENSORS) is None


class _PixelAges:
    def estimate(self, image, faces):
        del faces
        pixel = image.to_bytes()[:3]
        if pixel == b"\x01\x00\x00":
            return [(0.0, 0.17)]
        if pixel == b"\x02\x00\x00":
            return [(0.0, 0.18)]
        return [(0.0, 0.0)]


def _gate() -> AgeGate:
    world = World()
    world.nsfw = 0.95
    world.faces = [Face(0, 0, 1, 1)]
    world.image_scores[IMAGE_ADULT_SEXUAL] = 0.9
    stack = world.stack()
    return AgeGate(SafetyStack(stack.guard, stack.zeroshot, stack.faces, _PixelAges(), stack.nsfw, stack.fine))


def _images(folder: Path) -> None:
    folder.mkdir()
    for name, color in (("band.png", (1, 0, 0)), ("edge.png", (2, 0, 0)), ("open.png", (0, 0, 0))):
        Image.new("RGB", (1, 1), color).save(folder / name)


def _counts(images: Path, lora: Path, declared: int | None) -> dict[str, int]:
    return build_report(images, lora, declared_age=declared, gate=_gate())


def test_the_report_counts_only_18_19_and_20_and_ignores_a_bad_binding(tmp_path: Path) -> None:
    images = tmp_path / "images"
    _images(images)
    token = encode(19, _TENSORS)
    assert token is not None
    lora = tmp_path / "ada.safetensors"
    _file(lora, _TENSORS, token)
    applied = {"frames": 3, "gate_drops": 2, "alternate_rule_drops": 1, "disagreements": 1}
    plain = {"frames": 3, "gate_drops": 2, "alternate_rule_drops": 2, "disagreements": 0}
    for value in (18, 19, 20):
        assert _counts(images, lora, value) == applied
    assert _counts(images, lora, None) == applied
    for value in (17, 21, 99):
        assert _counts(images, lora, value) == plain
    _file(lora, _OTHER, token)
    assert _counts(images, lora, None) == plain
    assert _counts(images, lora, 19) == applied
    report = tmp_path / "out" / "report.json"
    write_report(report, applied)
    body = report.read_text(encoding="utf-8")
    assert json.loads(body) == applied
    assert "19" not in body
    assert set(json.loads(body)) == {"frames", "gate_drops", "alternate_rule_drops", "disagreements"}


def test_the_report_is_not_written_under_the_state_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    images = tmp_path / "images"
    _images(images)
    lora = tmp_path / "ada.safetensors"
    lora.write_bytes(b"not a lora")
    blocked = tmp_path / "home" / "state" / "canvas" / "report.json"
    assert main([str(images), str(lora), "--report", str(blocked)]) == 2
    assert not blocked.exists()
    assert not (tmp_path / "home" / "state").exists()
    seen: dict[str, object] = {}

    def fake_report(*args: object, **kwargs: object) -> dict[str, int]:
        del args
        seen["age"] = kwargs.get("declared_age")
        return {"frames": 0, "gate_drops": 0, "alternate_rule_drops": 0, "disagreements": 0}

    monkeypatch.setattr("tools.declared_age_shadow.build_report", fake_report)
    report = tmp_path / "out" / "report.json"
    assert main([str(images), str(lora), "--report", str(report), "--declared-age", "19"]) == 0
    assert seen["age"] == 19
    assert report.is_file()
    assert not (tmp_path / "home" / "state").exists()


def test_canvas_does_not_import_the_shadow_tool() -> None:
    for path in Path("canvas").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all("declared_age" not in alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert "declared_age" not in node.module
