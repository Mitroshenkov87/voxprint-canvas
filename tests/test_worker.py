"""The worker re-screens, writes no image files, and the parent reviews the result."""
import base64
import io
import json
import sys

from canvas.core.formats import save_image
from canvas.pipeline.worker import handle_message, main, release_worker_image, worker_generate
from canvas.safety.classifiers import Face
from canvas.safety.gate import GenerationBlocked
from canvas.safety.image import ScreenedInput
from tests.fakes import World, rgb


class _Gen:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, screened: ScreenedInput):
        self.calls += 1
        assert type(screened) is ScreenedInput
        return rgb(color=(4, 5, 6))


def test_worker_screens_again_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    world = World()
    gen = _Gen()
    image = worker_generate(world.gate(), gen, "a red boat", "text", [])
    assert gen.calls == 1
    assert image.to_bytes() == bytes([4, 5, 6]) * 4
    image.wipe()
    blocked = _Gen()
    result = handle_message({"prompt": "nude child", "mode": "text"}, world.gate(), blocked)
    assert result["ok"] is False
    assert result["message"] == "This request cannot be completed."
    assert "data_b64" not in result
    assert blocked.calls == 0
    assert list(tmp_path.iterdir()) == []


def test_parent_reviews_worker_bytes_before_anything_is_kept(tmp_path):
    world = World()
    payload = handle_message({"prompt": "a red boat", "mode": "text"}, world.gate(), _Gen())
    assert payload["ok"] is True
    released = release_worker_image(world.gate(), payload)
    path = tmp_path / "out.avif"
    save_image(released, path)
    assert path.read_bytes().startswith(b"VOXPRINT avif\n")
    world.nsfw = 0.99
    world.faces = [Face(0, 0, 1, 1)]
    world.ages = [6.0]
    secret = bytes([9, 1, 2])
    blocked = {
        "ok": True,
        "width": 1,
        "height": 1,
        "data_b64": base64.b64encode(secret).decode("ascii"),
    }
    refused = tmp_path / "refused.avif"
    try:
        release_worker_image(world.gate(), blocked)
    except GenerationBlocked:
        pass
    else:
        raise AssertionError("expected a refusal")
    assert not refused.exists()


def test_worker_process_refuses_without_pins(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"prompt": "a red boat", "mode": "text"}'))
    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["message"] == "Image generation is unavailable."
    images = [path for path in tmp_path.rglob("*") if path.suffix.lower() in {".avif", ".png", ".jpg", ".webp", ".jxl", ".heic"}]
    assert images == []
