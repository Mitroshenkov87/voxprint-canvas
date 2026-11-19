"""``canvas generate --count`` loads the model once and gates every picture."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canvas.app.generate_batch import GeneratePlanError, item_path, load_prompt_file, parse_seeds, plan_items
from canvas.app.main import main
from canvas.app.platform import GpuCheck, OsCheck
from canvas.infra.gpu import GpuInfo
from canvas.models.modes import Tier
from canvas.safety.gate import GenerationBlocked


def test_plan_items_counts_up_from_a_seed_and_accepts_a_seed_list() -> None:
    items = plan_items(prompt="a red boat", count=3, seed=10)
    assert [item.seed for item in items] == [10, 11, 12]
    assert [item.index for item in items] == [1, 2, 3]
    listed = plan_items(prompt="a red boat", seeds=[1, 2, 3])
    assert [item.seed for item in listed] == [1, 2, 3]
    with pytest.raises(GeneratePlanError):
        plan_items(prompt="a red boat", seed=1, seeds=[2, 3])
    with pytest.raises(GeneratePlanError):
        plan_items(prompt="a red boat", count=0)
    with pytest.raises(GeneratePlanError):
        parse_seeds("1,nope")


def test_prompt_files_and_item_paths(tmp_path: Path) -> None:
    text_file = tmp_path / "prompts.txt"
    text_file.write_text("a red boat\na blue cup\n", encoding="utf-8")
    rows = load_prompt_file(text_file)
    assert rows == [("a red boat", None), ("a blue cup", None)]
    items = plan_items(prompts=rows, seed=5)
    assert [item.prompt for item in items] == ["a red boat", "a blue cup"]
    assert [item.seed for item in items] == [5, 6]
    jsonl = tmp_path / "prompts.jsonl"
    jsonl.write_text(
        '{"prompt": "one", "seed": 3}\n{"prompt": "two"}\n',
        encoding="utf-8",
    )
    loaded = load_prompt_file(jsonl)
    planned = plan_items(prompts=loaded, seed=9)
    assert planned[0].seed == 3 and planned[1].seed == 10
    path, _warning = item_path(tmp_path / "out.webp", planned[0], total=2, out_arg=str(tmp_path / "out.webp"))
    assert path.name == "out-001-3.webp"
    folder, _warning = item_path(tmp_path / "shots", planned[1], total=2, out_arg=str(tmp_path / "shots"))
    assert folder == tmp_path / "shots" / "002-10.webp"


def _stub_generate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, export) -> list[object]:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr("canvas.app.main.check_gpu", lambda _info: GpuCheck(True, "", Tier.GB24))
    monkeypatch.setattr(
        "canvas.app.main.probe_gpu",
        lambda: GpuInfo("RTX 4090", 23.99, 20, 8, 9, "fake"),
    )
    monkeypatch.setattr("canvas.app.main.install_supporting", lambda *args, **kwargs: None)
    monkeypatch.setattr("canvas.app.main.load_catalog_stack", lambda *args, **kwargs: object())
    constructed: list[object] = []

    class FakeScreened:
        def wipe(self) -> None:
            return None

    class FakeGate:
        def __init__(self, stack: object) -> None:
            return None

        def warm_prompt(self) -> None:
            return None

        def release_request(self, *args: object, **kwargs: object) -> FakeScreened:
            return FakeScreened()

        def review_prompt(self, prompt: str) -> None:
            return None

    class FakeGenerator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.seed = 0
            self.models = None
            constructed.append(self)

    class FakePipeline:
        def __init__(self, gate: object, generator: object, *args: object, **kwargs: object) -> None:
            self.generator = generator
            constructed.append(self)

        def export(self, *args: object, **kwargs: object) -> None:
            export(self, *args, **kwargs)

    monkeypatch.setattr("canvas.app.main.AgeGate", FakeGate)
    monkeypatch.setattr("canvas.app.main.build_plan", lambda *args, **kwargs: object())
    monkeypatch.setattr("canvas.app.main.ensure_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr("canvas.app.main.ZImageGenerator", FakeGenerator)
    monkeypatch.setattr("canvas.app.main.Pipeline", FakePipeline)
    monkeypatch.setattr("canvas.app.main.EnlargeClient", lambda *args, **kwargs: object())
    return constructed


def test_count_loads_the_model_once_and_writes_indexed_webp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    exported: list[tuple[int, Path]] = []

    def export(self, *args: object, **kwargs: object) -> None:
        path = Path(args[3])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")
        exported.append((int(self.generator.seed), path))

    constructed = _stub_generate(monkeypatch, tmp_path, export)
    out = tmp_path / "boat.webp"
    code = main([
        "generate", "--prompt", "a red boat", "--out", str(out),
        "--count", "3", "--seed", "10", "--json",
    ])
    captured = capsys.readouterr()
    body = json.loads(captured.out)
    assert code == 0
    assert body["ok"] is True and body["written"] == 3 and body["refused"] == 0
    assert [row["seed"] for row in body["items"]] == [10, 11, 12]
    assert [path.name for _seed, path in exported] == [
        "boat-001-10.webp", "boat-002-11.webp", "boat-003-12.webp",
    ]
    assert all(path.suffix == ".webp" and path.is_file() for _seed, path in exported)
    # One generator and one pipeline; three exports.
    assert sum(1 for item in constructed if item.__class__.__name__ == "FakeGenerator") == 1
    assert sum(1 for item in constructed if item.__class__.__name__ == "FakePipeline") == 1
    assert "1/3 seed=10 wrote" in captured.err


def test_a_refused_item_is_skipped_and_the_batch_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    def export(self, *args: object, **kwargs: object) -> None:
        prompt = str(args[0])
        if "blocked" in prompt:
            raise GenerationBlocked("prompt")
        path = Path(args[3])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")

    _stub_generate(monkeypatch, tmp_path, export)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a blocked prompt\na red boat\n", encoding="utf-8")
    folder = tmp_path / "shots"
    code = main([
        "generate", "--prompts", str(prompts), "--out", str(folder),
        "--seeds", "1,2", "--json",
    ])
    body = json.loads(capsys.readouterr().out)
    assert code == 1
    assert body["ok"] is False and body["written"] == 1 and body["refused"] == 1
    assert body["items"][0]["status"] == "refused" and body["items"][0]["path"] == ""
    assert body["items"][1]["ok"] is True and body["items"][1]["seed"] == 2
    assert (folder / "002-2.webp").is_file()
    assert not (folder / "001-1.webp").exists()


def test_single_generate_json_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    def export(self, *args: object, **kwargs: object) -> None:
        Path(args[3]).write_bytes(b"ok")

    _stub_generate(monkeypatch, tmp_path, export)
    out = tmp_path / "one.webp"
    code = main(["generate", "--prompt", "a red boat", "--out", str(out), "--seed", "7", "--json"])
    body = json.loads(capsys.readouterr().out)
    assert code == 0
    assert body["ok"] is True and body["path"] == str(out) and body["seed"] == 7
    assert "items" not in body
    assert out.is_file()
