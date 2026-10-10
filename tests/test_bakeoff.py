"""The bake-off uses one prompt list, one seed list, and the same content checks."""
import json
from pathlib import Path

import pytest

from canvas.models.loader import ModelNotReady
from canvas.models.modes import Tier
from canvas.models.plan import CallPlan
from canvas.models.registry import load_catalog
from canvas.safety.classifiers import Face
from canvas.safety.gate import SafetyUnavailable
from canvas.safety.image import ScreenedInput
from tests.fakes import World, rgb
from tools.bakeoff import PROMPTS, SEEDS, main, run_bakeoff


def _plan() -> CallPlan:
    return CallPlan("0.41.0", "QwenImagePipeline", ("Qwen/Qwen-Image-2512",), "fp8", 50, 4.0, "", "24gb", "text")


def test_same_prompts_and_seeds_and_kept_images_stay_in_temp(tmp_path: Path):
    world = World()
    seen: list[tuple[str, str, int]] = []

    def runner(base: str, prompt: str, seed: int, screened: ScreenedInput):
        assert type(screened) is ScreenedInput
        seen.append((base, prompt, seed))
        return rgb()

    rows = run_bakeoff(
        load_catalog(),
        world.gate(),
        Tier.GB24,
        runner=runner,
        out_dir=tmp_path,
        vram=lambda: 140,
    )
    pairs = [(prompt, seed) for prompt in PROMPTS for seed in SEEDS]
    by_base: dict[str, list[tuple[str, int]]] = {}
    for row in rows:
        by_base.setdefault(row.base, []).append((row.prompt, row.seed))
    assert list(by_base) == ["z-image-turbo", "qwen-image-2512"]
    assert by_base["z-image-turbo"] == by_base["qwen-image-2512"] == pairs
    assert [(base, prompt, seed) for base, prompt, seed in seen] == [
        (base, prompt, seed) for base in ("z-image-turbo", "qwen-image-2512") for prompt, seed in pairs
    ]
    assert {row.status for row in rows} == {"kept"}
    assert {row.peak_vram_mb for row in rows} == {140}
    repo = Path(".").resolve()
    for row in rows:
        path = Path(row.path)
        assert path.is_file()
        assert path.resolve().is_relative_to(tmp_path.resolve())
        assert repo not in path.resolve().parents
    report = json.loads((tmp_path / "bakeoff.json").read_text(encoding="utf-8"))
    assert len(report) == len(rows)
    assert "guard" in world.calls


def test_a_refused_image_is_not_written(tmp_path: Path):
    world = World()
    world.nsfw = 0.95
    world.faces = [Face(0, 0, 1, 1)]
    world.ages = [8.0]

    def runner(base: str, prompt: str, seed: int, screened: ScreenedInput):
        del base, prompt, seed, screened
        return rgb(color=(9, 9, 9))

    rows = run_bakeoff(load_catalog(), world.gate(), Tier.GB16, runner=runner, out_dir=tmp_path, vram=lambda: 10)
    assert {row.status for row in rows} == {"refused"}
    assert all(row.path == "" for row in rows)
    binaries = list(tmp_path.glob("*.bin"))
    assert binaries == []


def test_a_missing_generator_records_not_ready(tmp_path: Path):
    world = World()

    def runner(base: str, prompt: str, seed: int, screened: ScreenedInput):
        del base, prompt, seed, screened
        raise ModelNotReady(_plan())

    rows = run_bakeoff(load_catalog(), world.gate(), Tier.GB24, runner=runner, out_dir=tmp_path)
    assert {row.status for row in rows} == {"not_ready"}
    assert list(tmp_path.glob("*.bin")) == []


def test_images_cannot_be_written_into_the_repo():
    world = World()
    with pytest.raises(ValueError):
        run_bakeoff(
            load_catalog(),
            world.gate(),
            Tier.GB16,
            runner=lambda base, prompt, seed, screened: rgb(),
            out_dir=Path("."),
        )


def test_main_stops_when_the_checks_cannot_load(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))

    def explode(root):
        del root
        raise SafetyUnavailable("missing_model")

    monkeypatch.setattr("tools.bakeoff.load_catalog_stack", explode)
    assert main([]) == 2
    assert capsys.readouterr().out.strip() == "Image generation is unavailable."
