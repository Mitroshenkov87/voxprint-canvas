"""Prompt rewrite: keep stated ages, screen both texts, unload before diffusion."""
import json

import pytest

from canvas.app.main import main
from canvas.app.shell_run import improve_prompt
from canvas.app.platform import OsCheck
from canvas.infra.gpu import GpuInfo
from canvas.models.modes import Tier
from canvas.pipeline.enhance import (
    SYSTEM_PROMPT,
    EnhanceResult,
    _visible_reply,
    causal_lm_ready,
    resolve_prompt,
    rewrite_text,
    stated_ages,
)
from canvas.safety.gate import GenerationBlocked
from tests.fakes import World


class _Rewriter:
    def __init__(self, text: str) -> None:
        self.text = text
        self.closed = False
        self.calls = 0

    def rewrite(self, prompt: str) -> str:
        del prompt
        self.calls += 1
        return self.text

    def close(self) -> None:
        self.closed = True


def test_generate_help_offers_enhance(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["generate", "--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out.casefold()
    assert "--enhance" in text
    assert "translat" not in text
    assert "nsfw" not in text
    assert "classifier" not in text


def test_system_prompt_translates_expands_and_keeps_facts():
    folded = SYSTEM_PROMPT.casefold()
    for phrase in (
        "natural english",
        "subject",
        "composition",
        "lighting",
        "lens",
        "style",
        "verbatim",
        "age",
        "adult",
        "youthful",
        "do not add people",
        "original language",
        "script",
    ):
        assert phrase in folded


def test_tied_embeddings_are_enough_for_generation():
    config = {"architectures": ["Qwen3ForCausalLM"], "tie_word_embeddings": True}
    names = {"model.embed_tokens.weight", "model.norm.weight"}
    assert causal_lm_ready(config, names) is True
    assert causal_lm_ready(config, {"model.norm.weight"}) is False
    untied = {"architectures": ["Qwen3ForCausalLM"], "tie_word_embeddings": False}
    assert causal_lm_ready(untied, names) is False
    assert causal_lm_ready(untied, {"lm_head.weight"}) is True
    assert causal_lm_ready({"architectures": ["Qwen3Model"], "tie_word_embeddings": True}, names) is False


def test_numeric_and_word_ages_match_across_a_translation():
    assert stated_ages("a 25-year-old woman") == (25,)
    assert stated_ages("a twenty-five-year-old woman") == (25,)
    assert stated_ages("женщина 40 лет") == (40,)
    assert stated_ages("a woman in her forties") == (40,)
    kept = resolve_prompt(
        World().gate(),
        "портрет 25-летней женщины",
        "portrait of a twenty-five-year-old woman, soft window light, 50mm lens",
    )
    assert kept.used_rewrite is True
    assert kept.prompt.startswith("portrait")


def test_a_rewrite_that_drops_or_changes_an_age_falls_back():
    original = "a 25-year-old woman in a black coat"
    dropped = resolve_prompt(World().gate(), original, "a woman in a black coat, studio light")
    assert dropped == EnhanceResult(original, original, "a woman in a black coat, studio light", False, "age")
    changed = resolve_prompt(World().gate(), original, "a 30-year-old woman in a black coat")
    assert changed.used_rewrite is False
    assert changed.prompt == original
    assert changed.fallback == "age"
    word = resolve_prompt(World().gate(), "a twenty-five-year-old woman", "a woman beside a window")
    assert word.fallback == "age"
    assert word.prompt == "a twenty-five-year-old woman"
    invented = resolve_prompt(World().gate(), "a woman in a studio", "a 40-year-old woman in a studio")
    assert invented.fallback == "age"


def test_quoted_signage_must_survive_in_the_original_script():
    original = 'a shop front with a sign "Аптека"'
    dropped = resolve_prompt(World().gate(), original, "a shop front with a painted sign, dusk, 35mm lens")
    assert dropped.fallback == "quote"
    assert dropped.prompt == original
    kept = resolve_prompt(
        World().gate(),
        original,
        'a shop front at dusk, 35mm lens, a sign reading "Аптека"',
    )
    assert kept.used_rewrite is True


def test_blocked_rewrite_or_original_stops_the_request():
    gate = World().gate()
    with pytest.raises(GenerationBlocked):
        resolve_prompt(gate, "a red boat", "nude child")
    with pytest.raises(GenerationBlocked):
        resolve_prompt(gate, "nude child", "a red boat on a lake")
    adult = resolve_prompt(gate, "a nude woman on a sofa", "a nude woman on a sofa, window light, 50mm lens")
    assert adult.used_rewrite is True


def test_improve_returns_the_rewrite_or_keeps_the_original(monkeypatch):
    gate = World().gate()
    monkeypatch.setattr("canvas.app.shell_run.selectable", lambda: True)
    monkeypatch.setattr(
        "canvas.app.shell_run.open_rewriter",
        lambda *args, **kwargs: _Rewriter("a red boat on calm water, 50mm lens"),
    )
    assert improve_prompt(gate=gate, prompt="красная лодка", tier=Tier.GB16) == "a red boat on calm water, 50mm lens"
    monkeypatch.setattr(
        "canvas.app.shell_run.open_rewriter",
        lambda *args, **kwargs: _Rewriter("a woman in a black coat"),
    )
    kept = improve_prompt(gate=gate, prompt="a 25-year-old woman in a black coat", tier=Tier.GB16)
    assert kept == "a 25-year-old woman in a black coat"
    monkeypatch.setattr("canvas.app.shell_run.open_rewriter", lambda *args, **kwargs: _Rewriter("nude child"))
    with pytest.raises(GenerationBlocked):
        improve_prompt(gate=gate, prompt="a red boat", tier=Tier.GB16)


def test_a_reasoning_trace_is_not_the_rewritten_prompt():
    assert _visible_reply("<think>plan the picture</think>\na red boat, 50mm lens") == "a red boat, 50mm lens"
    assert _visible_reply("<think>still thinking") == ""


def test_rewrite_unloads_before_the_caller_continues():
    rewriter = _Rewriter("a red boat on calm water")
    text = rewrite_text(rewriter, "красная лодка")
    assert text == "a red boat on calm water"
    assert rewriter.closed is True
    assert rewriter.calls == 1


def test_enhance_flag_unloads_before_diffusion_and_json_returns_both_prompts(monkeypatch, tmp_path, capsys):
    order: list[str] = []
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr(
        "canvas.app.main.probe_gpu",
        lambda: GpuInfo("RTX 4060 Ti", 16, 14, 8, 9, "fake"),
    )
    monkeypatch.setattr("canvas.app.main.install_supporting", lambda *args, **kwargs: order.append("supporting"))
    monkeypatch.setattr("canvas.app.main.load_catalog_stack", lambda *args, **kwargs: order.append("stack") or object())

    class FakeScreened:
        def wipe(self) -> None:
            order.append("wipe")

    class FakeGate:
        def __init__(self, stack: object) -> None:
            order.append("gate")

        def warm(self) -> None:
            order.append("warm")

        def review_prompt(self, text: str) -> None:
            order.append(f"review:{text}")

        def release_request(self, *args: object, **kwargs: object) -> FakeScreened:
            order.append("release")
            return FakeScreened()

    class FakePipeline:
        def __init__(self, *args: object, **kwargs: object) -> None:
            order.append("pipeline")

        def export(self, prompt: str, *args: object, **kwargs: object) -> None:
            order.append(f"export:{prompt}")

    class FakeGenerator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.seed = 0
            self.models = None

    monkeypatch.setattr("canvas.app.main.AgeGate", FakeGate)
    monkeypatch.setattr("canvas.app.main.build_plan", lambda *args, **kwargs: order.append("plan") or object())
    monkeypatch.setattr("canvas.app.main.ensure_plan", lambda *args, **kwargs: order.append("ensure"))
    monkeypatch.setattr("canvas.app.main.ZImageGenerator", FakeGenerator)
    monkeypatch.setattr("canvas.app.main.Pipeline", FakePipeline)
    monkeypatch.setattr("canvas.app.main.EnlargeClient", lambda *args, **kwargs: object())
    class RecordingRewriter(_Rewriter):
        def close(self) -> None:
            super().close()
            order.append("unload")

    monkeypatch.setattr(
        "canvas.app.main.open_rewriter",
        lambda *args, **kwargs: RecordingRewriter("a red boat on calm water, 50mm lens"),
    )
    code = main([
        "generate", "--prompt", "красная лодка", "--aspect", "1:1",
        "--out", str(tmp_path / "out.avif"), "--enhance", "--json",
    ])
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert body["ok"] is True
    assert body["prompt"] == "красная лодка"
    assert body["enhanced_prompt"] == "a red boat on calm water, 50mm lens"
    assert order.index("unload") < order.index("export:a red boat on calm water, 50mm lens")
    assert any(item.startswith("review:") for item in order)


def test_enhance_age_mismatch_falls_back_and_still_returns_both_prompts(monkeypatch, tmp_path, capsys):
    seen: list[str] = []
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr(
        "canvas.app.main.probe_gpu",
        lambda: GpuInfo("RTX 4090", 24, 20, 8, 9, "fake"),
    )
    monkeypatch.setattr("canvas.app.main.install_supporting", lambda *args, **kwargs: None)
    monkeypatch.setattr("canvas.app.main.load_catalog_stack", lambda *args, **kwargs: object())

    class FakeScreened:
        def wipe(self) -> None:
            return None

    class FakeGate:
        def __init__(self, stack: object) -> None:
            return None

        def warm(self) -> None:
            return None

        def review_prompt(self, text: str) -> None:
            return None

        def release_request(self, *args: object, **kwargs: object) -> FakeScreened:
            return FakeScreened()

    class FakePipeline:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def export(self, prompt: str, *args: object, **kwargs: object) -> None:
            seen.append(prompt)

    class FakeGenerator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.seed = 0
            self.models = None

    monkeypatch.setattr("canvas.app.main.AgeGate", FakeGate)
    monkeypatch.setattr("canvas.app.main.build_plan", lambda *args, **kwargs: object())
    monkeypatch.setattr("canvas.app.main.ensure_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr("canvas.app.main.ZImageGenerator", FakeGenerator)
    monkeypatch.setattr("canvas.app.main.Pipeline", FakePipeline)
    monkeypatch.setattr("canvas.app.main.EnlargeClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "canvas.app.main.open_rewriter",
        lambda *args, **kwargs: _Rewriter("a woman in a black coat"),
    )
    original = "a 25-year-old woman in a black coat"
    code = main([
        "generate", "--prompt", original, "--out", str(tmp_path / "out.avif"), "--enhance", "--json",
    ])
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert body["prompt"] == original
    assert body["enhanced_prompt"] == "a woman in a black coat"
    assert body["used_prompt"] == original
    assert body["fallback"] == "age"
    assert seen == [original]


def test_a_blocked_rewrite_does_not_generate(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr(
        "canvas.app.main.probe_gpu",
        lambda: GpuInfo("RTX 4090", 24, 20, 8, 9, "fake"),
    )
    monkeypatch.setattr("canvas.app.main.install_supporting", lambda *args, **kwargs: None)
    monkeypatch.setattr("canvas.app.main.load_catalog_stack", lambda *args, **kwargs: object())
    exported: list[str] = []

    class FakeScreened:
        def wipe(self) -> None:
            return None

    class FakeGate:
        def __init__(self, stack: object) -> None:
            self.real = World().gate()

        def warm(self) -> None:
            return None

        def review_prompt(self, text: str) -> None:
            self.real.review_prompt(text)

        def release_request(self, *args: object, **kwargs: object) -> FakeScreened:
            return FakeScreened()

    class FakePipeline:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def export(self, prompt: str, *args: object, **kwargs: object) -> None:
            exported.append(prompt)

    class FakeGenerator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.seed = 0
            self.models = None

    monkeypatch.setattr("canvas.app.main.AgeGate", FakeGate)
    monkeypatch.setattr("canvas.app.main.build_plan", lambda *args, **kwargs: object())
    monkeypatch.setattr("canvas.app.main.ensure_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr("canvas.app.main.ZImageGenerator", FakeGenerator)
    monkeypatch.setattr("canvas.app.main.Pipeline", FakePipeline)
    monkeypatch.setattr("canvas.app.main.EnlargeClient", lambda *args, **kwargs: object())
    monkeypatch.setattr("canvas.app.main.open_rewriter", lambda *args, **kwargs: _Rewriter("nude child"))
    code = main([
        "generate", "--prompt", "a red boat", "--out", str(tmp_path / "out.avif"), "--enhance", "--json",
    ])
    assert code == 1
    body = json.loads(capsys.readouterr().out)
    assert body["ok"] is False
    assert exported == []
