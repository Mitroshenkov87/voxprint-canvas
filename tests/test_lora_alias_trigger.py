"""rc.41: LoRA alias, trigger auto-insert, recommended strength, Photoshoot trigger lead, writer retry.

CPU only. Fakes stand in for the classifiers and the picture model. The existing
gate tests are not touched: the "gate still sees the final prompt" cases here
drive a recording stub in front of the unchanged gate.
"""
from __future__ import annotations

import json
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from canvas.models.user_lora import (
    ALIAS_MAX,
    apply_trigger,
    chip_word,
    inspect_folder,
    installed_loras,
    parse_alias,
    parse_recommended_strength,
    prepare_cli_lora,
    resolve_aliases,
    rewrite_alias,
)
from canvas.pipeline.photoshoot import (
    ADULT_FICTION_CONTEXT,
    ENCODER_TOKEN_LIMIT,
    Shot,
    assemble_line,
    truncate_for_encoder,
)
from canvas.safety.gate import AgeGate
from tests.fakes import World
from tests.lora_files import write_lora, write_safetensors
from tests.test_photoshoot import IDEA, _frame, _lines, _run

TRIGGER = "zqxv7ada"
ALIAS = "vox_ada"
_A = "layers.0.attention.to_q.lora_A.weight"
_B = "layers.0.attention.to_q.lora_B.weight"


def _folder_lora(tmp_path: Path, lora_id: str = "ada-1", **meta: str):
    write_lora(tmp_path, lora_id, trigger=TRIGGER, metadata=dict(meta))
    found, reason = inspect_folder(tmp_path / "user-models" / "lora" / lora_id)
    assert found is not None, reason
    return found


# --- alias and strength parsing ---------------------------------------------------------------


def test_alias_must_be_vox_slug_lowercase_ascii_at_most_32() -> None:
    assert parse_alias("vox_ada") == "vox_ada"
    assert parse_alias("vox_a1_b2") == "vox_a1_b2"
    assert parse_alias("vox_" + "a" * (ALIAS_MAX - 4)) != ""
    for bad in ("vox_" + "a" * (ALIAS_MAX - 3), "Vox_ada", "vox_Ada", "ada", "vox_", "vox-ada", "vox_adá", "", None, 7, ["vox_a"]):
        assert parse_alias(bad) == ""


def test_recommended_strength_reads_a_decimal_string() -> None:
    assert parse_recommended_strength("1.0") == 1.0
    assert parse_recommended_strength(" 0.75 ") == 0.75
    for bad in ("", "abc", "1,0", "-1", "nan", "inf", "1e3", None, 1.0, "0", "9"):
        assert parse_recommended_strength(bad) is None


def test_header_alias_and_strength_reach_the_lora(tmp_path: Path) -> None:
    lora = _folder_lora(tmp_path, **{
        "voxprint.subject.alias": ALIAS, "voxprint.lora.recommended_strength": "1.0",
    })
    assert lora.trigger == TRIGGER
    assert lora.alias == ALIAS
    # The sidecar says 0.8; the weight header's recommended strength is the default.
    assert lora.default_strength == 1.0
    other = _folder_lora(tmp_path, "ada-2", **{"voxprint.lora.recommended_strength": "0.65"})
    assert other.default_strength == 0.65


def test_a_bad_alias_is_treated_as_absent(tmp_path: Path) -> None:
    lora = _folder_lora(tmp_path, **{"voxprint.subject.alias": "Not An Alias"})
    assert lora.alias == ""
    assert lora.trigger == TRIGGER


def _bare(tmp_path: Path, **meta: str):
    path = tmp_path / "kettle.safetensors"
    header = {"voxprint.lora.base": "z-image-turbo", "voxprint.subject.trigger": TRIGGER, **meta}
    write_safetensors(path, {_A: b"\x01\x02", _B: b"\x03\x04"}, header)
    return prepare_cli_lora(path)


def test_recommended_strength_default_and_the_fall_back_to_one(tmp_path: Path) -> None:
    assert _bare(tmp_path, **{"voxprint.lora.recommended_strength": "0.9"}).default_strength == 0.9
    assert _bare(tmp_path).default_strength == 1.0
    assert _bare(tmp_path, **{"voxprint.lora.recommended_strength": "not a number"}).default_strength == 1.0


# --- rewrite and insert ---------------------------------------------------------------------


def _aliased():
    return types.SimpleNamespace(trigger=TRIGGER, alias=ALIAS)


def test_alias_is_rewritten_to_the_trigger_and_the_trigger_leads() -> None:
    lora = _aliased()
    assert apply_trigger(f"a portrait of {ALIAS} by a window", lora) == f"a portrait of {TRIGGER} by a window"
    assert apply_trigger("a red boat", lora) == f"{TRIGGER}, a red boat"
    assert apply_trigger(f"{ALIAS.upper()} smiling", lora) == f"{TRIGGER} smiling"
    # A longer word that merely starts with the alias is not the alias.
    assert apply_trigger(f"{ALIAS}_2 and a boat", lora) == f"{TRIGGER}, {ALIAS}_2 and a boat"


def test_the_trigger_is_not_duplicated() -> None:
    lora = _aliased()
    once = apply_trigger(f"{TRIGGER}, a red boat", lora)
    assert once == f"{TRIGGER}, a red boat"
    assert apply_trigger(once, lora) == once
    mixed = apply_trigger(f"{ALIAS} and {TRIGGER}", lora)
    assert mixed.count(TRIGGER) == 2 and ALIAS not in mixed
    assert apply_trigger(apply_trigger("x", lora), lora) == apply_trigger("x", lora)


def test_a_legacy_lora_without_an_alias_is_auto_insert_only(tmp_path: Path) -> None:
    lora = _folder_lora(tmp_path)
    assert lora.alias == ""
    assert chip_word(lora) == TRIGGER
    # Nothing is derived from the file name (the id is "ada-1").
    assert apply_trigger("a red boat", lora) == f"{TRIGGER}, a red boat"
    assert "ada-1" not in apply_trigger("a red boat", lora)
    assert rewrite_alias("vox_ada-1", lora) == "vox_ada-1"
    assert chip_word(_aliased()) == ALIAS
    assert chip_word(None) == ""


def test_no_lora_or_no_trigger_changes_nothing() -> None:
    assert apply_trigger("a red boat", None) == "a red boat"
    assert apply_trigger("a red boat", types.SimpleNamespace(trigger="", alias="")) == "a red boat"


# --- uniqueness -------------------------------------------------------------------------------


def test_clashing_aliases_get_a_suffix_and_a_note(tmp_path: Path) -> None:
    for lora_id in ("a-first", "b-second", "c-third"):
        write_lora(tmp_path, lora_id, trigger=f"tr{lora_id[0]}x9", metadata={"voxprint.subject.alias": ALIAS})
    write_lora(tmp_path, "d-other", trigger="trdx9", metadata={"voxprint.subject.alias": "vox_other"})
    loras, notes = installed_loras(tmp_path, library=tmp_path / "no-library")
    aliases = {item.id: item.alias for item in loras}
    assert aliases == {
        "a-first": ALIAS, "b-second": f"{ALIAS}_2", "c-third": f"{ALIAS}_3", "d-other": "vox_other",
    }
    assert notes == (("b-second", ALIAS, f"{ALIAS}_2"), ("c-third", ALIAS, f"{ALIAS}_3"))
    # Each alias now reaches its own trigger. Nothing is shadowed.
    by_id = {item.id: item for item in loras}
    assert apply_trigger(f"{ALIAS}_2 portrait", by_id["b-second"]) == "trbx9 portrait"
    assert len({item.alias for item in loras}) == len(loras)


@dataclass(frozen=True)
class _Item:
    id: str
    alias: str


def test_a_suffix_keeps_the_alias_inside_the_length_limit() -> None:
    long_alias = "vox_" + "a" * (ALIAS_MAX - 4)
    out, notes = resolve_aliases((_Item("1", long_alias), _Item("2", long_alias)))  # type: ignore[arg-type]
    assert out[0].alias == long_alias
    assert len(out[1].alias) <= ALIAS_MAX and out[1].alias.endswith("_2") and out[1].alias != out[0].alias
    assert notes == (("2", long_alias, out[1].alias),)


def test_the_selected_lora_carries_the_unique_alias(tmp_path: Path) -> None:
    from canvas.models.user_lora import resolve_selected

    write_lora(tmp_path, "a-first", trigger="trax9", metadata={"voxprint.subject.alias": ALIAS})
    write_lora(tmp_path, "b-second", trigger="trbx9", metadata={"voxprint.subject.alias": ALIAS})
    chosen = resolve_selected("b-second", tmp_path)
    assert chosen is not None and chosen.alias == f"{ALIAS}_2"


# --- gate and encoder see the final prompt ---------------------------------------------------


class _Recorder(AgeGate):
    """The unchanged gate with a note of every prompt it is asked to review."""

    def __init__(self) -> None:
        super().__init__(World().stack())
        self.prompts: list[str] = []

    def review_prompt(self, text: str):
        self.prompts.append(text)
        return super().review_prompt(text)


def test_the_worker_gate_and_runner_get_the_rewritten_prompt(tmp_path: Path) -> None:
    from canvas.app.shell_run import RunStopped
    from canvas.core.formats import OutputFormat
    from canvas.models.modes import Mode
    from canvas.ui.shell_layout import PictureJob
    from canvas.worker_jobs import Session, _generate

    lora = _folder_lora(tmp_path, **{"voxprint.subject.alias": ALIAS})
    reviewed: list[str] = []
    seen: list[str] = []

    class Check:
        def review_prompt(self, text: str):
            reviewed.append(text)
            return None

    class Runner:
        bound_lora = lora

        def __call__(self, job, inputs):
            seen.append(job.prompt)
            raise RunStopped("stop after the prompt")

    job = PictureJob(
        f"{ALIAS} reading on a bench", "", Mode.TEXT, "1:1", "native", OutputFormat.PNG, 1, 8, 0.0, "euler", 1,
        lora_path=str(lora.weights), lora_weight=1.0,
    )
    fields = job._asdict()
    fields["mode"] = job.mode.value
    fields["fmt"] = job.fmt.value
    ctx = types.SimpleNamespace(progress=lambda **_kw: None, cancelled=lambda: False)
    current = Session(gate=None, check=Check(), tier=None, runner=Runner(), improver=lambda **_kw: "")
    payload = {"job": fields, "layout_prompt": f"{ALIAS} reading", "out_dir": str(tmp_path / "out")}
    reply = _generate(ctx, current, payload)
    assert reply.get("status") != "ok"
    final = f"{TRIGGER} reading on a bench"
    assert seen == [final]
    assert final in reviewed and f"{TRIGGER} reading" in reviewed
    assert all(ALIAS not in item for item in reviewed + seen)


def test_the_batch_session_rewrites_before_the_gate_hook(tmp_path: Path) -> None:
    from canvas.tools.render_batch import _Session, Hooks

    lora = _folder_lora(tmp_path, **{"voxprint.subject.alias": ALIAS})
    got: list[str] = []

    def generate(prompt: str, seed: int, *, explicit: bool):
        got.append(prompt)
        return None

    session = _Session(Hooks(generate=generate), adult=False, aspect="1:1", lora_bind=(lora, 1.0))
    session.generate(f"{ALIAS} on a bench", 1)
    session.generate(f"{TRIGGER}, already there", 2)
    assert got == [f"{TRIGGER} on a bench", f"{TRIGGER}, already there"]


# --- Photoshoot ----------------------------------------------------------------------------


def test_the_trigger_leads_a_shot_line_after_truncation() -> None:
    lora = _aliased()
    shot = Shot("close-up, soft light", "seated", "85 mm", "a loft")
    subject = "a woman, 34 years old, in a red coat, " + "wearing a long elaborate costume made of silk, " * 90
    plain = assemble_line(subject, shot, include_camera=True)
    assert len(plain.split()) <= ENCODER_TOKEN_LIMIT - 48 and TRIGGER not in plain
    line = assemble_line(subject, shot, include_camera=True, lora=lora)
    assert line.startswith(f"{TRIGGER}, ")
    assert len(line.split()) <= ENCODER_TOKEN_LIMIT - 48
    # A line the cut would otherwise take the head from still starts with the trigger.
    cut = truncate_for_encoder(f"{TRIGGER}, " + " ".join(["word"] * 900))
    assert TRIGGER in cut
    assert assemble_line(f"{ALIAS}, " + subject, shot, include_camera=True, lora=lora).count(ALIAS) == 0


def _lora_generate(lora, drawn: list[str]):
    def generate(line: str, seed: int, hero: object, strength: object) -> object:
        drawn.append(line)
        return _frame()

    generate.lora = lora  # type: ignore[attr-defined]
    return generate


def test_photoshoot_lines_lead_with_the_trigger_and_the_gate_sees_them(tmp_path: Path) -> None:
    lora = types.SimpleNamespace(trigger=TRIGGER, alias=ALIAS)
    recorder = _Recorder()
    drawn: list[str] = []
    idea = f"{ALIAS}, {IDEA}"
    result = _run(
        tmp_path, idea=idea, count=4, age_gate=recorder, generate=_lora_generate(lora, drawn),
        rewrite=lambda prompt: _lines(int(json.loads(prompt.splitlines()[0])["count"])),
    )
    assert result.stopped == "" and len(result.kept) == 4
    assert len(drawn) == 4 and all(line.startswith(f"{TRIGGER}, ") for line in drawn)
    shot_prompts = [item for item in recorder.prompts if "\n" in item]
    assert len(shot_prompts) >= 4
    # plan_series reviews idea + line: the idea part is already rewritten and the line part leads with the trigger.
    for item in shot_prompts:
        first, _newline, line = item.partition("\n")
        assert ALIAS not in item
        assert TRIGGER in first
        assert line.startswith(f"{TRIGGER}, ")
    assert all(ALIAS not in item for item in recorder.prompts)


def test_without_a_lora_the_lines_do_not_change(tmp_path: Path) -> None:
    drawn: list[str] = []
    plain = _run(tmp_path / "a", generate=lambda line, seed, hero, strength: drawn.append(line) or _frame(), count=3,
                 rewrite=lambda prompt: _lines(int(json.loads(prompt.splitlines()[0])["count"])))
    assert plain.stopped == "" and drawn and all(TRIGGER not in line for line in drawn)


# --- writer retry and the template fallback -------------------------------------------------


@pytest.mark.parametrize("reply", ["Sorry, I cannot write that.", "Lorem ipsum, nothing you can use.", ""])
@pytest.mark.parametrize("adult", [True, False])
def test_a_declining_or_unusable_writer_is_asked_once_more_then_the_template_runs(
    tmp_path: Path, reply: str, adult: bool,
) -> None:
    calls: list[str] = []

    def rewrite(prompt: str) -> str:
        calls.append(prompt)
        return reply

    result = _run(tmp_path, count=5, adult=adult, rewrite=rewrite)
    assert len(calls) == 2
    assert ("context" in json.loads(calls[0].splitlines()[0])) is adult
    assert (ADULT_FICTION_CONTEXT in calls[1]) is adult
    assert result.stopped == "" and result.code == ""
    assert result.source == "planned"
    assert len(result.kept) == 5
    document = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert len(document["lines"]) == 5 and not document["blocked"]
    assert all(line.startswith("A single photograph") for line in document["lines"])


def test_the_template_scenario_is_still_reviewed_by_the_gate(tmp_path: Path) -> None:
    recorder = _Recorder()
    result = _run(tmp_path, count=3, adult=True, age_gate=recorder, rewrite=lambda _prompt: "I cannot help with that.")
    assert result.stopped == "" and len(result.kept) == 3
    lines = [item.partition("\n")[2] for item in recorder.prompts if "\n" in item]
    assert len(lines) >= 3 and all(line.startswith("A single photograph") for line in lines)


# --- batch screen and strength default --------------------------------------------------------


def test_the_batch_first_screen_and_the_writer_see_the_trigger_not_the_alias(tmp_path: Path) -> None:
    from canvas.app.batch_cli import run_batch_command
    from canvas.pipeline.batch import BatchEntry
    from tests.test_batch import _Events, _gate, _rewrite

    lora = types.SimpleNamespace(trigger=TRIGGER, alias=ALIAS)
    events = _Events()
    written: list[str] = []

    def rewrite(prompt: str) -> str:
        written.append(prompt)
        return _rewrite(prompt)

    def frame_for(_entry):
        def draw(line, seed, hero, strength):
            return _frame()

        draw.lora = lora  # type: ignore[attr-defined]
        return draw

    folder = tmp_path / "out"
    folder.mkdir()
    entries = [BatchEntry("a", f"{ALIAS}, {IDEA}", "", "", 1, 1), BatchEntry("b", "", f"{ALIAS} on a bench", "", 2, 1)]
    run_batch_command(
        entries=entries, folder=folder, style="photo", places="same", adult=False,
        age_gate=_gate(events), rewrite=rewrite, frame_for=frame_for,
    )
    checked = events.of("prompt_check")
    # Every prompt check, the first screen included, sees the rewritten and inserted text.
    assert checked and all(ALIAS not in item for item in checked)
    assert checked[0] == f"{TRIGGER}, {IDEA}"
    assert f"{TRIGGER} on a bench" in checked
    assert written and all(ALIAS not in item for item in written)


def test_a_missing_or_unreadable_recommended_strength_is_one_even_with_a_sidecar_default(tmp_path: Path) -> None:
    write_lora(tmp_path, "ada-1", trigger=TRIGGER, default_strength=0.8, metadata={})
    absent, _ = inspect_folder(tmp_path / "user-models" / "lora" / "ada-1")
    assert absent is not None and absent.default_strength == 1.0
    write_lora(tmp_path, "ada-2", trigger=TRIGGER, default_strength=0.8, metadata={"voxprint.lora.recommended_strength": "x"})
    bad, _ = inspect_folder(tmp_path / "user-models" / "lora" / "ada-2")
    assert bad is not None and bad.default_strength == 1.0
    write_lora(tmp_path, "ada-3", trigger=TRIGGER, default_strength=0.8, metadata={"voxprint.lora.recommended_strength": "0.5"})
    present, _ = inspect_folder(tmp_path / "user-models" / "lora" / "ada-3")
    assert present is not None and present.default_strength == 0.5
    # A non-Z-Image third-party file without the key also falls back to 1.0.
    assert _bare(tmp_path, **{"voxprint.lora.base": "sdxl"}).default_strength == 1.0


# --- GUI series carries the window LoRA -----------------------------------------------------


def test_gui_series_binds_the_window_lora_and_strength(tmp_path: Path, monkeypatch) -> None:
    """A Photoshoot from the window uses the same LoRA and strength as Generate.

    The worker series path used to leave the runner unbound, so cook/precompute
    drew with the base model even when the job named a LoRA.
    """
    from canvas.app.shell_run import RunStopped
    from canvas.core.formats import OutputFormat
    from canvas.models.modes import Mode
    from canvas.pipeline.series import SeriesResult
    from canvas.ui.shell_layout import PictureJob
    from canvas.worker_jobs import Session, _series

    lora = _folder_lora(tmp_path, **{"voxprint.subject.alias": ALIAS})
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    seen: dict[str, object] = {}

    class Check:
        def review_prompt(self, text: str):
            return None

    class Runner:
        bound_lora = None
        lora_strength = 0.25
        cook = False

        def __call__(self, job, inputs):
            seen["job_path"] = job.lora_path
            seen["job_weight"] = job.lora_weight
            seen["job_prompt"] = job.prompt
            seen["bound"] = self.bound_lora
            seen["strength"] = self.lora_strength
            raise RunStopped("stop after bind")

    runner = Runner()

    def fake_run_series(**kwargs):
        draw = kwargs["draw_frame"]
        seen["draw_lora"] = getattr(draw, "lora", None)
        seen["during_bound"] = runner.bound_lora
        seen["during_strength"] = runner.lora_strength
        try:
            draw("a red boat at dusk", 1, None, None)
        except RunStopped:
            pass
        return SeriesResult(stopped="", reason="", kept=(), folder=None, code="", source="")

    reviewed: list[str] = []

    def record_review(_current, prompts):
        reviewed.extend(prompts)
        return None

    class Writer:
        def __call__(self, prompt: str) -> str:
            return prompt

        def release(self) -> None:
            return None

    monkeypatch.setattr("canvas.pipeline.series.run_series", fake_run_series)
    monkeypatch.setattr("canvas.worker_jobs._review_prompts", record_review)
    monkeypatch.setattr("canvas.worker_jobs._selected_lora", lambda _path, _runner: lora)
    monkeypatch.setattr("canvas.worker_jobs._writer_session", lambda *_a, **_k: Writer())

    job = PictureJob(
        "a red boat", "", Mode.TEXT, "1:1", "native", OutputFormat.PNG, 1, 8, 0.0, "euler", 1,
        lora_path=str(lora.weights), lora_weight=1.0,
    )
    fields = job._asdict()
    fields["mode"] = job.mode.value
    fields["fmt"] = job.fmt.value
    ctx = types.SimpleNamespace(progress=lambda **_kw: None, cancelled=lambda: False)
    current = Session(gate=None, check=Check(), tier=None, runner=runner, improver=lambda **_kw: "")
    payload = {
        "job": fields,
        "idea": f"{ALIAS} on a bench",
        "count": 2,
        "style": "photo",
        "place": "same",
        "seed": 1,
        "adult": False,
        "container": "png",
        "series_id": "abcd",
        "out_dir": str(tmp_path / "session"),
        "template": "",
        "suffix": "",
        "seed_mode": "locked",
    }
    reply = _series(ctx, current, payload)
    assert reply.get("status") == "ok"
    assert seen["during_bound"] is lora
    assert seen["during_strength"] == pytest.approx(1.0)
    assert seen["draw_lora"] is lora
    assert seen["job_path"] == str(lora.weights)
    assert seen["job_weight"] == pytest.approx(1.0)
    assert seen["bound"] is lora
    assert seen["strength"] == pytest.approx(1.0)
    assert str(seen["job_prompt"]).startswith(f"{TRIGGER}, ")
    # Alias is rewritten before the gate reviews the idea.
    assert reviewed and all(ALIAS not in item for item in reviewed)
    assert any(TRIGGER in item for item in reviewed)
    # The bind is restored so a later picture does not keep the series LoRA.
    assert runner.bound_lora is None
    assert runner.lora_strength == pytest.approx(0.25)
