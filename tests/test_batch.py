"""canvas batch: manifest rules, checks before any picture, one writer, one pipeline load."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from canvas.app.batch_cli import (
    MAX_BATCH_FRAMES,
    MAX_ENTRIES,
    ManifestInvalid,
    default_out,
    load_manifest,
    parse_manifest,
    run_batch_command,
    sanitise_name,
)
from canvas.app.main import main
from canvas.models import runtime
from canvas.pipeline.batch import BatchEntry
from canvas.pipeline.enhance import WriterSession
from canvas.safety.gate import GenerationBlocked, SafetyUnavailable, StylisedBlocked
from canvas.safety.image import RawImage
from tests.fakes import World, rgb
from tests.test_photoshoot import IDEA, _payload
from tests.test_photoshoot_cli import _ready
from tests.test_runtime_reuse import _install, _render

CUP = "a red ceramic cup on a wooden table, studio light"
LAMP = "a brass desk lamp on an oak desk, evening light"
NEGATIVE = "blurry, washed out colours"


def _rewrite(prompt: str) -> str:
    body = json.loads(prompt)
    return _payload(str(body["idea"]), int(body["count"]), start=len(body.get("previous", [])))


def _entries(*items: dict[str, object]) -> list[BatchEntry]:
    return parse_manifest({"entries": list(items)})


class _Events:
    """One ordered list of every check and every draw."""

    def __init__(self) -> None:
        self.log: list[tuple[str, str]] = []

    def of(self, kind: str) -> list[str]:
        return [value for event, value in self.log if event == kind]

    def first(self, kind: str) -> int:
        return next(index for index, (event, _value) in enumerate(self.log) if event == kind)

    def last(self, kind: str) -> int:
        return max(index for index, (event, _value) in enumerate(self.log) if event == kind)


def _gate(events: _Events, *, refuse: str = "", refuse_image: str = "", unavailable: bool = False):
    """The fake gate with stubbed checks that record what they saw.

    ``refuse`` refuses that exact prompt text. ``refuse_image`` refuses every
    picture whose line contains it. ``unavailable`` makes the first picture
    check fail to finish.
    """
    gate = World().gate()
    real_prompt = gate.review_prompt
    real_image = gate.review_image
    failed = [False]

    def review_prompt(prompt: str) -> None:
        events.log.append(("prompt_check", prompt))
        if refuse and prompt == refuse:
            raise GenerationBlocked("prompt")
        real_prompt(prompt)

    def review_image(raw: RawImage, line: str):
        events.log.append(("image_check", line))
        if unavailable and not failed[0]:
            failed[0] = True
            raise SafetyUnavailable("load")
        if refuse_image and refuse_image in line:
            raise GenerationBlocked("image")
        return real_image(raw, line)

    gate.review_prompt = review_prompt
    gate.review_image = review_image
    return gate


def _frame_for(events: _Events):
    def frame_for(entry: BatchEntry):
        def draw(line: str, seed: int, hero: object | None, strength: float | None) -> RawImage:
            del hero, strength
            events.log.append(("render", f"{entry.name}|{seed}|{line}"))
            return rgb(4, 4)

        return draw

    return frame_for


def _run(tmp_path: Path, entries: list[BatchEntry], gate, events: _Events, rewrite=_rewrite):
    folder = tmp_path / "out"
    folder.mkdir(parents=True, exist_ok=True)
    return run_batch_command(
        entries=entries, folder=folder, style="photo", places="same", adult=False,
        age_gate=gate, rewrite=rewrite, frame_for=_frame_for(events),
    )


def _rows(body: dict[str, object]) -> list[dict[str, object]]:
    rows = body["entries"]
    assert isinstance(rows, list)
    return rows


# Manifest rules


def test_a_valid_manifest_parses_with_defaults() -> None:
    entries = _entries(
        {"prompt": IDEA},
        {"shot": CUP, "negative": NEGATIVE, "seed": 7, "count": 3, "name": "My cup/shot 1"},
    )
    assert (entries[0].name, entries[0].prompt, entries[0].shot, entries[0].count) == ("001", IDEA, "", 1)
    assert entries[0].seeded is False  # no seed in the entry: random per picture, not a silent seed 1
    assert 0 <= entries[0].seed <= 2**32 - 1
    assert entries[1] == BatchEntry("My_cup_shot_1", "", CUP, NEGATIVE, 7, 3)


@pytest.mark.parametrize(
    ("document", "fragment"),
    [
        ([], "JSON object"),
        ({}, "entries"),
        ({"entries": []}, "entries"),
        ({"entries": [{"prompt": IDEA}], "style": "photo"}, "Unknown manifest key"),
        ({"entries": ["a cup"]}, "Entry 1: must be a JSON object"),
        ({"entries": [{}]}, 'exactly one of "prompt" or "shot"'),
        ({"entries": [{"prompt": IDEA, "shot": CUP}]}, 'exactly one of "prompt" or "shot"'),
        ({"entries": [{"prompt": "   "}]}, "'prompt' is empty"),
        ({"entries": [{"prompt": 3}]}, "'prompt' must be text"),
        ({"entries": [{"prompt": IDEA, "negative": ["x"]}]}, "'negative' must be text"),
        ({"entries": [{"prompt": IDEA, "count": 0}]}, "'count' must be a whole number"),
        ({"entries": [{"prompt": IDEA, "count": 100}]}, "'count' must be a whole number"),
        ({"entries": [{"prompt": IDEA, "count": True}]}, "'count' must be a whole number"),
        ({"entries": [{"prompt": IDEA, "count": 2.0}]}, "'count' must be a whole number"),
        ({"entries": [{"prompt": IDEA, "seed": -1}]}, "'seed' must be a whole number"),
        ({"entries": [{"prompt": IDEA, "steps": 4}]}, "unknown key 'steps'"),
        ({"entries": [{"prompt": IDEA, "name": "///"}]}, "'name' has no letters"),
        ({"entries": [{"prompt": IDEA, "name": 5}]}, "'name' must be text"),
        ({"entries": [{"prompt": IDEA, "name": "batch.json"}]}, "cannot be batch.json"),
        ({"entries": [{"prompt": IDEA, "name": "a"}, {"prompt": IDEA, "name": "A"}]}, "already used by entry 1"),
        ({"entries": [{"prompt": IDEA}, {"prompt": IDEA, "name": "001"}]}, "already used by entry 1"),
        ({"entries": [{"prompt": IDEA}] * (MAX_ENTRIES + 1)}, "the limit is"),
        ({"entries": [{"prompt": IDEA, "count": 99}] * (MAX_BATCH_FRAMES // 99 + 1)}, "pictures; the limit is"),
    ],
)
def test_a_broken_manifest_is_refused_with_a_clear_message(document: object, fragment: str) -> None:
    with pytest.raises(ManifestInvalid) as caught:
        parse_manifest(document)
    assert fragment in str(caught.value)


def test_names_are_made_safe_for_a_folder() -> None:
    assert sanitise_name("  ../../etc/passwd ") == "etc_passwd"
    assert sanitise_name("Café shot #2") == "Caf_shot_2"
    assert sanitise_name("con") == "_con"
    assert sanitise_name("...") == ""
    assert len(sanitise_name("x" * 200)) == 64


def test_the_file_must_be_json(tmp_path: Path) -> None:
    path = tmp_path / "shoot.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestInvalid, match="not valid JSON"):
        load_manifest(path)
    with pytest.raises(ManifestInvalid, match="cannot be read"):
        load_manifest(tmp_path / "missing.json")
    assert default_out(path) == tmp_path / "shoot_batch"


def test_a_bad_manifest_exits_2_before_any_model_or_folder(monkeypatch, tmp_path: Path, capsys) -> None:
    _ready(monkeypatch)
    touched: list[str] = []
    monkeypatch.setattr("canvas.app.main.AgeGate", lambda _stack: touched.append("gate"))
    monkeypatch.setattr("canvas.app.main.load_catalog_stack", lambda *_a, **_k: touched.append("stack"))
    monkeypatch.setattr("canvas.app.main.prepare_output_folder", lambda folder: touched.append("folder"))
    manifest = tmp_path / "shoot.json"
    manifest.write_text(json.dumps({"entries": [{"prompt": IDEA, "count": 0}]}), encoding="utf-8")
    code = main(["batch", "--manifest", str(manifest), "--out", str(tmp_path / "out"), "--json"])
    assert code == 2
    body = json.loads(capsys.readouterr().out)
    assert body["error"] == "bad_manifest" and "Entry 1" in body["message"]
    assert touched == []


def test_a_bad_output_folder_exits_2_before_any_model(monkeypatch, tmp_path: Path, capsys) -> None:
    _ready(monkeypatch)
    touched: list[str] = []
    monkeypatch.setattr("canvas.app.main.load_catalog_stack", lambda *_a, **_k: touched.append("stack"))
    manifest = tmp_path / "shoot.json"
    manifest.write_text(json.dumps({"entries": [{"prompt": IDEA}]}), encoding="utf-8")
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    code = main(["batch", "--manifest", str(manifest), "--out", str(blocker), "--json"])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["code"] == "output_folder"
    assert touched == []


def test_the_cli_takes_the_gpu_lock_once_and_writes_the_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    _ready(monkeypatch)
    events = _Events()
    locks: list[str] = []
    real_pinned = __import__("canvas.app.main", fromlist=["pinned"]).pinned

    @contextmanager
    def counting(job: str, *args, **kwargs):
        locks.append(job)
        with real_pinned(job, *args, **kwargs) as path:
            yield path

    negatives: list[str] = []

    def fake_generate(gate, tier, style, adult, lora, scale, negative):
        del gate, tier, style, adult, lora, scale
        negatives.append(negative)
        return _frame_for(events)(BatchEntry("x", "", "", negative, 1, 1))

    monkeypatch.setattr("canvas.app.main.pinned", counting)
    monkeypatch.setattr("canvas.app.main.model_rewrite", lambda *_args: _rewrite)
    monkeypatch.setattr("canvas.app.main.model_generate", fake_generate)
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "gpu.lock"))
    manifest = tmp_path / "shoot.json"
    manifest.write_text(json.dumps({"entries": [
        {"prompt": IDEA, "name": "teapot"},
        {"shot": CUP, "negative": NEGATIVE, "count": 2},
    ]}), encoding="utf-8")
    code = main(["batch", "--manifest", str(manifest), "--json"])
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert locks == ["canvas-batch"]
    assert negatives == ["", NEGATIVE]
    out = tmp_path / "shoot_batch"
    assert json.loads((out / "batch.json").read_text(encoding="utf-8"))["made"] == 3
    assert body["made"] == 3 and body["ok"] is True
    assert sorted(path.name for path in (out / "teapot").glob("0*.png")) == ["01.png"]
    assert sorted(path.name for path in (out / "002").glob("0*.png")) == ["01.png", "02.png"]


# Checks before any picture


def test_every_prompt_and_negative_is_checked_before_any_render(tmp_path: Path) -> None:
    events = _Events()
    entries = _entries(
        {"prompt": IDEA, "negative": "dull, flat light", "count": 2, "name": "teapot"},
        {"shot": CUP, "negative": NEGATIVE, "count": 2, "name": "cup"},
        {"shot": LAMP, "name": "lamp"},
    )
    body, code = _run(tmp_path, entries, _gate(events), events)
    assert code == 0, body
    checked = events.of("prompt_check")
    for text in (IDEA, "dull, flat light", CUP, NEGATIVE, LAMP):
        assert text in checked
    assert events.of("render")
    assert events.last("prompt_check") < events.first("render")
    assert all(row["status"] == "ok" for row in _rows(body))


def test_a_refused_entry_renders_nothing_and_the_others_continue(tmp_path: Path) -> None:
    events = _Events()
    entries = _entries(
        {"shot": CUP, "name": "cup", "count": 2},
        {"shot": LAMP, "negative": NEGATIVE, "name": "lamp", "count": 2},
        {"prompt": IDEA, "name": "teapot"},
    )
    body, code = _run(tmp_path, entries, _gate(events, refuse=NEGATIVE), events)
    assert code == 1
    rows = {row["name"]: row for row in _rows(body)}
    assert rows["lamp"]["status"] == "refused"
    assert rows["lamp"]["made"] == 0 and rows["lamp"]["frames"] == []
    assert rows["lamp"]["folder"] is None
    assert not (tmp_path / "out" / "lamp").exists()
    assert not any(line.startswith("lamp|") for line in events.of("render"))
    assert rows["cup"]["status"] == "ok" and rows["cup"]["made"] == 2
    assert rows["teapot"]["status"] == "ok" and rows["teapot"]["made"] == 1
    summary = json.loads((tmp_path / "out" / "batch.json").read_text(encoding="utf-8"))
    assert summary["refused"] == 1
    assert [row["status"] for row in summary["entries"]] == ["ok", "refused", "ok"]


def test_a_refused_prompt_entry_does_not_reach_the_writer(tmp_path: Path) -> None:
    events = _Events()
    written: list[str] = []

    def rewrite(prompt: str) -> str:
        written.append(json.loads(prompt)["idea"])
        return _rewrite(prompt)

    entries = _entries({"prompt": IDEA, "name": "teapot"}, {"shot": CUP, "name": "cup"})
    body, _code = _run(tmp_path, entries, _gate(events, refuse=IDEA), events, rewrite)
    rows = {row["name"]: row for row in _rows(body)}
    assert rows["teapot"]["status"] == "refused"
    assert written == []
    assert rows["cup"]["status"] == "ok"


# Every picture is checked before it is saved


def test_every_saved_picture_was_checked_and_a_failed_check_writes_nothing(tmp_path: Path) -> None:
    events = _Events()
    entries = _entries(
        {"shot": CUP, "name": "cup", "count": 2},
        {"shot": LAMP, "name": "lamp", "count": 2},
    )
    body, code = _run(tmp_path, entries, _gate(events, refuse_image="brass desk lamp"), events)
    assert code == 1
    rendered = events.of("render")
    checked = events.of("image_check")
    assert len(checked) == len(rendered) == 4
    assert events.last("render") < events.first("image_check")
    saved = sorted((tmp_path / "out").rglob("0*.webp"))
    assert [path.relative_to(tmp_path / "out").as_posix() for path in saved] == ["cup/01.webp", "cup/02.webp"]
    assert list((tmp_path / "out" / "lamp").glob("0*.webp")) == []
    rows = {row["name"]: row for row in _rows(body)}
    assert rows["lamp"]["made"] == 0 and rows["lamp"]["status"] != "ok"


def test_a_check_that_cannot_finish_stops_every_later_write(tmp_path: Path) -> None:
    events = _Events()
    entries = _entries({"shot": CUP, "name": "cup", "count": 2}, {"shot": LAMP, "name": "lamp"})
    body, code = _run(tmp_path, entries, _gate(events, unavailable=True), events)
    assert code == 1
    assert list((tmp_path / "out").rglob("0*.webp")) == []
    assert len(events.of("image_check")) == 3
    assert {row["status"] for row in _rows(body)} == {"error"}
    assert {row["code"] for row in _rows(body)} == {"check.load"}


# One writer, one pipeline


class _Writer:
    def __init__(self, opened: list[int], closed: list[int]) -> None:
        self.max_new_tokens = 0
        self._closed = closed
        opened.append(1)

    def complete(self, prompt: str, *, system: str) -> str:
        del system
        return _rewrite(prompt)

    def close(self) -> None:
        self._closed.append(1)


def test_the_writer_is_loaded_once_for_the_batch(tmp_path: Path) -> None:
    events = _Events()
    opened: list[int] = []
    closed: list[int] = []
    session = WriterSession(lambda _budget: _Writer(opened, closed), "system", lambda _prompt: 64)  # type: ignore[arg-type,return-value]
    entries = _entries(
        {"prompt": IDEA, "name": "one", "count": 2},
        {"prompt": "Green glass vase with white tulips", "name": "two"},
        {"prompt": "Blue enamel kettle on a gas stove", "name": "three"},
    )
    body, code = _run(tmp_path, entries, _gate(events), events, session)
    assert code == 0, body
    assert opened == [1]
    assert closed == [1]


def test_the_pipeline_loads_once_across_entries_below_64gb(tmp_path: Path, monkeypatch) -> None:
    loads, _places = _install(monkeypatch)
    gate = World().gate()
    weights = tmp_path / "weights"

    def frame_for(entry: BatchEntry):
        def draw(line: str, seed: int, hero, strength):
            del seed, strength
            return _render(gate, weights, line, hero if isinstance(hero, RawImage) else None)

        return draw

    entries = _entries(
        {"shot": CUP, "name": "cup", "count": 2},
        {"shot": LAMP, "name": "lamp", "count": 2},
        {"prompt": IDEA, "name": "teapot", "count": 2},
    )
    folder = tmp_path / "out"
    folder.mkdir()
    body, code = run_batch_command(
        entries=entries, folder=folder, style="photo", places="same", adult=False,
        age_gate=gate, rewrite=_rewrite, frame_for=frame_for,
    )
    assert code == 0, body
    assert body["made"] == 6
    assert loads[0] == 1
    # ram32 keeps nothing once the batch is over.
    assert runtime._slot is None


# No entry is dropped silently


def test_an_entry_that_saved_fewer_pictures_than_asked_is_partial_not_ok(tmp_path: Path) -> None:
    events = _Events()
    gate = _gate(events)
    inner = gate.review_image
    seen = [0]

    def review_image(raw: RawImage, line: str):
        seen[0] += 1
        if seen[0] == 1:
            raise StylisedBlocked("prompt")
        return inner(raw, line)

    gate.review_image = review_image
    entries = _entries({"shot": CUP, "name": "arched_recline", "count": 3}, {"shot": LAMP, "name": "lamp"})
    body, code = _run(tmp_path, entries, gate, events)
    rows = _rows(body)
    assert [row["name"] for row in rows] == ["arched_recline", "lamp"]
    first = rows[0]
    assert first["requested"] == 3 and first["made"] == 2
    assert first["status"] == "partial" and first["code"] and first["message"]
    assert rows[1]["status"] == "ok"
    assert code == 1 and body["ok"] is False


def test_an_error_inside_one_entry_is_recorded_and_the_others_continue(tmp_path: Path) -> None:
    events = _Events()
    entries = _entries({"shot": CUP, "name": "cup"}, {"shot": LAMP, "name": "lamp"})
    plain = _frame_for(events)

    def frame_for(entry: BatchEntry):
        if entry.name == "cup":
            raise RuntimeError("boom")
        return plain(entry)

    folder = tmp_path / "out"
    folder.mkdir()
    body, code = run_batch_command(
        entries=entries, folder=folder, style="photo", places="same", adult=False,
        age_gate=_gate(events), rewrite=_rewrite, frame_for=frame_for,
    )
    rows = {row["name"]: row for row in _rows(body)}
    assert set(rows) == {"cup", "lamp"}
    assert rows["cup"]["status"] == "error" and rows["cup"]["code"] == "internal" and rows["cup"]["message"]
    assert rows["lamp"]["status"] == "ok" and rows["lamp"]["made"] == 1
    assert code == 1
    summary = json.loads((folder / "batch.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in summary["entries"]] == ["cup", "lamp"]


def test_an_unexpected_error_mid_batch_still_gives_every_entry_a_row(tmp_path: Path) -> None:
    events = _Events()
    entries = _entries({"shot": CUP, "name": "cup"}, {"shot": LAMP, "name": "lamp"})

    def frame_for(entry: BatchEntry):
        def draw(line: str, seed: int, hero: object | None, strength: float | None) -> RawImage:
            raise RuntimeError("render failed")

        return draw

    folder = tmp_path / "out"
    folder.mkdir()
    body, code = run_batch_command(
        entries=entries, folder=folder, style="photo", places="same", adult=False,
        age_gate=_gate(events), rewrite=_rewrite, frame_for=frame_for,
    )
    rows = _rows(body)
    assert [row["name"] for row in rows] == ["cup", "lamp"]
    assert all(row["status"] in {"error", "not_produced", "partial"} for row in rows)
    assert all(row["made"] == 0 and row["message"] for row in rows)
    assert code == 1 and (folder / "batch.json").is_file()
