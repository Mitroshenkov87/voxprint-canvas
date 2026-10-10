"""v2 prompt sets, the age lint, and the young-adult false-block report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bakeoff.prompt_lint import MIN_STATED_AGE, lint_jsonl, lint_text, stated_ages
from bakeoff.run_bakeoff import _REVIEW_SKIP, load_prompts, main, parse_args, summarise

_V1_SHA256 = {
    "general.jsonl": "8e0e3a8a243d9ba0e713ce8f73159ffae043d76941de6b99f7173abb998a1f27",
    "text.jsonl": "5dd04a4bf2825ac40e9c8fc5d577a517acf22df058cfa9e72e03170ac1f52b9c",
    "adult.jsonl": "9416fdf7f23901be53b21f165dc65f37971a346d994186b42020477ace841565",
}
_EXPLICIT = {"nude", "topless", "lingerie", "erotic", "nudity", "boudoir"}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _band(age: int) -> str:
    if 25 <= age <= 30:
        return "25-30"
    if 20 <= age <= 24:
        return "20-24"
    if age > 30:
        return "older"
    return "under"


def _prompt_band(row: dict) -> str:
    ages = row["ages"]
    assert ages
    assert stated_ages(row["prompt"]) == ages
    bands = {_band(age) for age in ages}
    assert len(bands) == 1
    return bands.pop()


def _assert_balance(rows: list[dict]) -> None:
    bands = [_prompt_band(row) for row in rows]
    count = len(bands)
    mid = bands.count("25-30")
    low = bands.count("20-24")
    older = bands.count("older")
    assert mid / count >= 0.70
    assert low >= 1
    assert older >= 1
    assert older <= max(2, count // 8)
    assert (mid + low) / count >= 0.85
    assert "under" not in bands


def _explicit(text: str) -> bool:
    tokens = {tok.strip(".,") for tok in text.lower().replace("-", " ").split()}
    return any(tok in _EXPLICIT for tok in tokens)


def test_lint_minimum_is_20_and_youth_terms_still_fail():
    assert MIN_STATED_AGE == 20
    assert lint_text("Portrait of a 20-year-old woman in a wool coat") == []
    assert lint_text("A 25 year old man in a denim jacket") == []
    assert lint_text("Мужчине 20 лет, портрет") == []
    assert any("stated age 19" in hit for hit in lint_text("a 19-year-old woman"))
    assert any("stated age 18" in hit for hit in lint_text("ей 18 лет"))
    assert any("stated age 9" in hit for hit in lint_text("the 9th birthday cake"))
    for sample in (
        "a young 25-year-old woman",
        "a petite 30-year-old woman",
        "a girl in a coat",
        "a teenage boy",
        "a student with a backpack",
        "a schoolgirl",
        "a school-girl",
        "школьница в пальто",
        "девушка 25 лет",
        "подросток на улице",
        "молодой мужчина",
    ):
        assert lint_text(sample), sample
    assert lint_text("a minority language poster") == []
    assert lint_text("a boyfriend waiting at a cafe") == []


def test_v1_prompt_files_are_unchanged_and_pass_the_lint():
    root = Path("bakeoff/prompts")
    for name, digest in _V1_SHA256.items():
        path = root / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert lint_jsonl(path) == []


def test_v2_rebalances_person_ages_and_keeps_the_text_set():
    root = Path("bakeoff/prompts/v2")
    general = _rows(root / "general.jsonl")
    adult = _rows(root / "adult.jsonl")
    text = _rows(root / "text.jsonl")
    v1_text = _rows(Path("bakeoff/prompts/text.jsonl"))
    assert len(general) == 40
    assert len(adult) == 30
    assert len(text) == 35
    people = [row for row in general if row.get("person")]
    assert len(people) == 20
    for row in general:
        if row.get("person"):
            continue
        assert stated_ages(row["prompt"]) == []
    _assert_balance(people)
    _assert_balance(adult)
    for row in adult:
        assert row["category"] == "adult"
        assert row["ages"]
        assert min(row["ages"]) >= 20
    assert [row["prompt"] for row in text] == [row["prompt"] for row in v1_text]
    assert [row["target"] for row in text] == [row["target"] for row in v1_text]
    assert [row["id"] for row in text] == ["v2" + row["id"] for row in v1_text]
    loaded = load_prompts("v2")
    assert len(loaded) == 105
    assert [row["category"] for row in loaded[:4]] == ["general", "text_en", "text_ru", "adult"]
    assert lint_jsonl(root / "general.jsonl") == []
    assert lint_jsonl(root / "adult.jsonl") == []
    assert lint_jsonl(root / "text.jsonl") == []


def test_fp_young_adults_cover_every_age_and_both_levels():
    rows = _rows(Path("bakeoff/prompts/v2/fp_young_adults.jsonl"))
    assert len(rows) == 24
    seen: dict[tuple[int, str], int] = {}
    for row in rows:
        assert row["category"] == "fp_young_adult"
        assert row["content_level"] in ("non_explicit", "explicit")
        assert stated_ages(row["prompt"]) == row["ages"]
        assert len(set(row["ages"])) == 1
        age = row["ages"][0]
        assert 20 <= age <= 25
        seen[(age, row["content_level"])] = seen.get((age, row["content_level"]), 0) + 1
        if row["content_level"] == "explicit":
            assert _explicit(row["prompt"])
        else:
            assert not _explicit(row["prompt"])
    for age in range(20, 26):
        assert seen[(age, "non_explicit")] == 2
        assert seen[(age, "explicit")] == 2
    assert lint_jsonl(Path("bakeoff/prompts/v2/fp_young_adults.jsonl")) == []
    loaded = load_prompts("fp_young_adults")
    assert [row["id"] for row in loaded] == [row["id"] for row in rows]


def test_default_prompt_set_is_v1():
    prompts = load_prompts()
    assert len(prompts) == 105
    assert prompts[0]["id"] == "gen00"
    assert [row["category"] for row in prompts[:4]] == ["general", "text_en", "text_ru", "adult"]


def test_fp_summary_reports_false_blocks_by_age_and_level(tmp_path: Path):
    prompts = load_prompts("fp_young_adults")
    for prompt in prompts:
        blocked = prompt["content_level"] == "explicit" and prompt["ages"][0] <= 22
        row = {
            "arm": "z_bf16",
            "prompt_id": prompt["id"],
            "category": prompt["category"],
            "seed": 0,
            "seconds": 1.0,
            "image": "unused.webp",
        }
        record_path = tmp_path / "records.jsonl"
        score_path = tmp_path / "scores.jsonl"
        with record_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        scored = dict(row)
        scored["gate_block"] = blocked
        with score_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(scored) + "\n")
    args = parse_args(["--prompt-set", "fp_young_adults", "--out", str(tmp_path), "--review", ""])
    summary = summarise(tmp_path, prompts, [], args)
    report = summary["fp_young_adults"]["z_bf16"]
    assert report["by_age_and_content_level"]["20"]["explicit"]["gate_false_block_rate"] == 1.0
    assert report["by_age_and_content_level"]["20"]["explicit"]["prompts"] == 2
    assert report["by_age_and_content_level"]["20"]["non_explicit"]["gate_false_block_rate"] == 0.0
    assert report["by_age_and_content_level"]["25"]["explicit"]["gate_false_block_rate"] == 0.0
    assert report["by_age"]["20"]["gate_false_block_rate"] == 0.5
    assert report["by_content_level"]["explicit"]["images"] == 12
    assert report["by_content_level"]["explicit"]["gate_false_block_rate"] == 0.5
    assert report["by_content_level"]["non_explicit"]["gate_false_block_rate"] == 0.0
    page = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Young-adult gate measurement" in page
    assert "| 20 | explicit |" in page
    assert "| all | non_explicit |" in page
    assert "0.10" in summary["gate_rule"]
    assert "age<20" in summary["gate_rule"]
    assert "fp_young_adult" in _REVIEW_SKIP
    assert "adult" in _REVIEW_SKIP


def test_v1_summary_has_no_measurement_block(tmp_path: Path):
    args = parse_args(["--out", str(tmp_path), "--review", ""])
    summary = summarise(tmp_path, load_prompts("v1"), [], args)
    assert summary["prompt_set"] == "v1"
    assert "fp_young_adults" not in summary
    assert "fp_young_adult" not in summary["prompt_sets"]
    assert "Young-adult gate measurement" not in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_prompt_set_flag_selects_v2_and_rejects_unknown(tmp_path: Path):
    assert main(["--dry-run", "--prompt-set", "v2", "--out", str(tmp_path)]) == 0
    with pytest.raises(SystemExit) as caught:
        main(["--prompt-set", "nope", "--out", str(tmp_path)])
    assert caught.value.code == 2
