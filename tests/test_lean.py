"""Lean testing: showcase review set, cheap checks, and budget sizing."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from bakeoff.lean import (
    Q_BASE_S,
    SHOWCASE_COUNTS,
    Z_BASE_S,
    _EXPLICIT,
    image_name,
    native_frame,
    preflight_errors,
    question_name,
    render_size,
    showcase_html,
    size_run,
    validate_plan,
    validate_summary,
)
from bakeoff.prompt_lint import stated_ages
from bakeoff.run_bakeoff import ALL_ARMS, Z_ARMS, default_arms, load_prompts, main, parse_args, summarise

_SHOWCASE_IDS = [
    "sc01", "sc04", "sc07", "sc10", "sc12", "sc14",
    "sc02", "sc05", "sc08", "sc11", "sc13", "sc15",
    "sc03", "sc06", "sc09",
]


def _plan(text: str) -> dict:
    checks_at = text.index("checks ok: prompt-lint, dry-run, schema")
    plan_at = text.index("plan ")
    assert checks_at < plan_at
    return json.loads(text[plan_at + len("plan "):].strip())


def test_showcase_is_fifteen_hard_prompts_without_explicit_content():
    rows = load_prompts("showcase")
    assert [row["id"] for row in rows] == _SHOWCASE_IDS
    assert dict(Counter(row["category"] for row in rows)) == SHOWCASE_COUNTS
    assert preflight_errors(rows, "showcase") == []
    assert question_name("showcase") == "showcase"
    for row in rows:
        assert row["category"] != "adult"
        assert row.get("content_level") != "explicit"
        ages = stated_ages(row["prompt"])
        assert all(25 <= age <= 30 for age in ages)
        tokens = set(re.findall(r"[a-z]+", row["prompt"].lower()))
        assert tokens.isdisjoint(_EXPLICIT)
        if row["category"] == "portrait":
            assert row["ages"] == ages
            assert ages
    portraits = [row for row in rows if row["category"] == "portrait"]
    assert [row["ages"] for row in portraits] == [[27], [30]]
    v2 = load_prompts("v2")
    assert len(v2) == 105
    assert "sc01" not in {row["id"] for row in v2}


def test_showcase_preflight_rejects_explicit_youth_and_out_of_band_ages():
    rows = load_prompts("showcase")
    explicit = [dict(row) for row in rows]
    explicit[0] = dict(explicit[0], prompt=explicit[0]["prompt"] + " nude")
    assert any("explicit term" in hit for hit in preflight_errors(explicit, "showcase"))

    youth = [dict(row) for row in rows]
    youth[6] = dict(youth[6], prompt="a girl in a red coat standing on a ferry")
    assert any("youth term" in hit for hit in preflight_errors(youth, "showcase"))

    young = [dict(row) for row in rows]
    portrait = next(index for index, row in enumerate(young) if row["category"] == "portrait")
    young[portrait] = dict(
        young[portrait],
        prompt="Studio portrait of a 19-year-old woman in a black turtleneck",
        ages=[19],
    )
    hits = preflight_errors(young, "showcase")
    assert any("stated age 19" in hit for hit in hits)
    assert any("portrait age must be 25-30" in hit for hit in hits)

    older = [dict(row) for row in rows]
    portrait = next(index for index, row in enumerate(older) if row["category"] == "portrait")
    older[portrait] = dict(
        older[portrait],
        prompt="Studio portrait of a 40-year-old woman in a black turtleneck",
        ages=[40],
    )
    assert any("portrait age must be 25-30" in hit for hit in preflight_errors(older, "showcase"))


def test_native_frames_differ_by_model_and_names_keep_the_size():
    assert Z_BASE_S == 3.0
    assert Q_BASE_S == 50.0
    assert native_frame("z", "16:9") == (1280, 720)
    assert native_frame("q", "16:9") == (1664, 928)
    assert native_frame("z", "1:1") == (1024, 1024)
    assert native_frame("q", "1:1") == (1328, 1328)
    assert render_size("z", {"aspect": "16:9"}, (1024, 1024)) == (1280, 720)
    assert render_size("q", {"id": "gen00"}, (1328, 1328)) == (1328, 1328)
    assert image_name({"id": "sc01", "aspect": "16:9"}, 0, 1280, 720) == "sc01_s0_1280x720.webp"
    assert image_name({"id": "gen00"}, 0, 1024, 1024) == "gen00_s0.webp"


def test_budget_sizes_the_showcase_and_leaves_metrics_whole():
    prompts = load_prompts("showcase")
    specs = default_arms()
    defaults = {name: (arm.width, arm.height) for name, arm in specs.items()}
    full = size_run("showcase", prompts, ["z_bf16", "q_bf16"], 25, 1, 0, set(), defaults)
    assert full["dropped"] == []
    assert [(row["arm"], len(row["prompt_ids"])) for row in full["kept"]] == [("z_bf16", 15), ("q_bf16", 15)]
    assert full["estimate_min"] == 19.44

    short = size_run("showcase", prompts, ["z_bf16", "q_bf16"], 12, 1, 0, set(), defaults)
    assert short["dropped"] == []
    kept = {row["arm"]: row for row in short["kept"]}
    assert len(kept["z_bf16"]["prompt_ids"]) == 15
    assert len(kept["q_bf16"]["prompt_ids"]) == 5
    categories = {row["category"] for row in prompts if row["id"] in set(kept["q_bf16"]["prompt_ids"])}
    assert len(categories) >= 3

    metrics = load_prompts("v2")
    subset = {row["id"] for row in metrics[:40]}
    sized = size_run("v2", metrics, list(ALL_ARMS), 0, 1, 15, subset, defaults)
    assert sized["question"] == "metrics"
    assert sized["dropped"] == []
    counts = {row["arm"]: len(row["prompt_ids"]) for row in sized["kept"]}
    assert counts["q_tuned"] == 40
    assert counts["z_bf16"] == counts["q_bf16"] == 105


def test_dry_run_checks_the_plan_before_writing_images(tmp_path: Path, capsys):
    out = tmp_path / "results"
    code = main([
        "--dry-run", "--prompt-set", "showcase", "--budget-min", "25", "--out", str(out),
    ])
    assert code == 0
    assert not out.exists()
    assert list(tmp_path.rglob("*.webp")) == []
    plan = _plan(capsys.readouterr().out)
    assert validate_plan(plan) == []
    assert plan["question"] == "showcase"
    assert plan["prompts"] == 15
    assert plan["budget_min"] == 25.0
    assert list(plan["arms"]) == Z_ARMS
    assert plan["arms"]["z_bf16"]["prompts"] == 15
    assert plan["arms"]["z_fp8"]["prompts"] == 15
    assert "q_bf16" not in plan["arms"]
    assert plan["dropped"] == []
    assert plan["frames"][0] == {"id": "sc01", "aspect": "16:9", "z": "1280x720", "q": "1664x928"}
    assert all(row["z"] != row["q"] for row in plan["frames"])

    code = main([
        "--dry-run", "--prompt-set", "showcase", "--budget-min", "25",
        "--arms", "z_bf16,q_bf16", "--out", str(tmp_path / "pair"),
    ])
    assert code == 0
    pair = _plan(capsys.readouterr().out)
    assert list(pair["arms"]) == ["z_bf16", "q_bf16"]
    assert pair["arms"]["q_bf16"]["prompts"] == 15

    code = main(["--dry-run", "--prompt-set", "v2", "--out", str(tmp_path / "v2")])
    assert code == 0
    metrics = _plan(capsys.readouterr().out)
    assert metrics["question"] == "metrics"
    assert metrics["prompts"] == 105
    assert list(metrics["arms"]) == Z_ARMS
    assert "frames" not in metrics
    assert metrics["budget_min"] == 0.0
    assert metrics["deadline_min"] == 150.0

    assert main(["--budget-min", "-1", "--dry-run"]) == 2


def test_showcase_page_is_one_full_size_image_with_prev_next():
    prompts = load_prompts("showcase")
    records = [
        {
            "arm": "q_bf16", "prompt_id": "sc01", "seed": 0, "width": 1664, "height": 928,
            "model": "Qwen-Image-2512", "image": "images/q_bf16/sc01_s0_1664x928.webp",
        },
        {
            "arm": "z_bf16", "prompt_id": "sc01", "seed": 0, "width": 1280, "height": 720,
            "model": "Z-Image-Turbo", "image": "images/z_bf16/sc01_s0_1280x720.webp",
        },
    ]
    page = showcase_html(records, prompts)
    assert page.count('data-page="') == 2
    assert 'data-page="0">' in page
    assert 'data-page="1" hidden>' in page
    assert ">Prev</button>" in page
    assert ">Next</button>" in page
    assert 'event.key === "ArrowLeft"' in page
    assert 'event.key === "ArrowRight"' in page
    assert "model Z-Image-Turbo · arm z_bf16 · prompt sc01 · seed 0 · size 1280×720" in page
    assert 'style="width:1280px;height:720px;max-width:none"' in page
    assert 'src="../images/z_bf16/sc01_s0_1280x720.webp"' in page
    assert "GATE 4B" in page
    assert "&quot;" in page
    tricky = showcase_html(
        [{"arm": "z_bf16", "prompt_id": "x", "seed": 1, "width": 8, "height": 8, "image": "images/a.webp"}],
        [{"id": "x", "prompt": '<script>alert("x")</script>'}],
    )
    assert "<script>alert" not in tricky
    assert "&lt;script&gt;" in tricky


def test_summary_schema_keeps_the_gate_rule(tmp_path: Path):
    prompts = load_prompts("showcase")
    args = parse_args(["--prompt-set", "showcase", "--budget-min", "25", "--out", str(tmp_path), "--review", ""])
    summary = summarise(tmp_path, prompts, ["budget 25 min keeps z_bf16"], args)
    assert validate_summary(summary) == []
    assert summary["question"] == "showcase"
    assert summary["budget_min"] == 25.0
    assert "0.10" in summary["gate_rule"]
    assert "age<20" in summary["gate_rule"]
    metrics = summarise(tmp_path, load_prompts("v1"), [], parse_args(["--out", str(tmp_path), "--review", ""]))
    assert validate_summary(metrics) == []
    assert metrics["question"] == "metrics"
    assert metrics["budget_min"] == 0.0
