"""The bake-off face detector is required before any GPU stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bakeoff.face_detector import (
    FACE_DETECTOR_ID,
    OPENCV_HEADLESS,
    YUNET_SHA256,
    face_crops,
    face_preflight,
)
from bakeoff.run_bakeoff import Scorer, load_prompts, main, parse_args, summarise

_MODEL = Path("bakeoff/models/face_detection_yunet_2023mar.onnx")


def test_vendored_yunet_matches_the_pin():
    data = _MODEL.read_bytes()
    assert hashlib.sha256(data).hexdigest() == YUNET_SHA256
    assert face_preflight() == []


def test_preflight_rejects_a_missing_package_a_bad_hash_and_a_missing_file(tmp_path: Path):
    missing_pkg = face_preflight(model=_MODEL, installed=set())
    assert any("opencv-python-headless is not installed" in hit for hit in missing_pkg)
    assert any("Whole-image age scoring is not allowed" in hit for hit in missing_pkg)

    gui_only = face_preflight(model=_MODEL, installed={"opencv-python"})
    assert any(OPENCV_HEADLESS in hit for hit in gui_only)

    absent = tmp_path / "missing.onnx"
    missing_file = face_preflight(model=absent, installed={OPENCV_HEADLESS})
    assert any(f"YuNet model missing: {absent}" in hit for hit in missing_file)
    assert any(YUNET_SHA256 in hit for hit in missing_file)

    bad = tmp_path / "face.onnx"
    bad.write_bytes(b"not-the-yunet-model")
    mismatch = face_preflight(model=bad, installed={OPENCV_HEADLESS})
    assert any("does not match pin" in hit and YUNET_SHA256 in hit for hit in mismatch)


def test_preflight_stops_the_run_before_creating_output(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("BAKEOFF_YUNET", str(tmp_path / "missing.onnx"))
    out = tmp_path / "results"
    code = main(["--dry-run", "--out", str(out), "--review", ""])
    captured = capsys.readouterr()
    assert code == 2
    assert not out.exists()
    assert list(tmp_path.rglob("*.webp")) == []
    assert "face detector preflight failed:" in captured.err
    assert "YuNet model missing" in captured.err
    assert "age check runs on the whole image" not in captured.err
    assert "plan " not in captured.out


def test_no_detected_face_is_not_scored_as_the_whole_image():
    scored: list[object] = []

    class Image:
        size = (32, 32)

        def crop(self, box):
            return ("crop", box)

    probe = Scorer.__new__(Scorer)
    probe.face_boxes = lambda image: []
    probe._minor_on_crop = lambda crop: scored.append(crop) or 0.9
    assert probe.minor_prob(Image()) == (0.0, 0)
    assert scored == []

    probe.face_boxes = lambda image: [(4, 4, 10, 12)]
    image = Image()
    assert probe.minor_prob(image) == (0.9, 1)
    assert scored == [("crop", (1, 1, 17, 19))]
    assert face_crops(image, [(4, 4, 10, 12)]) == [("crop", (1, 1, 17, 19))]


def test_summary_records_face_counts_and_drops_gate_rates_without_the_detector(tmp_path: Path):
    prompts = load_prompts("v1")
    record = {
        "arm": "z_bf16",
        "prompt_id": prompts[0]["id"],
        "category": prompts[0]["category"],
        "seed": 0,
        "seconds": 1.0,
        "image": "images/z_bf16/x.webp",
    }
    fallback = dict(record)
    fallback.update(nsfw=0.2, gate_block=True, faces=0, minor_prob=0.8)
    with (tmp_path / "records.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    with (tmp_path / "scores.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(fallback) + "\n")
    args = parse_args(["--out", str(tmp_path), "--review", ""])
    summary = summarise(tmp_path, prompts, [], args)
    assert summary["gate_stats_valid"] is False
    assert summary["face_counts"] == [{
        "arm": "z_bf16",
        "prompt_id": prompts[0]["id"],
        "seed": 0,
        "faces": None,
    }]
    metrics = summary["arms"]["z_bf16"]["metrics"]
    assert metrics["nonadult_gate_block_rate"] is None
    assert metrics["adult_gate_false_block_rate"] is None
    assert metrics["adult_permissive_rate"] is None
    page = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Gate stats: invalid (face detector did not run)." in page
    assert "gate stats invalid: the face detector did not run on every scored image" in summary["notes"]

    detected = dict(fallback)
    detected["faces"] = 2
    detected["face_detector"] = FACE_DETECTOR_ID
    detected["gate_block"] = False
    with (tmp_path / "scores.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(detected) + "\n")
    summary = summarise(tmp_path, prompts, [], args)
    assert summary["gate_stats_valid"] is True
    assert summary["face_counts"][0]["faces"] == 2
    assert summary["arms"]["z_bf16"]["metrics"]["nonadult_gate_block_rate"] == 0.0
