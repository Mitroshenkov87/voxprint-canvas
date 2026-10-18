"""The 16 GB smoke tool stays neutral, fail-closed on the gate, and strict on the manifest."""
import json
from pathlib import Path

from tools import smoke_16gb
from tools.smoke_16gb import PROMPTS, gate_ready, main, manifest_failures, manifest_ids, summarise


def test_prompts_are_five_neutral_scenes():
    assert len(PROMPTS) == 5
    text = " ".join(PROMPTS).lower()
    assert "teapot" in text and "landscape" in text and "city street" in text
    for word in ("woman", "man", "girl", "boy", "person", "nude", "child"):
        assert word not in text.split()


def test_no_gpu_is_exit_3(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(smoke_16gb, "gpu_name", lambda: "")
    assert main(["--out", str(tmp_path)]) == 3
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["exit_code"] == 3 and saved["ok"] is False
    capsys.readouterr()


def test_gate_is_unavailable_when_a_classifier_is_missing():
    assert not gate_ready({})
    assert not gate_ready({"classifiers": []})
    assert not gate_ready({"classifiers": [{"id": "nsfw", "ok": True, "detail": "not_downloaded"}]})
    assert not gate_ready({"classifiers": [{"id": "nsfw", "ok": False, "detail": "hash_mismatch"}]})
    assert gate_ready({"classifiers": [{"id": "nsfw", "ok": True, "detail": ""}]})


def test_manifest_failures_cover_empty_missing_and_hash(tmp_path):
    good = {
        "models": [{"id": "z-image-turbo-bf16-16gb", "ok": True, "detail": ""},
                   {"id": "z-image-creative-bf16", "ok": True, "detail": "not_downloaded"}],
        "classifiers": [{"id": "nsfw-image-detection", "ok": True, "detail": ""}],
    }
    assert manifest_failures(good, set()) == ["mdl.json lists no models"]
    assert manifest_failures(good, {"nsfw-image-detection"}) == []
    bad = json.loads(json.dumps(good))
    bad["classifiers"][0] = {"id": "nsfw-image-detection", "ok": False, "detail": "hash_mismatch"}
    assert manifest_failures(bad, {"nsfw-image-detection"}) == ["nsfw-image-detection: hash_mismatch"]
    assert manifest_failures(good, {"nsfw-image-detection", "fairface-age"}) == ["fairface-age: not in the self-test"]
    path = tmp_path / "mdl.json"
    path.write_text('{"schema": 1, "app": "canvas", "models": [{"id": "a"}]}', encoding="utf-8")
    assert manifest_ids(path) == {"a"}


def test_summary_counts_gate_results_and_peaks():
    rows = [
        {"gate": "pass", "peak_allocated_bytes": 10},
        {"gate": "pass", "peak_allocated_bytes": 30},
        {"gate": "fail", "peak_allocated_bytes": None},
    ]
    body = summarise(rows, 14000, "NVIDIA GeForce RTX 4060 Ti", "16gb")  # type: ignore[arg-type]
    assert body["gate"] == {"pass": 2, "fail": 1}
    assert body["peak_vram"] == {"torch_max_memory_allocated_bytes": 30, "nvidia_smi_peak_mib": 14000}
    assert body["size"] == [1024, 1024]


def test_wrappers_install_fetch_through_mdl_and_run_the_tool():
    root = Path("tools")
    shell = (root / "smoke_16gb.sh").read_text(encoding="utf-8")
    windows = (root / "smoke_16gb.ps1").read_text(encoding="utf-8")
    for text in (shell, windows):
        assert "voxprint_common.mdl" in text and "mdl.json" in text
        assert "tools.smoke_16gb" in text
        assert "constraints-cuda.txt" in text
        assert "exit 2" in text
    assert "set -eu" in shell
    assert "exit 5" in shell
    assert "exit $LASTEXITCODE" in windows
