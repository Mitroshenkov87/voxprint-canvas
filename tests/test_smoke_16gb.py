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
    monkeypatch.setattr(smoke_16gb, "_canvas_version", lambda python: "0.1.0")
    assert main(["--out", str(tmp_path)]) == 3
    saved = json.loads((tmp_path / "result-16gb.json").read_text(encoding="utf-8"))
    assert saved["exit_code"] == 3 and saved["ok"] is False
    assert saved["pass"] == "16gb" and saved["vram_cap_gb"] == 16.0
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


COMMON_SHA = "aeb2b180025967a02a81e206dfa0dd01c3d77a2aa20447485288496123851505"


def test_wrappers_pin_common_025_fetch_through_mdl_and_run_the_tool():
    root = Path("tools")
    shell = (root / "smoke_16gb.sh").read_text(encoding="utf-8")
    windows = (root / "smoke_16gb.ps1").read_text(encoding="utf-8")
    installer = Path("installer/linux/install.sh").read_text(encoding="utf-8")
    assert COMMON_SHA in installer and "v0.2.6/voxprint_common-0.2.6-py3-none-any.whl" in installer
    for text in (shell, windows):
        assert "voxprint_common.mdl" in text and "mdl.json" in text
        assert "smoke_16gb" in text
        assert COMMON_SHA in text and "v0.2.6/voxprint_common-0.2.6-py3-none-any.whl" in text
        assert "0.2.4" not in text and "3051bc64" not in text
        assert "exit 2" in text and "exit 5" in text
    assert "constraints-cuda.txt" in shell and "set -eu" in shell
    assert "exit $code" in windows


def test_the_windows_script_needs_no_bash_and_runs_both_passes():
    windows = Path("tools/smoke_16gb.ps1").read_text(encoding="utf-8")
    lowered = windows.lower()
    for word in ("bash", "wsl ", "wsl.exe", "/bin/sh"):
        assert word not in lowered.replace("no wsl and no bash", "")
    assert "[double]$VramCapGB" in windows and "[string]$Python" in windows
    assert 'VOXPrint AI Suite\\Canvas\\venv\\Scripts\\python.exe' in windows
    assert '$Pass = "16gb"' in windows and '$Pass = "24gb"' in windows
    assert "InvariantCulture" in windows
    # Windows PowerShell 5.1 has none of these.
    for token in (" ?? ", " && ", " || ", "?.", "-Parallel"):
        assert token not in windows


def test_passes_choose_the_cap_tier_and_generator():
    assert smoke_16gb._cap_for("16gb", None) == 16.0
    assert smoke_16gb._cap_for("16gb", 16) == 16.0
    assert smoke_16gb._cap_for("24gb", 16) is None
    assert smoke_16gb._canvas("py", None, "x") == ["py", "-m", "canvas.app.main", "x"]
    assert smoke_16gb._canvas("py", 16.0, "x") == ["py", "-m", "canvas.app.main", "--vram-cap-gb", "16", "x"]
    report = {"models": [{"id": "z-image-turbo-bf16-16gb", "ok": True, "detail": ""}], "classifiers": []}
    assert manifest_failures(report, {"z-image-turbo-bf16-16gb"}, "z-image-turbo-bf16") == [
        "z-image-turbo-bf16: not in the self-test"]


def test_the_24gb_pass_refuses_a_16gb_tier(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(smoke_16gb, "gpu_name", lambda: "NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(smoke_16gb, "_canvas_version", lambda python: "0.1.0")
    report = {"gpu": {"tier": "16gb"},
              "models": [{"id": "z-image-turbo-bf16", "ok": True, "detail": ""}],
              "classifiers": [{"id": "nsfw-image-detection", "ok": True, "detail": ""}]}

    class Proc:
        stdout = json.dumps(report)
        stderr = ""
        returncode = 0

    seen: list[list[str]] = []
    monkeypatch.setattr(smoke_16gb.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or Proc())
    monkeypatch.setattr(smoke_16gb, "manifest_ids", lambda path=None: {"nsfw-image-detection"})
    assert main(["--out", str(tmp_path), "--pass", "24gb", "--vram-cap-gb", "16"]) == 2
    assert "--vram-cap-gb" not in seen[0]
    saved = json.loads((tmp_path / "result-24gb.json").read_text(encoding="utf-8"))
    assert saved["vram_cap_gb"] is None and "expected the 24gb tier" in saved["error"]
    capsys.readouterr()


def test_gate_bench_times_every_stage_and_forces_the_age_stage(monkeypatch, tmp_path):
    from PIL import Image as PILImage

    from canvas.models import loader
    from canvas.safety.classifiers import GuardVerdict

    calls: list[str] = []

    class Stub:
        def assess(self, text):
            calls.append("guard")
            return GuardVerdict("safe", ())

        def score_image(self, raw, labels):
            calls.append("siglip")
            return {label: 0.0 for label in labels}

        def detect(self, raw):
            calls.append("yunet")
            return ()

        def estimate(self, raw, faces):
            calls.append(f"fairface:{len(faces)}")
            return tuple((0.0, 0.0) for _ in faces)

        def sexual_score(self, raw):
            calls.append("nsfw")
            return 0.01

    class Fine:
        def estimate(self, raw, faces):
            calls.append(f"fine:{len(faces)}")
            return tuple(30.0 for _ in faces)

    stub = Stub()
    stack = loader.SafetyStack(stub, stub, stub, stub, stub, Fine())
    monkeypatch.setattr(loader, "load_catalog_stack", lambda root, catalog=None: stack)
    image = tmp_path / "a.png"
    PILImage.new("RGB", (64, 64), "white").save(image)
    from canvas.infra.vram import set_vram_cap_gb

    try:
        body = smoke_16gb.gate_bench([PROMPTS[0]], [image], 16.0)
    finally:
        set_vram_cap_gb(None)
    frame = body["frames"][0]
    assert body["forced"] is True
    for key in ("prompt_check_s", "nsfw_pass_s", "age_stage_s"):
        assert isinstance(frame[key], float)
    assert frame["age_stage"]["synthetic_face"] is True and frame["age_stage"]["faces"] == 0
    assert calls == ["guard", "nsfw", "siglip", "yunet", "fairface:1", "fine:1"]
    assert isinstance(frame["age_stage"]["fine_s"], float)
    text = json.dumps(body)
    assert PROMPTS[0] not in text and "0.01" not in text
