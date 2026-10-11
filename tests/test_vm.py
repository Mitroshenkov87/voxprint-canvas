"""The machine-check scripts stay non-interactive and use the exit-code contract."""
import json
from pathlib import Path

from tools.vm_test import main, results_dir


def test_no_gpu_is_exit_3(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("tools.vm_test._gpu_present", lambda: False)
    monkeypatch.setenv("VOXPRINT_VM_RESULTS", str(tmp_path))
    assert main(["--smoke", "--json", "--tag", "v0.1.0-rc.2"]) == 3
    body = json.loads(capsys.readouterr().out)
    assert body["ok"] is False
    assert body["exit_code"] == 3
    assert body["mode"] == "smoke"
    saved = json.loads((tmp_path / "canvas" / "v0.1.0-rc.2" / "result.json").read_text(encoding="utf-8"))
    assert saved["exit_code"] == 3


def test_results_dir_uses_the_sync_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_VM_RESULTS", str(tmp_path))
    path = results_dir("v0.1.0-rc.2")
    assert path == tmp_path / "canvas" / "v0.1.0-rc.2"
    assert path.is_dir()


def test_scripts_are_noninteractive_and_name_the_exit_codes():
    root = Path("scripts")
    for name in ("vm_test.sh", "vm_test.ps1"):
        text = (root / name).read_text(encoding="utf-8")
        assert "--smoke" in text or "vm_test" in text
        assert "tools.vm_test" in text
        assert "exit 2" in text or "exit 2" in text.lower()
    shell = (root / "vm_test.sh").read_text(encoding="utf-8")
    assert "set -eu" in shell
    windows = (root / "vm_test.ps1").read_text(encoding="utf-8")
    assert "exit $LASTEXITCODE" in windows
    assert "--vram-cap-gb" in shell
    assert "--vram-cap-gb" in windows


def test_vram_cap_jobs_cover_t2i_inpaint_and_edit(tmp_path, monkeypatch):
    from tools.vm_test import _child_env, edit_comparisons, vram_cap_jobs

    assert edit_comparisons(False) == (("qwen-edit", "fp8"),)
    assert edit_comparisons(True) == (("qwen-edit", None), ("qwen-edit-fp8", "fp8"))
    monkeypatch.setenv("CANVAS_EDIT_ROUTE", "fp8")
    plain = _child_env(16)
    assert plain is not None and "CANVAS_EDIT_ROUTE" not in plain
    forced = _child_env(16, "fp8")
    assert forced is not None and forced["CANVAS_EDIT_ROUTE"] == "fp8"
    jobs = vram_cap_jobs(tmp_path, 16)
    names = [name for name, _dest, _args in jobs]
    assert names == ["t2i-2k", "inpaint-2k", "qwen-edit"]
    text = " ".join(jobs[0][2])
    assert "--vram-cap-gb 16" in text
    assert "--output-size 2k" in text
    assert "--mask" in " ".join(jobs[1][2])
    assert "--edit-instruction" in " ".join(jobs[2][2])


def test_vram_cap_without_a_gpu_is_exit_3(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("tools.vm_test._gpu_present", lambda: False)
    monkeypatch.setenv("VOXPRINT_VM_RESULTS", str(tmp_path))
    assert main(["--vram-cap-gb", "16", "--json", "--tag", "cap"]) == 3
    body = json.loads(capsys.readouterr().out)
    assert body["mode"] == "vram-cap"
    assert body["peak_vram_mib"] == 0
    assert body["peak_allocated_bytes"] == 0
    assert body["seconds_per_image"] == {}
