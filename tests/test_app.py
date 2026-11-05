"""Startup commands. The window itself is not opened here."""
import json
import os
import sys
import time

from canvas.app.info import RELEASES_API
from canvas.app.main import main
from canvas.app.platform import GpuCheck, OsCheck
from canvas.app.selftest import selftest_report
from canvas.app.updates import check_for_update, fetch_release
from canvas.infra.enlarge import EnlargeError
from canvas.infra.gpu import GpuInfo
from canvas.infra.models_users import read_users
from canvas.models.modes import Tier
from canvas.models.registry import load_catalog
from canvas.models.snapshots import weight_dir
from canvas.safety.gate import GenerationBlocked
from canvas.ui.backdrop import apply_backdrop


def test_selftest_register_and_update(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    assert main(["--selftest", "--json"]) == 2
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert "Collecting" not in captured.out
    assert "Looking in indexes" not in captured.out
    assert report["ok"] is False
    assert report["encoders"] == {"avif": True, "jxl": True, "webp": True}
    assert isinstance(report["models"], list) and report["models"]
    assert all(row["detail"] == "not_downloaded" and row["ok"] for row in report["models"])
    assert all(row["detail"] == "not_downloaded" and row["ok"] for row in report["classifiers"])
    assert report["deep"] is False
    assert isinstance(report["notes"], list)
    assert main(["--register-models"]) == 0
    assert read_users(tmp_path / "shared")["canvas"] is True
    assert main(["--unregister-models"]) == 0
    assert "canvas" not in read_users(tmp_path / "shared")
    monkeypatch.setattr("canvas.app.main.check_for_update", lambda current: "v9.9.9")
    assert main(["--check-update"]) == 0
    assert capsys.readouterr().out.strip() == "v9.9.9"


def test_vram_cap_flag_selects_the_16gb_tier(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("VOXPRINT_FAKE_VRAM", "RTX 4090,24,20,8.9")
    monkeypatch.setenv("CANVAS_VRAM_CAP_GB", "16")
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr("canvas.infra.gpu._limit_fraction", lambda _fraction: None)
    launched = []
    monkeypatch.setattr("canvas.app.main._launch", lambda tier, splash=None, app=None: launched.append(tier))
    assert main(["--vram-cap-gb", "16"]) == 0
    assert launched == [Tier.GB16]


def test_generate_vram_cap_uses_the_same_name(monkeypatch):
    monkeypatch.setenv("CANVAS_VRAM_CAP_GB", "16")
    seen: dict[str, str] = {}

    def fake(_args: object) -> int:
        seen["cap"] = os.environ.get("CANVAS_VRAM_CAP_GB", "")
        return 0

    monkeypatch.setattr("canvas.app.main._generate", fake)
    assert main(["generate", "--vram-cap-gb", "16", "--prompt", "a red boat", "--out", "out.avif"]) == 0
    assert seen["cap"] == "16"


def test_selftest_deep_is_still_read_only(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr("canvas.app.selftest.python_supports_tensorrt", lambda: False)
    assert main(["--selftest", "--deep", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["deep"] is True
    assert report["notes"] == ["TensorRT is skipped on Python 3.14."]
    assert "Collecting" not in capsys.readouterr().out


def test_update_check_uses_only_the_releases_endpoint():
    seen = []

    def fetcher(url: str) -> dict:
        seen.append(url)
        return {"tag_name": "v0.1.0"}

    assert check_for_update("0.1.0", fetcher) is None
    assert seen == [RELEASES_API]
    assert check_for_update("0.1.0", lambda url: {"tag_name": "v0.2.0"}) == "v0.2.0"

    def offline(url: str) -> dict:
        raise OSError("offline")

    assert check_for_update("0.1.0", offline) is None
    try:
        fetch_release("https://example.invalid/other")
    except ValueError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("expected a refused url")


def test_startup_refuses_or_launches(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(False, "Windows 11 24H2 or newer is required."))
    assert main([]) == 2
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr("canvas.app.main.probe_gpu", lambda: None)
    assert main([]) == 2
    monkeypatch.setattr(
        "canvas.app.main.probe_gpu",
        lambda: GpuInfo("RTX 4090", 24, 20, 8, 9, "fake"),
    )
    launched = []
    monkeypatch.setattr(
        "canvas.app.main._launch",
        lambda tier, splash=None, app=None: launched.append(tier),
    )
    assert main([]) == 0
    assert launched == [Tier.GB24]


def test_missing_models_do_not_fail_selftest_and_a_bad_hash_does(monkeypatch, tmp_path):
    monkeypatch.setattr("canvas.app.selftest.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr(
        "canvas.app.selftest.probe_gpu",
        lambda: GpuInfo("RTX 4090", 23.99, 20, 8, 9, "fake"),
    )
    report = selftest_report(tmp_path)
    assert report["gpu"]["tier"] == "24gb"
    assert report["ok"] is True
    assert report["models"][0]["detail"] == "not_downloaded"
    spec = load_catalog().by_role("face-detector")[0]
    folder = weight_dir(tmp_path, spec.repo)
    folder.mkdir(parents=True)
    name = next(iter(spec.files))
    (folder / name).write_bytes(b"not-the-model")
    ignored = selftest_report(tmp_path)
    row = next(item for item in ignored["models"] if item["id"] == spec.id)
    assert row["detail"] == "not_downloaded"
    assert ignored["ok"] is True
    assert (folder / name).read_bytes() == b"not-the-model"
    digest = spec.files[name]
    blob = tmp_path / "blobs" / digest
    blob.parent.mkdir()
    blob.write_bytes(b"not-the-model")
    broken = selftest_report(tmp_path)
    row = next(item for item in broken["models"] if item["id"] == spec.id)
    assert row["ok"] is False
    assert row["detail"] == "hash_mismatch"
    assert broken["ok"] is False


def test_generate_screens_the_prompt_before_the_generator(monkeypatch, tmp_path):
    order: list[str] = []
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr(
        "canvas.app.main.probe_gpu",
        lambda: GpuInfo("RTX 4090", 23.99, 20, 8, 9, "fake"),
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

        def release_request(self, *args: object, **kwargs: object) -> FakeScreened:
            order.append("review")
            return FakeScreened()

    class FakePipeline:
        def __init__(self, *args: object, **kwargs: object) -> None:
            order.append("pipeline")

        def export(self, *args: object, **kwargs: object) -> None:
            order.append("export")

    monkeypatch.setattr("canvas.app.main.AgeGate", FakeGate)
    monkeypatch.setattr("canvas.app.main.build_plan", lambda *args, **kwargs: order.append("plan") or object())
    monkeypatch.setattr("canvas.app.main.ensure_plan", lambda *args, **kwargs: order.append("ensure"))
    class FakeGenerator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.seed = 0
            self.models = None

    monkeypatch.setattr("canvas.app.main.ZImageGenerator", FakeGenerator)
    monkeypatch.setattr("canvas.app.main.Pipeline", FakePipeline)
    monkeypatch.setattr("canvas.app.main.EnlargeClient", lambda *args, **kwargs: object())
    code = main(["generate", "--prompt", "a red boat", "--aspect", "1:1", "--out", str(tmp_path / "out.avif"), "--json"])
    assert code == 0
    assert order.index("warm") < order.index("review")
    assert order.index("review") < order.index("ensure")
    assert order.index("supporting") < order.index("review")


def test_generate_json_reports_the_enlarge_error(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr(
        "canvas.app.main.probe_gpu",
        lambda: GpuInfo("RTX 4090", 23.99, 20, 8, 9, "fake"),
    )
    monkeypatch.setattr("canvas.app.main.check_gpu", lambda info: GpuCheck(True, "", Tier.GB24))
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

        def release_request(self, *args: object, **kwargs: object) -> FakeScreened:
            return FakeScreened()

    class FakePipeline:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def export(self, *args: object, **kwargs: object) -> None:
            raise EnlargeError("bad_input", "bad_input: cannot identify image file", False)

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
    code = main(
        ["generate", "--prompt", "a red boat", "--aspect", "1:1", "--out", str(tmp_path / "out.avif"), "--json"]
    )
    assert code == 1
    body = json.loads(capsys.readouterr().out)
    assert body == {"ok": False, "error": "bad_input: cannot identify image file"}


def test_edit_instruction_is_removed(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("LC_ALL", "ru_RU.UTF-8")
    monkeypatch.setenv("LANG", "ru_RU.UTF-8")
    monkeypatch.setenv("LC_MESSAGES", "ru_RU.UTF-8")
    code = main([
        "generate",
        "--edit-instruction",
        "paint the door blue",
        "--input",
        "in.png",
        "--out",
        str(tmp_path / "out.avif"),
        "--json",
    ])
    assert code == 2
    assert json.loads(capsys.readouterr().out) == {
        "ok": False,
        "error": "removed",
        "message": "Instruction editing has been removed.",
    }
    plain = main([
        "generate",
        "--edit-instruction",
        "paint the door blue",
        "--input",
        "in.png",
        "--out",
        str(tmp_path / "out.avif"),
    ])
    captured = capsys.readouterr()
    assert plain == 2
    assert captured.out == ""
    assert "Instruction editing has been removed." in captured.err
    assert "temporarily" not in captured.err.casefold()


def test_generate_command_refuses_an_unsupported_machine(capsys):
    code = main(["generate", "--prompt", "a red boat", "--aspect", "16:9", "--out", "out.avif", "--output-size", "4k", "--json"])
    assert code == 2
    body = json.loads(capsys.readouterr().out)
    assert body["ok"] is False
    assert body["error"]


def test_backdrop_is_plain_off_windows():
    assert apply_backdrop(0) == "plain"


def test_photoshoot_writes_a_heartbeat_from_the_entry_point(monkeypatch, tmp_path):
    """pythonw has no stderr. The photoshoot command still heartbeats during prep."""
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr("canvas.infra.stage_timer.HEARTBEAT_S", 0.05)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr("canvas.app.main.check_gpu", lambda _info: GpuCheck(True, "", Tier.GB16))
    monkeypatch.setattr("canvas.app.main.probe_gpu", lambda: None)
    monkeypatch.setattr("canvas.app.main.install_supporting", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("canvas.app.main.load_catalog_stack", lambda *_args, **_kwargs: object())

    class _Gate:
        def review_prompt(self, prompt: str) -> None:
            time.sleep(0.25)
            raise GenerationBlocked("prompt")

        def release_weights(self) -> None:
            return None

    monkeypatch.setattr("canvas.app.main.AgeGate", lambda _stack: _Gate())
    monkeypatch.setattr("canvas.app.main.model_rewrite", lambda *_args, **_kwargs: (lambda _prompt: _prompt))
    monkeypatch.setattr("canvas.app.main.model_generate", lambda *_args, **_kwargs: (lambda *_ignored: None))
    code = main([
        "photoshoot",
        "--prompt",
        "a red cup on a table",
        "--out",
        str(tmp_path / "frames"),
        "--frames",
        "1",
        "--json",
    ])
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "stage prep heartbeat" in log
    assert code == 1
