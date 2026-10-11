"""The Enlarge CLI contract v1 is the only inter-app contract."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from canvas.infra.discovery import find_app, _split_windows_command
from canvas.infra.manifest import register_component
from canvas.infra.paths import shared_dir
from canvas.infra.enlarge import (
    EnlargeClient,
    EnlargeError,
    capabilities_ready,
    enlarge_argv,
    exit_error,
    loads,
    message_kind,
)

FAKE = Path(__file__).resolve().parent / "fixtures" / "enlarge_fake.py"
_EXIT_CODES = (1, 2, 3, 4, 5, 6, 7, 8, 130)


def _events(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _shape(events: list[dict]) -> list[dict]:
    shaped = []
    for event in events:
        if event.get("type") == "done":
            shaped.append({key: event[key] for key in ("type", "id", "width", "height", "format", "ms")})
        else:
            shaped.append(event)
    return shaped


def test_progress_lines_are_protocol_1_json():
    message = loads('{"protocol": 1, "type": "progress", "id": "1", "pct": 5, "stage": "waiting_gpu", "eta_s": 1.5}\n')
    assert message["type"] == "progress"
    assert message["stage"] == "waiting_gpu"
    with pytest.raises(EnlargeError):
        loads('{"type": "done"}')


def test_enlarge_argv_is_the_only_command_line(tmp_path):
    source = tmp_path / "in.avif"
    dest = tmp_path / "out.avif"
    argv = enlarge_argv(
        ["voxprint-enlarge"], source=source, output=dest, long_side=3840, format_name="avif", parent_pid=4242,
    )
    assert argv == [
        "voxprint-enlarge",
        "upscale",
        "--input",
        str(source.resolve()),
        "--output",
        str(dest),
        "--long-side",
        "3840",
        "--format",
        "avif",
        "--preset",
        "photo",
        "--json-progress",
        "--parent-lock-pid",
        "4242",
    ]
    root = Path(__file__).resolve().parents[1]
    enlarge_text = (root / "canvas" / "infra" / "enlarge.py").read_text(encoding="utf-8")
    pipeline_text = (root / "canvas" / "pipeline" / "pipeline.py").read_text(encoding="utf-8")
    assert "QLocalSocket" not in enlarge_text
    assert "--serve" not in enlarge_text
    assert "release_for_peer" not in pipeline_text
    assert "def enlarge_argv" in enlarge_text


def _png(path: Path, width: int = 16, height: int = 9) -> None:
    Image.new("RGB", (width, height), (1, 2, 3)).save(path, format="PNG")


def test_fake_cli_mirrors_dry_run_and_captures_stderr(tmp_path):
    source = tmp_path / "in.png"
    _png(source)
    dest = tmp_path / "out.avif"
    dry_dest = tmp_path / "dry.avif"
    parent = "42"
    env = os.environ.copy()
    env["VOXPRINT_GPU_PARENT_PID"] = parent
    dry = subprocess.run(
        enlarge_argv(
            [sys.executable, str(FAKE)],
            source=source,
            output=dry_dest,
            long_side=3840,
            format_name="avif",
            parent_pid=int(parent),
            dry_run=True,
        ),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert dry.returncode == 0
    assert "enlarge: log" in dry.stderr
    assert "raw cli detail" not in dry.stdout
    seen: list[dict] = []
    client = EnlargeClient([sys.executable, str(FAKE)])
    result = client.enlarge(source, output=dest, on_event=seen.append)
    assert result.output == dest
    assert result.width == 3840
    assert result.height == 2160
    assert result.format == "avif"
    assert _shape(seen) == _shape(_events(dry.stdout))
    assert [item["stage"] for item in seen if item["type"] == "progress"] == [
        "queued", "waiting_gpu", "loading", "tiles", "encoding",
    ]
    accepted = seen[0]
    assert accepted["type"] == "accepted"
    assert accepted["id"] == "1"
    assert accepted["queue_pos"] == 0
    done = seen[-1]
    assert done["ms"] == 12
    call = json.loads(Path(str(dest) + ".call.json").read_text(encoding="utf-8"))
    assert call["parent_env"] == str(os.getpid())
    assert call["argv"][call["argv"].index("--parent-lock-pid") + 1] == str(os.getpid())
    assert call["argv"][call["argv"].index("--long-side") + 1] == "3840"
    assert call["argv"][call["argv"].index("--preset") + 1] == "photo"
    assert "--scale" not in call["argv"]
    assert seen[0]["op"] == "accepted"
    assert "enlarge: log" not in str(result)


@pytest.mark.parametrize("code", _EXIT_CODES)
def test_exit_codes_map_to_user_messages(tmp_path, monkeypatch, code):
    source = tmp_path / "in.avif"
    source.write_bytes(b"in")
    dest = tmp_path / f"out-{code}.avif"
    monkeypatch.setenv("ENLARGE_FAKE_EXIT", str(code))
    expected = exit_error(code)
    with pytest.raises(EnlargeError) as caught:
        EnlargeClient([sys.executable, str(FAKE)]).enlarge(source, output=dest)
    assert caught.value.code == expected.code
    assert caught.value.retryable is expected.retryable
    assert "enlarge: log" not in str(caught.value)
    assert str(caught.value).startswith(f"{expected.code}: ")
    if code == 7:
        assert "Download the upscale model in Voxprint AI Enlarge" in str(caught.value)
        assert "raw cli detail" not in str(caught.value)
        assert "foo" not in str(caught.value)
    else:
        assert "raw cli detail model-id foo" in str(caught.value)


def test_message_kind_accepts_type_or_op():
    assert message_kind({"protocol": 1, "type": "done"}) == "done"
    assert message_kind({"protocol": 1, "op": "accepted"}) == "accepted"
    assert message_kind({"protocol": 1, "type": "progress", "op": "progress"}) == "progress"
    assert message_kind({"protocol": 1, "type": "done", "op": "error"}) == "done"
    assert message_kind({"protocol": 1}) == ""


def test_shipped_retryable_flags():
    assert exit_error(7).retryable is True
    assert exit_error(6).retryable is False
    assert exit_error(8).retryable is False
    assert exit_error(1).retryable is False
    assert exit_error(5).retryable is True


def test_op_only_progress_still_finishes(tmp_path, monkeypatch):
    monkeypatch.setenv("ENLARGE_FAKE_KIND", "op")
    source = tmp_path / "in.png"
    _png(source)
    seen: list[dict] = []
    result = EnlargeClient([sys.executable, str(FAKE)]).enlarge(
        source, output=tmp_path / "out.avif", on_event=seen.append,
    )
    assert result.output.name == "out.avif"
    assert all("type" not in item for item in seen)
    assert [message_kind(item) for item in seen][0] == "accepted"
    assert message_kind(seen[-1]) == "done"


def test_unknown_exit_uses_the_internal_message(tmp_path, monkeypatch):
    source = tmp_path / "in.avif"
    source.write_bytes(b"in")
    monkeypatch.setenv("ENLARGE_FAKE_EXIT", "99")
    with pytest.raises(EnlargeError) as caught:
        EnlargeClient([sys.executable, str(FAKE)]).enlarge(source, output=tmp_path / "out.avif")
    assert caught.value.code == "internal"
    assert "raw cli detail model-id foo" in str(caught.value)
    assert "internal: " in str(caught.value)


def test_windows_enlarge_override_keeps_backslashes(monkeypatch):
    raw = r"C:\hostedtoolcache\windows\Python\3.12.10\x64\python.exe C:\temp\caps.py"
    assert _split_windows_command(raw) == [
        r"C:\hostedtoolcache\windows\Python\3.12.10\x64\python.exe",
        r"C:\temp\caps.py",
    ]
    quoted = r'"C:\Program Files\Voxprint\voxprint-enlarge.exe"'
    assert _split_windows_command(quoted) == [r"C:\Program Files\Voxprint\voxprint-enlarge.exe"]
    monkeypatch.setattr("canvas.infra.discovery.sys.platform", "win32")
    monkeypatch.setenv("CANVAS_ENLARGE_EXE", raw)
    monkeypatch.setattr("canvas.infra.discovery.Path.is_file", lambda self: True)
    found = find_app("Enlarge", "enlarge")
    assert found is not None
    assert found.command[0].endswith(r"python.exe")
    assert found.command[1] == r"C:\temp\caps.py"


def test_capabilities_probe_is_cached(tmp_path, monkeypatch):
    script = tmp_path / "caps.py"
    body = {"protocol": 1, "app": "enlarge", "version": "0.1.0", "gpu": {"status": "ok", "reason": None}}
    script.write_text(
        "import json, sys\n"
        f"json.dump({body!r}, sys.stdout)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CANVAS_ENLARGE_EXE", f"{sys.executable} {script}")
    monkeypatch.setenv("CANVAS_ENLARGE_VERSION", "0.1.0")
    monkeypatch.setenv("CANVAS_CACHE", str(tmp_path / "cache"))
    client = EnlargeClient()
    assert client.installed()
    first = client.capabilities()
    assert first is not None and first["app"] == "enlarge"
    script.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    cached = client.capabilities()
    assert cached is not None and cached["version"] == "0.1.0"
    assert Path(tmp_path / "cache" / "voxprint" / "enlarge-capabilities.json").is_file()


@pytest.mark.parametrize(
    ("protocol", "gpu", "enabled"),
    [
        (1, "ok", True),
        (1, "warn", True),
        (1, "none", False),
        (0, "ok", False),
    ],
)
def test_capabilities_json_is_the_enable_state(tmp_path, monkeypatch, protocol, gpu, enabled):
    monkeypatch.setenv("CANVAS_ENLARGE_EXE", f"{sys.executable} {FAKE}")
    monkeypatch.setenv("CANVAS_ENLARGE_VERSION", f"{protocol}-{gpu}")
    monkeypatch.setenv("CANVAS_CACHE", str(tmp_path / "cache" / f"{protocol}-{gpu}"))
    monkeypatch.setenv("ENLARGE_FAKE_PROTOCOL", str(protocol))
    monkeypatch.setenv("ENLARGE_FAKE_GPU", gpu)
    client = EnlargeClient()
    assert client.installed()
    assert client.ready() is enabled
    assert capabilities_ready(client.capabilities()) is enabled


def test_enlarge_is_found_on_a_standard_path_and_in_the_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.delenv("CANVAS_ENLARGE_EXE", raising=False)
    binary = tmp_path / "enlarge" / "venv" / "bin" / "voxprint-enlarge"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    found = find_app("Enlarge", "enlarge")
    assert found is not None
    assert os.path.normcase(found.command[0]) == os.path.normcase(str(binary))
    binary.unlink()
    recorded = tmp_path / "shared" / "apps" / "voxprint-enlarge"
    recorded.parent.mkdir(parents=True)
    recorded.write_text("#!/bin/sh\n", encoding="utf-8")
    recorded.chmod(0o755)
    register_component(
        shared_dir(), "enlarge-cli", kind="tool", version="0",
        path="apps", sha256="ab" * 32, app="enlarge",
    )
    found = find_app("Enlarge", "enlarge")
    assert found is not None
    assert os.path.normcase(found.command[0]) == os.path.normcase(str(recorded))


def test_fake_enlarge_rejects_a_file_that_is_not_a_png(tmp_path):
    source = tmp_path / "native.avif"
    source.write_bytes(b"VOXPRINT rgb 2 2\n" + bytes(12))
    with pytest.raises(EnlargeError) as caught:
        EnlargeClient([sys.executable, str(FAKE)]).enlarge(source, output=tmp_path / "out.avif")
    assert caught.value.code == "bad_input"
    assert str(caught.value) == "bad_input: cannot identify image file"
    assert "Enlarge could not read this picture." not in str(caught.value)


def test_fake_enlarge_accepts_a_decodable_png(tmp_path):
    source = tmp_path / "native.png"
    _png(source, 16, 9)
    dest = tmp_path / "out.avif"
    result = EnlargeClient([sys.executable, str(FAKE)]).enlarge(source, output=dest)
    assert result.width == 3840
    assert result.height == 2160
    with Image.open(dest) as image:
        assert image.format == "AVIF"
        assert image.size == (3840, 2160)


def test_missing_capabilities_disables_fourk(tmp_path, monkeypatch):
    script = tmp_path / "down.py"
    script.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    monkeypatch.setenv("CANVAS_ENLARGE_EXE", f"{sys.executable} {script}")
    monkeypatch.setenv("CANVAS_CACHE", str(tmp_path / "cache"))
    client = EnlargeClient()
    assert client.installed()
    assert client.ready() is False
    assert client.capabilities() is None
