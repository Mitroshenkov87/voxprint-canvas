"""Protocol 1 messages and the CLI fallback."""
import sys
from pathlib import Path

from canvas.infra.discovery import find_app, _split_windows_command
from canvas.infra.enlarge import EnlargeClient, dumps, loads


def test_messages_carry_protocol_1():
    encoded = dumps({"type": "hello", "client": "canvas", "version": "0.1.0", "protocol": 9})
    message = loads(encoded)
    assert message["protocol"] == 1
    assert message["client"] == "canvas"


def test_cli_fallback_streams_json_progress(tmp_path, monkeypatch):
    script = tmp_path / "enlarge_stub.py"
    script.write_text(
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "output = args[args.index('--output') + 1]\n"
        "scale = args[args.index('--scale') + 1]\n"
        "Path = __import__('pathlib').Path\n"
        "Path(output).write_bytes(b'ok')\n"
        "print(json.dumps({'protocol': 1, 'type': 'accepted', 'id': '1'}))\n"
        "print(json.dumps({'protocol': 1, 'type': 'progress', 'id': '1', 'pct': 10, 'stage': 'waiting_gpu'}))\n"
        "print(json.dumps({'protocol': 1, 'type': 'done', 'id': '1', 'output': output, 'width': 3840, 'height': 2160, 'format': 'avif'}))\n",
        encoding="utf-8",
    )
    source = tmp_path / "in.avif"
    source.write_bytes(b"in")
    dest = tmp_path / "out.avif"
    seen = []
    client = EnlargeClient([sys.executable, str(script)])
    result = client.enlarge(source, scale=2, output=dest, transport="cli", on_event=seen.append)
    assert result.output == dest
    assert result.width == 3840
    assert [item["type"] for item in seen] == ["accepted", "progress", "done"]
    assert seen[1]["stage"] == "waiting_gpu"
    assert dest.read_bytes() == b"ok"
    monkeypatch.setenv("CANVAS_ENLARGE_EXE", f"{sys.executable} {script}")
    found = find_app("Enlarge", "enlarge")
    assert found is not None and found.command[0] == sys.executable
    monkeypatch.setenv("CANVAS_ENLARGE_EXE", str(tmp_path / "missing-enlarge"))
    assert find_app("Enlarge", "enlarge") is None


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
