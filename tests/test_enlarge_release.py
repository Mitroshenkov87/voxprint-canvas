"""Live checks against the pinned Enlarge Linux release. Skipped unless the tarball is extracted."""
from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

import pytest

from canvas.infra.enlarge import EnlargeClient, EnlargeError, message_kind
from canvas.infra.enlarge_release import LINUX_SHA256, TAG, parse_sha256sums
from canvas.infra.gpu_lock import lock_path, release, try_acquire


def _png(width: int, height: int, rgb: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _note(text: str) -> None:
    path = os.environ.get("CANVAS_ENLARGE_REPORT", "")
    if path:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EnlargeClient:
    binary = os.environ.get("CANVAS_ENLARGE_BIN", "")
    if not binary:
        pytest.skip("set CANVAS_ENLARGE_BIN to the extracted voxprint-enlarge launcher")
    monkeypatch.setenv("CANVAS_CACHE", str(tmp_path / "cache"))
    return EnlargeClient([binary])


def test_pin_parses_sha256sums():
    text = (
        "a05642963218193d1fe4f92daf9abe66da7811a132786614dc0472472f2ec1c7  VoxprintEnlarge-Setup.exe\n"
        f"{LINUX_SHA256}  voxprint-enlarge-linux-x64.tar.gz\n"
    )
    parsed = parse_sha256sums(text)
    assert parsed["voxprint-enlarge-linux-x64.tar.gz"] == LINUX_SHA256
    assert TAG == "v0.2.0-rc.1"


def test_capabilities_disable_fourk_without_a_gpu(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    body = client.capabilities()
    assert body is not None
    assert body.get("protocol") == 1
    assert body.get("app") == "enlarge"
    gpu = body.get("gpu")
    assert isinstance(gpu, dict)
    assert gpu.get("status") == "none"
    assert client.ready() is False
    _note(f"capabilities: protocol 1, gpu.status={gpu.get('status')}, reason={gpu.get('reason')}, ready=false")


def test_dry_run_honours_explicit_output_and_exit_codes(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "voxprint-gpu.lock"))
    source = tmp_path / "in.png"
    source.write_bytes(_png(960, 1))
    missing = tmp_path / "missing.png"
    explicit = tmp_path / "chosen.avif"
    default_name = tmp_path / "in.enlarged-4x.avif"
    default_name.write_bytes(b"occupied")
    errors: list[dict] = []
    with pytest.raises(EnlargeError) as missing_image:
        client.enlarge(missing, output=explicit, dry_run=True, on_event=errors.append)
    assert missing_image.value.code == "bad_input"
    assert missing_image.value.retryable is False
    error = errors[-1]
    assert message_kind(error) == "error"
    assert error.get("type") == "error"
    assert error.get("op") == "error"
    assert error.get("protocol") == 1
    assert error.get("code") == "bad_input"
    assert error.get("retryable") is False
    _note("dry-run missing image: exit bad_input, retryable false, type and op are error")

    monkeypatch.delenv("VOXPRINT_CI_LANCZOS", raising=False)
    holder = try_acquire("canvas-generate", 600, lock_path())
    assert holder is None
    try:
        bare: list[dict] = []
        with pytest.raises(EnlargeError) as blocked:
            client.enlarge(source, output=tmp_path / "blocked.avif", dry_run=True, on_event=bare.append)
        blocked_error = next(item for item in bare if message_kind(item) == "error")
        assert blocked_error.get("type") == blocked_error.get("op")
        assert blocked.value.code == blocked_error.get("code")
        assert blocked.value.retryable is blocked_error.get("retryable")
        assert blocked.value.code in {"model_missing", "gpu_unsupported"}
        if blocked.value.code == "model_missing":
            assert blocked.value.retryable is True
            assert "Download the upscale model in Voxprint AI Enlarge" in str(blocked.value)
            assert "nomos" not in str(blocked.value)
        else:
            assert blocked.value.retryable is False
        _note(
            f"dry-run without CI lanczos: code={blocked.value.code}, retryable={blocked.value.retryable}, "
            f"message={blocked_error.get('message')}"
        )
        monkeypatch.setenv("VOXPRINT_CI_LANCZOS", "1")
        seen: list[dict] = []
        result = client.enlarge(source, output=explicit, dry_run=True, on_event=seen.append)
    finally:
        release(lock_path())
    kinds = [message_kind(item) for item in seen]
    assert kinds[0] == "accepted"
    assert kinds[-1] == "done"
    assert all(item.get("protocol") == 1 for item in seen)
    assert all(item.get("type") == item.get("op") for item in seen)
    assert result.output == explicit
    assert explicit.is_file() is False
    assert (tmp_path / "chosen-2.avif").is_file() is False
    assert result.output != default_name
    done = seen[-1]
    assert done.get("output") == str(explicit)
    assert done.get("ms") == 0
    assert max(int(done["width"]), int(done["height"])) == 3840
    _note(
        f"dry-run with CI lanczos: exit 0, output={done.get('output')}, "
        f"size={done.get('width')}x{done.get('height')}, ms={done.get('ms')}"
    )
