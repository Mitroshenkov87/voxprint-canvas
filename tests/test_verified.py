"""Verified hashes are reused until the file size or time changes."""
import hashlib
import json
from pathlib import Path

from canvas.infra.manifest import load_manifest, register_component
from canvas.models.pin import verify_file
from canvas.models.verified import VerifiedCache


def test_unchanged_file_is_not_hashed_again(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    path = tmp_path / "weight.bin"
    payload = b"abc"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    first = VerifiedCache.open(False)
    assert first.matches(path, digest) is True
    first.flush()
    calls: list[Path] = []
    real = verify_file

    def counting(target: Path, expected: str) -> None:
        calls.append(target)
        real(target, expected)

    monkeypatch.setattr("canvas.models.verified.verify_file", counting)
    second = VerifiedCache.open(False)
    assert second.matches(path, digest) is True
    assert calls == []
    deep = VerifiedCache.open(True)
    assert deep.matches(path, digest) is True
    assert calls == [path]
    path.write_bytes(b"abcd")
    third = VerifiedCache.open(False)
    assert third.matches(path, digest) is False


def test_verified_records_keep_the_component_map(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    shared = tmp_path / "home" / "shared"
    register_component(shared, "python-3.14", kind="runtime", version="3.14.8", path="runtime", sha256="ab" * 32)
    path = tmp_path / "weight.bin"
    path.write_bytes(b"xyz")
    digest = hashlib.sha256(b"xyz").hexdigest()
    cache = VerifiedCache.open(False)
    assert cache.matches(path, digest) is True
    cache.flush()
    document = json.loads((shared / "manifest.json").read_text(encoding="utf-8"))
    assert "python-3.14" in document["components"]
    stored = document["verified"][path.resolve().as_posix()]
    assert stored["sha256"] == digest
    assert stored["size"] == 3
    assert load_manifest(shared)["python-3.14"].version == "3.14.8"
