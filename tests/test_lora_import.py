"""«Import LoRA»: Model Forge packages into the Canvas LoRA library."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canvas.models import lora_library
from canvas.models.lora_library import (
    ACCEPTED_KINDS,
    import_lora,
    list_library,
    remove_entry,
    resolve_library,
)
from canvas.models.user_lora import LoraRejected, prepare_cli_lora, resolve_selected
from canvas.safety.gate import GenerationBlocked
from tests.fakes import World
from tests.lora_files import write_safetensors
from tests.voxlora_files import TENSORS, manifest, write_voxlora


class _Refusing:
    """An input check that refuses every picture."""

    def __init__(self) -> None:
        self.calls = 0

    def release_request(self, prompt, mode, inputs=(), *, explicit=False):
        self.calls += 1
        raise GenerationBlocked("input")


class _Counting:
    def __init__(self) -> None:
        self.gate = World().gate()
        self.calls = 0

    def release_request(self, prompt, mode, inputs=(), *, explicit=False):
        self.calls += 1
        return self.gate.release_request(prompt, mode, inputs, explicit=explicit)


@pytest.fixture
def home(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    return tmp_path


def _ready() -> bool:
    return True


def _import(path: Path, gate=None, **options):
    options.setdefault("base_ready", _ready)
    return import_lora(path, gate if gate is not None else World().gate(), **options)


def _code(path: Path, gate=None, **options) -> str:
    with pytest.raises(LoraRejected) as caught:
        _import(path, gate, **options)
    return caught.value.code


def test_the_library_takes_objects_styles_and_subjects():
    assert ACCEPTED_KINDS == frozenset({"object", "style", "subject"})


def test_a_valid_package_is_copied_with_name_trigger_and_previews(home: Path):
    package = write_voxlora(home / "in", previews=3)
    gate = _Counting()
    entry = _import(package, gate)
    assert gate.calls == 4
    assert entry.manifest.name == "Brass Lamp"
    assert entry.manifest.trigger == "brsslmp"
    assert [path.name for path in entry.previews] == ["preview.webp", "01.webp", "02.webp", "03.webp"]
    assert entry.folder.parent == lora_library.library_root()
    assert [item.id for item in list_library()] == [entry.id]
    lora = resolve_library(entry.id)
    assert lora is not None and lora.name == "Brass Lamp" and lora.strength_range == (0.0, 1.2)
    assert resolve_selected(entry.id) == lora
    assert remove_entry(entry.id) and list_library() == ()


def test_the_strength_range_comes_from_the_manifest_and_stays_inside_0_to_2(home: Path):
    package = write_voxlora(home / "in", record=manifest(scale_min=0.2, scale_max=3.5, scale_default=0.9))
    lora = _import(package).as_lora()
    assert lora.strength_range == (0.2, 2.0)
    assert lora.default_strength == 0.9


def test_attest_is_kept_as_information(home: Path):
    record = manifest(attest={"consent_obtained": False})
    entry = _import(write_voxlora(home / "in", record=record))
    assert entry.manifest.attest == {"consent_obtained": False}


def test_a_weights_sha_mismatch_is_invalid_and_nothing_is_stored(home: Path):
    package = write_voxlora(home / "in", sha="0" * 64)
    assert _code(package) == "lora.invalid"
    assert list_library() == ()


def test_a_header_that_disagrees_with_the_json_is_invalid(home: Path):
    header = {"voxprint.lora.kind": "style", "voxprint.lora.base": "z-image", "voxprint.subject.trigger": "brsslmp"}
    assert _code(write_voxlora(home / "in", header=header)) == "lora.invalid"
    header = {"voxprint.lora.kind": "object", "voxprint.lora.base": "z-image", "voxprint.subject.trigger": "other"}
    assert _code(write_voxlora(home / "b", header=header)) == "lora.invalid"
    header = {"voxprint.lora.kind": "object", "voxprint.lora.base": "sdxl", "voxprint.subject.trigger": "brsslmp"}
    assert _code(write_voxlora(home / "c", header=header)) == "lora.invalid"


def test_a_different_base_is_a_base_mismatch(home: Path):
    package = write_voxlora(home / "in", record=manifest(base="flux-dev"))
    assert _code(package) == "lora.base_mismatch"


def test_a_missing_base_is_requested_from_mdl(home: Path):
    asked: list[str] = []

    def fetch(progress):
        asked.append("z-image")
        progress("queued")
        return True

    lines: list[str] = []
    entry = _import(write_voxlora(home / "in"), base_ready=lambda: False, fetch_base=fetch, progress=lines.append)
    assert asked == ["z-image"] and lines == ["queued"]
    assert entry.id in {item.id for item in list_library()}


def test_a_base_that_cannot_be_requested_refuses_the_import(home: Path):
    package = write_voxlora(home / "in")
    assert _code(package, base_ready=lambda: False, fetch_base=lambda _progress: False) == "lora.base_missing"
    assert list_library() == ()


def test_the_default_fetch_queues_the_generator_with_mdl(home: Path, monkeypatch):
    queued: list[list[str]] = []
    import voxprint_common.mdl as mdl

    monkeypatch.setattr(mdl, "enqueue", lambda app, ids, manifests=None: queued.append(list(ids)))
    lines: list[str] = []
    assert lora_library._fetch_with_mdl(lines.append)
    assert queued and all(item.startswith("z-image-turbo") for item in queued[0])
    assert len(lines) == 1


def test_a_refused_preview_refuses_the_whole_package(home: Path):
    gate = _Refusing()
    assert _code(write_voxlora(home / "in"), gate) == "lora.preview"
    assert gate.calls == 1
    assert list_library() == ()


def test_a_preview_that_is_not_webp_or_too_large_is_refused(home: Path):
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, "PNG")
    assert _code(write_voxlora(home / "a", preview=buffer.getvalue())) == "lora.preview"
    big = io.BytesIO()
    Image.new("RGB", (1100, 64)).save(big, "WEBP")
    assert _code(write_voxlora(home / "b", preview=big.getvalue())) == "lora.preview"


def test_a_subject_lora_is_accepted_and_its_previews_are_checked(home: Path):
    gate = _Counting()
    entry = _import(write_voxlora(home / "in", record=manifest(kind="subject")), gate)
    assert entry.manifest.kind == "subject" and gate.calls == 3
    assert _code(write_voxlora(home / "r", record=manifest(kind="subject")), _Refusing()) == "lora.preview"


def test_an_unknown_kind_is_refused_with_the_kind_code(home: Path):
    package = write_voxlora(home / "in", record=manifest(kind="vehicle"))
    gate = _Counting()
    assert _code(package, gate) == "lora.kind"
    assert gate.calls == 0 and [item.manifest.kind for item in list_library()] == []


def test_a_compressed_or_unexpected_entry_is_invalid(home: Path):
    assert _code(write_voxlora(home / "in", compress=True)) == "lora.invalid"


def test_too_many_previews_is_invalid(home: Path):
    assert _code(write_voxlora(home / "in", previews=7)) == "lora.invalid"


def test_a_bare_safetensors_with_its_json_is_accepted_without_previews(home: Path):
    folder = home / "bare"
    folder.mkdir()
    weights = folder / "model.safetensors"
    write_safetensors(weights, TENSORS, {"voxprint.lora.kind": "style", "voxprint.lora.base": "z-image"})
    record = manifest(kind="style", trigger="")
    (folder / "voxprint-model.json").write_text(json.dumps(record), encoding="utf-8")
    gate = _Counting()
    entry = _import(weights, gate)
    assert entry.manifest.kind == "style" and entry.previews == () and gate.calls == 0


def test_a_stored_entry_of_a_refused_kind_is_not_listed(home: Path, monkeypatch):
    entry = _import(write_voxlora(home / "in"))
    monkeypatch.setattr(lora_library, "ACCEPTED_KINDS", frozenset({"style"}))
    assert list_library() == ()
    assert resolve_library(entry.id) is None


def test_a_changed_stored_file_drops_out_of_the_library(home: Path):
    entry = _import(write_voxlora(home / "in"))
    entry.weights.write_bytes(entry.weights.read_bytes() + b"\x00")
    assert list_library() == ()


def test_the_cli_takes_a_package_or_a_library_name(home: Path):
    package = write_voxlora(home / "in")
    opened = prepare_cli_lora(package)
    assert opened.trigger == "brsslmp" and opened.strength_range == (0.0, 2.0)
    entry = _import(package)
    named = prepare_cli_lora(entry.id)
    assert named.name == "Brass Lamp"
    unknown = write_voxlora(home / "s", record=manifest(kind="vehicle"))
    with pytest.raises(LoraRejected) as caught:
        prepare_cli_lora(unknown)
    assert caught.value.code == "lora.kind"


def test_a_name_without_latin_letters_gets_a_stable_id(home: Path):
    entry = _import(write_voxlora(home / "in", record=manifest(name="Акварель", kind="style")))
    assert entry.id.startswith("lora-") and len(entry.id) == 13


def test_an_encrypted_or_damaged_member_is_invalid(home: Path, monkeypatch):
    package = write_voxlora(home / "in")

    def broken(self, *args, **kwargs):
        raise RuntimeError("File is encrypted, password required for extraction")

    import zipfile

    monkeypatch.setattr(zipfile.ZipFile, "read", broken)
    assert _code(package) == "lora.invalid"
