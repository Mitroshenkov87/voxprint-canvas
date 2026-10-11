"""16 GB INT4 edit route: registry order, the private Nunchaku runtime, and the loader."""
import platform
import sys
import tarfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from canvas.app.selftest import selftest_report
from canvas.models.edit import EditLoadError
from canvas.models.features import instruction_edit_available
from canvas.models.edit_route import (
    FP8_EDIT_ID,
    MIRROR_R256_ID,
    QUANTFUNC_R128_ID,
    QUANTFUNC_R256_ID,
    int4_weight_name,
    select_edit_route,
)
from canvas.models.int4_edit import load_int4_edit_pipeline
from canvas.models.loader import ModelNotReady, ZImageGenerator
from canvas.models.modes import Mode, Tier
from canvas.models.nunchaku import (
    NUNCHAKU_LICENCE,
    NUNCHAKU_REVISION,
    PUBLISHED_CU130_TORCH211,
    _extract_python,
    install_runtime_files,
    nunchaku_ready,
    select_nunchaku_wheel,
)
from canvas.models.plan import build_plan
from canvas.models.provision import ensure_plan, format_bytes, format_duration, nunchaku_runtime_notice
from canvas.models.registry import Catalog, load_catalog
from canvas.models.snapshots import _urls, snapshot_repo, weight_dir
from canvas.pipeline.bf16_schedule import GpuLedger, VramRefused
from canvas.pipeline.schedule import NUNCHAKU_PEAK_GB, Placement, choose_schedule, int4_edit_peak_gb, run_edit_int4
from canvas.safety.inputs import ImageRole, PendingInput
from tests.fakes import World, rgb

QUANT_REV = "4ca543a1e91db4183ee16fd5aa5950e0b02803e5"
R256_SHA = "d47054ff853d8d9cd6812386d4611cfe36fa9d1a5447549c8f0aa37c8aa79302"
R128_SHA = "ec36d349309057f245e77255c008fc97330f2bc280dc8d704de715bcbaf6841b"
R256_FILE = "nunchaku_qwen_image_edit_2511_best_quality_int4.safetensors"
R128_FILE = "nunchaku_qwen_image_edit_2511_balance_int4.safetensors"


def _screened(mode: Mode = Mode.INSTRUCTION):
    return World().gate().release_request(
        "paint the door blue", mode.value, [PendingInput(ImageRole.INIT, rgb())],
    )


def _with(catalog: Catalog, spec_id: str, **changes) -> Catalog:
    models = tuple(replace(spec, **changes) if spec.id == spec_id else spec for spec in catalog.models)
    return replace(catalog, models=models)


def test_16gb_prefers_quantfunc_r256_while_the_mirror_is_pending():
    catalog = load_catalog()
    mirror = catalog.by_id(MIRROR_R256_ID)
    assert mirror.mirror_status == "pending"
    assert mirror.status == "pending"
    assert mirror.files == {}
    assert mirror.licence == "Apache-2.0"
    assert mirror.repo == "Mitroshenkov87/voxprint-mirror-qwen-image-edit-2511-int4"
    selected = select_edit_route(catalog, Tier.GB16)
    assert selected.id == QUANTFUNC_R256_ID
    assert selected.quant == "int4"
    assert selected.variant == "r256"
    published = _with(
        catalog,
        MIRROR_R256_ID,
        status="ready",
        mirror_status="published",
        files={R256_FILE: "a" * 64},
    )
    assert select_edit_route(published, Tier.GB16).id == MIRROR_R256_ID
    assert select_edit_route(published, Tier.GB24).id == FP8_EDIT_ID
    without_r256 = _with(catalog, QUANTFUNC_R256_ID, status="disabled")
    assert select_edit_route(without_r256, Tier.GB16).id == QUANTFUNC_R128_ID
    neither = _with(without_r256, QUANTFUNC_R128_ID, status="disabled")
    assert select_edit_route(neither, Tier.GB16).id == FP8_EDIT_ID


def test_edit_route_env_selects_fp8_on_16gb(monkeypatch):
    monkeypatch.setenv("CANVAS_EDIT_ROUTE", "fp8")
    assert select_edit_route(load_catalog(), Tier.GB16).id == FP8_EDIT_ID


def test_quantfunc_pins_and_base_files_do_not_resolve_against_quantfunc():
    catalog = load_catalog()
    for spec_id, filename, digest, size in (
        (QUANTFUNC_R256_ID, R256_FILE, R256_SHA, 14_164_392_776),
        (QUANTFUNC_R128_ID, R128_FILE, R128_SHA, 12_654_443_120),
    ):
        spec = catalog.by_id(spec_id)
        assert spec.revision == QUANT_REV
        assert spec.licence == "Apache-2.0"
        assert spec.repo == "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511"
        assert spec.pinned()
        assert int4_weight_name(spec) == filename
        assert spec.files[filename] == digest
        assert spec.origins[filename][1] == filename
        urls = _urls(spec, filename, spec.revision, fallback=False)
        assert urls == (
            f"https://huggingface.co/QuantFunc/Nunchaku-Qwen-Image-EDIT-2511/resolve/{QUANT_REV}/{filename}",
        )
        meta_size = spec.nbytes
        assert meta_size > size
    encoder = "qwen2.5-vl-7b-text-encoder-fp8.safetensors"
    spec = catalog.by_id(QUANTFUNC_R128_ID)
    encoder_urls = _urls(spec, encoder, spec.revision, fallback=False)
    assert encoder_urls[0].startswith("https://huggingface.co/Mitroshenkov87/voxprint-mirror-qwen-image-edit-2511/")
    assert "QuantFunc" not in encoder_urls[0]
    index_urls = _urls(spec, "model_index.json", spec.revision, fallback=False)
    assert index_urls[0].startswith("https://huggingface.co/Qwen/Qwen-Image-Edit-2511/resolve/")
    left = weight_dir(Path("/models"), snapshot_repo(catalog.by_id(QUANTFUNC_R256_ID)))
    right = weight_dir(Path("/models"), snapshot_repo(catalog.by_id(QUANTFUNC_R128_ID)))
    assert left != right


def test_selftest_reports_the_active_edit_route(tmp_path: Path):
    report = selftest_report(tmp_path)
    route = report["edit_route"]
    assert isinstance(route, dict)
    assert set(route) == {"active_tier", "active", "16gb", "24gb"}
    low = route["16gb"]
    assert isinstance(low, dict)
    assert low["id"] == QUANTFUNC_R256_ID
    assert low["quant"] == "int4"
    assert low["rank"] == "r256"
    assert low["licence"] == "Apache-2.0"
    assert low["revision"] == QUANT_REV
    assert low["on_disk"] is False
    assert low["fallback"] == FP8_EDIT_ID
    high = route["24gb"]
    assert isinstance(high, dict)
    assert high["id"] == FP8_EDIT_ID
    assert high["quant"] == "fp8"
    assert high["fallback"] == ""


def test_nunchaku_wheel_is_a_private_side_environment():
    assert NUNCHAKU_LICENCE == "Apache-2.0"
    assert NUNCHAKU_REVISION == "85bad5510d409570527b018387491fd0c81e4727"
    assert ("linux_x86_64", "cp314") not in PUBLISHED_CU130_TORCH211
    assert ("linux_x86_64", "cp312") not in PUBLISHED_CU130_TORCH211
    assert ("win_amd64", "cp312") in PUBLISHED_CU130_TORCH211
    linux = select_nunchaku_wheel("linux", "x86_64")
    assert linux.abi == "cp311" and linux.python == "3.11.17" and linux.side_venv
    assert linux.nbytes == 147_399_162 + 48_949_967 + 531_045_934
    assert linux.file("nunchaku").sha256 == "02a762d94ed25ac572dbc59bb8109b0a9df6f9d875aa51228564e70307b3b68c"
    assert "cp311" in linux.reason and "private side environment" in linux.reason
    windows = select_nunchaku_wheel("win32", "amd64")
    assert windows.abi == "cp312" and windows.python == "3.12.15" and windows.side_venv
    assert windows.nbytes == 92_083_834 + 46_352_482 + 1_915_224_818
    assert windows.file("nunchaku").sha256 == "ddc015bfa6fc6b84957532823c23418f5b986cb61e3c02c97a646cfe2a1fd955"
    host = select_nunchaku_wheel(sys.platform, platform.machine())
    seconds = max(1, (host.nbytes + 8 * 1024 * 1024 - 1) // (8 * 1024 * 1024))
    notice = nunchaku_runtime_notice()
    assert notice == f"Nunchaku INT4 runtime is {format_bytes(host.nbytes)}. Estimated time {format_duration(seconds)}."


@pytest.mark.skipif(not instruction_edit_available(), reason="instruction_edit is model_in_preparation")
def test_declined_int4_runtime_installs_the_fp8_pack(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr("canvas.models.provision.runtime_is_installed", lambda plan: False)
    monkeypatch.setattr(
        "canvas.models.provision.install_runtime_files",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("runtime download")),
    )
    downloaded: list[str] = []

    def fake_download(spec, root, progress=None):
        del root, progress
        downloaded.append(spec.id)
        return tmp_path

    monkeypatch.setattr("canvas.models.provision.download_snapshot", fake_download)
    notes: list[str] = []

    def confirm(notice: str) -> bool:
        notes.append(notice)
        return not notice.startswith("Nunchaku")

    catalog = load_catalog()
    plan = build_plan(catalog, _screened(), Tier.GB16)
    ensure_plan(catalog, plan, notes.append, confirm)
    assert any(item.startswith("Nunchaku INT4 runtime is ") for item in notes)
    assert downloaded == [FP8_EDIT_ID]
    assert select_edit_route(catalog, Tier.GB16).id == FP8_EDIT_ID


@pytest.mark.skipif(not instruction_edit_available(), reason="instruction_edit is model_in_preparation")
def test_accepted_int4_downloads_the_runtime_and_the_r256_pack(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr("canvas.models.provision.runtime_is_installed", lambda plan: False)
    installed: list[str] = []
    monkeypatch.setattr(
        "canvas.models.provision.install_runtime_files",
        lambda plan, root, progress=None: installed.append(plan.file("nunchaku").filename),
    )
    downloaded: list[str] = []
    monkeypatch.setattr(
        "canvas.models.provision.download_snapshot",
        lambda spec, root, progress=None: downloaded.append(spec.id) or tmp_path,
    )
    notes: list[str] = []
    catalog = load_catalog()
    plan = build_plan(catalog, _screened(), Tier.GB16)
    ensure_plan(catalog, plan, notes.append, confirm=None)
    assert installed and "cp31" in installed[0]
    assert downloaded == [QUANTFUNC_R256_ID]
    assert any("Qwen-Image-Edit-2511 INT4 is " in item for item in notes)


def test_int4_loader_offloads_one_block_and_unloads_the_encoder(tmp_path: Path, monkeypatch):
    seen: dict[str, object] = {}

    class Transformer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen["path"] = path
            seen["kwargs"] = kwargs
            return Transformer()

        def set_offload(self, enabled, use_pin_memory=False, num_blocks_on_gpu=0):
            seen["offload"] = (enabled, use_pin_memory, num_blocks_on_gpu)

    class Module:
        NunchakuQwenImageTransformer2DModel = Transformer

    encoder = SimpleNamespace(to=lambda device: seen.setdefault("devices", []).append(device))
    pipe = SimpleNamespace(
        enable_vae_tiling=lambda: seen.__setitem__("tiled", True),
        encode_prompt=lambda *args, **kwargs: "embeds",
        text_encoder=encoder,
    )
    monkeypatch.setattr("canvas.models.int4_edit._import_nunchaku", lambda: Module)
    monkeypatch.setattr(
        "canvas.models.int4_edit.load_edit_pipeline",
        lambda folder, dtype, *, low_vram=False, transformer=None: seen.__setitem__("transformer", transformer) or pipe,
    )
    monkeypatch.setattr(
        "canvas.models.int4_edit._import_torch",
        lambda: SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: seen.__setitem__("cache", True)),
        ),
    )
    weight = tmp_path / R256_FILE
    weight.write_bytes(b"weights")
    loaded = load_int4_edit_pipeline(weight, tmp_path, "bf16")
    assert seen["kwargs"] == {"torch_dtype": "bf16", "zero_cond_t": True}
    assert seen["offload"] == (True, True, 1)
    assert seen["tiled"] is True
    assert seen["transformer"] is not None
    assert loaded.encode_prompt() == "embeds"
    assert loaded.text_encoder is None
    assert seen["devices"] == ["cuda", "cpu"]
    assert seen["cache"] is True


def test_int4_peak_stays_at_or_under_12_gb_and_a_larger_stage_is_refused():
    for width, height in ((2048, 1152), (2048, 2048)):
        assert int4_edit_peak_gb(width, height) <= NUNCHAKU_PEAK_GB
        trace = run_edit_int4(width=width, height=height, ledger=GpuLedger(16))
        assert trace.peak_gb <= NUNCHAKU_PEAK_GB
        assert trace.tiled is True
    with pytest.raises(VramRefused):
        run_edit_int4(width=8192, height=8192, ledger=GpuLedger(16))
    choice = choose_schedule(16, Mode.INSTRUCTION, ram_gb=64, int4_edit=True)
    assert choice.placement is Placement.NUNCHAKU_INT4
    assert choice.peak_target_gb == NUNCHAKU_PEAK_GB
    assert choose_schedule(24, Mode.INSTRUCTION, ram_gb=64, int4_edit=True).placement is Placement.RESIDENT


def test_text_to_image_stays_bf16_while_int4_generator_is_disabled():
    catalog = load_catalog()
    plan = build_plan(catalog, World().gate().release_request("a red boat", Mode.TEXT.value, []), Tier.GB16)
    assert plan.quant == "bf16"
    assert plan.model_id == "z-image-turbo-bf16-16gb"
    spec = catalog.by_id("z-image-turbo-svdquant-int4-r256")
    assert spec.status == "disabled"
    assert spec.licence == "Apache-2.0"
    assert spec.variant == "r256"
    assert spec.pinned()
    assert nunchaku_ready(spec) is False
    screened = _screened()
    generator = ZImageGenerator(catalog, Tier.GB16, ledger=GpuLedger(16), ram_gb=64)
    with pytest.raises(ModelNotReady):
        generator.generate(screened, 2048, 1152)
    assert generator.last_plan is not None
    assert generator.last_plan.model_id == QUANTFUNC_R256_ID
    assert generator.last_plan.quant == "int4"


def test_python_extract_keeps_the_downloaded_wheels(tmp_path: Path):
    source = tmp_path / "stage" / "python" / "bin"
    source.mkdir(parents=True)
    (source / "python3").write_text("interp", encoding="utf-8")
    archive = tmp_path / "runtime" / "cpython.tar.gz"
    archive.parent.mkdir()
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(tmp_path / "stage" / "python", arcname="python")
    wheel = tmp_path / "runtime" / "torch.whl"
    wheel.write_bytes(b"wheel")
    _extract_python(archive, tmp_path / "runtime")
    assert wheel.read_bytes() == b"wheel"
    assert (tmp_path / "runtime" / "python" / "bin" / "python3").is_file()
    assert archive.is_file()


def test_a_bad_runtime_hash_is_refused(tmp_path: Path):
    plan = select_nunchaku_wheel("linux", "x86_64")

    def fetch(url: str, dest: Path) -> None:
        del url
        dest.write_bytes(b"not-the-wheel")

    with pytest.raises(OSError, match="could not be installed"):
        install_runtime_files(plan, tmp_path, fetch=fetch, extract=lambda archive, dest: None, pip=lambda *args: None)


def test_loader_refuses_a_plan_over_the_peak(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("canvas.models.int4_edit.int4_edit_peak_gb", lambda width, height: 13)
    weight = tmp_path / "transformer.safetensors"
    weight.write_bytes(b"x")
    with pytest.raises(EditLoadError) as caught:
        load_int4_edit_pipeline(weight, tmp_path, "bf16")
    assert "VRAM" in caught.value.detail
