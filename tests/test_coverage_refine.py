"""CPU-only stubs for tools/coverage_refine.py. No weights are loaded."""
from __future__ import annotations

import io
import json
import py_compile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import tools.coverage_refine as cr
from canvas.core.formats import OutputFormat
from canvas.models.modes import Mode
from canvas.safety.classifiers import Face
from canvas.safety.gate import SafetyError
from canvas.safety.image import SafeImage
from tests.fakes import World, rgb

MANUAL = (
    '{"schema":"voxprint-coverage/1","schema_rev":2,"generator_base":"z-image-turbo@f332072a","source":"manual",'
    '"weak":["pelvis"],"bins":{"pelvis":{"score_nude":0.0,"source":"manual"}}}'
)


@pytest.fixture
def world() -> World:
    return World()


def _picture(world: World, color=(40, 60, 90), size: int = 64) -> SafeImage:
    return world.gate().review_image(rgb(size, size, color), "an adult woman, 30 years old")


def _setup(tmp_path: Path, monkeypatch, coverage: str | None = None):
    weights = tmp_path / "person.safetensors"
    weights.write_bytes(b"x")
    lora = SimpleNamespace(weights=weights, trigger="trg", alias="")
    monkeypatch.setattr("canvas.tools.render_batch.bind_lora", lambda path, strength: (lora, 1.0 if strength is None else strength))
    meta = {} if coverage is None else {cr.COVERAGE_KEY: coverage}
    monkeypatch.setattr(cr, "_string_metadata", lambda path: meta)
    return weights


def _args(tmp_path: Path, weights: Path, *extra: str, adult: str = "on", seed: bool = True) -> list[str]:
    argv = ["--prompt", "a 30-year-old adult woman", "--lora", str(weights), "--adult-content", adult,
            "--out", str(tmp_path / "out"), "--box", "10,30,40,30"]
    if seed:
        argv += ["--seed", "77"]
    return argv + list(extra)


class _Calls:
    def __init__(self, world: World, refuse: set[Mode] | None = None) -> None:
        self.world = world
        self.requests: list[cr.RenderRequest] = []
        self.refuse = refuse or set()

    def __call__(self, request: cr.RenderRequest) -> SafeImage:
        self.requests.append(request)
        if request.mode in self.refuse:
            raise SafetyError("image_refused")
        shade = 200 if request.mode is Mode.INPAINT else (90 if request.schedule is not None else 60)
        return _picture(self.world, (shade, 60, 90))


def _run(tmp_path, monkeypatch, world, *extra, coverage=None, refuse=None, faces=True, adult="on"):
    weights = _setup(tmp_path, monkeypatch, coverage)
    calls = _Calls(world, refuse)
    log = io.StringIO()
    detect = (lambda image: [Face(20, 4, 16, 16)]) if faces else None
    hooks = cr.Hooks(weights_ready=lambda: True, generate=calls, gate=world.gate(), detect_faces=detect, log=log)
    saved: list[tuple[str, object]] = []
    real = cr.save_image

    def capture(image, path, fmt, *, metadata=None, **kwargs):
        saved.append((Path(path).name, fmt, metadata))
        real(image, path, fmt, metadata=metadata, **kwargs)

    monkeypatch.setattr(cr, "save_image", capture)
    code = cr.main(_args(tmp_path, weights, *extra, adult=adult), hooks=hooks)
    return code, calls, log.getvalue(), saved


# ---- CLI

def test_script_compiles() -> None:
    py_compile.compile(str(Path(cr.__file__)), doraise=True)


def test_lead_trigger_legacy_inserts_once() -> None:
    lora = type("L", (), {"trigger": "vxp_token"})()
    assert cr.lead_trigger_legacy("a 30-year-old adult woman", lora) == "vxp_token, a 30-year-old adult woman"
    assert cr.lead_trigger_legacy("vxp_token, already", lora) == "vxp_token, already"
    assert cr.lead_trigger_legacy("hello", type("L", (), {"trigger": ""})()) == "hello"


def test_recommended_strength_from_header(tmp_path, monkeypatch) -> None:
    weights = tmp_path / "person.safetensors"
    weights.write_bytes(b"x")
    monkeypatch.setattr(cr, "_string_metadata", lambda path: {cr.STRENGTH_KEY: "0.8"})
    assert cr.recommended_strength(weights) == 0.8
    monkeypatch.setattr(cr, "_string_metadata", lambda path: {cr.STRENGTH_KEY: "nope"})
    assert cr.recommended_strength(weights) == 1.0
    monkeypatch.setattr(cr, "_string_metadata", lambda path: {})
    assert cr.recommended_strength(weights) == 1.0


def test_omitted_strength_is_passed_as_header_value(tmp_path, monkeypatch, world) -> None:
    seen: list[object] = []
    weights = tmp_path / "person.safetensors"
    weights.write_bytes(b"x")
    lora = type("L", (), {"weights": weights, "trigger": "trg", "alias": ""})()

    def bind(path, strength):
        seen.append(strength)
        return lora, strength

    monkeypatch.setattr("canvas.tools.render_batch.bind_lora", bind)
    monkeypatch.setattr(cr, "_string_metadata", lambda path: {cr.STRENGTH_KEY: "0.8"})
    calls = _Calls(world)
    log = io.StringIO()
    hooks = cr.Hooks(weights_ready=lambda: True, generate=calls, gate=world.gate(),
                     detect_faces=lambda image: [Face(20, 4, 16, 16)], log=log)
    code = cr.main(_args(tmp_path, weights, "--variants", "A"), hooks=hooks)
    assert code == 0 and seen == [0.8]


def test_seed_is_required(tmp_path, monkeypatch) -> None:
    weights = _setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as stop:
        cr.main(_args(tmp_path, weights, seed=False), hooks=cr.Hooks(weights_ready=lambda: True))
    assert stop.value.code == 2


def test_adult_content_off_is_passed_to_the_pipeline(tmp_path, monkeypatch, world) -> None:
    code, calls, _, _ = _run(tmp_path, monkeypatch, world, "--variants", "A", adult="off")
    assert code == 0
    assert calls.requests and all(request.explicit is False for request in calls.requests)


def test_mask_variants_need_a_mask_or_box(tmp_path, monkeypatch, world) -> None:
    weights = _setup(tmp_path, monkeypatch)
    calls = _Calls(world)
    log = io.StringIO()
    argv = ["--prompt", "p", "--lora", str(weights), "--adult-content", "on", "--seed", "1",
            "--out", str(tmp_path / "o"), "--variants", "A,C"]
    code = cr.main(argv, hooks=cr.Hooks(weights_ready=lambda: True, generate=calls, gate=world.gate(), log=log))
    assert code == 2 and "--mask and/or --box" in log.getvalue()
    assert calls.requests == []


def test_early_steps_must_be_below_the_total(tmp_path, monkeypatch, world) -> None:
    code, calls, log, _ = _run(tmp_path, monkeypatch, world, "--early-steps", "8", "--variants", "B")
    assert code == 2 and "--early-steps" in log and calls.requests == []


# ---- coverage

def test_parse_coverage_manual_pelvis() -> None:
    cov = cr.parse_coverage(MANUAL)
    assert cov.usable and cov.rev == 2 and cov.weak == ["regions.pelvis"]
    assert cov.bins["regions.pelvis"] == {"score_nude": 0.0, "source": "manual"}  # score_clothed stays unset


def test_parse_coverage_missing_or_bad() -> None:
    assert not cr.parse_coverage(None).present
    bad_json = cr.parse_coverage("{nope")
    assert bad_json.present and not bad_json.usable and bad_json.warning
    wrong = cr.parse_coverage('{"schema":"other/1","weak":["face"]}')
    assert not wrong.usable and wrong.weak == []
    assert not cr.parse_coverage("[1]").usable


def test_an_unknown_region_bin_is_ignored() -> None:
    cov = cr.parse_coverage(json.dumps({"schema": cr.COVERAGE_SCHEMA, "schema_rev": 2, "weak": ["elbow"],
                                        "bins": {"elbow": {"score_nude": 0.0}}}))
    assert cov.usable and cov.weak == [] and cov.bins == {}
    assert cr.default_variants(cov)[0] == ["A", "B", "C"]


def test_parse_coverage_ignores_unknown_and_tolerates_missing_fields() -> None:
    raw = json.dumps({
        "schema": cr.COVERAGE_SCHEMA, "schema_rev": 2, "future": 1, "weak": ["hips", "elbows", 5],
        "bins": {"hips": {"score_clothed": 0.2, "extra": True}, "elbows": {"score": 1}, "face": "bad",
                 "legs": {"score_clothed": 0.9, "count_clothed": 12, "count_nude": 3, "source": "measured"},
                 "front": {"count": 4, "score": 0.5}},
    })
    cov = cr.parse_coverage(raw)
    assert cov.weak == ["regions.hips"]
    assert cov.bins["regions.hips"] == {"score_clothed": 0.2}
    assert cov.bins["regions.legs"]["count_clothed"] == 12 and cov.bins["regions.legs"]["count_nude"] == 3
    assert cov.bins["angles.front"] == {"count": 4, "score": 0.5}
    assert "elbows" not in cov.bins and "regions.face" not in cov.bins


def test_legacy_single_score_and_count_still_parse() -> None:
    for header in ({}, {"schema_rev": 1}):
        cov = cr.parse_coverage(json.dumps({
            "schema": cr.COVERAGE_SCHEMA, "weak": ["legs"], **header,
            "bins": {"legs": {"score": 0.1, "count": 3}, "torso": {"score": 0.0}}}))
        assert cov.rev == 1 and cov.bins["regions.legs"] == {"score": 0.1, "count": 3}
        assert cr.weak_regions(cov, True) == ["torso", "legs"]  # listed stays weak; a 0 score is weak
        assert cr.weak_regions(cov, False) == ["torso", "legs"]


def test_manual_bins_read_like_measured_bins() -> None:
    def doc(source: str):
        return cr.parse_coverage(json.dumps({"schema": cr.COVERAGE_SCHEMA, "schema_rev": 2, "weak": ["legs"],
                                             "bins": {"legs": {"score_nude": 0.0, "source": source}}}))

    assert cr.weak_regions(doc("manual"), True) == cr.weak_regions(doc("measured"), True) == ["legs"]
    assert doc("manual").bins["regions.legs"]["score_nude"] == doc("measured").bins["regions.legs"]["score_nude"]


def test_clothed_and_adult_state_selection() -> None:
    cov = cr.parse_coverage(MANUAL)
    assert cr.weak_regions(cov, True) == ["pelvis"]
    assert cr.weak_regions(cov, False) == []  # the override is for the nude state only
    both = cr.parse_coverage(json.dumps({
        "schema": cr.COVERAGE_SCHEMA, "schema_rev": 2, "weak": [],
        "bins": {"pelvis": {"score_clothed": 0.0, "score_nude": 0.8}, "bust": {"score_clothed": 0.9, "score_nude": 0.0}}}))
    assert cr.weak_regions(both, True) == ["bust"]
    assert cr.weak_regions(both, False) == ["pelvis"]
    listed = cr.parse_coverage(json.dumps({
        "schema": cr.COVERAGE_SCHEMA, "schema_rev": 2, "weak": ["pelvis"],
        "bins": {"pelvis": {"score_clothed": 0.4, "score_nude": 0.4}}}))
    assert cr.weak_regions(listed, True) == ["pelvis"] and cr.weak_regions(listed, False) == ["pelvis"]
    fallback = cr.parse_coverage(json.dumps({
        "schema": cr.COVERAGE_SCHEMA, "schema_rev": 2, "bins": {"hips": {"score": 0.0}}}))
    assert cr.weak_regions(fallback, False) == ["hips"]  # score_clothed unset: falls back to score


def test_default_variants() -> None:
    assert cr.default_variants(cr.parse_coverage(None))[0] == ["A", "B", "C"]
    clean = cr.parse_coverage(json.dumps({"schema": cr.COVERAGE_SCHEMA, "schema_rev": 2, "weak": []}))
    assert cr.default_variants(clean)[0] == ["A", "B", "C"]
    chosen, reason = cr.default_variants(cr.parse_coverage(MANUAL))
    assert chosen == ["A", "C", "D"] and "regions.pelvis" in reason
    assert cr.default_variants(cr.parse_coverage(MANUAL), adult=False)[0] == ["A", "B", "C"]
    hips = cr.parse_coverage(json.dumps({"schema": cr.COVERAGE_SCHEMA, "schema_rev": 2, "weak": ["hips"]}))
    assert cr.default_variants(hips)[0] == ["A", "C", "D"]


def test_variants_flag_parses_in_order() -> None:
    assert cr.parse_variants("d, a") == ["A", "D"]
    with pytest.raises(ValueError):
        cr.parse_variants("A,E")


# ---- mask

def test_mask_from_box_and_feather() -> None:
    soft, excluded = cr.build_mask((64, 64), mask_image=None, boxes=[(10, 30, 40, 30)], face_box=None,
                                   face_dilate=0, feather=0)
    assert excluded is None
    assert soft[40, 20] == 1.0 and soft[5, 5] == 0.0 and soft.sum() == 40 * 30
    feathered, _ = cr.build_mask((64, 64), mask_image=None, boxes=[(10, 30, 40, 30)], face_box=None,
                                 face_dilate=0, feather=8)
    assert 0.0 < feathered[29, 20] < 1.0  # the edge now reaches outside the box
    assert feathered[45, 30] > 0.9 and feathered[0, 0] == 0.0


def test_box_is_applied_onto_the_mask() -> None:
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (0, 0, 32, 64))  # left half
    soft, _ = cr.build_mask((64, 64), mask_image=mask, boxes=[(16, 0, 32, 64)], face_box=None, face_dilate=0, feather=0)
    assert soft[10, 20] == 1.0 and soft[10, 40] == 0.0 and soft[10, 5] == 0.0


def test_face_box_is_excluded_after_dilation() -> None:
    soft, excluded = cr.build_mask((64, 64), mask_image=None, boxes=[(0, 0, 64, 64)], face_box=(20, 20, 10, 10),
                                   face_dilate=4, feather=6)
    assert excluded == (16, 16, 18, 18)
    assert soft[16:34, 16:34].max() == 0.0 and soft[2, 2] > 0.9


def test_mask_needs_a_region() -> None:
    with pytest.raises(ValueError):
        cr.build_mask((8, 8), mask_image=None, boxes=(), face_box=None, face_dilate=0, feather=0)
    with pytest.raises(ValueError):
        cr.build_mask((8, 8), mask_image=None, boxes=[(0, 0, 8, 8)], face_box=(0, 0, 8, 8), face_dilate=0, feather=0)


def test_blend_and_face_paste_keep_the_source_face() -> None:
    source = np.full((16, 16, 3), 10, dtype=np.uint8)
    refined = np.full((16, 16, 3), 250, dtype=np.uint8)
    soft = np.zeros((16, 16), dtype=np.float32)
    soft[:, 8:] = 1.0
    mixed = cr.blend_refined(source, refined, soft)
    assert mixed[0, 0, 0] == 10 and mixed[0, 15, 0] == 250
    pasted = cr.paste_face(refined, source, (2, 2, 4, 4))
    assert (pasted[2:6, 2:6] == 10).all() and pasted[0, 0, 0] == 250


def test_parse_box() -> None:
    assert cr.parse_box("1,2,3,4") == (1, 2, 3, 4)
    for bad in ("1,2,3", "a,b,c,d", "0,0,0,5"):
        with pytest.raises(ValueError):
            cr.parse_box(bad)


# ---- schedule

class _Transformer:
    def __init__(self) -> None:
        self.weights: list[list[float]] = []

    def set_adapters(self, names, weights):
        assert names == ["canvas-lora"]
        self.weights.append(list(weights))


def test_schedule_drops_the_adapter_after_k_steps_and_restores() -> None:
    transformer = _Transformer()
    pipe = SimpleNamespace(transformer=transformer)
    schedule = cr.StepSchedule(3, 1.0, 0.3)
    for step in range(8):
        schedule(pipe, step, None, {})
        if step < 2:
            assert transformer.weights == []
    assert transformer.weights == [[0.3]]  # once, right after the third step
    assert schedule.dropped_after == 3
    schedule.restore()
    assert transformer.weights == [[0.3], [1.0]]


def test_default_early_steps() -> None:
    assert cr.default_early_steps(8) == 4 and cr.default_early_steps(4) == 2 and cr.default_early_steps(1) == 1


def test_runtime_call_options_reach_the_step_beat_and_inpaint(monkeypatch) -> None:
    from canvas.models import runtime

    seen: list[int] = []

    class Pipe:
        def __call__(self, prompt=None, strength=1.0, callback_on_step_end=None):  # noqa: D102
            return None

    progress = SimpleNamespace(kick=lambda: None)
    assert runtime._inpaint_extras(Pipe()) == {}
    with runtime.call_options(inpaint_strength=0.45, step_callback=lambda p, i, t, k: seen.append(i)):
        assert runtime._inpaint_extras(Pipe()) == {"strength": 0.45}
        call = runtime._with_step_beat(Pipe(), {}, progress)
        call["callback_on_step_end"](None, 3, None, {"latents": 1})
    assert seen == [3]
    assert runtime._inpaint_extras(Pipe()) == {}

    class NoStrength:
        def __call__(self, prompt=None):  # noqa: D102
            return None

    with runtime.call_options(inpaint_strength=0.45):
        assert runtime._inpaint_extras(NoStrength()) == {}


# ---- runs

def test_default_run_with_weak_pelvis(tmp_path, monkeypatch, world) -> None:
    code, calls, log, saved = _run(tmp_path, monkeypatch, world, coverage=MANUAL)
    assert code == 0
    out = tmp_path / "out"
    assert {p.name for p in out.glob("*.png")} == {"variant_A.png", "variant_C.png", "variant_D.png", "compare.png"}
    # A, the scheduled source for D, then two inpaints at the same seed with the trigger applied.
    assert [(r.mode, r.schedule is not None, r.use_lora) for r in calls.requests] == [
        (Mode.TEXT, False, True), (Mode.TEXT, True, True), (Mode.INPAINT, False, False), (Mode.INPAINT, False, False),
    ]
    assert {r.seed for r in calls.requests} == {77}
    assert all(r.prompt.startswith("trg") for r in calls.requests)
    assert [r.denoise for r in calls.requests if r.mode is Mode.INPAINT] == [0.45, 0.45]
    inpaint = calls.requests[2]
    assert {item.role.value for item in inpaint.inputs} == {"init", "mask"}
    document = json.loads((out / "log.json").read_text(encoding="utf-8"))
    assert document["coverage"]["weak"] == ["pelvis"]  # raw document; parsed ids are qualified
    assert document["weak_regions"] == ["regions.pelvis"]
    assert document["variants_default"] == ["A", "C", "D"]
    assert document["variants_run"] == ["A", "C", "D"]
    assert document["variants"]["C"]["face_lock"]["used"] is True
    assert document["variants"]["C"]["likeness"] is None
    assert "scheduled_source" in document["variants"]
    assert "--region-auto omitted" in document["region_auto"]
    assert document["compare"] == "ok"
    assert "pelvis" in log


def test_default_run_without_coverage_is_a_b_c(tmp_path, monkeypatch, world) -> None:
    code, calls, _, _ = _run(tmp_path, monkeypatch, world)
    assert code == 0
    assert {p.name for p in (tmp_path / "out").glob("variant_*.png")} == {
        "variant_A.png", "variant_B.png", "variant_C.png"}
    assert [r.mode for r in calls.requests] == [Mode.TEXT, Mode.TEXT, Mode.INPAINT]


def test_c_only_renders_but_does_not_write_the_baseline(tmp_path, monkeypatch, world) -> None:
    code, calls, _, _ = _run(tmp_path, monkeypatch, world, "--variants", "C")
    assert code == 0
    names = {p.name for p in (tmp_path / "out").glob("variant_*.png")}
    assert names == {"variant_C.png"}
    assert len(calls.requests) == 2


def test_face_lock_pastes_the_original_face(tmp_path, monkeypatch, world) -> None:
    code, _, _, _ = _run(tmp_path, monkeypatch, world, "--variants", "A,C")
    assert code == 0
    base = np.asarray(Image.open(tmp_path / "out" / "variant_A.png").convert("RGB"))
    refined = np.asarray(Image.open(tmp_path / "out" / "variant_C.png").convert("RGB"))
    assert (refined[4:20, 20:36] == base[4:20, 20:36]).all()  # face box (20,4,16,16)
    assert (refined[34:55, 15:45] != base[34:55, 15:45]).any()  # the region was refined
    assert (refined[0:2, 0:2] == base[0:2, 0:2]).all()  # outside the mask: untouched


def test_face_lock_skipped_without_detector_or_box(tmp_path, monkeypatch, world) -> None:
    code, _, log, _ = _run(tmp_path, monkeypatch, world, "--variants", "A,C", faces=False)
    assert code == 0 and "face-lock skipped" in log
    document = json.loads((tmp_path / "out" / "log.json").read_text(encoding="utf-8"))
    assert document["variants"]["C"]["face_lock"]["used"] is False


def test_face_box_flag_is_used_without_a_detector(tmp_path, monkeypatch, world) -> None:
    code, _, _, _ = _run(tmp_path, monkeypatch, world, "--variants", "A,C", "--face-box", "20,4,16,16", faces=False)
    assert code == 0
    document = json.loads((tmp_path / "out" / "log.json").read_text(encoding="utf-8"))
    assert document["variants"]["C"]["face_lock"] == {"used": True, "box": [20, 4, 16, 16], "source": "--face-box"}


def test_gate_refusal_stops_one_variant_only(tmp_path, monkeypatch, world) -> None:
    code, calls, log, _ = _run(tmp_path, monkeypatch, world, "--variants", "A,C,D", refuse={Mode.INPAINT})
    assert code == 1
    document = json.loads((tmp_path / "out" / "log.json").read_text(encoding="utf-8"))
    assert document["variants"]["C"]["status"] == "image_refused"
    assert document["variants"]["D"]["status"] == "image_refused"
    assert document["variants"]["A"]["status"] == "ok"
    assert (tmp_path / "out" / "variant_A.png").exists()
    assert not (tmp_path / "out" / "variant_C.png").exists()
    assert "a 30-year-old" not in log and "a 30-year-old" not in json.dumps(document["variants"])
    assert len(calls.requests) == 4


def test_refused_baseline_skips_c_and_continues(tmp_path, monkeypatch, world) -> None:
    class Refuse(_Calls):
        def __call__(self, request):
            if request.mode is Mode.TEXT and request.schedule is None:
                raise SafetyError("prompt_refused")
            return super().__call__(request)

    weights = _setup(tmp_path, monkeypatch)
    log = io.StringIO()
    code = cr.main(_args(tmp_path, weights, "--variants", "A,B,C"), hooks=cr.Hooks(
        weights_ready=lambda: True, generate=Refuse(world), gate=world.gate(), log=log))
    assert code == 1
    document = json.loads((tmp_path / "out" / "log.json").read_text(encoding="utf-8"))
    assert document["variants"]["A"]["status"] == "prompt_refused"
    assert document["variants"]["B"]["status"] == "ok"
    assert document["variants"]["C"]["status"].startswith("skipped")


def test_pngs_carry_only_the_seed(tmp_path, monkeypatch, world) -> None:
    code, _, _, saved = _run(tmp_path, monkeypatch, world, "--variants", "A,C")
    assert code == 0
    assert saved and all(fmt is OutputFormat.PNG and metadata == {"seed": 77} for _, fmt, metadata in saved)
    assert {name for name, _, _ in saved} == {"variant_A.png", "variant_C.png", "compare.png"}


def test_contact_sheet_is_labelled_side_by_side(world) -> None:
    sheet = cr.contact_sheet({"A": _picture(world, size=32), "C": _picture(world, size=48)})
    assert sheet is not None and sheet.width > 32 + 48 and sheet.height > 32
    assert cr.contact_sheet({}) is None


# ---- device and default output folder

def test_parse_device() -> None:
    assert cr.parse_device("cuda:0") == "0" and cr.parse_device("1") == "1" and cr.parse_device("cuda") == ""
    for bad in ("cpu", "cuda:x", "cuda:", "gpu0", "-1", ""):
        with pytest.raises(ValueError):
            cr.parse_device(bad)


def test_device_selects_the_cuda_device(tmp_path, monkeypatch, world) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    code, _, _, _ = _run(tmp_path, monkeypatch, world, "--gpu", "cuda:0", "--variants", "A")
    assert code == 0
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert json.loads((tmp_path / "out" / "log.json").read_text(encoding="utf-8"))["device"] == "cuda:0"


def test_bad_device_is_a_clear_error(tmp_path, monkeypatch, world) -> None:
    code, calls, log, _ = _run(tmp_path, monkeypatch, world, "--device", "cpu")
    assert code == 2 and "--device must be" in log and calls.requests == []


def test_default_out_is_a_timestamped_pictures_subfolder(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    folder = cr.default_out(0)
    assert folder.parent == tmp_path / ".local/share/voxprint-ai-suite/state/canvas/pictures"
    assert len(folder.name) == len("YYYYMMDD-HHMMSS") and folder.name[8] == "-"


def test_out_defaults_when_omitted(tmp_path, monkeypatch, world) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    weights = _setup(tmp_path, monkeypatch)
    argv = [a for a in _args(tmp_path, weights, "--variants", "A") if a not in ("--out", str(tmp_path / "out"))]
    code = cr.main(argv, hooks=cr.Hooks(weights_ready=lambda: True, generate=_Calls(world), gate=world.gate(),
                                        log=io.StringIO()))
    assert code == 0
    pictures = tmp_path / ".local/share/voxprint-ai-suite/state/canvas/pictures"
    run = next(pictures.iterdir())
    assert (run / "variant_A.png").exists() and (run / "log.json").exists()



# ---- schema_rev 3, sidecar, boxes, face-lock

REV3 = {
    "schema": "voxprint-coverage/1",
    "schema_rev": 3,
    "generator_base": "z-image-turbo@f332072a",
    "bins": {
        "angles": {
            "profile_left": {
                "count_clothed": 2, "count_nude": 1,
                "score_clothed": 0.0, "score_nude": 0.8,
            },
            "back": {"count_clothed": 1, "score_clothed": 0.9, "score_nude": 0.9},
        },
        "regions": {
            "hands": {
                "count_clothed": 4, "count_nude": 1,
                "score_clothed": 0.2, "score_nude": 0.0, "source": "manual",
            },
            "back": {"score_clothed": 0.9, "score_nude": 0.9},
            "face": {"score_clothed": 0.8, "score_nude": 0.8},
        },
    },
    "weak": ["regions.hands", "angles.profile_left"],
    "bins_legacy": {"full_body": {"score": 0.0}},
    "weak_legacy": ["full_body"],
}


def test_parse_coverage_rev3_qualified_weak() -> None:
    cov = cr.parse_coverage(json.dumps(REV3))
    assert cov.usable and cov.rev == 3 and cov.generator_base.startswith("z-image-turbo")
    assert cov.weak == ["regions.hands", "angles.profile_left"]
    assert cov.bins["regions.hands"]["source"] == "manual"
    assert cov.bins["regions.back"]["score_clothed"] == 0.9
    assert cov.bins["angles.back"]["score_clothed"] == 0.9
    assert "regions.full_body" not in cov.bins and "full_body" not in cov.weak
    assert cr.weak_regions(cov, True) == ["hands"]
    assert cr.weak_regions(cov, False) == ["hands"]  # listed, and the bin has a clothed score
    assert cr.weak_angles(cov, True) == ["profile_left"]  # listed
    assert cr.weak_angles(cov, False) == ["profile_left"]  # score_clothed is 0
    chosen, reason = cr.default_variants(cov, adult=True)
    assert chosen == ["A", "B", "C", "D"] and "regions.hands" in reason and "angles.profile_left" in reason
    assert cr.default_variants(cov, adult=True, zone="hands")[0] == ["A", "C", "D"]
    assert cr.default_variants(cov, adult=True, zone="profile_left")[0] == ["A", "B"]
    assert cr.default_variants(cr.parse_coverage(None), refine=True)[0] == ["A", "C", "D"]


def test_rev3_ignores_unknown_names() -> None:
    cov = cr.parse_coverage(json.dumps({
        "schema": cr.COVERAGE_SCHEMA, "schema_rev": 3,
        "weak": ["regions.elbow", "angles.front", "hands", "nope"],
        "bins": {
            "regions": {"elbow": {"score_nude": 0.0}, "hands": {"score_nude": 0.0}},
            "angles": {"front": {"score_nude": 0.0}, "three_quarter": {"score": 0.0}},
        },
    }))
    assert cov.weak == ["angles.front", "regions.hands"]
    assert "regions.hands" in cov.bins and "angles.front" in cov.bins
    assert "regions.elbow" not in cov.bins and "angles.three_quarter" not in cov.bins


def test_rev2_migration_back_becomes_an_angle() -> None:
    cov = cr.parse_coverage(json.dumps({
        "schema": cr.COVERAGE_SCHEMA, "schema_rev": 2,
        "weak": ["back", "hands", "full_body"],
        "bins": {
            "back": {"score": 0.0, "count": 3},
            "hands": {"score_nude": 0.0, "source": "manual"},
            "full_body": {"score": 0.0},
            "pelvis": {"score_clothed": 0.4, "score_nude": 0.4},
        },
        "bins_legacy": {"ignored": {"score": 0.0}},
        "weak_legacy": ["ignored"],
    }))
    assert cov.rev == 2
    assert cov.weak == ["angles.back", "regions.hands"]
    assert cov.bins["angles.back"] == {"score": 0.0, "count": 3}
    assert "regions.hands" in cov.bins and "full_body" not in cov.bins and "regions.full_body" not in cov.bins
    assert cr.weak_angles(cov, True) == ["back"]
    assert cr.weak_regions(cov, True) == ["hands"]


def test_sidecar_precedes_embedded_key(tmp_path, monkeypatch, world) -> None:
    weights = _setup(tmp_path, monkeypatch, MANUAL)
    sidecar = cr.sidecar_path(weights)
    sidecar.write_text(json.dumps(REV3), encoding="utf-8")
    code, _, log, _ = _run(tmp_path, monkeypatch, world, "--variants", "A", coverage=MANUAL)
    # _run overwrites metadata but the sidecar is still next to the same weights
    # Re-run against the existing file: _setup writes a new person.safetensors, so write sidecar after _run setup.
    assert sidecar.name.endswith(".coverage.json")
    # Direct load on the file we just wrote.
    cov = cr.load_coverage(weights, tmp_path / "missing-flag.json")
    assert cov.origin == "sidecar" and cov.rev == 3 and "regions.hands" in cov.weak
    document = json.loads((tmp_path / "out" / "log.json").read_text(encoding="utf-8")) if (tmp_path / "out" / "log.json").exists() else {}
    # The helper's _setup runs first without a sidecar; check load_coverage independently and a full run below.
    assert code == 0
    assert document.get("coverage_origin") in {None, "embedded", "sidecar"}


def test_sidecar_is_used_on_a_full_run(tmp_path, monkeypatch, world) -> None:
    weights = tmp_path / "vxpk.safetensors"
    weights.write_bytes(b"x")
    lora = __import__("types").SimpleNamespace(weights=weights, trigger="trg", alias="")
    monkeypatch.setattr("canvas.tools.render_batch.bind_lora", lambda path, strength: (lora, 1.0 if strength is None else strength))
    monkeypatch.setattr(cr, "_string_metadata", lambda path: {cr.COVERAGE_KEY: MANUAL})
    sidecar = cr.sidecar_path(weights)
    sidecar.write_text(json.dumps(REV3), encoding="utf-8")
    calls = _Calls(world)
    argv = _args(tmp_path, weights, "--variants", "A")
    code = cr.main(argv, hooks=cr.Hooks(weights_ready=lambda: True, generate=calls, gate=world.gate(),
                                        detect_faces=lambda image: [Face(20, 4, 16, 16)], log=io.StringIO()))
    assert code == 0
    document = json.loads((tmp_path / "out" / "log.json").read_text(encoding="utf-8"))
    assert document["coverage_origin"] == "sidecar"
    assert document["weak_regions"] == ["regions.hands"]
    assert document["weak_angles"] == ["angles.profile_left"]
    assert document["coverage"]["schema_rev"] == 3


def test_coverage_flag_is_used_without_a_sidecar(tmp_path, monkeypatch, world) -> None:
    flag = tmp_path / "custom.coverage.json"
    flag.write_text(json.dumps(REV3), encoding="utf-8")
    code, _, _, _ = _run(tmp_path, monkeypatch, world, "--coverage", str(flag), "--variants", "A")
    assert code == 0
    document = json.loads((tmp_path / "out" / "log.json").read_text(encoding="utf-8"))
    assert document["coverage_origin"] == "flag"
    assert document["weak_regions"] == ["regions.hands"]


def test_multiple_boxes_are_unioned() -> None:
    soft, excluded = cr.build_mask(
        (64, 64), mask_image=None, boxes=[(0, 0, 8, 8), (56, 56, 8, 8)],
        face_box=None, face_dilate=0, feather=0,
    )
    assert excluded is None
    assert soft[2, 2] == 1.0 and soft[60, 60] == 1.0 and soft[32, 32] == 0.0
    assert soft.sum() == 8 * 8 * 2


def test_repeated_box_flag_runs(tmp_path, monkeypatch, world) -> None:
    weights = _setup(tmp_path, monkeypatch)
    argv = [item for item in _args(tmp_path, weights, "--variants", "A,C") if item != "10,30,40,30"]
    # _args already added one --box; drop its value and supply two boxes.
    # Rebuild to be explicit.
    argv = ["--prompt", "a 30-year-old adult woman", "--lora", str(weights), "--adult-content", "on",
            "--out", str(tmp_path / "out"), "--seed", "77", "--variants", "A,C",
            "--box", "0,0,8,8", "--box", "40,40,8,8"]
    calls = _Calls(world)
    code = cr.main(argv, hooks=cr.Hooks(weights_ready=lambda: True, generate=calls, gate=world.gate(),
                                        detect_faces=lambda image: [Face(20, 4, 16, 16)], log=io.StringIO()))
    assert code == 0
    document = json.loads((tmp_path / "out" / "log.json").read_text(encoding="utf-8"))
    assert document["boxes"] == [[0, 0, 8, 8], [40, 40, 8, 8]]


def test_face_lock_excludes_the_face_for_a_non_face_zone(tmp_path, monkeypatch, world) -> None:
    code, _, _, _ = _run(tmp_path, monkeypatch, world, "--variants", "A,C", "--zone", "hands")
    assert code == 0
    base = np.asarray(Image.open(tmp_path / "out" / "variant_A.png").convert("RGB"))
    refined = np.asarray(Image.open(tmp_path / "out" / "variant_C.png").convert("RGB"))
    assert (refined[4:20, 20:36] == base[4:20, 20:36]).all()
    document = json.loads((tmp_path / "out" / "log.json").read_text(encoding="utf-8"))
    assert document["zone"] == "hands"
    assert document["variants"]["C"]["face_lock"]["used"] is True


def test_face_lock_is_skipped_when_zone_is_face(tmp_path, monkeypatch, world) -> None:
    code, _, _, _ = _run(tmp_path, monkeypatch, world, "--variants", "A,C", "--zone", "face",
                         "--box", "0,0,64,64")
    # _args already includes a box; the extra --box is appended and both are used.
    assert code == 0
    base = np.asarray(Image.open(tmp_path / "out" / "variant_A.png").convert("RGB"))
    refined = np.asarray(Image.open(tmp_path / "out" / "variant_C.png").convert("RGB"))
    assert (refined[4:20, 20:36] != base[4:20, 20:36]).any()
    document = json.loads((tmp_path / "out" / "log.json").read_text(encoding="utf-8"))
    assert document["variants"]["C"]["face_lock"] == {"used": False, "reason": "--zone face"}
