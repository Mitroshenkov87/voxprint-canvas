"""Owner rule 2026-11-16: an uploaded photo is refused only when it shows a minor AND the edit is 18+."""
import json
import threading
from pathlib import Path

import pytest

from canvas.safety import gate as gate_module
from canvas.safety.classifiers import Face
from canvas.safety.constants import IMAGE_ANIME, IMAGE_CLASSROOM, IMAGE_MINOR_SEXUAL, IMAGE_SCHOOL_UNIFORM
from canvas.safety.gate import GenerationBlocked, SafetyUnavailable, input_minor_reason
from canvas.safety.image import ScreenedInput
from canvas.safety.inputs import ImageRole, PendingInput
from tests.fakes import World, rgb

AGE_MODELS = ("image-score", "faces", "ages", "fine")
EDIT_ROLES = [ImageRole.INIT, ImageRole.REFERENCE, ImageRole.STYLE, ImageRole.POSE, ImageRole.DEPTH, ImageRole.EDGE]


def _minor(world: World) -> World:
    world.faces = [Face(0, 0, 2, 2)]
    world.ages = [(1.0, 0.0)]
    return world


def _adult(world: World) -> World:
    world.faces = [Face(0, 0, 2, 2)]
    world.ages = [(0.0, 0.0)]
    world.fine = [31.0]
    return world


def _ran_age_models(world: World) -> bool:
    return any(name in world.calls for name in AGE_MODELS)


@pytest.mark.parametrize("role", EDIT_ROLES)
def test_minor_photo_and_18_plus_edit_is_refused(role):
    world = _minor(World())
    source = rgb()
    with pytest.raises(GenerationBlocked) as caught:
        world.gate().release_request("nude", "restyle", [PendingInput(role, source)])
    assert caught.value.code == "input"
    assert source.to_bytes() == b""


def test_minor_photo_and_18_plus_inpaint_is_refused():
    world = _minor(World())
    with pytest.raises(GenerationBlocked):
        world.gate().release_request(
            "lingerie", "inpaint", [PendingInput(ImageRole.INIT, rgb()), PendingInput(ImageRole.MASK, rgb())],
        )


@pytest.mark.parametrize("role", EDIT_ROLES)
def test_minor_photo_and_a_neutral_edit_passes_without_running_the_image_models(role):
    world = _minor(World())
    world.nsfw = 0.99
    world.image_scores[IMAGE_MINOR_SEXUAL] = 0.99
    screened = world.gate().release_request("a red boat at sunset", "restyle", [PendingInput(role, rgb())])
    assert type(screened) is ScreenedInput
    assert world.calls == ["guard"]


def test_adult_photo_and_18_plus_edit_passes():
    world = _adult(World())
    screened = world.gate().release_request("nude, boudoir", "restyle", [PendingInput(ImageRole.INIT, rgb())])
    assert type(screened) is ScreenedInput
    assert _ran_age_models(world) and "nsfw" not in world.calls


def test_adult_photo_and_neutral_edit_passes_without_running_the_image_models():
    world = _adult(World())
    world.gate().release_request("a portrait, soft light", "restyle", [PendingInput(ImageRole.INIT, rgb())])
    assert world.calls == ["guard"]


@pytest.mark.parametrize("error", ["zero-image", "faces", "ages", "fine"])
def test_a_checker_error_on_an_18_plus_edit_refuses(error):
    world = _adult(World())
    world.error = error
    with pytest.raises(SafetyUnavailable):
        world.gate().release_request("nude", "restyle", [PendingInput(ImageRole.INIT, rgb())])


def test_junk_checker_output_on_an_18_plus_edit_refuses():
    world = _adult(World())
    world.fine = [31.0, 32.0]  # one face, two estimates
    with pytest.raises(SafetyUnavailable):
        world.gate().release_request("nude", "restyle", [PendingInput(ImageRole.INIT, rgb())])
    world = _adult(World())
    del world.image_scores[IMAGE_MINOR_SEXUAL]
    with pytest.raises(SafetyUnavailable):
        world.gate().release_request("nude", "restyle", [PendingInput(ImageRole.INIT, rgb())])
    gate = _adult(World()).gate()
    gate._stack.faces.detect = lambda _raw: None  # type: ignore[attr-defined]
    with pytest.raises(SafetyUnavailable):
        gate.release_request("nude", "restyle", [PendingInput(ImageRole.INIT, rgb())])


def test_a_checker_load_failure_on_an_18_plus_edit_refuses():
    gate = _adult(World()).gate()

    def broken() -> None:
        raise RuntimeError("no weights")

    gate._stack.faces.load = broken  # type: ignore[attr-defined]
    gate.release_request("a red boat", "restyle", [PendingInput(ImageRole.INIT, rgb())])  # not 18+: no load
    with pytest.raises(SafetyUnavailable):
        gate.release_request("nude", "restyle", [PendingInput(ImageRole.INIT, rgb())])


def test_a_checker_timeout_on_an_18_plus_edit_refuses(monkeypatch):
    monkeypatch.setattr(gate_module, "IMAGE_TIMEOUT_S", 0.05)
    gate = _adult(World()).gate()
    release = threading.Event()
    gate._stack.faces.detect = lambda _raw: release.wait(5)  # type: ignore[attr-defined]
    try:
        with pytest.raises(SafetyUnavailable):
            gate.release_request("nude", "restyle", [PendingInput(ImageRole.INIT, rgb())])
    finally:
        release.set()


def test_the_18_plus_switch_with_a_neutral_prompt_checks_and_refuses_a_minor_photo():
    world = _minor(World())
    with pytest.raises(GenerationBlocked):
        world.gate().release_request("a portrait", "restyle", [PendingInput(ImageRole.INIT, rgb())], explicit=True)
    world = _adult(World())
    world.gate().release_request("a portrait", "restyle", [PendingInput(ImageRole.INIT, rgb())], explicit=True)


def test_the_guard_sexual_category_makes_the_edit_18_plus():
    world = _minor(World())
    world.safety, world.categories = "unsafe", ("Sexual Content or Sexual Acts",)
    with pytest.raises(GenerationBlocked):
        world.gate().release_request("a portrait", "restyle", [PendingInput(ImageRole.INIT, rgb())])


@pytest.mark.parametrize("label", [IMAGE_SCHOOL_UNIFORM, IMAGE_CLASSROOM, IMAGE_ANIME])
def test_school_or_stylised_photo_without_minor_signals_passes_an_18_plus_edit(label):
    """Refused before 2026-11-16 (school-context / stylised), allowed now."""
    world = _adult(World())
    world.image_scores[label] = 0.95
    world.gate().release_request("nude", "restyle", [PendingInput(ImageRole.INIT, rgb())])
    assert input_minor_reason([(0.0, 0.0)], dict(world.image_scores), fine=[31.0]) == "allow"


def test_an_nsfw_adult_photo_with_a_neutral_edit_passes():
    world = _adult(World())
    world.nsfw = 0.99
    world.gate().release_request("a red boat", "restyle", [PendingInput(ImageRole.INIT, rgb())])
    assert world.calls == ["guard"]


def test_input_minor_reason_keeps_every_minor_criterion():
    scores = {label: 0.0 for label in gate_module.IMAGE_LABELS}
    assert input_minor_reason([(1.0, 0.0)], scores) == "clear-child"
    assert input_minor_reason([(0.0, 0.9)], scores) == "adolescent"
    assert input_minor_reason([(0.0, 0.0)], scores, fine=[10.0]) == "low-estimate"
    assert input_minor_reason([], {**scores, IMAGE_MINOR_SEXUAL: 0.9}) == "minor-sexual"
    assert input_minor_reason([], {**scores, gate_module.IMAGE_MINOR: 0.9}) == "minor-body"
    assert input_minor_reason([(0.0, 0.9)], scores, support="strong") == "allow"


# Once per run


def test_a_run_checks_one_photo_once_and_never_shares_across_photos():
    world = _adult(World())
    gate = world.gate()
    with gate.input_run():
        for _ in range(3):
            gate.release_request("nude", "restyle", [PendingInput(ImageRole.INIT, rgb(color=(1, 2, 3)))])
        assert world.calls.count("faces") == 1
        gate.release_request("nude", "restyle", [PendingInput(ImageRole.INIT, rgb(color=(9, 9, 9)))])
        assert world.calls.count("faces") == 2
        gate.release_request("a red boat", "restyle", [PendingInput(ImageRole.INIT, rgb(color=(5, 5, 5)))])
        assert world.calls.count("faces") == 2  # not 18+: no verdict needed
    gate.release_request("nude", "restyle", [PendingInput(ImageRole.INIT, rgb(color=(1, 2, 3)))])
    assert world.calls.count("faces") == 3  # the run ended, so its verdicts are gone


def test_a_checker_error_is_not_cached_as_a_pass():
    world = _adult(World())
    gate = world.gate()
    with gate.input_run():
        world.error = "ages"
        with pytest.raises(SafetyUnavailable):
            gate.release_request("nude", "restyle", [PendingInput(ImageRole.INIT, rgb())])
        world.error = ""
        _minor(world)
        with pytest.raises(GenerationBlocked):
            gate.release_request("nude", "restyle", [PendingInput(ImageRole.INIT, rgb())])


def test_the_verdict_is_keyed_by_the_adult_age_support():
    world = _adult(World())
    world.ages = [(0.0, 0.9)]
    world.fine = [17.0]
    gate = world.gate()
    with gate.input_run():
        with pytest.raises(GenerationBlocked):
            gate.release_request("nude", "restyle", [PendingInput(ImageRole.INIT, rgb())])
        gate.release_request("nude, 25 years old", "restyle", [PendingInput(ImageRole.INIT, rgb())])
    assert world.calls.count("faces") == 2


def _cli(monkeypatch, tmp_path: Path, world: World):
    from PIL import Image

    from tests.test_generate_batch import _stub_generate

    written: list[Path] = []

    def export(self, *args: object, **kwargs: object) -> None:
        self.gate.release_request(str(args[0]), args[1].value, args[2]).wipe()
        path = Path(args[3])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")
        written.append(path)

    _stub_generate(monkeypatch, tmp_path, export)
    original = __import__("canvas.app.main", fromlist=["Pipeline"]).Pipeline

    class GatedPipeline(original):  # type: ignore[misc, valid-type]
        def __init__(self, gate: object, generator: object, *args: object, **kwargs: object) -> None:
            super().__init__(gate, generator, *args, **kwargs)
            self.gate = gate

    monkeypatch.setattr("canvas.app.main.Pipeline", GatedPipeline)
    monkeypatch.setattr("canvas.app.main.AgeGate", lambda _stack: world.gate())
    source, mask = tmp_path / "in.png", tmp_path / "mask.png"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(source)
    Image.new("RGB", (4, 4), (255, 255, 255)).save(mask)
    return written, ["--input", str(source), "--mask", str(mask)]


@pytest.mark.parametrize("plan", [["--count", "3", "--seed", "10"], ["--seeds", "4,5,6"]])
def test_generate_count_runs_the_photo_age_check_once(monkeypatch, tmp_path, capsys, plan):
    from canvas.app.main import main

    world = _adult(World())
    written, images = _cli(monkeypatch, tmp_path, world)
    code = main(["generate", "--prompt", "nude, adult woman", "--out", str(tmp_path / "o.webp"), *plan, *images,
                 "--json"])
    body = json.loads(capsys.readouterr().out)
    assert code == 0 and body["written"] == 3
    assert len(written) == 3
    assert world.calls.count("faces") == 2  # the photo and the mask, once each for the whole run


@pytest.mark.parametrize("plan", [["--count", "3", "--seed", "10"], ["--seeds", "4,5,6"]])
def test_generate_count_with_a_minor_photo_refuses_every_item_and_writes_nothing(monkeypatch, tmp_path, capsys, plan):
    from canvas.app.main import main

    world = _minor(World())
    written, images = _cli(monkeypatch, tmp_path, world)
    main(["generate", "--prompt", "nude", "--out", str(tmp_path / "o.webp"), *plan, *images, "--json"])
    body = json.loads(capsys.readouterr().out)
    assert body["written"] == 0 and body["refused"] == 3
    assert written == []
    assert world.calls.count("faces") == 1
