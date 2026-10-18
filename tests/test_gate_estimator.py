"""gate: estimator adapter + tests."""
import json
import math
import time

import numpy as np
import pytest

from canvas.safety import gate as gate_module
from canvas.safety.adapters import (
    FaceAgeRegressor,
    face_crop,
    fairface_age_bands,
    letterbox,
    regressor_ages,
    regressor_scale,
)
from canvas.safety.classifiers import Face
from canvas.safety.constants import (
    CLEAR_CHILD_AT,
    ESTIMATE_CONFIRM_BELOW,
    ESTIMATE_DROP_BELOW,
    IMAGE_ADULT_SEXUAL,
    IMAGE_LABELS,
)
from canvas.safety.gate import GenerationBlocked, SafetyUnavailable, image_block_reason
from canvas.safety.image import SafeImage
from tests.fakes import World, rgb

LABELS = {0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 4: "30-39", 5: "40-49", 6: "50-59", 7: "60-69", 8: "more than 70"}


def _bands(**probs: float) -> tuple[float, float]:
    values = [0.0] * len(LABELS)
    for index, name in LABELS.items():
        values[index] = probs.get(name.replace("-", "_").replace(" ", "_"), 0.0)
    return fairface_age_bands(LABELS, values)


TEEN_READING = _bands(**{"10_19": 0.85, "20_29": 0.15})
SOFT_READING = _bands(**{"10_19": 0.60, "20_29": 0.40})


def _world(bands, fine, **scores: float) -> World:
    world = World()
    world.nsfw = 0.9
    world.image_scores.update({IMAGE_ADULT_SEXUAL: 0.9, **scores})
    world.faces = [Face(0, 0, 2, 2)] * len(bands)
    world.ages = list(bands)
    world.fine = None if fine is None else list(fine)
    return world


def _passes(world: World, prompt: str = "") -> bool:
    try:
        released = world.gate().review_image(rgb(), prompt)
    except GenerationBlocked:
        return False
    assert type(released) is SafeImage
    return True


def test_constants():
    assert ESTIMATE_CONFIRM_BELOW == 18.0
    assert ESTIMATE_DROP_BELOW == 13.0
    assert CLEAR_CHILD_AT == 0.70
    assert TEEN_READING[1] >= gate_module.ADOLESCENT_16_17_AT
    assert TEEN_READING[0] < CLEAR_CHILD_AT


@pytest.mark.parametrize("estimate", [18.5, 19.0, 19.5, 20.5])
def test_adults_18_to_20_pass(estimate):
    assert _passes(_world([TEEN_READING], [estimate]))


def test_babyface_22_passes():
    assert _passes(_world([TEEN_READING], [22.0]))
    assert _passes(_world([SOFT_READING], [22.0]))


def test_teen_reading_with_low_estimate_drops():
    assert not _passes(_world([TEEN_READING], [17.9]))
    assert not _passes(_world([TEEN_READING], [ESTIMATE_CONFIRM_BELOW - 0.01]))
    assert _passes(_world([TEEN_READING], [ESTIMATE_CONFIRM_BELOW]))


def test_each_face_is_paired_with_its_own_estimate():
    adult = (0.0, 0.0)
    assert _passes(_world([adult, TEEN_READING], [15.0, 19.0]))
    assert not _passes(_world([adult, TEEN_READING], [19.0, 15.0]))


def test_low_estimate_drops_on_its_own():
    assert not _passes(_world([(0.0, 0.0)], [12.9]))
    assert not _passes(_world([(0.0, 0.0)], [12.9]), "a woman, 30 years old")
    assert _passes(_world([(0.0, 0.0)], [ESTIMATE_DROP_BELOW]))
    scores = {label: 0.0 for label in IMAGE_LABELS}
    assert image_block_reason(0.95, [(0.0, 0.0)], scores, fine=[12.0]) == "low-estimate"
    assert image_block_reason(0.10, [(0.0, 0.0)], scores, fine=[12.0]) == "allow"


def test_clear_child_line_is_unchanged():
    for bands in ([(1.0, 0.0)], [(0.80, 0.0)], [(CLEAR_CHILD_AT, 0.0)]):
        assert not _passes(_world(bands, [30.0]))


def test_child_probes_stay_blocked():
    probes = ([(1.0, 0.0)], [(0.80, 0.0)], [(0.70, 0.0)], [(0.0, 0.20)], [(0.0, 0.16)])
    blocked = sum(not _passes(_world(bands, [15.0])) for bands in probes)
    assert blocked == len(probes)
    blocked = sum(not _passes(_world(bands, None)) for bands in probes)
    assert blocked == len(probes)


def test_stated_age_paths_are_unchanged():
    scores = {label: 0.0 for label in IMAGE_LABELS}
    scores[IMAGE_ADULT_SEXUAL] = 0.9
    face = [(0.0, gate_module.ADOLESCENT_16_17_AT_AGE_18)]
    assert image_block_reason(0.95, face, scores, support="adult", fine=[25.0]) == "adolescent"
    assert image_block_reason(0.95, face, scores, support="strong", fine=[25.0]) == "allow"


def test_without_estimates_the_line_is_unchanged():
    scores = {label: 0.0 for label in IMAGE_LABELS}
    assert image_block_reason(0.95, [TEEN_READING], scores) == "adolescent"
    assert image_block_reason(0.95, [TEEN_READING], scores, fine=[19.0]) == "allow"


def test_stage_one_skips_the_estimator():
    world = _world([TEEN_READING], [30.0])
    world.nsfw = 0.1
    world.image_scores[IMAGE_ADULT_SEXUAL] = 0.0
    assert _passes(world)
    assert "fine" not in world.calls


def test_faceless_frames_skip_the_estimator():
    world = _world([], [])
    assert _passes(world)
    assert "faces" in world.calls and "fine" not in world.calls


def test_explicit_frames_with_faces_run_the_estimator():
    world = _world([(0.0, 0.0)], [30.0])
    assert _passes(world)
    assert world.calls.index("faces") < world.calls.index("fine")


@pytest.mark.parametrize("fine", [[], [19.0, 19.0], [float("nan")], [float("inf")], [True], ["19"], [None]])
def test_bad_estimates_fail_closed(fine):
    with pytest.raises(SafetyUnavailable):
        _world([TEEN_READING], fine).gate().review_image(rgb())
    with pytest.raises(SafetyUnavailable):
        image_block_reason(0.95, [TEEN_READING], {label: 0.0 for label in IMAGE_LABELS}, fine=fine)


def test_estimator_error_fails_closed():
    world = _world([TEEN_READING], [19.0])
    world.error = "fine"
    with pytest.raises(SafetyUnavailable):
        world.gate().review_image(rgb())


def test_estimator_timeout_fails_closed(monkeypatch):
    monkeypatch.setattr(gate_module, "IMAGE_TIMEOUT_S", 0.05)
    world = _world([TEEN_READING], [19.0])

    class Slow:
        def estimate(self, image, faces):
            time.sleep(0.5)
            return [19.0]

    stack = world.stack()
    object.__setattr__(stack, "fine", Slow())
    with pytest.raises(SafetyUnavailable):
        gate_module.AgeGate(stack).review_image(rgb())


def test_input_images_use_the_estimator_on_explicit_requests():
    from canvas.safety.inputs import ImageRole, PendingInput

    world = _world([TEEN_READING], [15.0])
    world.nsfw = 0.1
    with pytest.raises(GenerationBlocked):
        world.gate().release_request("nude", "img2img", [PendingInput(ImageRole.INIT, rgb())], explicit=True)
    assert "fine" in world.calls


def test_adapter_without_a_folder_fails_closed(tmp_path):
    with pytest.raises(SafetyUnavailable):
        FaceAgeRegressor().estimate(rgb(), [Face(0, 0, 2, 2)])
    with pytest.raises(SafetyUnavailable):
        FaceAgeRegressor(tmp_path / "missing").load()
    with pytest.raises(SafetyUnavailable):
        FaceAgeRegressor(tmp_path).load()


GOOD_CONFIG = {
    "model_name": "mivolo_d1_384",
    "in_chans": 6,
    "num_classes": 3,
    "input_size": 384,
    "with_persons_model": True,
    "only_age": False,
    "min_age": 0,
    "max_age": 122,
    "avg_age": 61.0,
}


def test_adapter_rejects_a_bad_config(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SafetyUnavailable):
        FaceAgeRegressor(tmp_path).load()
    (tmp_path / "config.json").write_text(json.dumps({**GOOD_CONFIG, "in_chans": 3}), encoding="utf-8")
    with pytest.raises(SafetyUnavailable):
        FaceAgeRegressor(tmp_path).load()


def test_config_scale():
    assert regressor_scale(GOOD_CONFIG) == (0.0, 122.0, 61.0)
    for key, value in (("model_name", "other"), ("only_age", True), ("min_age", float("nan")), ("avg_age", 200)):
        with pytest.raises(SafetyUnavailable):
            regressor_scale({**GOOD_CONFIG, key: value})


def test_output_scaling_and_count():
    scale = (0.0, 122.0, 61.0)
    assert regressor_ages([0.0, -0.5], scale, 2) == [61.0, 0.0]
    with pytest.raises(SafetyUnavailable):
        regressor_ages([0.0], scale, 2)
    with pytest.raises(SafetyUnavailable):
        regressor_ages([math.nan], scale, 1)


def test_crop_and_letterbox():
    pytest.importorskip("cv2")
    pixels = np.zeros((40, 60, 3), dtype=np.uint8)
    assert face_crop(pixels, Face(-5, -5, 20, 20)).shape == (15, 15, 3)
    assert face_crop(pixels, Face(50, 30, 20, 20)).shape == (10, 10, 3)
    with pytest.raises(SafetyUnavailable):
        face_crop(pixels, Face(70, 50, 5, 5))
    assert letterbox(np.zeros((10, 30, 3), dtype=np.uint8), 384).shape == (384, 384, 3)
    assert letterbox(np.zeros((384, 384, 3), dtype=np.uint8), 384).shape == (384, 384, 3)
