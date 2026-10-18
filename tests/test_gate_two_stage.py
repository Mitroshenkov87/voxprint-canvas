"""Two-stage output net: the NSFW pass first, the age/school/stylised stage only on flagged frames."""
import pytest

from canvas.safety.classifiers import Face
from canvas.safety.constants import (
    IMAGE_ANIME,
    IMAGE_CLASSROOM,
    IMAGE_MINOR_SEXUAL,
    IMAGE_SCHOOL_UNIFORM,
    SEXUAL_SCORE_BLOCK_AT,
)
from canvas.safety.gate import GenerationBlocked, SafetyUnavailable
from canvas.safety.image import SafeImage, ScreenedInput
from canvas.safety.inputs import ImageRole, PendingInput
from tests.fakes import World, rgb

STAGE_TWO = {"image-score", "faces", "ages"}


def _child(world: World) -> None:
    world.faces = [Face(0, 0, 2, 2)]
    world.ages = [(1.0, 0.0)]


@pytest.mark.parametrize("nsfw", [0.0, 0.2, SEXUAL_SCORE_BLOCK_AT - 0.01])
def test_sfw_frames_never_call_the_age_adapters(nsfw):
    world = World()
    world.nsfw = nsfw
    _child(world)  # even a child reading is never asked for on a clothed frame
    world.image_scores[IMAGE_MINOR_SEXUAL] = 0.9
    world.image_scores[IMAGE_SCHOOL_UNIFORM] = 0.9
    assert type(world.gate().review_image(rgb())) is SafeImage
    assert world.calls == ["nsfw"]


@pytest.mark.parametrize("nsfw", [SEXUAL_SCORE_BLOCK_AT, 0.9, 1.0])
def test_flagged_frames_run_the_full_stage(nsfw):
    world = World()
    world.nsfw = nsfw
    world.faces = [Face(0, 0, 2, 2)]
    world.ages = [(0.0, 0.0)]
    assert type(world.gate().review_image(rgb())) is SafeImage
    assert world.calls[0] == "nsfw"
    assert set(world.calls[1:]) == STAGE_TWO


def test_fairface_runs_only_when_a_face_is_present():
    world = World()
    world.nsfw = 0.9
    world.gate().review_image(rgb())
    assert "image-score" in world.calls and "faces" in world.calls and "ages" not in world.calls


@pytest.mark.parametrize("failure", ["error", "nan", "bool"])
def test_an_nsfw_failure_runs_the_full_stage(failure):
    world = World()
    if failure == "error":
        world.error = "nsfw"
    elif failure == "nan":
        world.nsfw = float("nan")
    else:
        world.nsfw = True  # type: ignore[assignment]
    world.faces = [Face(0, 0, 2, 2)]
    world.ages = [(0.0, 0.0)]
    assert type(world.gate().review_image(rgb())) is SafeImage
    assert {"image-score", "faces", "ages"} <= set(world.calls)
    _child(world)
    with pytest.raises(GenerationBlocked):
        world.gate().review_image(rgb())


def test_an_nsfw_failure_and_a_stage_two_error_refuses_the_frame():
    for stage_error in ("zero-image", "faces", "ages"):
        world = World()
        world.faces = [Face(0, 0, 2, 2)]
        world.ages = [(0.0, 0.0)]
        world.error = stage_error  # the NSFW pass works and flags the frame
        world.nsfw = 0.9
        with pytest.raises(SafetyUnavailable):
            world.gate().review_image(rgb())


def test_minor_sexual_label_counts_only_in_a_flagged_frame():
    world = World()
    world.image_scores[IMAGE_MINOR_SEXUAL] = 0.9
    assert type(world.gate().review_image(rgb())) is SafeImage
    world.nsfw = 0.9
    with pytest.raises(GenerationBlocked):
        world.gate().review_image(rgb())


@pytest.mark.parametrize("label", [IMAGE_SCHOOL_UNIFORM, IMAGE_CLASSROOM, IMAGE_ANIME])
def test_school_and_stylised_labels_still_drop_a_flagged_frame(label):
    world = World()
    world.nsfw = 0.9
    world.image_scores[label] = 0.9
    with pytest.raises(GenerationBlocked):
        world.gate().review_image(rgb())


def test_child_probes_hit_a_full_block():
    probes = ([(1.0, 0.0)], [(0.80, 0.0)], [(0.70, 0.0)], [(0.0, 0.20)], [(0.0, 0.16)])
    for nsfw in (SEXUAL_SCORE_BLOCK_AT, 0.9):
        blocked = 0
        for bands in probes:
            world = World()
            world.nsfw = nsfw
            world.faces = [Face(0, 0, 2, 2)]
            world.ages = bands
            try:
                world.gate().review_image(rgb())
            except GenerationBlocked:
                blocked += 1
        assert blocked == len(probes)


def _inputs():
    return (PendingInput(ImageRole.INIT, rgb()), PendingInput(ImageRole.MASK, rgb()))


def test_clothed_inputs_get_the_nsfw_pass_only():
    world = World()
    screened = world.gate().release_request("a red boat", "inpaint", _inputs())
    assert type(screened) is ScreenedInput
    assert world.calls == ["guard", "nsfw", "nsfw"]


def test_flagged_inputs_run_the_full_stage_and_block_a_child():
    world = World()
    world.nsfw = 0.9
    _child(world)
    with pytest.raises(GenerationBlocked):
        world.gate().release_request("a red boat", "restyle", _inputs()[:1])
    assert STAGE_TWO <= set(world.calls)


@pytest.mark.parametrize("prompt,explicit", [("nude adult woman, 30 years old", False), ("a red boat", True)])
def test_an_explicit_request_forces_the_full_stage_on_inputs(prompt, explicit):
    world = World()
    world.faces = [Face(0, 0, 2, 2)]
    world.ages = [(0.0, 0.0)]
    world.gate().release_request(prompt, "restyle", _inputs()[:1], explicit=explicit)
    assert STAGE_TWO <= set(world.calls)
    world = World()
    world.image_scores[IMAGE_MINOR_SEXUAL] = 0.9
    with pytest.raises(GenerationBlocked):
        world.gate().release_request(prompt, "restyle", _inputs()[:1], explicit=explicit)


def test_the_adult_content_setting_reaches_the_gate_as_explicit(tmp_path):
    from canvas.ui.shell_layout import job_from, open_layout

    shell = open_layout(tmp_path, gate=World().gate())
    assert job_from(shell.layout).adult_content is False
    shell.layout.set_value("image.adult_content", True)
    assert job_from(shell.layout).adult_content is True


def test_toggle_off_with_an_explicit_prompt_still_runs_stage_two_on_inputs(tmp_path):
    from canvas.ui.shell_layout import job_from, open_layout

    world = World()
    shell = open_layout(tmp_path, gate=world.gate())
    shell.layout.set_value("image.adult_content", False)
    job = job_from(shell.layout)
    assert job.adult_content is False
    world.faces = [Face(0, 0, 2, 2)]
    world.ages = [(0.0, 0.0)]
    world.gate().release_request("nude adult woman, 30 years old", "restyle", _inputs()[:1],
                                 explicit=job.adult_content)
    assert world.calls[:2] == ["guard", "nsfw"]
    assert STAGE_TWO <= set(world.calls)
    world = World()  # a clothed input (NSFW 0.0) still meets the stage-2 labels
    world.image_scores[IMAGE_MINOR_SEXUAL] = 0.9
    with pytest.raises(GenerationBlocked):
        world.gate().release_request("nude adult woman, 30 years old", "restyle", _inputs()[:1],
                                     explicit=job.adult_content)
