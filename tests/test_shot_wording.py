"""Each shot's framing, camera, and pose lead its line, also when the idea describes a persona."""
from canvas.pipeline.enhance import stated_ages
from canvas.pipeline.frame_prompt import split_frame_prompts, strip_series_words
from canvas.pipeline.photoshoot import (
    Shot,
    assemble_line,
    planned_shots,
    shot_identity,
    single_frame,
    subject_parts,
)

# The idea from the rc.30 dual-GPU run, where every frame came out as a multi-view sheet.
PERSONA = (
    "Fictional persona: a 30-year-old adult woman, East Asian, blonde shoulder-length hair, very athletic, "
    "slim with visible muscle definition, mature adult facial features, faint smile lines; the same woman and "
    "same face in every frame. Clothed in a fitted grey athletic top and black leggings. Neutral plain "
    "light-grey studio backdrop, soft even light, for software screenshots. Vary pose and framing across "
    "frames: head-and-shoulders portrait front, portrait three-quarter, waist-up front, waist-up three-quarter, "
    "full body standing front, full body three-quarter, arms crossed, hands on hips, walking."
)
FRAMINGS = (
    "head-and-shoulders portrait front", "portrait three-quarter", "waist-up front", "waist-up three-quarter",
    "full body standing front", "full body three-quarter", "arms crossed", "hands on hips", "walking",
)


def _line(subject: str, shot: Shot, index: int = 0) -> str:
    """The line the series sends, as ``run_series`` builds it."""
    return single_frame(strip_series_words(assemble_line(subject, shot, include_camera=True, index=index, seed=7)))


def test_the_persona_is_split_into_identity_framings_and_setting() -> None:
    parts = subject_parts(PERSONA)
    assert parts.identity.startswith("Fictional persona: a 30-year-old adult woman")
    assert parts.identity.endswith("black leggings")
    assert parts.framings == FRAMINGS
    assert parts.setting.startswith("Neutral plain light-grey studio backdrop")
    assert "Vary" not in parts.identity and "Vary" not in parts.setting
    assert shot_identity(PERSONA).endswith("same face. Clothed in a fitted grey athletic top and black leggings")


def test_shot_camera_and_pose_lead_and_the_persona_follows() -> None:
    shot = Shot("low-angle shot, strong foreshortening", "arms raised overhead", "24 mm lens, low angle", "loft")
    line = _line(PERSONA, shot)
    lead = "A single photograph, low-angle shot, strong foreshortening, 24 mm lens, low angle, arms raised overhead"
    assert line.startswith(lead)
    assert line.index("foreshortening") < line.index("Fictional persona")
    assert line.index("arms raised overhead") < line.index("Fictional persona")
    assert shot_identity(PERSONA) in line
    assert sorted(stated_ages(line)) == [30]
    assert line.index("Fictional persona") < line.index("loft") < line.index("studio backdrop")
    # The persona's own framing list follows the shot's words; it does not lead.
    assert line.index("arms raised overhead") < line.index(FRAMINGS[0]) < line.index("Fictional persona")


def test_the_idea_framing_list_gives_one_framing_per_shot() -> None:
    shot = Shot("soft light", "standing", "50 mm lens", "studio")
    lines = [_line(PERSONA, shot, index) for index in range(len(FRAMINGS))]
    for index, line in enumerate(lines):
        mine = FRAMINGS[index]
        assert line.startswith(f"A single photograph, soft light, 50 mm lens, standing, {mine}, of ")
        others = [item for item in FRAMINGS if item != mine and item not in mine]
        assert not any(f", {item}," in line for item in others), line
    named = Shot("full body standing front, soft light", "standing", "50 mm lens", "studio")
    assert _line(PERSONA, named, 0).startswith("A single photograph, full body standing front, soft light")


def test_series_words_leave_no_dangling_punctuation() -> None:
    line = _line(PERSONA, Shot("soft light", "standing", "50 mm lens", "studio"))
    assert "across :" not in line and "across:" not in line
    assert " ." not in line and " ," not in line and " :" not in line
    assert "every" not in line
    assert strip_series_words("the same face in every frame. Clothed") == "the same face. Clothed"
    assert strip_series_words("Vary pose across frames: a, b") == "Vary pose: a, b"


def test_planned_persona_shots_follow_the_idea_framings() -> None:
    shots = planned_shots(PERSONA, 25, "same", place="", taken=())
    assert len(shots) == 25
    for index, shot in enumerate(shots):
        line = _line(PERSONA, shot, index)
        assert line.startswith(f"A single photograph, {FRAMINGS[index % len(FRAMINGS)]}, ")
        assert sum(f" {item}," in line for item in FRAMINGS if item not in ("arms crossed",)) <= 2
        assert shot_identity(PERSONA) in line


def test_a_one_sentence_idea_keeps_its_words_together() -> None:
    idea = "Fixed adult woman, age 30, shoulder-length brown hair, wearing a navy blazer and jeans, natural light"
    assert subject_parts(idea).identity == idea
    line = _line(idea, Shot("close-up", "hands on hips", "35 mm lens, high angle", "loft"))
    assert line == f"A single photograph, close-up, 35 mm lens, high angle, hands on hips, of {idea}, loft"


def test_new_lines_still_split_one_embedding_per_frame() -> None:
    shot = Shot("close-up", "standing", "50 mm lens", "loft")
    joined = " ".join(_line(PERSONA, shot, index) for index in range(3))
    frames = split_frame_prompts((joined,))
    assert len(frames) == 3
    assert all(frame.startswith("A single photograph, ") for frame in frames)
