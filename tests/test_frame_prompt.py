"""A joined scenario is not one prompt embedding."""
from canvas.pipeline.frame_prompt import split_frame_prompts, strip_series_words


def test_series_words_are_stripped_and_a_joined_scenario_splits() -> None:
    raw = "same woman in every frame, varied shots and angles, a red coat"
    assert "every frame" not in strip_series_words(raw)
    assert "shots" not in strip_series_words(raw)
    assert "angles" not in strip_series_words(raw)
    assert "red coat" in strip_series_words(raw)
    joined = "\n".join(
        f"A single photograph of a red cup, pose {index}, kitchen" for index in range(3)
    )
    frames = split_frame_prompts((joined,))
    assert len(frames) == 3
    assert joined not in frames
    assert all(item.startswith("A single photograph of a red cup") for item in frames)
    numbered = "1. wide view | still | 50 mm | loft\n2. close view | seated | 85 mm | loft"
    pieces = split_frame_prompts((numbered,))
    assert len(pieces) == 2
    assert all(item.startswith("A single photograph of") for item in pieces)
    assert numbered not in pieces
