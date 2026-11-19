"""A joined scenario is not one prompt embedding."""
import re

from canvas.pipeline.frame_prompt import frame_identity, split_frame_prompts, strip_series_words, with_choices

_DENY = re.compile(
    r"\b(?:frames?|borders?|photoshoots?|photo[-\s]+shoots?|panels?|sheets?|series)\b",
    re.IGNORECASE,
)


def test_series_words_are_stripped_and_a_joined_scenario_splits() -> None:
    raw = "same woman in every frame, varied shots and angles, a red coat"
    cleaned = strip_series_words(raw)
    assert "every frame" not in cleaned
    assert "every shot" not in cleaned
    assert "shots" not in cleaned
    assert "angles" not in cleaned
    assert "red coat" in cleaned
    assert " in," not in cleaned
    assert not cleaned.endswith(" in")
    assert _DENY.search(cleaned) is None
    assert strip_series_words("a woman in a red coat") == "a woman in a red coat"
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


def test_frame_becomes_shot_and_the_photograph_lead_stays() -> None:
    lead = "A single photograph of a red FRAME and a white BORDER"
    cleaned = strip_series_words(lead)
    assert cleaned == "A single photograph of a red shot and a white"
    assert "photograph" in cleaned
    assert re.search(r"\bframes?\b", cleaned, re.IGNORECASE) is None
    assert _DENY.search(cleaned) is None
    assert strip_series_words("a framed portrait by the doorframe") == "a framed portrait by the doorframe"
    listed = "FRAMES, Borders, Photo Shoot, photo-shoots, Photoshoots, PANELS, Sheets, Series"
    assert strip_series_words(listed) == ""
    assert _DENY.search("A single photograph of a red cup") is None


def test_one_item_is_chosen_from_each_parenthesized_list() -> None:
    outfits = ("jeans and sweater", "summer dress", "business suit", "sportswear", "winter coat")
    look = "East Asian, straight black hair, blue eyes"
    text = (
        f"{look}, a woman ({', '.join(outfits)}), different lighting, "
        "(soft light or golden hour), (loft, rooftop)"
    )
    first = strip_series_words(with_choices(text, 0, 0))
    second = strip_series_words(with_choices(text, 1, 0))
    seeded = strip_series_words(with_choices(text, 0, 3))
    assert first == f"{look}, a woman, {outfits[0]}, soft light, loft"
    assert second.endswith(f"{outfits[1]}, golden hour, rooftop")
    assert seeded.endswith(f"{outfits[3]}, golden hour, rooftop")
    identity = frame_identity(text)
    assert identity == f"{look}, a woman"
    assert identity in first
    assert "different" not in first
    assert "(" not in first
