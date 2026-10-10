"""Bucket mapping and the unwired adapter refusal."""
import pytest

from canvas.safety.adapters import (
    FairFaceAge,
    NsfwViT,
    Qwen3GuardGen,
    SiglipZeroShot,
    YuNetFaces,
    apparent_age_from_bucket,
    parse_qwen3guard,
)
from canvas.safety.gate import SafetyUnavailable
from tests.fakes import rgb


def test_fairface_lower_bound_and_guard_parse():
    assert apparent_age_from_bucket("20-29") == 20.0
    assert apparent_age_from_bucket("30-39") == 30.0
    with pytest.raises(SafetyUnavailable):
        apparent_age_from_bucket("unknown")
    verdict = parse_qwen3guard("Safety: Controversial\nCategories: Sexual Content or Sexual Acts\n")
    assert verdict.safety == "controversial"
    assert verdict.categories == ("Sexual Content or Sexual Acts",)
    with pytest.raises(SafetyUnavailable):
        parse_qwen3guard("Safety: maybe\n")


def test_unwired_adapters_refuse():
    image = rgb()
    with pytest.raises(SafetyUnavailable) as guard:
        Qwen3GuardGen().assess("a red boat")
    assert guard.value.code == "model_not_ready"
    with pytest.raises(SafetyUnavailable) as text:
        SiglipZeroShot().score_text("a red boat", ("a label",))
    assert text.value.code == "model_not_ready"
    with pytest.raises(SafetyUnavailable) as scored:
        SiglipZeroShot().score_image(image, ("a label",))
    assert scored.value.code == "model_not_ready"
    with pytest.raises(SafetyUnavailable):
        YuNetFaces().detect(image)
    with pytest.raises(SafetyUnavailable):
        FairFaceAge().estimate(image, ())
    with pytest.raises(SafetyUnavailable):
        NsfwViT().sexual_score(image)
