"""Bucket mapping and the unwired adapter refusal."""
import pytest

from canvas.safety.adapters import (
    FairFaceAge,
    NsfwViT,
    Qwen3GuardGen,
    SiglipZeroShot,
    YuNetFaces,
    _pooled_features,
    apparent_age_from_bucket,
    fairface_under_18,
    parse_qwen3guard,
    probability_under_18,
)
from canvas.safety.gate import SafetyUnavailable
from tests.fakes import rgb


def test_fairface_lower_bound_and_guard_parse():
    assert apparent_age_from_bucket("20-29") == 20.0
    assert apparent_age_from_bucket("30-39") == 30.0
    assert apparent_age_from_bucket("more than 70") == 70.0
    with pytest.raises(SafetyUnavailable):
        apparent_age_from_bucket("unknown")
    verdict = parse_qwen3guard("Safety: Controversial\nCategories: Sexual Content or Sexual Acts\n")
    assert verdict.safety == "controversial"
    assert verdict.categories == ("Sexual Content or Sexual Acts",)
    with pytest.raises(SafetyUnavailable):
        parse_qwen3guard("Safety: maybe\n")


def _bucket_row(chosen: str) -> dict[str, float]:
    labels = ("0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+")
    return {label: (1.0 if label == chosen else 0.0) for label in labels}


def test_under_18_probability_ignores_the_20_29_bucket():
    labels = ("0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+")
    assert probability_under_18(_bucket_row("20-29")) == 0.0
    assert probability_under_18(_bucket_row("30-39")) == 0.0
    assert probability_under_18(_bucket_row("10-19")) == pytest.approx(0.8)
    older = {label: 0.0 for label in labels if label != "70+"}
    older["more than 70"] = 1.0
    assert probability_under_18(older) == 0.0
    mixed = _bucket_row("30-39")
    mixed["30-39"] = 0.25
    mixed["0-2"] = 0.25
    mixed["10-19"] = 0.50
    assert probability_under_18(mixed) == pytest.approx(0.65)
    with pytest.raises(SafetyUnavailable):
        probability_under_18({"20-29": 0.5})
    with pytest.raises(SafetyUnavailable):
        probability_under_18({"nope": 1.0})
    with pytest.raises(SafetyUnavailable):
        probability_under_18({})
    broken = _bucket_row("20-29")
    broken["20-29"] = float("nan")
    with pytest.raises(SafetyUnavailable):
        probability_under_18(broken)
    broken = _bucket_row("20-29")
    broken["20-29"] = True  # type: ignore[assignment]
    with pytest.raises(SafetyUnavailable):
        probability_under_18(broken)
    ids = {str(index): label for index, label in enumerate(labels)}
    assert fairface_under_18(ids, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == 0.0
    ints = {index: label for index, label in enumerate(labels)}
    assert fairface_under_18(ints, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == pytest.approx(0.8)
    with pytest.raises(SafetyUnavailable):
        fairface_under_18({}, [1.0])
    with pytest.raises(SafetyUnavailable):
        fairface_under_18({0: "20-29", 1: "20-29"}, [0.5, 0.5])


def test_pooled_features_accepts_a_tensor_or_a_pooling_output():
    class Tensor:
        def norm(self, dim: int, keepdim: bool):
            return self

    tensor = Tensor()
    assert _pooled_features(tensor) is tensor

    class Output:
        pooler_output = object()

    assert _pooled_features(Output()) is Output.pooler_output
    with pytest.raises(SafetyUnavailable) as empty:
        _pooled_features(object())
    assert empty.value.code == "classifier_error"


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
