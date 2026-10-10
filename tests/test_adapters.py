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
    parse_qwen3guard,
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
