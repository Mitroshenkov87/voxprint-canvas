"""Deny-list normalisation for English, Russian and German."""
from canvas.safety.denylist import explain, scan

SHOWER = (
    "Эротическая художественная фотография: обнажённая женщина 30 лет с длинными тёмными волосами "
    "стоит под струями воды в душе из тёмного камня, капли на коже, пар, мягкий контровой свет, "
    "уверенный взгляд взрослой женщины, реалистичная текстура кожи, кинематографичная цветокоррекция, "
    "среднеформатная камера, 85 мм"
)


def test_plain_scene_has_no_signals():
    signals = scan("a red boat on a quiet lake at sunset")
    assert signals.minor is False
    assert signals.sexual is False


def test_leet_spacing_and_confusables():
    assert scan("ch1ld").minor is True
    assert scan("n4ked").sexual is True
    assert scan("k!d").minor is True
    assert scan("c h i l d porn").minor is True
    assert scan("c h i l d porn").sexual is True
    assert scan("k i d").minor is False  # three letters are not glued
    assert scan("k i d s").minor is True
    child = scan("\u0441hild nude")
    assert child.minor is True
    assert child.sexual is True


def test_russian_german_and_translit():
    assert scan("голый ребенок").minor is True
    assert scan("голый ребенок").sexual is True
    both = scan("rebenok seks")
    assert both.minor is True and both.sexual is True
    german = scan("ein nacktes Kind")
    assert german.minor is True and german.sexual is True
    assert scan("Schulmädchen").minor is True
    assert scan("minderjährig").minor is True
    spaced = scan("п о р н о")
    assert spaced.sexual is True


def test_age_patterns_are_a_minor_signal_only():
    assert scan("5 years old").minor is True
    assert scan("5 years old").sexual is False
    assert scan("17yo").minor is True
    assert scan("18 years old").minor is False
    assert scan("under 16").minor is True
    assert scan("unter 12").minor is True
    assert scan("младше 12").minor is True
    assert scan("fünf Jahre alt").minor is True
    assert scan("6 jahre alt").minor is True
    assert scan("5 лет").minor is True
    assert scan("teenager").minor is True
    assert scan("teenager").sexual is False
    assert scan("два года").minor is True
    assert scan("семнадцать лет").minor is True
    assert scan("восемнадцать лет").minor is False
    assert scan("возраст 16").minor is True
    assert scan("aged 17").minor is True
    assert scan("age 25").minor is False
    kite = scan("a child flying a kite in a park")
    assert kite.minor is True and kite.sexual is False and kite.minor_sexual is False
    hard = scan("loli")
    assert hard.minor_sexual is True and hard.sexual is False
    assert scan("лоликон").minor_sexual is True
    assert scan("schoolgirl").minor is True and scan("schoolgirl").minor_sexual is False


def test_adult_russian_shower_prompt_is_sexual_and_not_a_minor():
    detail = explain(SHOWER)
    assert detail.minor is False
    assert detail.minor_sexual is False
    assert detail.age is False
    assert detail.sexual is True
    assert "эротич" in detail.sexual_terms
    assert "обнаж" in detail.sexual_terms
    assert detail.minor_terms == ()
    for fragment in ("лет", "душ", "мягк", "средн"):
        assert fragment not in detail.minor_terms
        assert fragment not in detail.sexual_terms
    assert scan("30 лет").minor is False
    assert scan("85 мм").minor is False
    assert scan("18 лет").minor is False
    for age in (20, 25, 28, 32, 35):
        assert scan(f"женщина {age} лет").minor is False
    assert scan("17 лет").minor is True
    assert scan("секстант").sexual is False
    assert scan("sporn").sexual is False
    assert scan("порнография").sexual is True
    assert scan("малолетний").minor is True
    assert scan("лолита").minor_sexual is True
    assert scan("малышка").minor is True
