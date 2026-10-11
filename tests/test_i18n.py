"""Locale selection for the short UI strings."""
from canvas.app.i18n import (
    INSTRUCTION_EDIT_UNAVAILABLE_EN,
    INSTRUCTION_EDIT_UNAVAILABLE_RU,
    normalize,
    text,
)


def test_russian_locale_uses_the_russian_note(monkeypatch):
    monkeypatch.setenv("LC_ALL", "ru_RU.UTF-8")
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    monkeypatch.delenv("LANG", raising=False)
    assert normalize("ru_RU") == "ru"
    assert text("instruction_edit_unavailable") == INSTRUCTION_EDIT_UNAVAILABLE_RU


def test_other_locales_use_english(monkeypatch):
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    monkeypatch.delenv("LANG", raising=False)
    assert normalize("C") == "en"
    assert text("instruction_edit_unavailable") == INSTRUCTION_EDIT_UNAVAILABLE_EN
    assert text("instruction_edit_unavailable", "de_DE") == INSTRUCTION_EDIT_UNAVAILABLE_EN
