"""The UI catalog is complete, and Qt widgets do not carry hard-coded sentences."""
import ast
import json
import re
from pathlib import Path

from canvas.app.i18n import shipped_locales, strings, text

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "canvas" / "app" / "i18n_allowlist.txt"
CATALOG = ROOT / "canvas" / "app" / "messages.json"

_SETTERS = {
    "setWindowTitle",
    "setPlaceholderText",
    "setToolTip",
    "setAccessibleName",
    "setAccessibleDescription",
    "setStatusTip",
    "setWhatsThis",
    "setText",
    "setTitle",
    "setWindowIconText",
    "addTab",
    "addItem",
    "addAction",
    "set_stage",
    "drawText",
    "progress",
    "stage",
}
_CTORS = {"QLabel", "QPushButton", "QCheckBox", "QRadioButton", "QLineEdit", "QGroupBox", "QMenu", "QAction"}
_BOXES = {"question", "information", "warning", "critical"}
_CATALOG_CALLS = {"text", "catalog_text"}
_PLURAL_KEYS = (
    "unit.minute.one",
    "unit.minute.many",
    "unit.second.one",
    "unit.second.many",
)


def _patterns() -> tuple[re.Pattern[str], ...]:
    found = []
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            found.append(re.compile(stripped))
    return tuple(found)


def _allowed(value: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.fullmatch(value) for pattern in patterns)


def _callee(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _literals(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.lineno, node.value
        return
    if isinstance(node, ast.JoinedStr):
        parts = [item.value for item in node.values if isinstance(item, ast.Constant) and isinstance(item.value, str)]
        joined = "".join(parts).strip()
        if joined:
            yield node.lineno, joined
        return
    if isinstance(node, ast.IfExp):
        yield from _literals(node.body)
        yield from _literals(node.orelse)
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        yield from _literals(node.left)
        yield from _literals(node.right)
        return
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            yield from _literals(value)


def hardcoded_ui_strings(source: str, filename: str = "sample.py") -> list[str]:
    """String literals passed to a Qt text setter, a tr-less widget, or a splash stage."""
    tree = ast.parse(source)
    patterns = _patterns()
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee(node.func)
        if name not in _SETTERS and name not in _CTORS and name not in _BOXES:
            continue
        args = node.args[1:] if name in _BOXES and node.args else node.args
        values = list(args)
        values.extend(keyword.value for keyword in node.keywords if keyword.arg in {None, "text", "title", "label"})
        for arg in values:
            for lineno, value in _literals(arg):
                if _allowed(value, patterns):
                    continue
                errors.append(f"{filename}:{lineno}: {value!r} passed to {name}")
    return errors


def locale_gaps(payload: dict[str, object]) -> list[str]:
    """Keys that miss a shipped locale, or name a locale Canvas does not ship."""
    locales = payload.get("locales")
    found = payload.get("strings")
    if not isinstance(locales, list) or not locales:
        return ["locales is empty"]
    if not isinstance(found, dict) or not found:
        return ["strings is empty"]
    gaps: list[str] = []
    shipped = {str(item) for item in locales}
    for key, entry in found.items():
        if not isinstance(entry, dict):
            gaps.append(f"{key} is not a locale map")
            continue
        for lang in shipped:
            value = entry.get(lang)
            if not isinstance(value, str) or not value.strip():
                gaps.append(f"{key} missing {lang}")
        for lang in entry:
            if str(lang) not in shipped:
                gaps.append(f"{key} has unshipped locale {lang}")
    return gaps


def _catalog_keys(source: str) -> list[str]:
    tree = ast.parse(source)
    keys: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _callee(node.func) not in _CATALOG_CALLS:
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            keys.append(node.args[0].value)
    return keys


def test_gate_unavailable_is_the_shipped_sentence():
    assert shipped_locales() == ("en",)
    assert text("gate.unavailable", "en") == "Image generation is unavailable."


def test_a_russian_system_locale_still_shows_english(monkeypatch):
    from voxprint_common.i18n import get_language, set_language, tr, tr_lang

    from canvas.app import main as app_main

    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.setenv(name, "ru_RU.UTF-8")
    monkeypatch.setenv("VOXPRINT_LANG", "ru")
    set_language("ru")
    assert app_main.main(["--version"]) == 0
    assert get_language() == "en"
    assert text("gate.unavailable") == "Image generation is unavailable."
    assert tr("simple.cancel") == tr_lang("simple.cancel", "en")
    assert not list((ROOT / "canvas" / "ui" / "locales").glob("ru.json"))


def test_every_catalog_key_has_every_shipped_locale():
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert locale_gaps(payload) == []
    assert set(strings()) >= set(_PLURAL_KEYS)


def test_a_missing_translation_fails_the_catalog_check():
    gaps = locale_gaps({"locales": ["en"], "strings": {"gate.unavailable": {"de": "Gestoppt"}}})
    assert gaps == ["gate.unavailable missing en", "gate.unavailable has unshipped locale de"]


def test_ui_code_has_no_hardcoded_text():
    errors: list[str] = []
    used: list[str] = []
    known = set(strings())
    for path in sorted((ROOT / "canvas").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        errors.extend(hardcoded_ui_strings(source, path.relative_to(ROOT).as_posix()))
        for key in _catalog_keys(source):
            used.append(key)
            if key not in known:
                errors.append(f"{path.relative_to(ROOT).as_posix()}: unknown catalog key {key}")
    assert errors == []
    assert "gate.unavailable" in used


def test_the_lint_flags_a_hardcoded_widget_and_allows_the_list():
    flagged = hardcoded_ui_strings('QLabel("Hello")\n')
    assert flagged
    assert hardcoded_ui_strings('QLabel("")\nQPushButton("16:9")\nQLabel(text("app.name"))\n') == []
    assert hardcoded_ui_strings('self.status.setText("Enhancing…")\n')
    assert hardcoded_ui_strings('progress(f"Installing {name}")\n')
