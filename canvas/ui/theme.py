"""Suite palette and stylesheet.

Adapted from Voxprint AI Audiobook Builder ``ui/main_window.py``
(Apache-2.0, Copyright Aleksandr Mitroshenkov). Tokens match
``suite/design/PALETTE.md``. Text colours are opaque. Only the window
background is translucent in Acrylic mode.
"""
from __future__ import annotations

TEXT = "#f2f2f5"
TEXT_MUTED = "#c4c4d0"
TEXT_FAINT = "#a8a8b8"
TEXT_DISABLED = "#8e8e9e"
TEXT_ON_ACCENT = "#0b1220"
ACCENT = "#60a5fa"
ACCENT_HOVER = "#7db9ff"
ACCENT_SOFT = "#93c5fd"
ACCENT_STRONG = "#2563eb"
ROOT_PLAIN = "#17171c"
ROOT_GLASS = "rgba(16,16,22,222)"
ROOT_GLASS_MORE = "rgba(16,16,22,167)"
CARD_PLAIN = "#202029"
CARD_GLASS = "rgba(34,34,44,238)"
CARD_GLASS_MORE = "rgba(34,34,44,214)"
CONTROL_BG = "#2c2c38"
CONTROL_HOVER = "#383846"
CONTROL_BORDER = "#4b4b5c"
AMBER = "#fcd34d"
NOTE_BG = "#2a2418"
NOTE_BORDER = "#6b5720"
GREEN = "#86efac"
BADGE_GREEN_TEXT = "#0b1f12"
BADGE_AMBER_TEXT = "#251a02"
PRIMARY = "#f5d76e"
PRIMARY_HOVER = "#f9e291"
PRIMARY_BORDER = "#fbe8a6"
TEXT_ON_PRIMARY = "#1c1604"
PRIMARY_DISABLED_BG = "#36321f"
PRIMARY_DISABLED_TEXT = "#b9b08a"
MENU_BG = "#1c2535"
MENU_SELECTION = "#2f4c78"
RED = "#fca5a5"
# Canvas splash identity uses the suite accent. The rest of the window uses the tokens above.
SPLASH = ACCENT

CONTRAST_PAIRS = [
    (TEXT, ROOT_PLAIN), (TEXT, CARD_PLAIN), (TEXT, CONTROL_BG), (TEXT, CONTROL_HOVER),
    (TEXT_MUTED, ROOT_PLAIN), (TEXT_MUTED, CARD_PLAIN), (TEXT_FAINT, ROOT_PLAIN), (TEXT_FAINT, CARD_PLAIN),
    (TEXT_ON_ACCENT, ACCENT), (TEXT_ON_ACCENT, ACCENT_HOVER), (TEXT_ON_ACCENT, ACCENT_SOFT),
    (TEXT, ACCENT_STRONG), ("#dcdce4", "#34343f"), ("#86efac", CARD_PLAIN),
    (BADGE_GREEN_TEXT, GREEN), (BADGE_AMBER_TEXT, AMBER), (AMBER, CARD_PLAIN), (AMBER, NOTE_BG), (AMBER, ROOT_PLAIN),
    ("#ffffff", ACCENT_STRONG), (ACCENT_SOFT, ROOT_PLAIN), (ACCENT_SOFT, CARD_PLAIN),
    (TEXT_ON_PRIMARY, PRIMARY), (TEXT_ON_PRIMARY, PRIMARY_HOVER), (PRIMARY_DISABLED_TEXT, PRIMARY_DISABLED_BG),
    (TEXT, MENU_BG), ("#ffffff", MENU_SELECTION),
]

_STYLE = """
* {{ font-family: "Segoe UI Variable Text", "Segoe UI", "Ubuntu", "Noto Sans", "Cantarell", "DejaVu Sans", sans-serif; font-size: 14px; color: {text}; }}
QWidget#root {{ background: {root_bg}; }}
QLabel#title {{ font-size: 22px; font-weight: 600; }}
QLabel#subtitle, QLabel#footer, QLabel#hint {{ color: {faint}; }}
QLabel#footer {{ font-size: 12px; }}
QFrame#card {{ background: {card_bg}; border: 1px solid {border}; border-radius: 12px; }}
QPushButton {{ background: {control}; border: 1px solid {border}; border-radius: 8px; padding: 6px 14px; }}
QPushButton:hover {{ background: {hover}; }}
QPushButton:disabled {{ color: {disabled}; background: #1f1f28; border-color: #34343f; }}
QPushButton#aspect {{ padding: 4px 8px; }}
QPushButton#aspect:checked {{ background: {menu_sel}; border-color: {splash}; }}
QLabel#aspect-group {{ color: {faint}; font-size: 12px; }}
QPushButton#primary {{ background: {primary}; border: 1px solid {primary_border}; font-size: 16px; font-weight: 600; padding: 10px 18px; color: {on_primary}; }}
QPushButton#primary:hover {{ background: {primary_hover}; }}
QPushButton#primary:disabled {{ background: {primary_dis_bg}; color: {primary_dis_text}; border-color: #4a4430; }}
QComboBox, QLineEdit, QPlainTextEdit {{ background: {control}; border: 1px solid {border}; border-radius: 8px; padding: 6px 12px; }}
QComboBox QAbstractItemView {{ background: {menu_bg}; border: 1px solid {border}; selection-background-color: {menu_sel}; selection-color: #ffffff; }}
QTabWidget::pane {{ border: 1px solid {border}; border-radius: 8px; background: {card_bg}; }}
QTabBar::tab {{ background: {control}; color: {text}; padding: 8px 16px; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
QTabBar::tab:selected {{ background: {menu_bg}; }}
QProgressBar {{ background: {control}; border: 1px solid {border}; border-radius: 8px; }}
QProgressBar::chunk {{ background: {splash}; border-radius: 7px; }}
QLabel#status {{ color: {muted}; }}
"""


def build_style(glass: bool, more: bool = False) -> str:
    """Acrylic tint, or the solid fallback. ``more`` is the lighter glass setting."""
    if glass and more:
        root_bg, card_bg = ROOT_GLASS_MORE, CARD_GLASS_MORE
    elif glass:
        root_bg, card_bg = ROOT_GLASS, CARD_GLASS
    else:
        root_bg, card_bg = ROOT_PLAIN, CARD_PLAIN
    return _STYLE.format(
        text=TEXT, faint=TEXT_FAINT, muted=TEXT_MUTED, disabled=TEXT_DISABLED,
        control=CONTROL_BG, hover=CONTROL_HOVER, border=CONTROL_BORDER,
        primary=PRIMARY, primary_border=PRIMARY_BORDER, primary_hover=PRIMARY_HOVER,
        on_primary=TEXT_ON_PRIMARY, primary_dis_bg=PRIMARY_DISABLED_BG, primary_dis_text=PRIMARY_DISABLED_TEXT,
        menu_bg=MENU_BG, menu_sel=MENU_SELECTION, splash=SPLASH,
        root_bg=root_bg, card_bg=card_bg,
    )
