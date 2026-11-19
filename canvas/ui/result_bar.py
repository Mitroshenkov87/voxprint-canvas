"""The buttons under the result picture, laid out so every label stays whole.

The shared view builds them in one fixed line. Canvas moves them into the
wrapping flow used by the action row, so a narrow window puts the last
buttons on a second line instead of clipping them.
"""
from __future__ import annotations

from PySide6.QtWidgets import QPushButton, QSizePolicy, QWidget

SPACING = 8
MARGINS = (12, 0, 12, 12)


def wrap(bar: QWidget) -> None:
    """Swap the bar's line layout for the wrapping flow, once."""
    from canvas.ui.shell_chrome import _Flow

    old = bar.layout()
    if isinstance(old, _Flow):
        return
    widgets = []
    if old is not None:
        while old.count():
            item = old.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widgets.append(widget)
        # A widget keeps its layout until the layout moves elsewhere.
        QWidget().setLayout(old)
    flow = _Flow()
    flow.may_wrap = True
    flow.setSpacing(SPACING)
    flow.setContentsMargins(*MARGINS)
    bar.setLayout(flow)
    for widget in widgets:
        if isinstance(widget, QPushButton):
            policy = widget.sizePolicy()
            policy.setHorizontalPolicy(QSizePolicy.Policy.Minimum)
            widget.setSizePolicy(policy)
        flow.addWidget(widget)
    policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    policy.setHeightForWidth(True)
    bar.setSizePolicy(policy)
