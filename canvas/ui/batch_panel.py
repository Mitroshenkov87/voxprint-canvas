"""The Batch section: an optional seed list and one prompt per line.

The section sits in the left control column under the actions. Empty fields
mean one picture from the prompt box with the picture count, exactly as
before. Filled fields plan the run through ``canvas.app.generate_batch``
(the CLI's ``--seeds``/``--prompts`` rules), so the window and the CLI agree
on what a batch is.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from canvas.app.i18n import text


def mount(window, column) -> None:
    """Add the collapsed Batch disclosure to the prompt ``column``.

    Args:
        window: The shell window. The fields are read when Generate plans the
            run; they hold no other state.
        column: The prompt panel's vertical layout.
    """
    toggle = QToolButton()
    toggle.setObjectName("batchToggle")
    toggle.setText(text("canvas.batch.section"))
    toggle.setCheckable(True)
    toggle.setChecked(False)
    toggle.setAutoRaise(True)
    toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.ArrowType.RightArrow)
    body = QWidget()
    body.setObjectName("batchBody")
    inner = QVBoxLayout(body)
    inner.setContentsMargins(16, 2, 0, 2)
    inner.setSpacing(6)

    seeds_row = QWidget()
    seeds_line = QHBoxLayout(seeds_row)
    seeds_line.setContentsMargins(0, 0, 0, 0)
    seeds_line.setSpacing(8)
    seeds_label = QLabel(text("canvas.batch.seeds"))
    seeds_label.setObjectName("batchSeedsLabel")
    seeds_label.setProperty("setupHint", True)
    seeds = QLineEdit()
    seeds.setObjectName("batchSeeds")
    seeds.setPlaceholderText(text("canvas.batch.seeds_tip"))
    seeds_line.addWidget(seeds_label)
    seeds_line.addWidget(seeds, 1)
    inner.addWidget(seeds_row)

    prompts_label = QLabel(text("canvas.batch.prompts"))
    prompts_label.setObjectName("batchPromptsLabel")
    prompts_label.setProperty("setupHint", True)
    prompts = QPlainTextEdit()
    prompts.setObjectName("batchPrompts")
    prompts.setPlaceholderText(text("canvas.batch.prompts_tip"))
    prompts.setFixedHeight(96)
    inner.addWidget(prompts_label)
    inner.addWidget(prompts)

    body.hide()

    def opened(on: bool) -> None:
        body.setVisible(bool(on))
        toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)

    toggle.toggled.connect(opened)
    # The shared screen refuses an empty prompt box before Generate reaches
    # the window, so the first line mirrors into the box: the run shows what
    # it draws. A typed prompt is left alone; the window calls that «mixed».
    prompts.textChanged.connect(lambda: _mirror_first_line(window, prompts))
    at = column.count()
    for index in range(column.count()):
        if column.itemAt(index).spacerItem() is not None:
            at = index
            break
    column.insertWidget(at, toggle)
    column.insertWidget(at + 1, body)
    window._batch_toggle = toggle
    window._batch_body = body
    window._batch_seeds = seeds
    window._batch_prompts = prompts


def _mirror_first_line(window, prompts) -> None:
    """Copy the first batch line into an empty prompt box, so Generate starts."""
    lines = [line.strip() for line in prompts.toPlainText().splitlines() if line.strip()]
    layout = window.shell.layout
    if not lines or str(layout.prompt).strip():
        return
    window.view.set_prompt_text(lines[0])


def batch_fields(window) -> tuple[str, list[str]]:
    """The raw batch inputs: the seed list text and the prompt lines.

    Args:
        window: The shell window.

    Returns:
        ``(seeds_text, prompt_lines)`` with blank lines dropped.
    """
    seeds = ""
    prompts = ""
    toggle = getattr(window, "_batch_toggle", None)
    if toggle is not None and toggle.isChecked():
        seeds = window._batch_seeds.text().strip()
        prompts = window._batch_prompts.toPlainText()
    lines = [line.strip() for line in prompts.splitlines() if line.strip()]
    return seeds, lines
