"""The shared screen's review hooks, answered from worker replies.

The image screen asks its gate object two questions: may this prompt start,
and may this finished picture be kept. In the window both answers come from
the worker. Checks run in the worker; this object only remembers which
pictures the worker released and which it refused. A picture it was not told
about is refused, so a missing or late reply never shows pixels.
"""
from __future__ import annotations

from pathlib import Path

from voxprint_common.ui.safety import ReviewedOutput

from canvas.app.i18n import text


class WorkerChecks:
    """The screen's gate object in the window process. It runs no checks itself."""

    def __init__(self, client: object | None = None) -> None:
        """Answer for ``client``'s worker. Without a client every question is refused."""
        self.client = client
        self._released: dict[str, Path] = {}
        self._refused: dict[str, str] = {}

    def review_prompt(self, prompt: str) -> str | None:
        """Let the screen start; the worker reviews the prompt as the job's first step.

        Without a worker there is nothing to review with, so the screen is refused.
        """
        del prompt
        if self.client is None:
            return text("gate.unavailable")
        return None

    def release(self, item_id: str, path: Path) -> None:
        """Remember that the worker released ``item_id`` as the file ``path``."""
        self._released[item_id] = Path(path)

    def refuse(self, item_id: str, reason: str) -> None:
        """Remember that the worker refused ``item_id``."""
        self._refused[item_id] = reason

    def review_result(self, prompt: str, item_id: str, output: ReviewedOutput | None = None) -> str | None:
        """None only for a picture the worker released, given back as the same file."""
        del prompt
        path = self._released.pop(item_id, None)
        refused = self._refused.pop(item_id, None)
        if path is None:
            return refused or text("gate.unavailable")
        if output is None or output.path is None or Path(output.path) != path or not path.is_file():
            return text("gate.unavailable")
        return None
