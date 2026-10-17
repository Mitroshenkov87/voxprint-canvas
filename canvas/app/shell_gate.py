"""Adapt the always-on content gate to the shared screen.

The shared screen asks for a plain reason. The Canvas gate raises. This
adapter returns that sentence. It has no switch and it does not replace the
gate instance.
"""
from __future__ import annotations

from canvas.app.i18n import text
from canvas.safety.gate import AgeGate, SafetyError
from canvas.safety.image import RawImage, SafeImage


class ContentCheck:
    """Reviews prompts and finished pictures through one gate instance."""

    def __init__(self, gate: AgeGate) -> None:
        """Keep ``gate`` and review through it.

        Args:
            gate: The process content gate. Any other object is refused.

        Raises:
            TypeError: ``gate`` is not the Canvas content gate.
        """
        if not isinstance(gate, AgeGate):
            raise TypeError("the screen needs the always-on content gate")
        self.age_gate = gate
        self._staged: dict[str, RawImage] = {}
        self.kept: dict[str, SafeImage] = {}

    def stage(self, item_id: str, image: RawImage) -> None:
        """Hold pixels so the result review can see them.

        Args:
            item_id: Id the screen will pass to ``finish``.
            image: A copy of the picture. The review consumes it.
        """
        self._staged[item_id] = image

    def review_prompt(self, prompt: str) -> str | None:
        """Return a plain refusal, or None when the prompt may continue.

        Args:
            prompt: The words that would be used.
        """
        try:
            self.age_gate.review_prompt(prompt)
        except SafetyError as exc:
            return exc.user_message
        except Exception:
            return text("gate.unavailable")
        return None

    def review_result(self, prompt: str, item_id: str) -> str | None:
        """Review the staged picture. A missing picture is refused.

        Args:
            prompt: The words that produced the picture.
            item_id: Id passed to ``stage``.
        """
        raw = self._staged.pop(item_id, None)
        if raw is None:
            return text("gate.unavailable")
        try:
            safe = self.age_gate.review_image(raw, prompt)
        except SafetyError as exc:
            return exc.user_message
        except Exception:
            return text("gate.unavailable")
        self.kept[item_id] = safe
        return None
