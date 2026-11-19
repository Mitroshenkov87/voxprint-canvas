"""Adapt the always-on content gate to the shared screen.

The shared screen asks for a plain reason. The Canvas gate raises. This
adapter returns that sentence. It has no switch and it does not replace the
gate instance.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from voxprint_common.ui.safety import ReviewedOutput

from canvas.app.i18n import text
from canvas.safety.gate import AgeGate, SafetyError, release_generated
from canvas.safety.image import RawImage, SafeImage
from canvas.ui.ui_call import run_off_ui


class ContentCheck:
    """Reviews prompts through one gate instance and releases finished pictures."""

    def __init__(self, *, gate: AgeGate) -> None:
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
        def work() -> str | None:
            try:
                self.age_gate.review_prompt(prompt)
            except SafetyError as exc:
                return exc.user_message
            except Exception:
                return text("gate.unavailable")
            return None

        return run_off_ui(work)

    def review_result(
        self, prompt: str, item_id: str, output: object = None, *, lora_files: Sequence[Path] = (),
    ) -> str | None:
        """Release the staged picture. A missing picture is refused.

        Args:
            prompt: The words that produced the picture.
            item_id: Id passed to ``stage``.
            output: The shell's copy of the output (Yitro). Not trusted: the
                staged pixels are what is released.
            lora_files: Weights of every LoRA loaded for this picture. Ignored:
                the request was screened before generation and no classifier
                runs on generated pixels, with any LoRA or none
                (:func:`canvas.safety.gate.release_generated`).
        """
        raw = self._staged.pop(item_id, None)
        if raw is None:
            return text("gate.unavailable")

        def work() -> str | None:
            try:
                safe = release_generated(raw)
            except SafetyError as exc:
                return exc.user_message
            except Exception:
                return text("gate.unavailable")
            self.kept[item_id] = safe
            return None

        return run_off_ui(work)


def reviewed_output(image: RawImage) -> ReviewedOutput:
    """The output copy Yitro's ``finish`` needs. A missing copy is blocked there.

    The staged pixels are what ``review_result`` releases, not this copy.

    Args:
        image: The finished, unreviewed picture.
    """
    return ReviewedOutput(data=image.to_bytes(), mime=f"image/x-raw-rgb;width={image.width};height={image.height}")
