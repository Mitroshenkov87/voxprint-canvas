"""Fields recorded on a saved picture when metadata is enabled."""
from __future__ import annotations

from typing import Any, Mapping

from canvas.models.user_lora import UserLora, clamp_strength, resolve_selected, running_base_id


def picture_fields(
    *,
    prompt: str,
    seed: int,
    steps: int = 8,
    save_metadata: bool = False,
    lora: UserLora | None = None,
    lora_path: str = "",
    lora_weight: float = 0.7,
) -> dict[str, Any]:
    """Metadata recorded on one saved picture, in the shared schema.

    With full metadata off the picture still carries its ``seed`` and nothing
    else: no prompt, model, steps or LoRA. With it on, ``seed`` sits in the
    full block.

    ``seed`` and ``steps`` are integers, the LoRA strength is a float, and a
    resolved LoRA is listed under ``loras`` with only the fields that are
    known. Every field is best-effort: a LoRA that cannot be re-read never
    stops the picture; its raw strength field is recorded instead.
    """
    if not save_metadata:
        return {"seed": int(seed)}
    fields: dict[str, Any] = {
        "prompt": prompt,
        "seed": int(seed),
        "steps": int(steps),
    }
    base = running_base_id()
    if base:
        fields["base"] = base
    selected = lora
    path = lora_path or ("" if lora is None else str(lora.weights))
    if selected is None and path:
        try:
            selected = resolve_selected(path)
        except Exception:  # noqa: BLE001 - metadata never blocks a job
            selected = None
    if not path and selected is None:
        return fields
    strength: float = lora_weight if lora is None else clamp_strength(lora, lora_weight)
    name = ""
    digest = ""
    lora_base = ""
    if selected is not None:
        strength = clamp_strength(selected, lora_weight)
        name = selected.name or selected.id
        digest = selected.sha256
        lora_base = selected.base
    fields["lora_strength"] = strength
    if not name:
        return fields
    record: dict[str, Any] = {"name": name, "strength": strength}
    if digest:
        record["sha256"] = digest
    if lora_base:
        record["base"] = lora_base
    fields["loras"] = [record]
    return fields


def fields_for_job(job: object) -> dict[str, Any]:
    """``picture_fields`` from a ``PictureJob`` or any object with the same attributes."""
    return picture_fields(
        prompt=str(getattr(job, "prompt", "") or ""),
        seed=int(getattr(job, "seed", 0) or 0),
        steps=int(getattr(job, "steps", 8) or 8),
        save_metadata=bool(getattr(job, "save_metadata", False)),
        lora_path=str(getattr(job, "lora_path", "") or ""),
        lora_weight=float(getattr(job, "lora_weight", 0.7) or 0.7),
    )


def merge_metadata(metadata: Mapping[str, Any] | None, seed: int | None) -> dict[str, Any] | None:
    """Prefer an explicit metadata mapping; otherwise a seed-only record."""
    if metadata is not None:
        return dict(metadata)
    if seed is None:
        return None
    return {"seed": int(seed)}
