"""Region-refine compare for a person LoRA (operator tool, GPU machine).

Renders the same prompt and seed up to four ways so an operator can judge
whether a coverage fallback helps a zone the LoRA covers weakly:

    A  baseline LoRA text render at --strength
    B  strength schedule: --strength for the first --early-steps, then --late-strength
    C  baseline, then a base-model inpaint over the feathered, face-locked mask
    D  B, then the same base-model inpaint

Any zone can be refined (hands, back, profile, feet, and so on). A weak angle
has no box: variant B applies the late-strength schedule to the whole render.
--face-lock is on by default and never repaints the face unless --zone face.

Every prompt, input image, and output goes through the normal Canvas content
gate. 18+ OFF/ON is the same flag as in Canvas (``explicit`` on the pipeline).
A refusal stops that variant only. The tool never downloads weights.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

if __package__ in (None, ""):  # run as a script from a checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from canvas.core.formats import OutputFormat, save_image
from canvas.models.modes import Mode
from canvas.models.user_lora import _string_metadata
from canvas.safety.classifiers import Face

try:
    from canvas.models.user_lora import apply_trigger as _library_apply_trigger
except ImportError:  # rc.40 (c83ba7cc) has the trigger header but no alias rewrite
    _library_apply_trigger = None
from canvas.safety.gate import SafetyError
from canvas.safety.image import RawImage, SafeImage
from canvas.safety.inputs import ImageRole, PendingInput

SCHEMA = "voxprint-coverage-refine/1"
COVERAGE_KEY = "voxprint.coverage"
STRENGTH_KEY = "voxprint.lora.recommended_strength"
COVERAGE_SCHEMA = "voxprint-coverage/1"
REGION_BINS = (
    "face", "hair", "neck", "shoulders", "arms", "hands",
    "bust", "torso", "back", "hips", "pelvis", "buttocks", "legs", "feet",
)
ANGLE_BINS = (
    "front", "three_quarter_left", "three_quarter_right",
    "profile_left", "profile_right", "back", "high", "low",
)
_SCORE_FIELDS = ("score", "score_clothed", "score_nude")
_COUNT_FIELDS = ("count", "count_clothed", "count_nude")
PRESETS = ("fast", "balanced", "best")
ALL_VARIANTS = ("A", "B", "C", "D")
ADAPTER = "canvas-lora"
SFACE_NAMES = ("face_recognition_sface_2021dec.onnx", "face_recognition_sface_2021dec_int8.onnx")
MASK_VARIANTS = ("C", "D")



def lead_trigger(prompt: str, lora: object) -> str:
    """Alias rewrite and trigger insert when Canvas has apply_trigger (rc.41+); otherwise trigger insert only."""
    if _library_apply_trigger is not None:
        return _library_apply_trigger(prompt, lora)
    return lead_trigger_legacy(prompt, lora)


def lead_trigger_legacy(prompt: str, lora: object) -> str:
    """Put the LoRA trigger at the start of the prompt when it is not already there (rc.40 path)."""
    trigger = (getattr(lora, "trigger", None) or "").strip()
    if not trigger:
        return prompt
    if trigger in prompt:
        return prompt
    body = prompt.strip()
    return f"{trigger}, {body}" if body else trigger


def recommended_strength(weights: Path) -> float:
    """``voxprint.lora.recommended_strength`` from the header, else 1.0. Invalid values are 1.0."""
    raw = _string_metadata(weights).get(STRENGTH_KEY)
    if raw is None or raw == "":
        return 1.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0


# --------------------------------------------------------------------- hooks

@dataclass
class RenderRequest:
    """One generator call. ``schedule`` and ``denoise`` are set only where a variant needs them."""

    prompt: str
    seed: int
    mode: Mode
    inputs: tuple[PendingInput, ...] = ()
    schedule: StepSchedule | None = None
    denoise: float | None = None
    use_lora: bool = True
    explicit: bool = True


@dataclass
class Hooks:
    """Test seams. Production leaves every field at None."""

    weights_ready: Callable[[], bool] | None = None
    generate: Callable[[RenderRequest], SafeImage] | None = None
    gate: Any = None  # object with review_image(RawImage, prompt) -> SafeImage
    detect_faces: Callable[[RawImage], Sequence[Face]] | None = None
    likeness: Callable[[RawImage, RawImage, Face], float | None] | None = None
    log: TextIO = field(default_factory=lambda: sys.stderr)


# ------------------------------------------------------------------ coverage

@dataclass
class Coverage:
    """A tolerant read of ``voxprint-coverage/1`` (schema_rev 3, with rev 2/1 migration)."""

    present: bool = False
    usable: bool = False
    rev: int = 1
    generator_base: str = ""
    weak: list[str] = field(default_factory=list)
    bins: dict[str, dict[str, Any]] = field(default_factory=dict)
    document: dict[str, Any] | None = None
    warning: str = ""
    origin: str = ""


def _qualify(kind: str, name: str) -> str:
    return f"{kind}.{name}"


def _bin_entry(row: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    for key in _SCORE_FIELDS:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            entry[key] = float(value)
    for key in _COUNT_FIELDS:
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            entry[key] = value
    if isinstance(row.get("source"), str):
        entry["source"] = row["source"]
    return entry


def _qualify_name(name: str) -> str | None:
    """Qualified id, or None when the name is unknown and should be ignored.

    Unqualified ``back`` is ``angles.back`` (the name exists in both maps).
    """
    if name.startswith("regions."):
        bare = name.split(".", 1)[1]
        return _qualify("regions", bare) if bare in REGION_BINS else None
    if name.startswith("angles."):
        bare = name.split(".", 1)[1]
        return _qualify("angles", bare) if bare in ANGLE_BINS else None
    if name == "back":
        return "angles.back"
    if name in REGION_BINS:
        return _qualify("regions", name)
    if name in ANGLE_BINS:
        return _qualify("angles", name)
    return None


def _as_document(raw: object) -> tuple[dict[str, Any] | None, str]:
    if raw is None or raw == "":
        return None, ""
    if isinstance(raw, dict):
        return raw, ""
    if not isinstance(raw, str):
        return None, "coverage is not a string or object; ignored"
    try:
        doc = json.loads(raw)
    except ValueError:
        return None, "coverage is not valid JSON; ignored"
    if not isinstance(doc, dict):
        return None, "coverage is not an object; ignored"
    return doc, ""


def _collect_weak(listed: object) -> list[str]:
    weak: list[str] = []
    for name in listed if isinstance(listed, list) else []:
        if not isinstance(name, str):
            continue
        qualified = _qualify_name(name)
        if qualified and qualified not in weak:
            weak.append(qualified)
    return weak


def parse_coverage(raw: object) -> Coverage:
    """Read a coverage document. Anything unusable counts as absent (with a warning).

    Schema_rev 3 uses ``bins.angles`` and ``bins.regions`` and qualified
    ``weak`` ids such as ``regions.hands``. Rev 2 (flat bins) is migrated:
    ``back`` becomes ``angles.back``. ``bins_legacy`` and ``weak_legacy`` are
    ignored, and so are unknown names. A ``manual`` source is read like a
    measured one. A document without ``schema_rev`` (or with 1) uses the
    single ``score``/``count``.
    """
    if raw is None or raw == "":
        return Coverage()
    doc, warning = _as_document(raw)
    if doc is None:
        return Coverage(present=True, warning=warning or "coverage is unusable; ignored")
    if doc.get("schema") != COVERAGE_SCHEMA:
        return Coverage(present=True, warning="voxprint.coverage has an unknown schema; ignored")
    rev = doc.get("schema_rev")
    rev = rev if isinstance(rev, int) and not isinstance(rev, bool) else 1
    generator = doc.get("generator_base") if isinstance(doc.get("generator_base"), str) else ""
    bins: dict[str, dict[str, Any]] = {}
    if rev >= 3:
        groups = doc.get("bins") if isinstance(doc.get("bins"), dict) else {}
        for kind, allowed in (("angles", ANGLE_BINS), ("regions", REGION_BINS)):
            group = groups.get(kind) if isinstance(groups, dict) else None
            if not isinstance(group, dict):
                continue
            for name, row in group.items():
                if name in allowed and isinstance(row, dict):
                    bins[_qualify(kind, name)] = _bin_entry(row)
    else:
        found = doc.get("bins")
        if isinstance(found, dict):
            for name, row in found.items():
                if not isinstance(name, str) or not isinstance(row, dict):
                    continue
                qualified = _qualify_name(name)
                if qualified is None:
                    continue
                bins[qualified] = _bin_entry(row)
    weak = _collect_weak(doc.get("weak"))
    return Coverage(
        present=True, usable=True, rev=rev, generator_base=generator,
        weak=weak, bins=bins, document=doc,
    )


def sidecar_path(weights: Path) -> Path:
    """``<lora stem>.coverage.json`` next to the LoRA file, or ``<dir>/<name>.coverage.json``."""
    path = Path(weights)
    if path.is_dir():
        return path / f"{path.name}.coverage.json"
    return path.with_name(f"{path.stem}.coverage.json")


def parse_coverage_file(path: Path) -> Coverage:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return Coverage(present=True, warning=f"could not read {path}: {exc}")
    return parse_coverage(text)


def read_coverage(weights: Path) -> Coverage:
    """Coverage from the LoRA's safetensors header strings (non-string values are ignored)."""
    return parse_coverage(_string_metadata(weights).get(COVERAGE_KEY))


def load_coverage(weights: Path, coverage_flag: Path | None = None) -> Coverage:
    """Sidecar next to the LoRA, else ``--coverage`` PATH, else the embedded key.

    The sidecar takes precedence over the embedded ``voxprint.coverage`` key.
    """
    side = sidecar_path(weights)
    if side.is_file():
        coverage = parse_coverage_file(side)
        coverage.origin = "sidecar"
        return coverage
    if coverage_flag is not None:
        coverage = parse_coverage_file(coverage_flag)
        coverage.origin = "flag"
        return coverage
    coverage = read_coverage(weights)
    if coverage.present:
        coverage.origin = "embedded"
    return coverage


def _active_score(row: dict[str, Any], adult: bool) -> float | None:
    if adult:
        value = row.get("score_nude", row.get("score"))
    else:
        value = row.get("score_clothed", row.get("score"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _listed_applies(row: dict[str, Any], adult: bool, rev: int) -> bool:
    if adult or rev < 2:
        return True
    # Clothed state: a listing is ignored when the bin holds only a nude score.
    return not ("score_nude" in row and row.get("score_clothed", row.get("score")) is None)


def _weak_of(coverage: Coverage, adult: bool, kind: str) -> list[str]:
    """Unqualified names of weak bins of this kind, in schema order."""
    if not coverage.usable:
        return []
    allowed = REGION_BINS if kind == "regions" else ANGLE_BINS
    out: list[str] = []
    for name in allowed:
        qualified = _qualify(kind, name)
        row = coverage.bins.get(qualified, {})
        listed = qualified in coverage.weak
        if coverage.rev < 2:
            active = row.get("score")
            if listed or (active is not None and active <= 0.0):
                out.append(name)
            continue
        active = _active_score(row, adult)
        if (active is not None and active <= 0.0) or (listed and _listed_applies(row, adult, coverage.rev)):
            out.append(name)
    return out


def weak_regions(coverage: Coverage, adult: bool) -> list[str]:
    """Region bins that are weak for the active state (``adult`` = 18+ on)."""
    return _weak_of(coverage, adult, "regions")


def weak_angles(coverage: Coverage, adult: bool) -> list[str]:
    """Angle bins that are weak for the active state (``adult`` = 18+ on)."""
    return _weak_of(coverage, adult, "angles")


def qualify_zone(zone: str | None) -> str | None:
    """Qualified id for a ``--zone`` label, or the trimmed label if it is not a known bin."""
    if zone is None:
        return None
    text = zone.strip()
    if not text:
        return None
    return _qualify_name(text) or text


def zone_matches(zone: str | None, kind: str, name: str) -> bool:
    """True when no zone was given, or the zone names this bin."""
    qualified = qualify_zone(zone)
    if qualified is None:
        return True
    target = _qualify(kind, name)
    return qualified == target or qualified == name


def default_variants(
    coverage: Coverage,
    adult: bool = True,
    *,
    refine: bool = False,
    zone: str | None = None,
) -> tuple[list[str], str]:
    """Variants to run when ``--variants`` is omitted, and the reason.

    Region refine (the toggle, or a weak region) defaults to A,C,D. A weak
    angle has no box: B applies the late-strength schedule to the whole render.
    """
    regions = [name for name in weak_regions(coverage, adult) if zone_matches(zone, "regions", name)]
    angles = [name for name in weak_angles(coverage, adult) if zone_matches(zone, "angles", name)]
    state = "adult (18+)" if adult else "clothed"
    parts: list[str] = []
    if regions:
        parts.append("weak regions for the " + state + " state: " + ", ".join(_qualify("regions", n) for n in regions))
    if angles:
        parts.append(
            "weak angles for the " + state + " state: "
            + ", ".join(_qualify("angles", n) for n in angles)
            + " (no box; late-strength schedule applies to the whole render)"
        )
    if refine and not regions:
        parts.append("region refine enabled")
    region_refine = refine or bool(regions)
    if region_refine and angles:
        return ["A", "B", "C", "D"], "; ".join(parts)
    if region_refine:
        return ["A", "C", "D"], "; ".join(parts)
    if angles:
        return ["A", "B"], "; ".join(parts)
    if coverage.usable:
        return ["A", "B", "C"], f"coverage lists no weak bins for the {state} state"
    return ["A", "B", "C"], "no usable coverage key"


def parse_variants(text: str) -> list[str]:
    """``A,C,D`` as an ordered list. Raises ValueError for anything else."""
    chosen = [part.strip().upper() for part in text.split(",") if part.strip()]
    if not chosen or any(item not in ALL_VARIANTS for item in chosen):
        raise ValueError("--variants takes a comma list among A,B,C,D")
    return [item for item in ALL_VARIANTS if item in chosen]


# -------------------------------------------------------------- mask / boxes

Box = tuple[int, int, int, int]


def parse_box(text: str) -> Box:
    """``x,y,w,h`` in pixels."""
    try:
        parts = [int(part) for part in text.split(",")]
    except ValueError as exc:
        raise ValueError("a box is x,y,w,h in whole pixels") from exc
    if len(parts) != 4 or parts[2] < 1 or parts[3] < 1 or parts[0] < 0 or parts[1] < 0:
        raise ValueError("a box is x,y,w,h with a positive width and height")
    return (parts[0], parts[1], parts[2], parts[3])


def _clip(box: Box, width: int, height: int) -> Box | None:
    x0, y0 = max(0, box[0]), max(0, box[1])
    x1, y1 = min(width, box[0] + box[2]), min(height, box[1] + box[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def dilate_box(box: Box, pixels: int, width: int, height: int) -> Box:
    """Grow a box by ``pixels`` on every side, kept inside the picture."""
    grown = (box[0] - pixels, box[1] - pixels, box[2] + 2 * pixels, box[3] + 2 * pixels)
    return _clip(grown, width, height) or box


def build_mask(
    size: tuple[int, int],
    *,
    mask_image: Image.Image | None,
    boxes: Sequence[Box] = (),
    face_box: Box | None,
    face_dilate: int,
    feather: int,
    face_lock: bool = True,
) -> tuple[np.ndarray, Box | None]:
    """A soft mask in 0..1 (height x width) and the excluded face box.

    A mask image is resized to ``size`` when needed. Boxes alone make
    rectangles (the union). With both, the union of boxes is applied onto the
    mask (intersection). When ``face_lock`` is on, the dilated face box is
    cleared, then the edges are feathered with a Gaussian, then the excluded
    box is cleared again so the face is never refined.

    Raises:
        ValueError: Neither a mask nor a box was given, or the region is empty.
    """
    import cv2

    width, height = size
    box_list = list(boxes)
    if mask_image is None and not box_list:
        raise ValueError("A mask or a box is required for variants C and D.")
    region = np.ones((height, width), dtype=np.float32)
    if mask_image is not None:
        gray = mask_image.convert("L")
        if gray.size != size:
            gray = gray.resize(size, Image.Resampling.NEAREST)
        region = (np.asarray(gray, dtype=np.uint8) > 127).astype(np.float32)
    if box_list:
        keep = np.zeros_like(region)
        inside = False
        for box in box_list:
            clipped = _clip(box, width, height)
            if clipped is None:
                continue
            inside = True
            keep[clipped[1]:clipped[1] + clipped[3], clipped[0]:clipped[0] + clipped[2]] = 1.0
        if not inside:
            raise ValueError("The box is outside the picture.")
        region = region * keep
    excluded: Box | None = None
    if face_lock and face_box is not None:
        excluded = dilate_box(face_box, max(0, face_dilate), width, height)
        region[excluded[1]:excluded[1] + excluded[3], excluded[0]:excluded[0] + excluded[2]] = 0.0
    if not region.any():
        raise ValueError("The refine region is empty (after the face exclusion).")
    if feather > 0:
        sigma = max(0.5, feather / 2.0)
        region = cv2.GaussianBlur(region, (0, 0), sigma)
        if excluded is not None:
            region[excluded[1]:excluded[1] + excluded[3], excluded[0]:excluded[0] + excluded[2]] = 0.0
    return np.clip(region, 0.0, 1.0), excluded


def hard_mask(soft: np.ndarray) -> np.ndarray:
    """The 0/255 mask the inpaint pipeline receives: every pixel the soft mask touches."""
    return ((soft > 0.01) * 255).astype(np.uint8)


def blend_refined(source: np.ndarray, refined: np.ndarray, soft: np.ndarray) -> np.ndarray:
    """Keep the source outside the mask and the refined pixels inside, feathered at the edge."""
    weight = soft[..., None]
    mixed = source.astype(np.float32) * (1.0 - weight) + refined.astype(np.float32) * weight
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def paste_face(result: np.ndarray, source: np.ndarray, face_box: Box | None) -> np.ndarray:
    """Copy the original face pixels back, exactly, inside ``face_box``."""
    out = result.copy()
    if face_box is None:
        return out
    x, y, w, h = face_box
    out[y:y + h, x:x + w] = source[y:y + h, x:x + w]
    return out


# ---------------------------------------------------------------- scheduling

def default_early_steps(total: int) -> int:
    """Half of the step count, at least 1 and below the total."""
    return max(1, min(total // 2, total - 1))


def set_adapter_weight(transformer: Any, weight: float) -> None:
    """Set the weight of the Canvas LoRA adapter (same call as ``apply_lora``)."""
    setter = getattr(transformer, "set_adapters", None)
    if not callable(setter):
        raise RuntimeError("the transformer has no adapter setter")
    try:
        setter([ADAPTER], weights=[weight])
    except TypeError:
        setter([ADAPTER], adapter_weights=[weight])


class StepSchedule:
    """LoRA weight ``early`` for the first ``early_steps`` steps, then ``late``.

    Used as the ``callback_on_step_end`` hook. Call :meth:`restore` afterwards
    so the late weight never stays on the generator.
    """

    def __init__(self, early_steps: int, early: float, late: float) -> None:
        self.early_steps = early_steps
        self.early = early
        self.late = late
        self.transformer: Any = None
        self.dropped_after: int | None = None

    def __call__(self, pipe: Any, step: int, _timestep: Any, tensors: dict[str, Any]) -> dict[str, Any]:
        transformer = getattr(pipe, "transformer", None)
        if transformer is not None:
            self.transformer = transformer
        # ``step`` is 0-based and the callback runs after that step finished.
        if self.dropped_after is None and step + 1 >= self.early_steps and transformer is not None:
            set_adapter_weight(transformer, self.late)
            self.dropped_after = step + 1
        return tensors

    def restore(self) -> None:
        """Put the early weight back on the transformer the callback saw."""
        if self.transformer is not None and self.dropped_after is not None:
            set_adapter_weight(self.transformer, self.early)
            self.dropped_after = None


# --------------------------------------------------------- local model files

def find_yunet_folder(root: Path) -> Path | None:
    """A folder with the face detector the Canvas gate already uses. Nothing is downloaded."""
    try:
        from canvas.models.registry import load_catalog
        from canvas.models.snapshots import snapshot_ready

        spec = load_catalog().by_id("yunet-face")
        found = snapshot_ready(spec, root)
        if found is not None:
            return Path(found)
    except Exception:  # noqa: BLE001 — any catalog or view problem means "not found"
        pass
    local = Path(__file__).resolve().parents[1] / "bakeoff" / "models"
    if local.is_dir() and len(list(local.glob("*.onnx"))) == 1:
        return local
    return None


def find_sface(root: Path | None) -> Path | None:
    """An SFace onnx already on this machine, or None."""
    for name in ("VOXPRINT_SFACE", "SFACE_MODEL", "BAKEOFF_SFACE"):
        value = os.environ.get(name, "").strip()
        if value and Path(value).expanduser().is_file():
            return Path(value).expanduser()
    folders: list[Path] = []
    if root is not None:
        folders.append(root)
    folders.append(Path(__file__).resolve().parents[1] / "bakeoff" / "models")
    for folder in folders:
        if not folder.is_dir():
            continue
        for name in SFACE_NAMES:
            for pattern in (name, f"*/{name}", f"*/*/{name}"):
                for match in sorted(folder.glob(pattern)):
                    if match.is_file():
                        return match
    return None


def has_region_detector() -> bool:
    """True only when a body-region detector is importable here. None ships with Canvas."""
    import importlib.util

    return any(importlib.util.find_spec(name) is not None for name in ("canvas.safety.regions", "canvas.pipeline.regions"))


def _make_face_detector(root: Path) -> Callable[[RawImage], Sequence[Face]] | None:
    folder = find_yunet_folder(root)
    if folder is None:
        return None
    from canvas.safety.adapters import YuNetFaces

    detector = YuNetFaces(folder)
    return detector.detect


def _make_likeness(root: Path) -> Callable[[RawImage, RawImage, Face], float | None] | None:
    path = find_sface(root)
    if path is None:
        return None
    import cv2

    recognizer = cv2.FaceRecognizerSF.create(str(path), "")

    def feature(image: RawImage, face: Face) -> Any:
        pixels = np.frombuffer(image.to_bytes(), dtype=np.uint8).reshape(image.height, image.width, 3)
        crop = pixels[face.y:face.y + face.height, face.x:face.x + face.width]
        if crop.size == 0:
            return None
        bgr = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_RGB2BGR)
        return recognizer.feature(bgr)

    def likeness(base: RawImage, other: RawImage, face: Face) -> float | None:
        # A box crop, not a landmark-aligned face: use it to rank variants, not as an absolute score.
        first, second = feature(base, face), feature(other, face)
        if first is None or second is None:
            return None
        return float(recognizer.match(first, second, cv2.FaceRecognizerSF_FR_COSINE))

    return likeness


# --------------------------------------------------------------- live session

class _Session:
    """One AgeGate + generator. Tests inject ``hooks.generate`` and ``hooks.gate``."""

    def __init__(self, hooks: Hooks, *, lora_bind: tuple[Any, float], aspect: str, steps: int) -> None:
        self.hooks = hooks
        self.lora, self.strength = lora_bind
        self.aspect = aspect
        self.steps = steps
        self._pipeline: Any = None

    @property
    def gate(self) -> Any:
        if self.hooks.gate is not None:
            return self.hooks.gate
        self._open()
        return self._pipeline.gate

    def render(self, request: RenderRequest) -> SafeImage:
        if self.hooks.generate is not None:
            return self.hooks.generate(request)
        return self._live(request)

    def _open(self) -> None:
        if self._pipeline is not None:
            return
        from canvas.app.platform import check_gpu
        from canvas.infra.enlarge import EnlargeClient
        from canvas.infra.gpu import probe_gpu
        from canvas.infra.paths import models_dir
        from canvas.models.loader import ZImageGenerator, load_catalog_stack
        from canvas.models.registry import load_catalog
        from canvas.pipeline.pipeline import Pipeline
        from canvas.safety.gate import AgeGate

        gpu = check_gpu(probe_gpu())
        if not gpu.ok or gpu.tier is None:
            raise RuntimeError(gpu.message or "No supported GPU is available.")
        catalog = load_catalog()
        root = models_dir()
        gate = AgeGate(load_catalog_stack(root, catalog))
        gate.warm()
        generator = ZImageGenerator(catalog, gpu.tier)
        generator.models = root
        self._pipeline = Pipeline(
            gate, generator, gpu.tier, use_gpu_lock=True, catalog=catalog, enlarge=EnlargeClient(),
        )

    def _live(self, request: RenderRequest) -> SafeImage:
        from canvas.models import runtime
        from canvas.models.plan import BaseNotSelected, ModeUnavailable, OptionalModelUnavailable

        self._open()
        generator = self._pipeline.generator
        generator.seed = request.seed
        if request.use_lora:
            generator.lora_path = str(self.lora.weights)
            generator.lora_strength = self.strength
            generator.bound_lora = self.lora
        else:
            # The base-model pass: no adapter at all (the runtime unloads it for this call).
            generator.lora_path = ""
            generator.lora_strength = 0.0
            generator.bound_lora = None
        try:
            with runtime.call_options(inpaint_strength=request.denoise, step_callback=request.schedule):
                return self._pipeline.generate(
                    request.prompt, request.mode, request.inputs,
                    aspect=self.aspect, output="native", explicit=request.explicit,
                )
        except (ModeUnavailable, BaseNotSelected, OptionalModelUnavailable) as exc:
            raise RuntimeError(type(exc).__name__) from exc
        finally:
            generator.lora_path = str(self.lora.weights)
            generator.lora_strength = self.strength
            generator.bound_lora = self.lora
            if request.schedule is not None:
                try:
                    request.schedule.restore()
                except Exception:  # noqa: BLE001 — never leave the late weight on a kept pipeline
                    runtime.release_pipeline()


# --------------------------------------------------------------------- runner

def _pixels(image: SafeImage | RawImage) -> np.ndarray:
    data = image.data if isinstance(image, SafeImage) else image.to_bytes()
    return np.frombuffer(bytes(data), dtype=np.uint8).reshape(image.height, image.width, 3)


def _raw(array: np.ndarray) -> RawImage:
    height, width = array.shape[:2]
    return RawImage(width, height, np.ascontiguousarray(array, dtype=np.uint8).tobytes())


def _face_box(face: Face) -> Box:
    return (max(0, face.x), max(0, face.y), face.width, face.height)


@dataclass
class Params:
    """Everything a run needs, after the CLI is parsed."""

    prompt: str
    seed: int
    strength: float
    late_strength: float
    early_steps: int
    steps: int
    denoise: float
    feather: int
    face_dilate: int
    boxes: list[Box]
    face_box: Box | None
    mask_image: Image.Image | None
    variants: list[str]
    out: Path
    face_lock: bool = True
    zone: str | None = None
    adult: bool = True


class Runner:
    """Runs the selected variants, in order, and keeps the log."""

    def __init__(self, session: _Session, hooks: Hooks, params: Params, prompt: str, log: TextIO) -> None:
        self.session = session
        self.hooks = hooks
        self.p = params
        self.prompt = prompt  # trigger applied, sent to the gate and the encoder
        self.log = log
        self.images: dict[str, SafeImage] = {}
        self.records: dict[str, dict[str, Any]] = {}
        self.baseline: SafeImage | None = None
        self.baseline_face: Face | None = None
        self.likeness = hooks.likeness
        self.detect = hooks.detect_faces

    # -- helpers
    def _fail(self, key: str, exc: BaseException, started: float) -> None:
        record = self.records.setdefault(key, {})
        if isinstance(exc, SafetyError):
            from canvas.tools.render_batch import safety_error_fields

            record["status"] = getattr(exc, "code", None) or "refused"
            record.update(safety_error_fields(exc))
        else:
            record["status"] = f"error:{type(exc).__name__}"
        record["seconds"] = round(time.perf_counter() - started, 3)
        print(f"{key}: {record['status']}", file=self.log)  # never the prompt

    def _text(self, key: str, scheduled: bool) -> SafeImage:
        schedule = StepSchedule(self.p.early_steps, self.p.strength, self.p.late_strength) if scheduled else None
        request = RenderRequest(self.prompt, self.p.seed, Mode.TEXT, schedule=schedule, explicit=self.p.adult)
        image = self.session.render(request)
        if schedule is not None:
            self.records.setdefault(key, {})["dropped_after_step"] = schedule.dropped_after
        return image

    def _first_face(self, image: SafeImage) -> Face | None:
        if self.detect is None:
            return None
        try:
            faces = list(self.detect(_raw(_pixels(image))))
        except Exception as exc:  # noqa: BLE001 — detection is optional
            print(f"face detection unavailable: {type(exc).__name__}", file=self.log)
            self.detect = None
            return None
        return max(faces, key=lambda f: f.width * f.height) if faces else None

    def _score(self, key: str, image: SafeImage) -> None:
        value: float | None = None
        if self.likeness is not None and self.baseline is not None and self.baseline_face is not None:
            face = self._first_face(image) or self.baseline_face
            try:
                value = self.likeness(_raw(_pixels(self.baseline)), _raw(_pixels(image)), face)
            except Exception as exc:  # noqa: BLE001 — likeness is optional
                print(f"likeness unavailable: {type(exc).__name__}", file=self.log)
                self.likeness = None
        self.records.setdefault(key, {})["likeness"] = value

    def _save(self, key: str, image: SafeImage) -> None:
        path = self.p.out / f"variant_{key}.png"
        save_image(image, path, OutputFormat.PNG, metadata={"seed": self.p.seed})
        record = self.records.setdefault(key, {})
        record["status"] = "ok"
        record["file"] = path.name
        self.images[key] = image

    # -- variants
    def _baseline_image(self) -> SafeImage | None:
        if self.baseline is not None:
            return self.baseline
        started = time.perf_counter()
        key = "A" if "A" in self.p.variants else "baseline"
        try:
            image = self._text(key, scheduled=False)
        except Exception as exc:  # noqa: BLE001 — logged, run goes on
            self._fail(key, exc, started)
            return None
        self.baseline = image
        self.baseline_face = self._first_face(image)
        if key == "A":
            self.records.setdefault("A", {})["seconds"] = round(time.perf_counter() - started, 3)
            self._save("A", image)
            self.records["A"]["likeness"] = 1.0 if self.baseline_face is not None and self.likeness is not None else None
        else:
            self.records["baseline"] = {"status": "ok", "written": False,
                                        "seconds": round(time.perf_counter() - started, 3)}
        return image

    def _run_b(self, write: bool) -> SafeImage | None:
        """Scheduled render. Written as variant B only when B was requested (D needs it as a source)."""
        started = time.perf_counter()
        key = "B" if write else "scheduled_source"
        try:
            image = self._text(key, scheduled=True)
            self.records[key]["seconds"] = round(time.perf_counter() - started, 3)
            if write:
                self._save("B", image)
                self._score("B", image)
            else:
                self.records[key]["status"] = "ok"
                self.records[key]["written"] = False
            return image
        except Exception as exc:  # noqa: BLE001
            self._fail(key, exc, started)
            return None

    def _refine(self, key: str, source: SafeImage) -> None:
        """Base-model inpaint over the face-locked, feathered mask, then the face pasted back."""
        started = time.perf_counter()
        record = self.records.setdefault(key, {})
        try:
            width, height = source.width, source.height
            zone_id = qualify_zone(self.p.zone)
            lock_face = self.p.face_lock and zone_id not in {"regions.face", "face"}
            if not lock_face:
                face_box = None
                record["face_lock"] = {
                    "used": False,
                    "reason": "--zone face" if zone_id in {"regions.face", "face"} else "--no-face-lock",
                }
            else:
                face = self._first_face(source)
                face_box = _face_box(face) if face is not None else self.p.face_box
                record["face_lock"] = (
                    {"used": True, "box": list(face_box), "source": "detector" if face is not None else "--face-box"}
                    if face_box is not None
                    else {"used": False, "reason": "no face found and no --face-box"}
                )
                if face_box is None:
                    print(f"{key}: face-lock skipped (no face found, no --face-box)", file=self.log)
            soft, excluded = build_mask(
                (width, height), mask_image=self.p.mask_image, boxes=self.p.boxes, face_box=face_box,
                face_dilate=self.p.face_dilate, feather=self.p.feather, face_lock=lock_face,
            )
            source_pixels = _pixels(source).copy()
            mask_rgb = np.repeat(hard_mask(soft)[..., None], 3, axis=2)
            inputs = (
                PendingInput(ImageRole.INIT, _raw(source_pixels)),
                PendingInput(ImageRole.MASK, _raw(mask_rgb)),
            )
            request = RenderRequest(
                self.prompt, self.p.seed, Mode.INPAINT, inputs=inputs, denoise=self.p.denoise, use_lora=False,
                explicit=self.p.adult,
            )
            inpainted = self.session.render(request)
            blended = blend_refined(source_pixels, _pixels(inpainted), soft)
            final = paste_face(blended, source_pixels, face_box)
            # The composite is a new picture: it gets its own gate review before it is kept.
            reviewed = self.session.gate.review_image(_raw(final), self.prompt)
            record["seconds"] = round(time.perf_counter() - started, 3)
            record["excluded_box"] = list(excluded) if excluded is not None else None
            self._save(key, reviewed)
            self._score(key, reviewed)
        except Exception as exc:  # noqa: BLE001
            self._fail(key, exc, started)

    def run(self) -> None:
        variants = self.p.variants
        if "A" in variants or "C" in variants:
            self._baseline_image()
        b_image = self._run_b("B" in variants) if ("B" in variants or "D" in variants) else None
        if "C" in variants:
            if self.baseline is None:
                self.records["C"] = {"status": "skipped:no baseline image"}
                print("C: skipped, the baseline render did not complete", file=self.log)
            else:
                self._refine("C", self.baseline)
        if "D" in variants:
            if b_image is None:
                self.records["D"] = {"status": "skipped:no scheduled image"}
                print("D: skipped, variant B did not complete", file=self.log)
            else:
                self._refine("D", b_image)


# ----------------------------------------------------------------- contact sheet

def contact_sheet(images: dict[str, SafeImage], pad: int = 16, label_height: int = 28) -> RawImage | None:
    """Side by side, same height, labelled A/B/C/D. None when nothing completed."""
    if not images:
        return None
    height = min(image.height for image in images.values())
    tiles: list[tuple[str, Image.Image]] = []
    for key in sorted(images):
        picture = Image.frombytes("RGB", (images[key].width, images[key].height), bytes(images[key].data))
        scale = height / picture.height
        tiles.append((key, picture.resize((max(1, round(picture.width * scale)), height), Image.Resampling.LANCZOS)))
    total = sum(tile.width for _, tile in tiles) + pad * (len(tiles) + 1)
    sheet = Image.new("RGB", (total, height + label_height + 2 * pad), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    x = pad
    for key, tile in tiles:
        draw.text((x + 4, pad // 2 + 4), key, fill=(235, 235, 235))
        sheet.paste(tile, (x, label_height + pad))
        x += tile.width + pad
    return RawImage(sheet.width, sheet.height, sheet.tobytes())


# ------------------------------------------------------------------------- CLI

PICTURES_DIR = "~/.local/share/voxprint-ai-suite/state/canvas/pictures"


def default_out(now: float | None = None) -> Path:
    """A timestamped run folder in the Canvas pictures folder (the one PG's sync copies)."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    return Path(PICTURES_DIR).expanduser() / stamp


def parse_device(value: str) -> str:
    """The CUDA index for ``cuda:N`` or ``N`` ("" for a bare ``cuda``).

    Raises:
        ValueError: The value is not a CUDA device.
    """
    text = value.strip().lower()
    if text == "cuda":
        return ""
    for prefix in ("cuda:", ""):
        if text.startswith(prefix) and text[len(prefix):].isdigit() and text[len(prefix):].isascii():
            return str(int(text[len(prefix):]))
    raise ValueError(f"--device must be cuda, cuda:N, or N (for example cuda:0), not {value!r}.")


def apply_device(value: str, log: TextIO) -> None:
    """Make this process see only that CUDA device. Must run before any pipeline work."""
    index = parse_device(value)
    if not index:
        return
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch.cuda, "is_initialized", lambda: False)():
        print("warning: CUDA was already initialised; --device may not take effect.", file=log)
    os.environ["CUDA_VISIBLE_DEVICES"] = index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare region-refine variants of a person LoRA (content gate stays on).",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--lora", required=True, type=Path, help="person LoRA file, package, or folder")
    parser.add_argument("--strength", type=float, default=None,
                        help="LoRA strength (default: the file's recommended strength, else 1.0)")
    parser.add_argument("--seed", required=True, type=int, help="fixed across all variants")
    parser.add_argument("--adult", "--adult-content", dest="adult_content", choices=("on", "off"), required=True,
                        help="adult content (18+); same meaning as in Canvas")
    parser.add_argument("--refine-region", action="store_true",
                        help="base-model inpaint over --box/--mask (default variants A,C,D)")
    parser.add_argument("--mask", type=Path, default=None, help="PNG: white marks the region to refine")
    parser.add_argument("--box", action="append", default=None,
                        help="x,y,w,h in pixels on the generated picture; repeat for several zones")
    parser.add_argument("--zone", default=None,
                        help="region or angle name, logged and matched to coverage weak bins")
    parser.add_argument("--coverage", type=Path, default=None,
                        help="coverage JSON, used when no <lora stem>.coverage.json sidecar is present")
    parser.add_argument("--face-box", default=None, help="x,y,w,h used when no face detector is available")
    parser.add_argument("--face-lock", dest="face_lock", action=argparse.BooleanOptionalAction, default=True,
                        help="exclude and paste back the face (on by default; skipped when --zone face)")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"output folder (default: a YYYYMMDD-HHMMSS folder in {PICTURES_DIR})")
    parser.add_argument("--device", "--gpu", dest="device", default=None,
                        help="CUDA device for this process: cuda, cuda:N, or N (for example cuda:0)")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--late-strength", type=float, default=0.3)
    parser.add_argument("--early-steps", type=int, default=None)
    parser.add_argument("--denoise", type=float, default=0.45)
    parser.add_argument("--preset", choices=PRESETS, default="balanced")
    parser.add_argument("--aspect", default="1:1")
    parser.add_argument("--variants", default=None,
                        help="comma list among A,B,C,D (default A,C,D when refining)")
    parser.add_argument("--feather", type=int, default=8)
    parser.add_argument("--face-dilate", type=int, default=16)
    return parser


def _validate(args: argparse.Namespace, log: TextIO) -> str | None:
    if args.device is not None:
        try:
            parse_device(args.device)
        except ValueError as exc:
            return str(exc)
    if not args.prompt.strip():
        return "The prompt is empty."
    if not 0.0 < args.denoise <= 1.0:
        return "--denoise must be above 0 and at most 1."
    if args.late_strength < 0:
        return "--late-strength must not be negative."
    if args.feather < 0 or args.face_dilate < 0:
        return "--feather and --face-dilate must not be negative."
    if args.mask is not None and not args.mask.is_file():
        return f"The mask file is missing: {args.mask}"
    if args.coverage is not None and not args.coverage.is_file():
        return f"The coverage file is missing: {args.coverage}"
    if not args.lora.exists():
        return f"The LoRA file is missing: {args.lora}"
    return None


def main(argv: Sequence[str] | None = None, hooks: Hooks | None = None) -> int:
    """CLI entry. 2 means a bad input or missing weights; 1 means no picture or a failed variant."""
    args = _parser().parse_args(argv)
    hooks = hooks or Hooks()
    log = hooks.log
    problem = _validate(args, log)
    if problem is not None:
        print(problem, file=log)
        return 2
    if args.out is None:
        args.out = default_out()
    if args.device is not None:
        apply_device(args.device, log)
    print(f"Output folder: {args.out}", file=log)
    try:
        boxes = [parse_box(item) for item in (args.box or [])]
        face_box_arg = parse_box(args.face_box) if args.face_box else None
        chosen = parse_variants(args.variants) if args.variants else None
    except ValueError as exc:
        print(str(exc), file=log)
        return 2
    if args.models_dir is not None:
        os.environ["VOXPRINT_MODELS_DIR"] = str(args.models_dir.expanduser().resolve())

    from canvas.infra.paths import resolved_models_dir
    from canvas.tools.render_batch import (
        apply_preset_steps,
        bind_lora,
        format_deps_error,
        missing_gate_imports,
        preset_values,
    )

    root, source = resolved_models_dir()
    print(f"Models folder: {root} (from {source})", file=log)
    try:
        steps, _guidance, _sampler = preset_values(args.preset)
        from canvas.models.user_lora import LoraRejected

        try:
            # Always pass a float: rc.40 bind_lora does not accept None, and this
            # tool reads voxprint.lora.recommended_strength itself (rc.40 does not).
            strength = args.strength if args.strength is not None else recommended_strength(args.lora)
            lora_bind = bind_lora(args.lora, strength)
        except LoraRejected as exc:
            from canvas.app.i18n import text

            print(text(exc.code, **exc.details) if exc.details else str(exc), file=log)
            return 2
    except (OSError, ValueError) as exc:
        print(f"The LoRA could not be opened: {exc}", file=log)
        return 2
    lora = lora_bind[0]
    if args.strength is None:
        strength = recommended_strength(Path(lora.weights))
        lora_bind = (lora, strength)

    adult = args.adult_content == "on"
    coverage = load_coverage(Path(lora.weights), args.coverage)
    if coverage.warning:
        print(f"warning: {coverage.warning}", file=log)
    if coverage.origin:
        print(f"Coverage source: {coverage.origin}", file=log)
    default, reason = default_variants(
        coverage, adult=adult, refine=args.refine_region, zone=args.zone,
    )
    variants = chosen or default
    print(f"Variants: {','.join(variants)} ({'requested' if chosen else 'default: ' + reason})", file=log)
    if args.zone:
        print(f"Zone: {args.zone}", file=log)
    regions = weak_regions(coverage, adult)
    angles = weak_angles(coverage, adult)
    if regions:
        print("Weak regions: " + ", ".join(_qualify("regions", name) for name in regions), file=log)
    if angles:
        print(
            "Weak angles (no box; late-strength schedule on the whole render): "
            + ", ".join(_qualify("angles", name) for name in angles),
            file=log,
        )

    if any(item in variants for item in MASK_VARIANTS) and args.mask is None and not boxes:
        print("Variants C and D need --mask and/or --box (the region to refine).", file=log)
        return 2
    early = args.early_steps if args.early_steps is not None else default_early_steps(steps)
    if ("B" in variants or "D" in variants) and not 1 <= early < steps:
        print(f"--early-steps must be at least 1 and less than the {steps} steps of this preset.", file=log)
        return 2

    if hooks.weights_ready is not None:
        if not hooks.weights_ready():
            print("The Z-Image-Turbo weights are missing. Install them with VOXPrint MDL, then run this command again.",
                  file=log)
            return 2
    else:
        missing = missing_gate_imports()
        if missing:
            print(format_deps_error(missing), file=log)
            return 2
        from canvas.app.platform import check_gpu
        from canvas.infra.gpu import probe_gpu
        from canvas.tools.render_batch import format_preflight_error, missing_model_ids

        gpu = check_gpu(probe_gpu())
        if not gpu.ok or gpu.tier is None:
            print(gpu.message or "No supported GPU is available.", file=log)
            return 2
        absent = missing_model_ids(root, gpu.tier)
        if absent:
            print(format_preflight_error(root, source, absent), file=log)
            return 2

    try:
        args.out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"The output folder could not be created: {args.out} ({exc})", file=log)
        return 2
    apply_preset_steps(steps)

    if hooks.weights_ready is None:
        if hooks.detect_faces is None:
            try:
                hooks.detect_faces = _make_face_detector(root)
            except Exception:  # noqa: BLE001 — optional
                hooks.detect_faces = None
        if hooks.likeness is None:
            try:
                hooks.likeness = _make_likeness(root)
            except Exception:  # noqa: BLE001 — optional
                hooks.likeness = None
    face_lock_available = hooks.detect_faces is not None
    if not face_lock_available:
        print("No face detector found: face-lock uses --face-box" + ("." if face_box_arg else " (not given: skipped)."),
              file=log)
    if hooks.likeness is None:
        print("No SFace model found: likeness is null.", file=log)

    mask_image = Image.open(args.mask) if args.mask is not None else None
    if mask_image is not None:
        mask_image.load()
    params = Params(
        prompt=args.prompt, seed=int(args.seed), strength=lora_bind[1], late_strength=float(args.late_strength),
        early_steps=early, steps=steps, denoise=float(args.denoise), feather=int(args.feather),
        face_dilate=int(args.face_dilate), boxes=boxes, face_box=face_box_arg, mask_image=mask_image,
        variants=variants, out=args.out, face_lock=bool(args.face_lock), zone=args.zone, adult=adult,
    )
    final_prompt = lead_trigger(args.prompt, lora)  # before the gate and the encoder
    session = _Session(hooks, lora_bind=lora_bind, aspect=args.aspect, steps=steps)
    runner = Runner(session, hooks, params, final_prompt, log)
    started = time.perf_counter()
    runner.run()

    sheet_status = "none"
    sheet = contact_sheet(runner.images)
    if sheet is not None:
        try:
            reviewed = session.gate.review_image(sheet, final_prompt)
            save_image(reviewed, args.out / "compare.png", OutputFormat.PNG, metadata={"seed": params.seed})
            sheet_status = "ok"
        except Exception as exc:  # noqa: BLE001
            sheet_status = getattr(exc, "code", None) or f"error:{type(exc).__name__}"
            print(f"compare.png: {sheet_status}", file=log)

    document = {
        "schema": SCHEMA,
        "prompt": args.prompt,
        "seed": params.seed,
        "lora": Path(lora.weights).name,
        "strength": params.strength,
        "late_strength": params.late_strength,
        "early_steps": params.early_steps,
        "steps": params.steps,
        "preset": args.preset,
        "denoise": params.denoise,
        "feather": params.feather,
        "face_dilate": params.face_dilate,
        "adult_content": adult,
        "refine_region": bool(args.refine_region),
        "zone": args.zone,
        "device": args.device,
        "out": str(args.out),
        "boxes": [list(item) for item in boxes],
        "face_box": list(face_box_arg) if face_box_arg else None,
        "face_lock": bool(args.face_lock),
        "mask": args.mask.name if args.mask else None,
        "coverage": coverage.document,
        "coverage_origin": coverage.origin or None,
        "coverage_warning": coverage.warning or None,
        "weak_regions": [_qualify("regions", name) for name in regions],
        "weak_angles": [_qualify("angles", name) for name in angles],
        "variants_requested": chosen,
        "variants_default": None if chosen else default,
        "variants_reason": None if chosen else reason,
        "variants_run": [key for key in ALL_VARIANTS if runner.records.get(key, {}).get("status") == "ok"],
        "variants": {key: runner.records[key] for key in sorted(runner.records)},
        "baseline_face": None if runner.baseline_face is None else list(_face_box(runner.baseline_face)),
        "face_detector": face_lock_available,
        "sface": hooks.likeness is not None,
        "compare": sheet_status,
        "region_auto": None if has_region_detector() else
        "--region-auto omitted: no body-region detector is present on this machine",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (args.out / "log.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    ok = [key for key in ALL_VARIANTS if key in variants and runner.records.get(key, {}).get("status") == "ok"]
    if not ok:
        print("No variant completed.", file=log)
        return 1
    return 0 if len(ok) == len(variants) else 1


if __name__ == "__main__":
    sys.exit(main())
