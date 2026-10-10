#!/usr/bin/env python3
"""Voxprint AI Canvas model bake-off: Z-Image-Turbo vs Qwen-Image-2512.

Runs on one A100 80GB inside a hard time budget. Every arm is resumable and
time-capped; results from several runs (for example tier arms re-run after
Model Forge part D) merge into the same results/summary.json.

Usage: see bakeoff/README.md.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import random
import re
import shutil
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SCHEMA = "voxprint-canvas-bakeoff/1"

Z_REPO, Z_REV = "Tongyi-MAI/Z-Image-Turbo", "f332072aa78be7aecdf3ee76d5c247082da564a6"
Q_REPO, Q_REV = "Qwen/Qwen-Image-2512", "25468b98e3276ca6700de15c6628e51b7de54a26"
SIGLIP_REPO, SIGLIP_REV = "google/siglip2-so400m-patch14-384", "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
NSFW_REPO, NSFW_REV = "Falconsai/nsfw_image_detection", "96cb0d0342c7afb80cab76ecc58b265fa44da256"
AGE_REPO, AGE_REV = "dima806/fairface_age_image_detection", "4e02ab8057ea7fd74b1670940995c5dfda3e6ec0"

# Official Qwen-Image-2512 negative prompt (model card), as unicode escapes.
Q_NEG_OFFICIAL = (
    "\u4f4e\u5206\u8fa8\u7387\uff0c\u4f4e\u753b\u8d28\uff0c\u80a2\u4f53\u7578\u5f62\uff0c"
    "\u624b\u6307\u7578\u5f62\uff0c\u753b\u9762\u8fc7\u9971\u548c\uff0c\u8721\u50cf\u611f\uff0c"
    "\u4eba\u8138\u65e0\u7ec6\u8282\uff0c\u8fc7\u5ea6\u5149\u6ed1\uff0c\u753b\u9762\u5177\u6709AI\u611f\u3002"
    "\u6784\u56fe\u6df7\u4e71\u3002\u6587\u5b57\u6a21\u7cca\uff0c\u626d\u66f2\u3002"
)
Q_NEG_TUNED = Q_NEG_OFFICIAL + (
    " smooth doll-like body, missing nipples, covered body, censored, blurred, mosaic, deformed anatomy"
)

ALL_ARMS = ["z_bf16", "z_fp8", "q_bf16", "q_tuned", "q_fp8", "q_int4"]


@dataclass
class Arm:
    name: str
    family: str  # "z" or "q"
    quant: str  # "bf16", "fp8-forge", "int4"
    subset: bool
    width: int
    height: int
    steps: int
    guidance: float
    negative: str | None = None
    max_min: float = 30.0
    notes: list[str] = field(default_factory=list)


def default_arms() -> dict[str, Arm]:
    return {
        "z_bf16": Arm("z_bf16", "z", "bf16", False, 1024, 1024, 9, 0.0, max_min=10),
        "z_fp8": Arm("z_fp8", "z", "fp8-forge", False, 1024, 1024, 9, 0.0, max_min=10,
                     notes=["Forge voxprint-fp8-v1 weights dequantized to BF16 (A100 has no FP8 tensor cores): quality of the 16 GB tier, not its speed"]),
        "q_bf16": Arm("q_bf16", "q", "bf16", False, 1328, 1328, 50, 4.0, Q_NEG_OFFICIAL, max_min=80,
                      notes=["native BF16 on A100 80GB: quality upper bound for the 24 GB tier"]),
        "q_tuned": Arm("q_tuned", "q", "bf16", True, 1328, 1328, 50, 4.5, Q_NEG_TUNED, max_min=28,
                       notes=["true CFG 4.5 + official negative + English anatomy terms"]),
        "q_fp8": Arm("q_fp8", "q", "fp8-forge", False, 1328, 1328, 50, 4.0, Q_NEG_OFFICIAL, max_min=80,
                     notes=["Forge FP8 dequantized to BF16 on A100: 24 GB tier quality"]),
        "q_int4": Arm("q_int4", "q", "int4", False, 1328, 1328, 50, 4.0, Q_NEG_OFFICIAL, max_min=60,
                      notes=["SVDQuant INT4 via Nunchaku: 16 GB tier"]),
    }


# ---------------------------------------------------------------- utils

def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def load_prompts() -> list[dict[str, Any]]:
    """Round-robin interleave the categories so a time-capped arm still covers all of them."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for name in ("general.jsonl", "text.jsonl", "adult.jsonl"):
        for row in read_jsonl(HERE / "prompts" / name):
            buckets.setdefault(row["category"], []).append(row)
    order = ["general", "text_en", "text_ru", "adult"]
    out: list[dict[str, Any]] = []
    while any(buckets.get(c) for c in order):
        for c in order:
            if buckets.get(c):
                out.append(buckets[c].pop(0))
    return out


def key(arm: str, pid: str, seed: int) -> str:
    return f"{arm}|{pid}|{seed}"


class Clock:
    def __init__(self, deadline_min: float, reserve_min: float) -> None:
        self.start = time.monotonic()
        self.deadline = self.start + deadline_min * 60
        self.reserve = reserve_min * 60

    def gen_left(self) -> float:
        return self.deadline - self.reserve - time.monotonic()

    def left(self) -> float:
        return self.deadline - time.monotonic()

    def elapsed_min(self) -> float:
        return (time.monotonic() - self.start) / 60


def find_one(patterns: list[str]) -> Path | None:
    for pattern in patterns:
        hits = sorted(glob.glob(os.path.expanduser(pattern), recursive=True))
        if hits:
            return Path(hits[0])
    return None


def safetensor_list(path: Path | None) -> list[Path]:
    if path is None or not path.exists():
        return []
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.safetensors"))


# ---------------------------------------------------------------- weights

def dequant_files(paths: list[Path]) -> dict[str, Any]:
    """Read voxprint-fp8-v1 files (float8_e4m3fn + <name>.scale_weight BF16) as BF16 tensors."""
    import torch
    from safetensors.torch import load_file

    raw: dict[str, Any] = {}
    for path in paths:
        raw.update(load_file(str(path)))
    scales = {k[: -len(".scale_weight")]: v for k, v in raw.items() if k.endswith(".scale_weight")}
    out: dict[str, Any] = {}
    for name, tensor in raw.items():
        if name.endswith(".scale_weight"):
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            if name not in scales:
                raise RuntimeError(f"{name} is fp8 without a scale")
            out[name] = tensor.to(torch.bfloat16) * scales[name].to(torch.bfloat16)
        elif tensor.is_floating_point():
            out[name] = tensor.to(torch.bfloat16)
        else:
            out[name] = tensor
    return out


def _alias_key(key: str, current: dict[str, Any]) -> str | None:
    """Match a text-encoder name with or without a leading ``model.`` prefix."""
    if key in current:
        return key
    if key.startswith("model.") and key[len("model.") :] in current:
        return key[len("model.") :]
    prefixed = "model." + key
    if prefixed in current:
        return prefixed
    return None


def apply_state(module: Any, state: dict[str, Any]) -> int:
    current = module.state_dict()
    device = next(module.parameters()).device
    mapped = {}
    for raw_name, value in state.items():
        name = _alias_key(raw_name, current)
        target = current.get(name) if name is not None else None
        if target is None or tuple(value.shape) != tuple(target.shape):
            continue
        mapped[name] = value.to(device=device, dtype=target.dtype)
    weights = [k for k in current if k.endswith(".weight")]
    if len(mapped) < max(1, len(weights) // 2):
        raise RuntimeError(f"fp8 file matches only {len(mapped)} of {len(current)} tensors")
    module.load_state_dict(mapped, strict=False)
    return len(mapped)


# ---------------------------------------------------------------- generation

class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = Path(args.out).resolve()
        self.records_path = self.out / "records.jsonl"
        self.arms_path = self.out / "arms.json"
        self.pipe: Any = None
        self.family: str | None = None
        self.loaded_quant: str | None = None

    def unload(self) -> None:
        import torch

        self.pipe = None
        self.family = None
        self.loaded_quant = None
        gc.collect()
        torch.cuda.empty_cache()

    def load(self, arm: Arm) -> str | None:
        """Load or switch the pipeline for an arm. Returns a skip reason or None."""
        import torch

        a = self.args
        if arm.quant == "int4":
            int4 = Path(os.path.expanduser(a.q_int4_dir))
            if not int4.exists() or not safetensor_list(int4):
                return f"deferred: no INT4 build at {int4}"
            try:
                from nunchaku import NunchakuQwenImageTransformer2DModel  # type: ignore
            except Exception as exc:  # noqa: BLE001
                return f"deferred: nunchaku not importable ({type(exc).__name__})"
            from diffusers import QwenImagePipeline

            self.unload()
            files = safetensor_list(int4)
            transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(str(files[0]))
            self.pipe = QwenImagePipeline.from_pretrained(
                a.q_repo, revision=a.q_rev, transformer=transformer, torch_dtype=torch.bfloat16
            ).to("cuda")
            self.family, self.loaded_quant = "q", "int4"
            return None

        if self.family != arm.family or self.loaded_quant not in ("bf16", None) and arm.quant == "bf16":
            self.unload()
            if arm.family == "z":
                from diffusers import ZImagePipeline

                self.pipe = ZImagePipeline.from_pretrained(a.z_repo, revision=a.z_rev, torch_dtype=torch.bfloat16)
            else:
                from diffusers import QwenImagePipeline

                self.pipe = QwenImagePipeline.from_pretrained(a.q_repo, revision=a.q_rev, torch_dtype=torch.bfloat16)
            self.pipe.to("cuda")
            self.pipe.set_progress_bar_config(disable=True)
            self.family, self.loaded_quant = arm.family, "bf16"

        if arm.quant == "fp8-forge" and self.loaded_quant != "fp8-forge":
            if arm.family == "z":
                tr = safetensor_list(Path(os.path.expanduser(a.z_fp8)) if a.z_fp8 else
                                     find_one(["~/forge-out/**/zimage-turbo-transformer-fp8.safetensors"]))
                enc = safetensor_list(Path(os.path.expanduser(a.z_encoder_fp8)) if a.z_encoder_fp8 else
                                      find_one(["~/forge-out/**/qwen3-4b-text-encoder-fp8.safetensors"]))
            else:
                qdir = Path(os.path.expanduser(a.q_fp8_dir))
                tr = [p for p in safetensor_list(qdir) if "encoder" not in p.name]
                enc = safetensor_list(Path(os.path.expanduser(a.q_encoder_fp8))) if a.q_encoder_fp8 else \
                    [p for p in safetensor_list(qdir) if "encoder" in p.name]
            if not tr:
                return "skipped: Forge FP8 transformer files not found (pass the path arg)"
            n = apply_state(self.pipe.transformer, dequant_files(tr))
            arm.notes.append(f"transformer: {n} tensors from {len(tr)} FP8 file(s)")
            if enc:
                n = apply_state(self.pipe.text_encoder, dequant_files(enc))
                arm.notes.append(f"text encoder: {n} tensors from FP8")
            else:
                arm.notes.append("text encoder: stock BF16 (no FP8 encoder file found)")
            self.loaded_quant = "fp8-forge"
        return None

    def generate(self, arm: Arm, prompt: dict[str, Any], seed: int) -> Any:
        import torch

        gen = torch.Generator(device="cuda").manual_seed(seed)
        kwargs: dict[str, Any] = dict(
            prompt=prompt["prompt"], width=arm.width, height=arm.height,
            num_inference_steps=arm.steps, generator=gen,
        )
        if arm.family == "z":
            kwargs["guidance_scale"] = arm.guidance
        else:
            kwargs["true_cfg_scale"] = arm.guidance
            kwargs["negative_prompt"] = arm.negative or " "
        return self.pipe(**kwargs).images[0]

    def run(self, arms: list[Arm], prompts: list[dict[str, Any]], clock: Clock) -> None:
        import torch

        done = {key(r["arm"], r["prompt_id"], r["seed"]) for r in read_jsonl(self.records_path)}
        arm_meta = load_json(self.arms_path, {})
        subset_ids = {p["id"] for p in prompts[: self.args.subset_n]}
        seeds = [int(s) for s in self.args.seeds.split(",")]
        for arm in arms:
            meta = arm_meta.get(arm.name, {})
            if clock.gen_left() < 60:
                meta.update(status="not-run", reason="deadline reached before start")
                arm_meta[arm.name] = meta
                save_json(self.arms_path, arm_meta)
                continue
            t0 = time.monotonic()
            log(f"arm {arm.name}: loading")
            try:
                reason = self.load(arm)
            except Exception as exc:  # noqa: BLE001
                reason = f"skipped: load failed: {type(exc).__name__}: {exc}"[:300]
                self.unload()
            if reason:
                log(f"arm {arm.name}: {reason}")
                meta.update(status="skipped", reason=reason)
                arm_meta[arm.name] = meta
                save_json(self.arms_path, arm_meta)
                continue
            load_s = time.monotonic() - t0
            torch.cuda.reset_peak_memory_stats()
            arm_deadline = time.monotonic() + arm.max_min * 60 - load_s
            todo = [p for p in prompts if not arm.subset or p["id"] in subset_ids]
            made, last_s = 0, 0.0
            status = "done"
            for prompt in todo:
                for seed in seeds:
                    k = key(arm.name, prompt["id"], seed)
                    if k in done and not self.args.force:
                        continue
                    budget = min(clock.gen_left(), arm_deadline - time.monotonic())
                    if budget < max(last_s * 1.1, 5):
                        status = "partial"
                        break
                    torch.cuda.synchronize()
                    s0 = time.perf_counter()
                    image = self.generate(arm, prompt, seed)
                    torch.cuda.synchronize()
                    last_s = time.perf_counter() - s0
                    rel = Path("images") / arm.name / f"{prompt['id']}_s{seed}.webp"
                    (self.out / rel).parent.mkdir(parents=True, exist_ok=True)
                    image.save(self.out / rel, "WEBP", quality=95)
                    append_jsonl(self.records_path, {
                        "arm": arm.name, "prompt_id": prompt["id"], "category": prompt["category"],
                        "seed": seed, "seconds": round(last_s, 3), "width": arm.width, "height": arm.height,
                        "steps": arm.steps, "guidance": arm.guidance, "image": str(rel),
                        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
                    done.add(k)
                    made += 1
                    if made % 10 == 0:
                        log(f"arm {arm.name}: {made} images, last {last_s:.1f}s, {clock.left()/60:.0f} min left")
                if status == "partial":
                    break
            peak = torch.cuda.max_memory_allocated() / 2**30
            meta.update(status=status, quant=arm.quant, load_s=round(load_s, 1),
                        peak_vram_gb=round(max(peak, meta.get("peak_vram_gb", 0) if not self.args.force else 0), 2),
                        settings={"width": arm.width, "height": arm.height, "steps": arm.steps,
                                  "guidance": arm.guidance, "negative": arm.negative is not None,
                                  "subset": arm.subset}, notes=arm.notes,
                        gpu=torch.cuda.get_device_name(0))
            meta.pop("reason", None)
            arm_meta[arm.name] = meta
            save_json(self.arms_path, arm_meta)
            log(f"arm {arm.name}: {status}, {made} new images, peak {peak:.1f} GiB")
        self.unload()


# ---------------------------------------------------------------- scoring

def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).upper().replace("\u0401", "\u0415")
    s = re.sub(r"[^\w%]+", " ", s)
    return " ".join(s.split())


def semiglobal_cer(target: str, text: str) -> float:
    """Edit distance of target against the best-matching substring of text, / len(target)."""
    t, s = target.replace(" ", ""), text.replace(" ", "")
    if not t:
        return 0.0
    prev = [0] * (len(s) + 1)
    for i in range(1, len(t) + 1):
        cur = [i] + [0] * len(s)
        for j in range(1, len(s) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (t[i - 1] != s[j - 1]))
        prev = cur
    return min(prev) / len(t)


class Scorer:
    def __init__(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, pipeline

        self.torch = torch
        self.device = device
        self.notes: list[str] = []
        self.clip = AutoModel.from_pretrained(SIGLIP_REPO, revision=SIGLIP_REV, torch_dtype=torch.float16).to(device).eval()
        self.clip_proc = AutoProcessor.from_pretrained(SIGLIP_REPO, revision=SIGLIP_REV)
        self.nsfw = pipeline("image-classification", model=NSFW_REPO, revision=NSFW_REV, device=0)
        self.age = pipeline("image-classification", model=AGE_REPO, revision=AGE_REV, device=0, top_k=None)
        try:
            import cv2  # type: ignore

            self.cv2 = cv2
            self.faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        except Exception:  # noqa: BLE001
            self.cv2 = None
            self.faces = None
            self.notes.append("opencv missing: age check runs on the whole image")
        self.ocr: dict[str, Any] = {}
        try:
            import easyocr  # type: ignore

            self.ocr["en"] = easyocr.Reader(["en"], gpu=True, verbose=False)
            self.ocr["ru"] = easyocr.Reader(["ru", "en"], gpu=True, verbose=False)
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"OCR unavailable ({type(exc).__name__}): text metrics skipped")

    def clip_t(self, image: Any, text: str) -> float:
        torch = self.torch
        inputs = self.clip_proc(text=[text], images=[image], padding="max_length", max_length=64,
                                truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img = self.clip.get_image_features(pixel_values=inputs["pixel_values"].half())
            txt = self.clip.get_text_features(input_ids=inputs["input_ids"])
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        return float((img * txt).sum()) * 100

    def nsfw_score(self, image: Any) -> float:
        for row in self.nsfw(image):
            if row["label"].lower() == "nsfw":
                return float(row["score"])
        return 0.0

    def minor_prob(self, image: Any) -> tuple[float, int]:
        crops = [image]
        n_faces = 0
        if self.faces is not None:
            import numpy as np

            gray = self.cv2.cvtColor(np.array(image.convert("RGB")), self.cv2.COLOR_RGB2GRAY)
            found = self.faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
            n_faces = len(found)
            if n_faces:
                crops = []
                for (x, y, w, h) in found:
                    m = int(0.25 * max(w, h))
                    crops.append(image.crop((max(0, x - m), max(0, y - m), x + w + m, y + h + m)))
        worst = 0.0
        for crop in crops:
            p = 0.0
            for row in self.age(crop):
                nums = [int(n) for n in re.findall(r"\d+", row["label"])]
                if nums and max(nums) < 20:
                    p += float(row["score"])
            worst = max(worst, p)
        return worst, n_faces

    def score(self, record: dict[str, Any], prompt: dict[str, Any], out: Path) -> dict[str, Any]:
        from PIL import Image

        image = Image.open(out / record["image"]).convert("RGB")
        row: dict[str, Any] = {"arm": record["arm"], "prompt_id": record["prompt_id"], "seed": record["seed"]}
        row["clip_t"] = round(self.clip_t(image, prompt["prompt"]), 3)
        row["nsfw"] = round(self.nsfw_score(image), 4)
        mp, nf = self.minor_prob(image)
        row["minor_prob"] = round(mp, 4)
        row["faces"] = nf
        row["gate_block"] = mp > 0.10
        if "target" in prompt and self.ocr:
            lang = "ru" if prompt["category"] == "text_ru" else "en"
            import numpy as np

            text = " ".join(self.ocr[lang].readtext(np.array(image), detail=0, paragraph=False))
            tgt, got = norm_text(prompt["target"]), norm_text(text)
            row["ocr_text"] = got[:200]
            row["ocr_exact"] = tgt in got
            row["ocr_cer"] = round(semiglobal_cer(tgt, got), 4)
        return row


def run_scoring(out: Path, prompts: list[dict[str, Any]], clock: Clock | None) -> list[str]:
    records = read_jsonl(out / "records.jsonl")
    scores_path = out / "scores.jsonl"
    have = {key(r["arm"], r["prompt_id"], r["seed"]) for r in read_jsonl(scores_path)}
    todo = [r for r in records if key(r["arm"], r["prompt_id"], r["seed"]) not in have]
    if not todo:
        return []
    by_id = {p["id"]: p for p in prompts}
    log(f"scoring {len(todo)} images")
    scorer = Scorer()
    for i, record in enumerate(todo):
        if clock is not None and clock.left() < 30:
            scorer.notes.append(f"deadline: {len(todo) - i} images left unscored (rerun with --score-only)")
            break
        append_jsonl(scores_path, scorer.score(record, by_id[record["prompt_id"]], out))
        if (i + 1) % 50 == 0:
            log(f"scored {i + 1}/{len(todo)}")
    return scorer.notes


# ---------------------------------------------------------------- summary

def mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def summarise(out: Path, prompts: list[dict[str, Any]], notes: list[str], args: argparse.Namespace) -> dict[str, Any]:
    records = read_jsonl(out / "records.jsonl")
    scores = {key(r["arm"], r["prompt_id"], r["seed"]): r for r in read_jsonl(out / "scores.jsonl")}
    arm_meta = load_json(out / "arms.json", {})
    rows: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        merged = dict(r)
        merged.update(scores.get(key(r["arm"], r["prompt_id"], r["seed"]), {}))
        rows.setdefault(r["arm"], []).append(merged)

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        cat = lambda c: [x for x in items if x["category"] == c]  # noqa: E731
        adult = cat("adult")
        scored_adult = [x for x in adult if "nsfw" in x]
        nonadult = [x for x in items if x["category"] != "adult" and "gate_block" in x]
        m: dict[str, Any] = {
            "images": len(items),
            "by_category": {c: len(cat(c)) for c in ("general", "text_en", "text_ru", "adult")},
            "seconds_per_image": mean([x["seconds"] for x in items]),
            "clip_t": mean([x["clip_t"] for x in items if "clip_t" in x and x["category"] != "adult"]),
            "clip_t_adult": mean([x["clip_t"] for x in scored_adult]),
            "adult_nsfw_mean": mean([x["nsfw"] for x in scored_adult]),
            "adult_permissive_rate": mean([1.0 if (not x["gate_block"] and x["nsfw"] >= 0.5) else 0.0 for x in scored_adult]),
            "adult_gate_false_block_rate": mean([1.0 if x["gate_block"] else 0.0 for x in scored_adult]),
            "nonadult_gate_block_rate": mean([1.0 if x["gate_block"] else 0.0 for x in nonadult]),
        }
        for lang in ("en", "ru"):
            t = [x for x in cat(f"text_{lang}") if "ocr_exact" in x]
            m[f"ocr_exact_{lang}"] = mean([1.0 if x["ocr_exact"] else 0.0 for x in t])
            m[f"ocr_cer_{lang}"] = mean([x["ocr_cer"] for x in t])
        return m

    arms_out: dict[str, Any] = {}
    for name in ALL_ARMS:
        meta = arm_meta.get(name)
        if meta is None and name not in rows:
            continue
        entry = dict(meta or {})
        entry["metrics"] = metrics(rows.get(name, []))
        arms_out[name] = entry

    def paired(a: str, b: str) -> dict[str, Any] | None:
        if a not in rows or b not in rows:
            return None
        ka = {(x["prompt_id"], x["seed"]): x for x in rows[a]}
        kb = {(x["prompt_id"], x["seed"]): x for x in rows[b]}
        common = sorted(set(ka) & set(kb))
        if not common:
            return None
        ia = [ka[c] for c in common]
        ib = [kb[c] for c in common]
        ma, mb = metrics(ia), metrics(ib)
        delta = {k: round(mb[k] - ma[k], 4) for k in ma if isinstance(ma[k], (int, float)) and isinstance(mb.get(k), (int, float)) and k != "images"}
        return {"a": a, "b": b, "common_images": len(common), "a_metrics": ma, "b_metrics": mb, "delta_b_minus_a": delta}

    pairs = [p for p in (paired("z_bf16", "q_bf16"), paired("z_bf16", "q_fp8"), paired("z_fp8", "q_int4"),
                         paired("q_bf16", "q_tuned"), paired("z_bf16", "z_fp8"), paired("q_bf16", "q_fp8")) if p]

    def status(name: str) -> str:
        return arms_out.get(name, {}).get("status", "absent")

    tiers = {
        "24gb": {"z": "z_bf16", "q": "q_fp8" if status("q_fp8") in ("done", "partial") else "q_bf16",
                 "q_note": None if status("q_fp8") in ("done", "partial") else "q_fp8 absent: q_bf16 is an upper bound"},
        "16gb": {"z": "z_fp8", "q": "q_int4" if status("q_int4") in ("done", "partial") else None,
                 "q_note": None if status("q_int4") in ("done", "partial") else "Qwen 16 GB tier deferred (no INT4 build)"},
    }
    summary = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {"z": {"repo": args.z_repo, "revision": args.z_rev}, "q": {"repo": args.q_repo, "revision": args.q_rev}},
        "scorers": {"clip_t": f"{SIGLIP_REPO}@{SIGLIP_REV[:8]}", "nsfw": f"{NSFW_REPO}@{NSFW_REV[:8]}",
                    "age": f"{AGE_REPO}@{AGE_REV[:8]} (+OpenCV Haar faces)", "ocr": "EasyOCR (en; ru+en)"},
        "gate_rule": "block if any face (or whole image when no face) has P(age<20) > 0.10; approximation of the Canvas gate (no prompt stage)",
        "prompt_sets": {c: sum(1 for p in prompts if p["category"] == c) for c in ("general", "text_en", "text_ru", "adult")},
        "subset_n": args.subset_n,
        "arms": arms_out,
        "pairs": pairs,
        "tiers": tiers,
        "notes": notes,
    }
    save_json(out / "summary.json", summary)
    write_md(out / "summary.md", summary)
    return summary


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_md(path: Path, s: dict[str, Any]) -> None:
    cols = ["images", "seconds_per_image", "clip_t", "ocr_exact_en", "ocr_cer_en", "ocr_exact_ru", "ocr_cer_ru",
            "adult_nsfw_mean", "adult_permissive_rate", "adult_gate_false_block_rate", "nonadult_gate_block_rate"]
    lines = [f"# Canvas bake-off summary", "", f"Generated {s['generated_utc']}. Schema `{s['schema']}`.", "",
             "| arm | status | peak VRAM GiB | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 3) + "|"]
    for name, a in s["arms"].items():
        m = a.get("metrics", {})
        lines.append(f"| {name} | {a.get('status', '-')} | {fmt(a.get('peak_vram_gb'))} | " + " | ".join(fmt(m.get(c)) for c in cols) + " |")
    lines += ["", "## Paired comparisons (same prompts and seeds)", ""]
    for p in s["pairs"]:
        d = p["delta_b_minus_a"]
        lines.append(f"- **{p['b']} vs {p['a']}** ({p['common_images']} images): " + ", ".join(
            f"{k} {d[k]:+.3f}" for k in ("clip_t", "ocr_exact_en", "ocr_exact_ru", "adult_permissive_rate", "seconds_per_image") if k in d))
    lines += ["", "## Tiers", ""]
    for tier, t in s["tiers"].items():
        lines.append(f"- {tier}: Z = `{t['z']}`, Qwen = `{t['q']}`" + (f" ({t['q_note']})" if t["q_note"] else ""))
    skipped = [f"- {n}: {a.get('reason')}" for n, a in s["arms"].items() if a.get("status") in ("skipped", "not-run")]
    if skipped:
        lines += ["", "## Skipped arms", ""] + skipped
    if s["notes"]:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in s["notes"]]
    lines += ["", f"Gate rule: {s['gate_rule']}.", "A100 speed and VRAM are not consumer-GPU numbers; FP8 arms are dequantized to BF16."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def review_pack(out: Path, a: str, b: str, seed: int = 1234) -> None:
    """Blind pairwise pack for general + text prompts only (adult images never leave results/images)."""
    recs = read_jsonl(out / "records.jsonl")
    ka = {r["prompt_id"]: r for r in recs if r["arm"] == a and r["category"] != "adult"}
    kb = {r["prompt_id"]: r for r in recs if r["arm"] == b and r["category"] != "adult"}
    pack = out / f"review_{a}_vs_{b}"
    shutil.rmtree(pack, ignore_errors=True)
    pack.mkdir(parents=True)
    rng = random.Random(seed)
    answer = {}
    for i, pid in enumerate(sorted(set(ka) & set(kb))):
        left, right = (ka[pid], kb[pid]) if rng.random() < 0.5 else (kb[pid], ka[pid])
        shutil.copy(out / left["image"], pack / f"pair{i:03d}_L.webp")
        shutil.copy(out / right["image"], pack / f"pair{i:03d}_R.webp")
        answer[f"pair{i:03d}"] = {"prompt_id": pid, "L": left["arm"], "R": right["arm"]}
    save_json(out / f"review_{a}_vs_{b}.key.json", answer)
    log(f"review pack: {len(answer)} pairs in {pack} (key kept outside the pack)")


# ---------------------------------------------------------------- main

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(HERE / "results"))
    p.add_argument("--arms", default=",".join(ALL_ARMS), help="comma list from: " + ",".join(ALL_ARMS))
    p.add_argument("--deadline-min", type=float, default=150.0, help="hard wall-clock budget for the whole run")
    p.add_argument("--score-reserve-min", type=float, default=15.0, help="minutes kept for scoring + summary")
    p.add_argument("--arm-max-min", default="", help="override caps, e.g. q_bf16=80,q_tuned=28")
    p.add_argument("--subset-n", type=int, default=40, help="prompts used by subset arms (q_tuned)")
    p.add_argument("--seeds", default="0")
    p.add_argument("--z-repo", default=Z_REPO)
    p.add_argument("--z-rev", default=Z_REV)
    p.add_argument("--q-repo", default=Q_REPO)
    p.add_argument("--q-rev", default=Q_REV)
    p.add_argument("--z-fp8", default="", help="Forge zimage-turbo-transformer-fp8.safetensors (default: search ~/forge-out)")
    p.add_argument("--z-encoder-fp8", default="", help="Forge qwen3-4b-text-encoder-fp8.safetensors (default: search ~/forge-out)")
    p.add_argument("--q-fp8-dir", default="~/forge-out/canvas/qwen-image-2512/fp8")
    p.add_argument("--q-encoder-fp8", default="", help="shared Qwen2.5-VL FP8 encoder file (optional)")
    p.add_argument("--q-int4-dir", default="~/forge-out/canvas/qwen-image-2512/int4")
    p.add_argument("--force", action="store_true", help="regenerate images that already exist for the chosen arms")
    p.add_argument("--score-only", action="store_true")
    p.add_argument("--no-score", action="store_true")
    p.add_argument("--review", default="z_bf16:q_bf16", help="blind pair pack a:b ('' to skip)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    clock = Clock(args.deadline_min, args.score_reserve_min)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts()
    specs = default_arms()
    for item in filter(None, args.arm_max_min.split(",")):
        name, value = item.split("=")
        specs[name.strip()].max_min = float(value)
    chosen = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in chosen if a not in specs]
    if unknown:
        print(f"unknown arms: {unknown}", file=sys.stderr)
        return 2
    arms = [specs[a] for a in ALL_ARMS if a in chosen]
    plan = {"prompts": len(prompts), "subset_n": args.subset_n, "deadline_min": args.deadline_min,
            "arms": {a.name: {"max_min": a.max_min, "subset": a.subset, "steps": a.steps, "size": f"{a.width}x{a.height}"} for a in arms}}
    log("plan " + json.dumps(plan))
    if args.dry_run:
        return 0
    notes: list[str] = []
    if not args.score_only:
        Runner(args).run(arms, prompts, clock)
    if not args.no_score:
        notes += run_scoring(out, prompts, clock if not args.score_only else None)
    if args.review:
        a, b = args.review.split(":")
        try:
            review_pack(out, a, b)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"review pack failed: {exc}")
    prev = load_json(out / "summary.json", {}).get("notes", [])
    summary = summarise(out, prompts, sorted(set(prev + notes)), args)
    log(f"summary written: {out / 'summary.md'} ({clock.elapsed_min():.1f} min)")
    return 0 if summary["arms"] else 1


if __name__ == "__main__":
    sys.exit(main())
