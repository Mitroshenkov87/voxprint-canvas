"""STEP 1 of the GPU acceptance run: base Z-Image-Turbo decode check (no LoRA).

Samples one 1024x1024 frame through the Canvas path (seed 1616, 8 steps, the
on-disk FlowMatchEulerDiscreteScheduler with shift 3.0, bfloat16, no LoRA, no
enhance), then decodes the same Canvas latent two ways and compares both plus a
full stock pipeline run:

1. ``canvas_decode.png``    - the Canvas decode path (``runtime._decode_latents``)
2. ``stock_vae_decode.png`` - stock diffusers ``AutoencoderKL`` in float32 from the same local VAE folder
3. ``stock_pipeline.png``   - stock ``ZImagePipeline(output_type="pil")`` at the same seed, steps, size, and prompt

Every image must pass the shared decode-noise assert
(``canvas.models.decode_noise``), and the Canvas decode must match the stock
float32 VAE decode of the same latent within 30 dB PSNR. If the stock float32
VAE decode of the Canvas latent is already noise, the latent itself is noise:
that points at sampling, not only at the VAE decode.

Usage (from the repo root, inside the Canvas ``[gen]`` venv):
    python tools/gpu_decode_check.py --out DIR [--models-dir DIR] [--gpu N] [--prompt TEXT]

Everything goes to ``DIR``: the three PNGs, ``report.json`` and ``run.log`` (a
copy of stderr). The last stdout line is the one-line verdict, always one of
``DECODE CHECK PASS: ...``, ``DECODE CHECK FAIL: ...`` or ``DECODE CHECK ERROR: ...``.

Exit codes: 0 pass (clean decode), 1 a check failed (noise or divergence),
2 the run could not start (missing weights, no CUDA, missing package).
Nothing is downloaded: the hub stays offline and every load is ``local_files_only``.
"""
from __future__ import annotations

# The hub stays offline: set before diffusers/transformers are first imported.
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

# Standalone: ``python tools/gpu_decode_check.py`` puts tools/ first on sys.path,
# so make the repo's ``canvas`` package importable without an editable install.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Below this much card memory the stock reference pipeline (about 20.5 GB of
# bf16 weights) cannot sit on the card whole, so it runs with model CPU offload.
_STOCK_RESIDENT_MIN_BYTES = 22 * 1024**3

PROMPT = "a red apple on a wooden table"
SEED = 1616
STEPS = 8
SIZE = 1024
PSNR_MIN_DB = 30.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode check for base Z-Image-Turbo (STEP 1 of the GPU acceptance run).",
    )
    parser.add_argument("--out", required=True, type=Path, help="output folder for the PNGs and report.json")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Weights folder (sets VOXPRINT_MODELS_DIR for this process)",
    )
    parser.add_argument("--gpu", type=int, default=None, metavar="N", help="use this CUDA device index")
    parser.add_argument("--prompt", default=PROMPT, help="the still-life prompt to sample")
    return parser


def _pin_gpu(index: int | None) -> None:
    """``--gpu N`` pins the card before torch initializes, like the Canvas CLI."""
    if index is None:
        return
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)


_OUT: Path | None = None
_LOG: Any = None


class _Tee:
    """Copy stderr into ``run.log`` in the output folder."""

    def __init__(self, stream: Any, log: Any) -> None:
        self._stream = stream
        self._log = log

    def write(self, data: str) -> int:
        self._log.write(data)
        self._log.flush()
        return self._stream.write(data)

    def flush(self) -> None:
        self._log.flush()
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _verdict(code: int, reason: str) -> None:
    """The single verdict line on stdout (and in run.log)."""
    word = "PASS" if code == 0 else ("FAIL" if code == 1 else "ERROR")
    line = f"DECODE CHECK {word}: {reason}"
    if _LOG is not None:
        _LOG.write(line + "\n")
        _LOG.flush()
    print(line, flush=True)


def _fail(message: str, code: int) -> int:
    """Stop early: reason on stderr, a minimal report.json, and the verdict line."""
    print(message, file=sys.stderr)
    if _OUT is not None and not (_OUT / "report.json").exists():
        try:
            partial = sorted(p.name for p in _OUT.glob("*.png"))
            (_OUT / "report.json").write_text(
                json.dumps({"ok": False, "exit_code": code, "reason": message, "images_written": partial}, indent=2)
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
    _verdict(code, message)
    return code


def _psnr(left: Any, right: Any) -> float:
    """PSNR in dB between two PIL images, on float [0, 1] pixels."""
    import numpy

    a = numpy.asarray(left.convert("RGB"), dtype=numpy.float32) / 255.0
    b = numpy.asarray(right.convert("RGB"), dtype=numpy.float32) / 255.0
    if a.shape != b.shape:
        return 0.0
    mse = float(numpy.mean((a - b) ** 2))
    if mse <= 0.0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _noise_report(image: Any) -> tuple[dict[str, float], str]:
    """Metrics plus ``pass`` or the fail code from the shared decode-noise assert."""
    from canvas.models.decode_noise import DecodeNoiseError, assert_not_noise, noise_metrics

    metrics = noise_metrics(image)
    try:
        assert_not_noise(image)
    except DecodeNoiseError as exc:
        return metrics, exc.code
    return metrics, "pass"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _pin_gpu(args.gpu)
    if args.models_dir is not None:
        os.environ["VOXPRINT_MODELS_DIR"] = str(args.models_dir.expanduser().resolve())

    global _OUT, _LOG
    try:
        out = args.out.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        stale = out / "report.json"
        if stale.exists():
            stale.unlink()
        _LOG = (out / "run.log").open("w", encoding="utf-8")
    except OSError as exc:
        return _fail(f"The output folder could not be created: {args.out} ({exc})", 2)
    _OUT = out
    sys.stderr = _Tee(sys.stderr, _LOG)
    print(f"Output folder: {out}", file=sys.stderr)

    try:
        import torch
    except ImportError:
        return _fail("torch is not installed in this environment.", 2)
    if not torch.cuda.is_available():
        return _fail("CUDA is not available: this check needs the GPU box.", 2)

    from canvas.app.platform import check_gpu
    from canvas.infra.gpu import probe_gpu
    from canvas.infra.paths import resolved_models_dir
    from canvas.models.modes import Tier
    from canvas.models.registry import load_catalog
    from canvas.models.snapshots import snapshot_ready
    from canvas.models.verified import flush_process_cache, process_cache

    gpu = check_gpu(probe_gpu())
    if not gpu.ok or gpu.tier is None:
        return _fail(f"GPU check failed: {gpu.message}", 2)
    tier = gpu.tier

    root, source = resolved_models_dir()
    catalog = load_catalog()
    wanted = ["z-image-turbo-bf16-16gb" if tier is Tier.GB16 else "z-image-turbo-bf16"]
    wanted.append("z-image-turbo-bf16" if tier is Tier.GB16 else "z-image-turbo-bf16-16gb")
    cache = process_cache()
    folder = None
    used_id = ""
    try:
        for model_id in wanted:
            spec = catalog.by_id(model_id)
            folder = snapshot_ready(spec, root, cache)
            if folder is not None:
                used_id = model_id
                break
    finally:
        flush_process_cache()
    if folder is None:
        return _fail(
            f"The Z-Image-Turbo snapshot is missing from {root} (from {source}): {' and '.join(wanted)}. "
            "Install it with VOXPrint MDL, then run this command again.",
            2,
        )
    print(f"Models folder: {root} (from {source})", file=sys.stderr)
    print(f"Snapshot: {folder} (id {used_id}, tier {tier.value})", file=sys.stderr)

    from canvas.models import runtime
    from canvas.models.decode_noise import DECODE_NOISE_CODE
    from canvas.models.plan import PIPELINE_TEXT, CallPlan
    from canvas.safety.gate import issue_input

    from canvas.infra import settings as canvas_settings

    canvas_settings._cached = canvas_settings.Settings(steps=STEPS)

    plan = CallPlan("diffusers", PIPELINE_TEXT, (used_id,), "bf16", STEPS, 0.0, "none", tier.value, "text")
    screened = issue_input(args.prompt, "text", ())
    latent = None
    canvas_image = None
    vae_dtype = ""
    shift = None
    try:
        with runtime.pipeline_scope():
            latent = runtime._render(
                plan, folder, screened, SIZE, SIZE, tier,
                seed=SEED, control_file=None, strength=None,
                lora=None, lora_strength=None, negative="", keep_latent=True,
            )
            pipe = runtime._slot.pipe if runtime._slot is not None else None
            if pipe is None:
                return _fail("Canvas kept no pipeline after sampling.", 1)
            shift = float(getattr(getattr(pipe.scheduler, "config", None), "shift", 0.0) or 0.0)
            decoded = runtime._decode_latents(pipe, latent, torch)
            canvas_image = runtime._latents_to_pil(pipe, decoded, torch)
            vae_dtype = str(next(pipe.vae.parameters()).dtype)
    except Exception as exc:  # the sampling or the Canvas decode failed before any comparison
        return _fail(f"Canvas sampling or decode failed: {type(exc).__name__}: {exc}", 1)
    finally:
        screened.wipe()
    runtime.release_pipeline()
    runtime.clear_series_embeds()
    gc.collect()
    torch.cuda.empty_cache()

    if canvas_image is None or latent is None:
        return _fail("Canvas produced no latent or no decode.", 1)
    canvas_image.save(out / "canvas_decode.png")

    # Stock float32 VAE decode of the same Canvas latent.
    import diffusers
    from diffusers.image_processor import VaeImageProcessor

    try:
        vae32 = diffusers.AutoencoderKL.from_pretrained(
            folder / "vae", torch_dtype=torch.float32, local_files_only=True,
        )
        vae32.to("cuda")
        scaled = (latent.to("cuda", torch.float32) / vae32.config.scaling_factor) + vae32.config.shift_factor
        with torch.inference_mode():
            sample = vae32.decode(scaled, return_dict=False)[0]
        stock_vae_image = VaeImageProcessor(vae_scale_factor=16, do_convert_rgb=True).postprocess(
            sample, output_type="pil",
        )[0]
    except Exception as exc:
        return _fail(f"The stock float32 VAE decode failed: {type(exc).__name__}: {exc}", 1)
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    stock_vae_image.save(out / "stock_vae_decode.png")

    # Stock pipeline, same seed, steps, size, and prompt.
    try:
        pipe = diffusers.ZImagePipeline.from_pretrained(folder, torch_dtype=torch.bfloat16, local_files_only=True)
        card_bytes = int(torch.cuda.get_device_properties(0).total_memory)
        if card_bytes < _STOCK_RESIDENT_MIN_BYTES:
            # A 16 GB card cannot hold the whole bf16 pipeline: offload per model, like stock docs do.
            print("Stock pipeline: model CPU offload (card under 22 GiB).", file=sys.stderr)
            pipe.enable_model_cpu_offload()
        else:
            try:
                pipe.to("cuda")
            except RuntimeError:
                pipe.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
                pipe.enable_model_cpu_offload()
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        stock_image = pipe(
            args.prompt,
            height=SIZE,
            width=SIZE,
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            generator=generator,
            output_type="pil",
        ).images[0]
    except Exception as exc:
        return _fail(f"The stock ZImagePipeline run failed: {type(exc).__name__}: {exc}", 1)
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    stock_image.save(out / "stock_pipeline.png")

    canvas_metrics, canvas_code = _noise_report(canvas_image)
    vae_metrics, vae_code = _noise_report(stock_vae_image)
    stock_metrics, stock_code = _noise_report(stock_image)
    psnr_vae = _psnr(canvas_image, stock_vae_image)
    psnr_pipe = _psnr(canvas_image, stock_image)

    latent_std = float(latent.float().std())
    latent_mean = float(latent.float().mean())
    reasons: list[str] = []
    if canvas_code != "pass":
        reasons.append(f"the Canvas decode failed the noise assert ({canvas_code})")
    if vae_code != "pass":
        reasons.append(
            "the stock float32 VAE decode of the Canvas latent is noise "
            "(the Canvas latent is already noise: sampling-side, not only the VAE decode)",
        )
    if stock_code != "pass":
        reasons.append(f"the stock ZImagePipeline image failed the noise assert ({stock_code})")
    if psnr_vae < PSNR_MIN_DB:
        reasons.append(
            f"the Canvas decode diverges from the stock VAE decode of the same latent "
            f"({psnr_vae:.1f} dB < {PSNR_MIN_DB:.0f} dB)",
        )
    ok = not reasons
    reason = (
        "every image passes the noise assert and the Canvas decode matches the stock float32 "
        f"VAE decode of the same latent ({psnr_vae:.1f} dB >= {PSNR_MIN_DB:.0f} dB)"
        if ok
        else "; ".join(reasons)
    )

    report = {
        "ok": ok,
        "exit_code": 0 if ok else 1,
        "reason": reason,
        "model": {"id": used_id, "path": str(folder), "source": source},
        "settings": {
            "seed": SEED,
            "steps": STEPS,
            "shift": shift,
            "width": SIZE,
            "height": SIZE,
            "dtype": "bfloat16",
            "prompt": args.prompt,
            "psnr_min_db": PSNR_MIN_DB,
        },
        "latent": {"std": latent_std, "mean": latent_mean},
        "images": {
            "canvas_decode": {
                "file": "canvas_decode.png",
                "noise": canvas_metrics,
                "noise_assert": canvas_code,
                "vae_dtype": vae_dtype,
            },
            "stock_vae_decode": {
                "file": "stock_vae_decode.png",
                "noise": vae_metrics,
                "noise_assert": vae_code,
                "vae_dtype": "float32",
            },
            "stock_pipeline": {"file": "stock_pipeline.png", "noise": stock_metrics, "noise_assert": stock_code},
        },
        "psnr": {"canvas_vs_stock_vae_db": psnr_vae, "canvas_vs_stock_pipeline_db": psnr_pipe},
        "decode_noise_code": DECODE_NOISE_CODE,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {out / 'report.json'}", file=sys.stderr)
    _verdict(0 if ok else 1, reason)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
