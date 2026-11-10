# REPO-MAP

One-page map of the VOXPrint AI Canvas repository. Read this before touching code.

## Top level

- `main.py` — thin shim calling `canvas.app.main:main`.
- `canvas/__main__.py` — the same entry for `python -m canvas`.
- `canvas/` — the application package (below).
- `tests/` — pytest suite; `conftest.py` sets offscreen Qt and per-test isolation.
- `tools/` — repo tooling (audit, icon, screen grabs, model-index check).
- `scripts/` — bench writer, PDF guide build, VM test drivers.
- `installer/` — Windows (Inno) and Linux install assets.
- `docs/` — mkdocs guide, architecture notes, release notes.
- `bakeoff/` — prompt bakeoff harness with its own requirements.
- `assets/` — splash and icon art.
- `pyproject.toml`, `constraints*.txt` — packaging and pinned dependencies (voxprint-common 0.7.0).
- `mdl.json` — VOXPrint MDL manifest, generated from `canvas/models/catalog.json`.
- `BUILD.json` — the only copy of version, channel, build and codename.

## canvas/ — the app

### canvas/app — entry points and CLI commands
- `main.py` — argparse CLI (`generate`, `photoshoot`, `batch`, …) and window startup: `main()`, `_parser()`.
- `shell_run.py` — `ShellRunner`: runs one `PictureJob` through the pipeline for the window.
- `shell_gate.py` — `ContentCheck`: adapts the content gate to the screen's review hooks.
- `batch_cli.py`, `photoshoot_cli.py`, `photoshoot_bench.py` — the `batch`, `photoshoot`, `photoshoot-bench` commands.
- `selftest.py` — headless GPU/model/encoder checks.
- `identity.py` — release identity from `BUILD.json`: `splash_line()`, `version()`.
- `i18n.py` + `messages.json` — user-visible strings (`tr()`, `text()`).
- `platform.py`, `preflight.py`, `provision.py`, `updates.py`, `branding.py`, `info.py`.

### canvas/core — formats and memory policy
- `formats.py` — `save_image()`, `read_metadata()`, `parse_format()`; WebP is the default. PNG/WebP/AVIF/JPEG go through `voxprint_common.image.save_image` (`purpose="picture"` is lossy WebP q90; `purpose="master"` / `lossless=True` is lossless for series and Forge). Metadata is off unless the caller passes a mapping; read-back is a flat string dict.
- `exchange.py` — pictures handed to Enlarge and Forge.
- `vram_policy.py` — how much VRAM a batch may plan for.

### canvas/models — model registry, tiers, loading
- `registry.py` — `load_catalog()` reads the shared model registry.
- `plan.py` — `build_plan()`: registry entries and diffusers classes per mode; loads nothing. The 16 GB MOSS row is `voxprint-moss-image-sft-compact` (formerly `moss-image-sft-compact-16gb`).
- `loader.py` — `ZImageGenerator`, `load_catalog_stack()`.
- `runtime.py` — diffusers inference; heavy libraries load only when a snapshot is on disk.
- `snapshots.py` — resolves a model through its MDL view and refuses a bad hash.
- `user_lora.py` — user LoRAs: `prepare_cli_lora()`, `resolve_selected()`, `running_base_id()`. A MOSS LoRA's base tag is `moss5b`. User-facing `moss` and `moss-image` (and the older tags `moss-image-sft` and `moss-image-base`) map to it; the Z-Image tag stays `z-image-turbo`. Safetensors metadata `voxprint.lora.base_variant` (expected `moss-image-sft`) and `voxprint.lora.base_sha256` / `base_sha256` are stored on the entry. A `moss5b` file whose variant is `moss-image-base` still loads, logs a warning, and returns it on `warnings` (also in `--json`).
- `tiers.py`, `modes.py`, `sizes.py` — VRAM tiers, `Mode`, aspect-ratio maps.
- `provision.py`, `install_fetch.py`, `downloader.py`, `mdl_*.py` — install, download, MDL views.

### canvas/pipeline — generation stages
- `pipeline.py` — `Pipeline.export()`: screen inputs, generate inside the worker boundary, screen the output.
- `worker.py` — the worker-process boundary.
- `engine.py` — the only calls into a generator; the input must already be screened.
- `series.py`, `photoshoot.py`, `series_text.py`, `frame_prompt.py` — Photoshoot series planning and writing.
- `enhance.py` — prompt rewriter: `WriterSession`, `rewrite_text()`.
- `schedule.py`, `bf16_schedule.py`, `offload.py`, `device_guard.py`, `gpu_release.py` — the 16 GB residency plan.
- `batch.py` — batch manifest runner.

### canvas/ui — PySide6 window
- `shell_screen.py` — `ShellWindow`, the Canvas window itself.
- `shell_layout.py` — the shared image screen: `PictureJob`, `open_layout()`/`open_worker_layout()`, `job_from()`, `_settings()`.
- `shell_chrome.py`, `theme.py`, `fonts.py`, `splash.py`, `backdrop.py` — suite look.
- `prompt_row.py`, `action_row.py`, `result_bar.py` — main screen blocks.
- `lora_panel.py` — LoRA import, library and strength picker.
- `worker_checks.py` — review hooks answered from worker replies.
- `locales/en.json` — UI strings.

### canvas/worker_jobs.py
- Worker-process handlers `run()` and `check()`; `_picture_metadata()` writes prompt/seed/LoRA fields into a saved picture when the job turns metadata on.

### canvas/safety
- minors gate - do not modify

### canvas/infra, canvas/ipc
- `infra/` — suite plumbing: `paths.py`, `gpu.py`/`gpu_lock.py`, `vram.py`, `settings.py` (`canvas.toml`), `enlarge.py` client.
- `ipc/` — window-to-worker protocol: `schema.py` (`canvas.worker/1`), `client.py` (`WorkerClient`).
