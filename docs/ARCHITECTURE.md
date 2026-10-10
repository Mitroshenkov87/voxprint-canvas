# Architecture

Voxprint AI Canvas is a local image application. It runs on the machine's NVIDIA GPU. The window, the checks, and the generator live in one Python package.

## Layout

| Path | Role |
| --- | --- |
| `canvas/app` | Startup, OS and GPU requirements, release check |
| `canvas/ui` | Two-tab window and the suite palette |
| `canvas/core` | Output formats and the VRAM budget |
| `canvas/infra` | Model folder, GPU lock, thermal pause |
| `canvas/models` | Registry, tier, diffusers plan, download pins |
| `canvas/safety` | Content safety checks |
| `canvas/pipeline` | In-process job and the worker-process boundary |
| `installer` | Windows installer and Linux packaging stub |
| `tools/audit.py` | Release audit |

## Modes

v1 has two tabs.

- Text to image uses Z-Image-Turbo (8 steps, guidance 0). Cards with 24 GB or more can also select the 50-step Z-Image Creative plan.
- Image to image covers restyle, inpaint, outpaint, pose, depth, and edge through the Z-Image pipelines and `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`. Instruction edit uses `Qwen/Qwen-Image-Edit-2511` and is offered only on the 24 GB tier.

The registry pins diffusers 0.41.0. A 16 GB card uses the FP8 turbo weights and the lite FP8 controlnet. A 24 GB card uses BF16 turbo weights and the full BF16 controlnet. Creative and instruction edit stay hidden on 16 GB.

Weight files stay unpinned until their SHA-256 values are frozen. The loader and the downloader refuse an empty pin. Inference through diffusers is not wired yet: the generator records the plan and stops.

## Content safety checks

The app runs content safety checks on a request before it keeps an image.

Do not use the app to create content that is illegal or socially unacceptable; you are responsible for what you create.

## Shared suite behaviour

- Models live in `%LOCALAPPDATA%\Voxprint\models` or `~/.local/share/voxprint/models`. `.users.json` records the key `canvas`.
- The GPU lock is `%TEMP%/voxprint-gpu.lock`. It is stale when the recorded process is gone, or when `eta` is more than two hours in the past.
- The memory budget is free memory minus a headroom of at least 2 GB and at most 3 GB, and at least 8 percent of the card.
- After 2.5 hours at full speed, work pauses while the 5-minute median temperature is at least 83 °C, or while a throttle has lasted more than 60 seconds. It resumes at 75 °C or below once the throttle has cleared.
- The only network call besides model downloads is `https://api.github.com/repos/Mitroshenkov87/voxprint-canvas/releases/latest`.

## What is still a stub

SHA-256 pins, the diffusers call itself, and the AVIF / JPEG XL / WebP bitstream encoders. The next step is to freeze those pins and wire diffusers 0.41.0.
