# Architecture

VOXPrint AI Canvas is a local image application. It runs on the machine's NVIDIA GPU. The window, the checks, and the generator live in one Python package.

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
| `installer` | Windows setup and Linux archive |
| `tools/audit.py` | Release audit |
| `tools/bakeoff.py` | Text-base comparison on a supported GPU |

## Modes

v1 has two tabs.

- Text to image uses Z-Image-Turbo (8 steps, guidance 0) in BF16 on a 16 GB card and on a 24 GB card. A 16 GB job loads the Qwen3-4B text encoder, frees it, denoises with one transformer block group streamed from pinned CPU memory (the full 6B weights, about 12.3 GB, stay in system RAM), then frees that and decodes with the VAE. The planning peak is 14.5 GB at 2048×1152 and 2048×2048. The encoder runs on CPU when the memory budget is short of its weights, and the VAE decode is tiled when a full decode is larger than the budget. Prompt embeddings stay cached, so a new seed skips the encoder. Cards with 24 GB or more can also select the 50-step Z-Image Creative plan. `CANVAS_VRAM_CAP_GB` or `--vram-cap-gb` lowers the card to that budget, so a 24 GB card capped at 16 GB takes this schedule.
- Image to image covers restyle, inpaint, outpaint, pose, depth, and edge through the Z-Image pipelines and `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`.

Enhance loads that same Qwen3-4B text encoder as a causal language model, improves the prompt in the same field, and unloads it before diffusion. An English prompt stays English. Any other language is improved and written in English. Revert restores the text from before that rewrite. On a 16 GB card the encoder is gone before the transformer is loaded. Tied embeddings are the output layer, so the shipped encoder is the rewriter and there is no extra pack. The original prompt and the rewrite both go through the content safety checks. A rewrite that drops or changes a stated age, or drops quoted lettering, is discarded and the original prompt is used.

The registry pins diffusers 0.41.0 and transformers 5.19.0. Z-Image weights are BF16 on every tier. A 16 GB card loads the lite BF16 controlnet only for inpaint and guided modes. A 24 GB card uses the full BF16 controlnet. Creative stays hidden on 16 GB. Each 16 GB stage asks for the free memory left after the headroom, and a stage that is larger than that budget refuses the picture. The 16 GB runtime installs diffusers group offload (`use_stream`, `record_stream`, pinned transformer blocks) and falls back to sequential CPU offload when that hook is missing. The 4K chain unloads the generator, text encoder, and VAE on every tier before Enlarge, and the suite GPU lock is released for that spawn. `CANVAS_VRAM_CAP_GB` and `--vram-cap-gb` limit the memory this process may use and select the same tier value, `16gb` or `24gb`, from that limit.

Both tabs offer the same aspect picker, grouped as Landscape (16:9, 21:9, 3:2, 4:3), Square (1:1), and Portrait (9:16, 2:3, 3:4). Each model has its own native frame in the registry, a multiple of 16, around 1–2 megapixels. Image to image starts from the input image's aspect. Output size 4K does not generate at 4K. It generates the model's ~2K frame for that aspect, then asks VOXPrint AI Enlarge through its headless CLI (`voxprint-enlarge-cli upscale --input … --output … --long-side 3840 --format avif --preset photo --json-progress --parent-lock-pid …`) to make the long side 3840. Canvas sets `VOXPRINT_GPU_PARENT_PID` to that same pid, releases the suite GPU lock while Enlarge runs, and takes the lock again afterwards. Enlarge is started with a clean environment: PATH, the home and temp variables, and the suite lock path. Discovery and the 4K checkbox use `voxprint-enlarge-cli --capabilities --json`. The option is disabled, with an install hint and an Install Enlarge button, when Enlarge is absent or cannot run. On every tier the generator, text encoder, and VAE are unloaded before the upscale, and Enlarge starts only when free GPU memory is above the amount Enlarge declares (6 GB when the capabilities document does not say). The enlarged image goes through the content safety checks before it is shown. A peak-VRAM probe marks generate, unload, and upscale so the chain can be measured.

`tools/bakeoff.py` runs the same prompts and seeds for each text base and records time and peak VRAM. It writes images only in the operating system's temp directory.

Third-party add-ons are off until you turn them on. The curated list has one model, Chroma1-HD, fetched from its upstream address when you ask. Files are stored in a `third-party` folder under the shared models directory. Voxprint does not host or redistribute those files.

Weight files in the main registry are pinned by SHA-256. The loader refuses a missing file or a bad hash. A picture uses the pinned snapshot through diffusers 0.41.0 when those files are already on disk. The setup downloads the default Z-Image-Turbo pack. Z-Image-Turbo is taken from the public mirror first, then from the upstream repository.

## Content safety checks

The app runs content safety checks on a request before it keeps an image.

Do not use the app to create content that is illegal or socially unacceptable; you are responsible for what you create.

## Shared suite behaviour

- The program is installed in `C:\Program Files\VOXPrint AI Suite\Canvas\` on Windows and `/opt/voxprint-ai-suite/canvas/` on Linux. The Windows Start menu folder is VOXPrint AI Suite. The window program is `voxprint-canvas.exe` and the command-line program is `voxprint-canvas-cli.exe`. An upgrade removes the previous install directory and its shortcuts, and leaves `%LOCALAPPDATA%\Voxprint` and `~/.local/share/voxprint` in place, including the shared manifest. 4K finds Enlarge in that shared manifest first, then `C:\Program Files\VOXPrint AI Suite\Enlarge\voxprint-enlarge-cli.exe` (Linux `/opt/voxprint-ai-suite/enlarge/voxprint-enlarge-cli`), then PATH.
- Shared files live in `%LOCALAPPDATA%\Voxprint\shared` or `~/.local/share/voxprint/shared`. `manifest.json` is schema 1 and records which apps ref each component. An uninstall removes a component only when no app still refs it. The setup installs Python 3.14 when the suite runtime is missing, the CUDA 13.0 libraries, and the default Z-Image-Turbo pack. Optional packs download when a picture needs them, after the window shows the size and the estimated time. Every launch re-checks the recorded hashes and replaces a missing or corrupt file.
- The GPU lock is `%TEMP%/voxprint-gpu.lock`. It is stale when the recorded process is gone, or when `eta` is more than two hours in the past.
- The memory budget is free memory minus a headroom of at least 2 GB and at most 3 GB, and at least 8 percent of the card.
- After 2.5 hours at full speed, work pauses while the 5-minute median temperature is at least 83 °C, or while a throttle has lasted more than 60 seconds. It resumes at 75 °C or below once the throttle has cleared.
- The only network call besides model downloads is `https://api.github.com/repos/Mitroshenkov87/voxprint-canvas/releases/latest`.

## Release identity

`BUILD.json` ships in the package and is the copy of the version, the channel, the build number and the codename. The build number is the `build` field. `GITHUB_RUN_NUMBER` is not used. The release title template, the About dialog, and `voxprint-canvas --version` show `X.Y.Z RC · build N "Tavnit"` from that file. The startup splash shows `X.Y.Z RC build N "Codename"`.

Save, download and export use the approved icons in `canvas/ui/icons`. A row marked PENDING is not loaded. Icons are recoloured with a palette token. The floppy-disk drawing and the theme name `document-save` are not used.

## What is still a stub

HEIC still needs a system codec on the machine. A 16 GB job still runs its memory schedule, then offloads the real model. The official upstream transformer remains a fallback and is larger than the mirror pack.
