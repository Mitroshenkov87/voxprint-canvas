# Voxprint AI Canvas

Local text-to-image and image-to-image generation for the Voxprint AI Media Suite. Everything runs on your own GPU; no cloud service is required.

> Status: early development. No releases yet.

Text to image can use Z-Image-Turbo or Qwen-Image-2512. The default is unset until a comparison on a supported GPU chooses one. Image to image covers restyle, inpaint, outpaint, pose, depth, and edge. On a 24 GB GPU, Creative and instruction edit are available as well. Third-party add-ons are off until you turn them on. Third-party weights are provided by their authors, who are responsible for the licence and training data. Voxprint does not host or redistribute them. Output defaults to AVIF, with JPEG XL and WebP beside it.

## Requirements

- **OS:** Windows 11 24H2 or newer, or a Linux distribution released in 2025 or later.
- **GPU:** NVIDIA GeForce RTX 40 series or newer, with 16 GB of memory (24 GB unlocks the larger plans). PCs without a supported NVIDIA GPU are not supported.
- **Python:** 3.12 or newer, when running from source.

## Responsibility

Do not use the app to create content that is illegal or socially unacceptable; you are responsible for what you create.

## Build and test

```bash
python -m pip install -e ".[dev]"
python -m pytest
python tools/audit.py --python python
```

Tests do not download weights and do not need a GPU. Image encoding and the diffusers 0.41.0 call are still stubs. The Windows installer and the Linux packaging stub select a VRAM tier and register the shared models folder; they do not download weights either.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Models downloaded by the app keep their own licences.

## Roadmap

The next version, and the plans after it, are in [ROADMAP.md](ROADMAP.md).

## Documentation

The docs site is published from this repository: <https://mitroshenkov87.github.io/voxprint-canvas/>.

## ☕ Support the Project

If you find this project useful and would like to support its development, you can buy me a coffee!

- **Network:** TRON (TRC-20)
- **Accepted:** USDT or TRX
- **Address:** `TYveZBXaSpM4FEHa6iGrc7A6zuLfS4z3x6`

> ⚠️ **Important:** Please ensure you are sending funds **only** via the TRON (TRC-20) network. 
> Sending assets from other networks (such as Ethereum ERC-20, BSC BEP-20, etc.) to this address will result in **permanent loss of funds**.
