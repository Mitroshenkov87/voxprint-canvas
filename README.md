# VOXPrint AI Canvas

Local text-to-image and image-to-image generation for the VOXPrint AI Suite. Everything runs on your own GPU; no cloud service is required.

> Status: 0.1.0 release candidate.

Text to image uses Z-Image-Turbo in BF16 on every tier. Image to image covers restyle, inpaint, outpaint, pose, depth, and edge. On a 24 GB GPU, Creative is available as well.

Third-party add-ons are off until you turn them on. Third-party weights are provided by their authors, who are responsible for the licence and training data. Voxprint does not host or redistribute them. Output defaults to AVIF, with JPEG XL and WebP beside it.

## Install

Windows setup places the app in `C:\Program Files\VOXPrint AI Suite\Canvas\`. The Start menu folder is VOXPrint AI Suite. The window program is `voxprint-canvas.exe` and the command line is `voxprint-canvas-cli.exe`.

Linux setup places the app in `/opt/voxprint-ai-suite/canvas/`.

Pictures, models, and the shared manifest stay in `%LOCALAPPDATA%\VOXPrint AI Suite` on Windows and `~/.local/share/voxprint-ai-suite` on Linux. An upgrade removes the previous program folder and its shortcuts, and leaves that data in place.

## Requirements

- **OS:** Windows 11 24H2 or newer, or a Linux distribution released in 2025 or later.
- **GPU:** NVIDIA GeForce RTX 40 series or newer, with 16 GB of memory (24 GB unlocks the larger plans). PCs without a supported NVIDIA GPU are not supported.
- **Python:** 3.14 or newer, when running from source.

## Responsibility

Do not use the app to create content that is illegal or socially unacceptable; you are responsible for what you create.

## Build and test

```bash
python -m pip install -e ".[dev]"
python -m pytest
python tools/audit.py --python python
```

Tests do not download weights and do not need a GPU. AVIF, JPEG XL, WebP and JPEG are written by the installed libraries. HEIC still needs a system codec. The Windows setup and the Linux archive install Python when it is missing, the GPU libraries, and the default Z-Image-Turbo pack. Optional packs download when you choose them, after the window shows the size and the estimated time.

### GPU smoke test on Windows

Proving Ground runs this on Windows Server 2025 or Windows 11 24H2 (build 26100 or newer, which the Setup requires) with an NVIDIA RTX 4090. PowerShell 7 (`pwsh`) is recommended and Windows PowerShell 5.1 also works; WSL and bash are not used.

1. Download the `VoxprintCanvasSetup` artifact from the `windows-install` job of the commit under test and unzip it.
2. Install from an elevated PowerShell prompt:

```powershell
$setup = Start-Process -FilePath .\VoxprintCanvasSetup.exe -ArgumentList "/VERYSILENT","/SUPPRESSMSGBOXES","/SP-","/NORESTART","/LOG=$env:TEMP\voxprint-setup.log" -Wait -PassThru
$setup.ExitCode   # 0 means installed
```

3. Run both passes from a normal PowerShell prompt in a folder you can write to:

```powershell
$smoke = "$env:ProgramFiles\VOXPrint AI Suite\Canvas\tools\smoke_16gb.ps1"
pwsh -NoProfile -ExecutionPolicy Bypass -File $smoke -VramCapGB 16 -Out .\smoke-results   # 16 GB tier, group offload
pwsh -NoProfile -ExecutionPolicy Bypass -File $smoke -Out .\smoke-results                 # 24 GB tier, uncapped
```

Without PowerShell 7, run the same lines with `powershell` instead of `pwsh`. A silent Setup uses PowerShell 7 when it is installed and Windows PowerShell 5.1 otherwise; an interactive Setup offers to install PowerShell 7 from Microsoft first.

The script uses the Python the Setup installed (`-Python <python.exe>` picks another one). It verifies the voxprint-common 0.3.1 wheel, downloads the models through VOXPrint MDL, generates five neutral 1024x1024 pictures through the content gate, and times the gate by stage. The results are `smoke-results\result-16gb.json` and `smoke-results\result-24gb.json`. Exit codes: 0 pass, 1 a picture failed, 2 setup (no Python, wheel hash, wrong tier), 3 no GPU, 4 gate unavailable, 5 an MDL download or hash check failed.

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
