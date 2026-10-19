<!--
VOXPrint AI Canvas user guide, English source.
Cover metadata (kept in the build script, as in Audiobook Builder's build_manual.py META):
  title: VOXPrint AI Canvas
  subtitle: User guide
  version line: Version 0.1.0 (release candidate) · build 12 · guide of 18 October 2026
  licence line: Apache-2.0 · © 2026 Aleksandr Mitroshenkov
Headings: "#" = chapter, "##" = section, "###" = sub-section. The build numbers chapters and sections.
UI labels are written exactly as in canvas/ui/locales/en.json, canvas/app/messages.json and the
voxprint-common 0.3.1 catalog, in bold. Screenshots: ![caption](screens/<id>.png), listed in screens.md.
Checked notes that used to sit in HTML comments have been resolved in the text below.
-->

# What Canvas is

**VOXPrint AI Canvas** makes pictures on your own computer. You describe a picture in words, or start from a picture you already have, and Canvas draws it on your NVIDIA graphics card. No cloud service and no account are needed. Canvas only goes online to download its models, to install add-on programs you ask for, and when you check for an update.

Canvas can:

- create a picture from a description (text to image);
- redraw an existing picture in a new look, replace one painted area, or grow a picture past its edges;
- follow the pose, the depth or the outlines of a reference picture (structure guidance);
- make a 4K version of a picture through VOXPrint Enlarge;
- shoot a whole series of the same subject with **Photoshoot**;
- use a style, object or person add-on (LoRA) that you trained in VOXPrint AI Model Forge.

The picture model is **Z-Image-Turbo** in BF16 on every graphics card that Canvas supports.

![The Canvas main window in simple mode.](screens/main-window-simple.png)

> **Release candidate.** This guide describes version 0.1.0, release candidate 13 (build 12). Some details can still change before the final release.

## What you need

| | Requirement |
|---|---|
| Operating system | Windows 11 24H2 (build 26100) or newer, 64-bit; or a Linux distribution released in 2025 or later |
| Graphics card | NVIDIA GeForce RTX 40 series or newer, with at least 16 GB of video memory (VRAM) |
| Without such a card | Not supported. Canvas has no processor-only mode. |
| Disk space | About 28 GB for the default model pack, plus room for the program libraries and your pictures. Optional packs need more (see below). |
| Internet | During installation and for each optional pack you download |

Setup also downloads the program libraries (Python, PyTorch and the shared suite runtime). This release does not fix their size; leave free space beyond the model pack.

## Graphics card tiers

Canvas reads the video memory of your card when it starts and picks one of two tiers. Every picture is computed on the graphics card; both tiers are designed to fit in their memory.

| Tier | Cards | What it adds |
|---|---|---|
| 16 GB | Cards with about 16 GB of VRAM (from 15.5 GB) | All text-to-image and picture-editing modes. Structure guidance uses the lighter ControlNet pack. |
| 24 GB | Cards with about 24 GB of VRAM or more (from 23.5 GB) | Everything in the 16 GB tier, plus **Bolder** variation (the Creative model) and the full ControlNet pack. |

A card with less than 15.5 GB of video memory, or a card older than the RTX 40 series, cannot run Canvas. In Expert mode you can make Canvas plan for less memory than the card has (**VRAM limit**, chapter 7); a 24 GB card limited to 16 GB uses the 16 GB tier.

## What gets downloaded

The setup file is small. It does not contain the graphics libraries or the models; Setup downloads them on your computer.

| Pack | When | Size |
|---|---|---|
| Default pack (Z-Image-Turbo and the supporting models) | During installation | about 28 GB |
| ControlNet lite (structure guidance, 16 GB tier) | When you choose it | about 2 GB |
| ControlNet (structure guidance, 24 GB tier) | When you choose it | about 6.7 GB |
| Creative Z-Image (**Bolder**, 24 GB tier) | When you choose it | about 20.5 GB |
| Scenario writer | On first use | about 8 GB |

Models are kept in the shared VOXPrint models folder, so other VOXPrint apps can use the same files without downloading them again.

On a 24 GB card the plan names a separate Z-Image-Turbo entry (`z-image-turbo-bf16`). That entry is the same files as the default 16 GB pack, stored once, so it is not downloaded again. The 24 GB tier still adds the optional Creative Z-Image pack (**Bolder**) and the full ControlNet pack when you choose them.

# Installing on Windows

## Download and check Setup

Download the setup from the project's releases page: <https://github.com/Mitroshenkov87/voxprint-canvas/releases>. The file is called `VoxprintCanvasSetup-<version>.exe`, for example `VoxprintCanvasSetup-0.1.0-rc.13.exe`. The release also has a `SHA256SUMS` file. To check the download, open PowerShell in the download folder and compare the result of

```powershell
Get-FileHash .\VoxprintCanvasSetup-0.1.0-rc.13.exe -Algorithm SHA256
```

with the line for that file in `SHA256SUMS`.

The setup file is not code-signed. Windows SmartScreen may say that it prevented an unrecognized app from starting. Choose **More info**, then **Run anyway**.

## Run Setup

1. Double-click the setup file. Windows asks for administrator rights; Setup installs for all users.
2. On an English Windows, Setup uses English and does not ask. On a Russian Windows, Setup uses Russian and does not ask. When Windows uses another language, Setup asks which language to use.
3. Setup installs the program into `C:\Program Files\VOXPrint AI Suite\Canvas\`. There is no folder page: the location is fixed.
4. Setup then runs its install steps. The wizard shows *Installing the program libraries and models. This can take a while; progress is written to the setup log.* There is no separate console window. This takes a while, mostly because of the model download.
5. When Setup finishes, the Start menu has a **VOXPrint AI Suite** folder with **VOXPrint AI Canvas**. Setup can also add a desktop shortcut; that box is off by default. The guides open from **More → Help** in the Canvas window.

## PowerShell 7

Setup runs its install steps with PowerShell 7 when it is on the computer. If PowerShell 7 is missing, Setup asks:

> PowerShell 7 is not installed. Install PowerShell 7 (Microsoft) now? Recommended.

![Setup offers to install PowerShell 7.](screens/setup-pwsh-offer.png)

- **Yes**: Setup shows the page **Installing PowerShell 7 (Microsoft)**, downloads the official PowerShell 7 installer from Microsoft, checks its checksum and its Microsoft signature, installs it and continues with it.
- **No**, or if the download or installation fails: Setup continues with the Windows PowerShell 5.1 that is built into Windows and shows this note:

> Setup will continue with Windows PowerShell 5.1. Please install PowerShell 7 when possible: https://aka.ms/powershell-release?tag=stable

![The note shown when Setup continues with Windows PowerShell 5.1.](screens/setup-pwsh-fallback.png)

Canvas works the same either way. PowerShell 7 is recommended; you can install it later from the link above.

## Model download

The default model pack (about 28 GB) is downloaded during Setup through VOXPrint MDL, the shared model downloader of the suite. Each file is checked against its pinned checksum before it is used. A file that is already complete, with a matching size and a verification marker, is not downloaded again. A stopped download is kept as a partial file. The next run continues that file from where it stopped when the source accepts a range request. If the source ignores the range, that file starts again.

## First launch

Open **Start → VOXPrint AI Suite → VOXPrint AI Canvas**. A startup window appears while Canvas checks the system and the graphics card (**Checking the GPU**) and prepares its models. Then the main window opens.

![The startup window.](screens/splash.png)

The main window follows, in order, `ui_language` in the suite settings file, the `VOXPRINT_LANG` environment variable, the operating-system display language, then English. Canvas dialogs follow the process locale: Russian when it starts with `ru`, otherwise English. There is no language control in the window. To use one language for both, set the display language (on Linux, `LANG`) and do not set a different `ui_language`. `VOXPRINT_LANG` changes the main window only. The suite settings file is `%LOCALAPPDATA%\VOXPrint AI Suite\state\suite.json` on Windows and `~/.local/share/voxprint-ai-suite/state/suite.json` on Linux. `VOXPRINT_HOME` replaces that data folder.

If the startup window closes and no main window appears, the computer does not meet the requirements. See chapter 15.

## Silent installation

Administrators can install without questions:

```powershell
.\VoxprintCanvasSetup-0.1.0-rc.13.exe /VERYSILENT /SUPPRESSMSGBOXES /SP- /NORESTART /LOG="$env:TEMP\voxprint-setup.log"
```

A silent Setup does not offer PowerShell 7. It uses PowerShell 7 when it is installed and Windows PowerShell 5.1 otherwise. Exit code 0 means Canvas was installed (chapter 15 lists the other codes).

# Installing on Linux

You need a Linux distribution released in 2025 or later, and `curl`, `tar` and `sha256sum`. The installer needs permission to create `/opt/voxprint-ai-suite/canvas/` (`sudo` when you are not root). It does not run `nvidia-smi` and it does not name a minimum driver version. The graphics-card rules in chapter 1 still apply.

1. Download `voxprint-canvas-<version>-linux-x86_64.tar.gz` and `SHA256SUMS` from the releases page and check the archive:

        sha256sum -c --ignore-missing SHA256SUMS

2. Unpack the archive and open its folder:

        tar -xzf voxprint-canvas-0.1.0-rc.13-linux-x86_64.tar.gz
        cd voxprint-canvas-0.1.0-rc.13-linux-x86_64

3. Run the installer. It installs the program into `/opt/voxprint-ai-suite/canvas/`, so it needs permission to write there:

        sudo ./install.sh

    When you use `sudo`, the program goes into `/opt`, and your own data and menu entry are still created in your home folder, as your user.

4. The script installs Python if needed, PyTorch for CUDA 13.0, the suite libraries and the default model pack. It prints **Installed.** at the end.

After installation, **VOXPrint AI Canvas** is in the applications menu, in the Graphics category. You can also start it from a terminal with `voxprint-canvas`; the command-line tool is `voxprint-canvas-cli`.

> The Linux installer ignores proxy settings from the environment (`HTTP_PROXY`, `HTTPS_PROXY` and similar) on purpose, and it has no proxy setting of its own. A machine that can reach the network only through a proxy cannot pass those variables into this installer.

# The main window

Canvas opens in simple mode. Everything you need for most pictures is on one screen.

![The main window in simple mode.](screens/main-window-simple.png)

From top to bottom:

| Area | What it holds |
|---|---|
| Top bar | **Use a picture** (and **Edit the mask** once a picture is chosen) on the left; **More** and **Expert mode** on the right |
| Prompt row | The description field (**Describe the picture**), **Improve prompt**, **Photoshoot**, **Revert to my prompt** and **Generate** |
| Preset rows | **Quality**, **Style**, **Size**, **Output size** and **Number of pictures** |
| Picture area | The finished picture, or *The picture will show up here.* while it is empty; the progress while a picture is being made |
| Action bar | Under a finished picture: **More like this**, **Save**, **Open folder**, **Use prompt again**, **Use as input picture**, **Delete** |
| History strip | Small previews of your earlier pictures |
| Footer | *Do not use the app to create content that is illegal or socially unacceptable; you are responsible for what you create.* |

## The More menu

**More** opens a menu with the tools that do not need room on the main screen:

- **Picture setup**: where a picture starts from, variation, what to change, and the file type (chapter 9);
- **Add-ons**: LoRA add-ons from Model Forge (chapter 12);
- **Check this PC**: system check and update check (chapter 15);
- **Open folder**: opens the folder with your pictures;
- **Manage models**: opens the VOXPrint MDL window to download, check or remove models;
- **Settings**: shows the program name and version (**About VOXPrint AI Canvas**);
- **Expert mode**: shows or hides the expert settings.

![The More menu.](screens/more-menu.png)

## The history strip

Every finished picture is added to the strip under the picture area. Click a preview to show it again. Right-click a preview for its menu: **Open**, **Show in folder**, **Copy**, **Use prompt again**, **Use as input picture** and **Delete**. Delete asks *Delete this picture?* and removes the picture only after you choose **Delete** a second time; **Keep** leaves it.

![The menu of a picture in the history strip.](screens/history-item-menu.png)

# Creating your first image

1. Click the description field (**Describe the picture**) and describe what you want to see. Write it as you would brief a photographer or an illustrator: the subject, the place, the light. For example:

    > A red bicycle leaning against a brick wall on a quiet street, morning sun, soft shadows

2. Leave **Quality** on **Balanced** and **Style** on **Photo** for now.
3. Choose the shape under **Size**: **Landscape**, **Square** or **Portrait**.
4. Click **Generate**.

![A description in the prompt field, ready to generate.](screens/first-image-prompt.png)

While Canvas works, the picture area shows *Creating the picture...*, a stage (**Preparing**, then **Drawing**), the progress and the estimated time left, with **Cancel**.

![A picture being made.](screens/first-image-progress.png)

When it is done, the picture appears with the action bar under it:

- **More like this** makes another picture from the same description, settings and seed;
- **Save** saves a copy of the picture wherever you choose;
- **Open folder** opens the folder where Canvas keeps your pictures;
- **Use prompt again** puts this picture's description and settings back on the screen;
- **Use as input picture** uses this picture as the starting point for editing (chapter 9);
- **Delete** removes the picture.

![The finished picture with the action bar.](screens/first-image-result.png)

Every picture is saved automatically (chapter 13); **Save** only makes an extra copy.

More example descriptions:

- *A ceramic coffee mug on a wooden desk next to an open notebook, soft window light, product photo*
- *A mountain lake at sunrise, mist over the water, pine trees on the shore*
- *A woman in her thirties in a knitted sweater reading a book in a café, natural light*
- *A chef in a white jacket plating a salad in a bright restaurant kitchen*

# Simple mode and presets

The preset rows set the most important values with one click. The highlighted chip is the current choice.

## Quality

| Chip | What it does |
|---|---|
| **Fast** | Fewer steps: quicker, a little less detail |
| **Balanced** | The default |
| **Best** | More steps and a different sampler: slower, more detail |

## Style

**Style** adds a look to your description: **Photo**, **Cinematic**, **Portrait**, **Illustration**, **Anime**, **3D render**, **Painting** or **Sketch**. You don't need to write the style into the description yourself.

## Size

**Landscape**, **Square** and **Portrait** choose the orientation. The list next to them chooses the exact aspect ratio within that orientation (chapter 8).

![The aspect ratio list for Landscape.](screens/size-ratio-menu.png)

## Output size

**Standard** is the normal size of the model. **4K** makes a picture whose long side is 3840 pixels, using VOXPrint Enlarge. Until Enlarge is installed, the chip reads **4K (needs Enlarge)** (chapter 10).

## Number of pictures

**1**, **2** or **4** pictures per click on **Generate**. With more than one, each picture starts from the next seed, so they differ from one another.

## Improve prompt and Revert to my prompt

**Improve prompt** rewrites your description in the same field, adding detail the picture model understands well. An English description stays in English; a description in another language is improved and written in English. **Revert to my prompt** brings back the text you had before the last rewrite.

# Expert mode

Click **Expert mode** (top right, or in the **More** menu) to open the **Expert settings** panel on the right. Click it again to close the panel. Canvas remembers the choice, and the other VOXPrint apps use the same setting.

![The Expert settings panel.](screens/expert-panel.png)

## Live values

The panel shows the real values behind the chips, and it changes with them. Choose **Best** under **Quality**, for example, and **Steps**, **Guidance (CFG)** and **Sampler** change at once.

The panel has two groups.

**Tuning**

| Field | What it does |
|---|---|
| **Steps** | How many steps a picture takes |
| **Guidance (CFG)** | How closely the picture follows the words |
| **Seed** | A number that repeats the same picture |
| **Sampler** | How each picture is drawn: **Euler**, **DPM** or **Heun** |
| **Picture size** | Width and height of the picture |
| **Output size** | How large the finished picture is |
| **Negative prompt** | What to leave out of the picture |
| **Add-ons (LoRA)** | The folder of the add-on in use (chapter 12) |
| **LoRA strength** | How strongly the add-on changes the picture |
| **VRAM limit** | How much graphics memory a picture may use: **Automatic**, **8 GB**, **12 GB**, **16 GB** or **24 GB** |

**Image**

| Field | What it does |
|---|---|
| **Adult content (18+)** | Off by default |
| **Exact size** | **Match the choice**, **Standard**, **2K** or **4K** (chapter 8) |

**Seed** starts at 0, and 0 is a real seed, not a new number each time. The same **Seed** repeats the picture. Change **Seed** for a different picture. When **Number of pictures** is greater than 1, Canvas uses that seed and then the next integers. **Use prompt again** restores that picture's seed. **More like this** copies the history seed and generates again, so the seed stays the same.

**Picture size** shows the preset text (for example `1344x768` for 16:9). The generator uses the aspect ratio from the **Size** row: 16:9 is 1280×720 (chapter 8). Typing a size in this field does not change the picture.

## Your own values and the presets

When you change a value yourself, the chip shows that it was changed, for example **Balanced + changes**, and a **Reset to Balanced** button appears at the top of the panel. Click it to go back to the preset values.

![A changed value: the Quality chip reads "Balanced + changes".](screens/expert-custom.png)

# Aspect ratios, file types and sizes

## Aspect ratios

| Orientation | Ratios | Standard size (pixels) | 2K size (pixels) |
|---|---|---|---|
| **Landscape** | 16:9 | 1280 × 720 | 1920 × 1088 |
| | 21:9 | 1344 × 576 | 1920 × 816 |
| | 3:2 | 1248 × 832 | 1920 × 1280 |
| | 4:3 | 1152 × 864 | 1920 × 1440 |
| **Square** | 1:1 | 1024 × 1024 | 1920 × 1920 |
| **Portrait** | 9:16 | 720 × 1280 | 1088 × 1920 |
| | 2:3 | 832 × 1248 | 1280 × 1920 |
| | 3:4 | 864 × 1152 | 1440 × 1920 |

When you start from a picture, Canvas picks the ratio closest to that picture's shape.

## Output sizes

| Choice | Where | Result |
|---|---|---|
| **Standard** | **Output size** row, or **Exact size** | The standard size from the table |
| **2K** | Expert: **Exact size** | The 2K size from the table |
| **4K** | **Output size** row, or **Exact size** | The 2K frame enlarged by VOXPrint Enlarge so that the long side is 3840 pixels |

**Exact size** wins over the **Output size** row until it is set back to **Match the choice**.

When **4K** is selected, a **Detail** row appears with **Light**, **Medium** and **Strong**.

![4K selected, with the Detail row.](screens/output-4k-detail.png)

The **Detail** row writes the same number as **LoRA strength**: **Light** is 0.4, **Medium** is 0.7 and **Strong** is 1.1. That number is applied only when an add-on is selected. With none selected, the row only stores the number.

## File types

Choose the file type in **More → Picture setup → File type**:

| Type | Notes |
|---|---|
| **AVIF** | The default. Small files with very high quality. |
| **JPEG XL** | Very high quality; supported by fewer viewers. |
| **WebP** | Widely supported. |

# Editing an existing image

Canvas can start from a picture instead of a description. You choose the picture with **Use a picture** and decide what to do with it in **Picture setup**.

## Picture setup

Open **More → Picture setup**.

![Picture setup.](screens/picture-setup.png)

| Group | Choices |
|---|---|
| **Start from** | **From text**: the picture is created from the description. **From a picture**: the chosen image is the base. |
| **Variation** | For **From text**: **Usual** (the familiar result) or **Bolder** (more unusual ideas, 24 GB tier only) |
| **Change** | For **From a picture**: **Restyle**, **Paint over**, **Extend**, **Copy the pose**, **Depth map**, **Outlines** |
| **File type** | **AVIF**, **JPEG XL**, **WebP** |

Click **Done** to close the window. Your choices stay until you change them.

## Choose a picture

1. Click **Use a picture** at the top left.
2. Click **Open** and choose an image file. The window shows a preview.
3. Click **Use this picture**.

![Use a picture, with a picture loaded.](screens/use-picture.png)

Canvas switches **Start from** to **From a picture**, sets **Size** to the ratio closest to the picture, and shows the **Edit the mask** button next to **Use a picture**. You can also click **Use as input picture** under any finished picture.

## Restyle (image to image)

**Restyle** redraws the whole picture in a new look and keeps its composition.

1. Choose a picture.
2. In **Picture setup**, choose **From a picture** and **Restyle**.
3. Describe the result, for example *the same street as a watercolour painting*, choose a **Style** if you like, and click **Generate**.

## Paint over (inpaint with a mask)

**Paint over** replaces one area and leaves the rest of the picture as it is.

1. Choose a picture.
2. Click **Edit the mask**.
3. Paint the area that should change. Use **Brush size** for the size of the brush, **Eraser** to correct, **Brush** to paint again and **Clear** to start over.
4. Click **Use this mask**. Canvas sets **Change** to **Paint over**.
5. Describe what should be in the painted area, for example *a green armchair*, and click **Generate**.

![The mask editor with a painted area.](screens/mask-editor.png)

If you open **Edit the mask** before choosing a picture, it says *Open a picture first.*

## Extend (outpaint)

**Extend** uses the same mask editor as **Paint over**. The editor paints only inside the picture, at the picture's size.

1. Choose a picture.
2. Click **Edit the mask** and paint the area that should change.
3. Click **Use this mask**. Canvas sets **Change** to **Paint over**.
4. Open **Picture setup**, choose **From a picture** and **Extend**.
5. Describe what should appear and click **Generate**.

## Structure guidance (ControlNet)

These three modes take the structure from your picture and draw a new picture on it according to the description:

| Change | What it keeps |
|---|---|
| **Copy the pose** | The pose of the person in the picture |
| **Depth map** | What is near and what is far |
| **Outlines** | The outlines in the picture |

1. Choose a picture.
2. In **Picture setup**, choose **From a picture** and one of the three modes.
3. Describe the new picture, for example *a dancer in a red dress on a theatre stage* with **Copy the pose**, and click **Generate**.

Structure guidance needs the ControlNet pack (ControlNet lite on the 16 GB tier, ControlNet on the 24 GB tier). If it is not installed, download it in **More → Manage models**.

Canvas does not offer those packs from the picture screen. Open **More → Manage models** and download the pack there. The same applies to Creative Z-Image (**Bolder**). When the default models are missing, the window says *Models are not ready yet. Use Manage models.*

# Upscaling through Enlarge

Canvas makes 4K pictures with VOXPrint Enlarge, a separate suite program. It draws the picture at 2K and Enlarge enlarges it so that the long side is 3840 pixels.

## Install Enlarge

1. Click **4K (needs Enlarge)** in the **Output size** row.
2. Canvas asks *4K needs VOXPrint Enlarge (≈ size). Install it?* Click **Install**, or **Cancel** to leave it.
3. A window shows *Installing VOXPrint Enlarge…* with the progress and **Cancel**.
4. When the installation is done, Canvas selects **4K** for you.

![Canvas asks before it installs Enlarge.](screens/enlarge-confirm.png)

![Enlarge is being installed.](screens/enlarge-progress.png)

If the installation fails, Canvas says *VOXPrint Enlarge could not be installed.* Try again later, or check the connection and the free disk space.

## Make a 4K picture

Choose **4K** in the **Output size** row (or **Exact size: 4K** in Expert mode) and click **Generate**. The picture takes longer, because Canvas frees the graphics card for Enlarge after drawing.

# Photoshoot

**Photoshoot** makes a series of shots of the same subject: different angles, framings and, if you like, places. It is meant for things like product pictures, portrait sessions or fashion looks.

## Start a series

1. Describe the subject in the description field, for example:

    > A yellow kettle with a blue handle

    or

    > A man in his forties in a navy suit, business portrait

2. Choose **Photo**, **Cinematic** or **Portrait** under **Style**. Other styles are treated as **Photo** for a series.
3. Click **Photoshoot** (next to **Improve prompt**) so that it is highlighted.
4. Click **Generate**.

![Photoshoot is switched on.](screens/photoshoot-on.png)

Canvas first writes a short scenario: one line per shot, with the subject and the framing. The list appears in the picture area and cannot be edited. Then the shots are drawn one by one; the list shows which shot is being drawn, and the line under it reads the shot number and the total, for example *Shot 2 of 6* when **Frames** is 6. The first finished shot becomes the reference for the others, so the subject stays the same across the series.

![A series in progress.](screens/photoshoot-running.png)

Set **Frames** next to **Photoshoot** before you click **Generate**. A series can have from 1 to 50 shots and starts at 6. The shots stay in one place (**Same place**), with **Seed** set to **In order** and **How close** set to **Usual**. The catalog also names **Place** (**Different places**), **Same seed**, **Random**, **Looser** and **Closer**. Those controls are not on the screen. The shot list cannot be edited.

## Stop a series

Click **Cancel** under the list. Canvas stops after the shot that is being drawn; the shots that are already finished stay.

## The finished series

When the series is ready, it appears in the history strip as one entry. **Open folder** (or **Show in folder** in the entry's menu) opens the series folder with all its shots. Click **Photoshoot** again to switch the mode off and go back to single pictures.

![A finished series in the history strip.](screens/photoshoot-done.png)

If a shot cannot be made, it is marked in the list and skipped; the rest of the series continues. If the scenario cannot be used, Canvas says *The scenario cannot be used. The series has stopped.* Rewrite the description and try again.

## From the command line

The `photoshoot` command writes the same series into a folder you name:

```text
voxprint-canvas-cli photoshoot --prompt "A yellow kettle with a blue handle" --frames 6 --out ./kettle
```

`--frames` sets the number of shots, from 1 to 50. The default is 6. `--style` takes the same style ids as the **Style** row (`photo` by default). `--places` is `same` or `different` (`same` by default). `--seed` starts at 1. Each frame is an AVIF file. The folder also contains `scenario.txt`, with one scenario line per shot. Add `--json` for the frame paths, the path of `scenario.txt`, and how many frames were made of those requested. `scenario_source` is `writer` when the text model wrote every line, `planned` when the lines were built from the idea, and `mixed` for both. `--adult` follows the Expert **Adult content (18+)** switch.

# LoRAs from Model Forge

VOXPrint AI Model Forge trains small add-ons (LoRAs) for Z-Image-Turbo: a style, an object or a person. Canvas lists the add-ons that Model Forge saved into the shared models folder.

## Where Canvas looks

Each add-on is one folder:

```text
<models folder>/user-models/lora/<id>/
    voxprint-model.json
    <id>.safetensors
```

The folder name is the add-on's id. Model Forge writes each add-on into that folder when it exports the add-on into the shared models folder. The models folder is, by default:

| System | Models folder |
|---|---|
| Windows | `%LOCALAPPDATA%\VOXPrint AI Suite\models` |
| Linux | `~/.local/share/voxprint-ai-suite/models` |

To use another models folder, set `VOXPRINT_MODELS_DIR` to an absolute path, or set `models_dir` to an absolute path in the suite settings file (`suite.json`, see chapter 2). `VOXPRINT_MODELS_DIR` wins when both are set. Canvas does not name a Model Forge screen or button for this export.

Only add-ons made for Z-Image-Turbo are listed.

## Choose an add-on

1. Open **More → Add-ons**. The window starts with a notice: *Third-party weights are provided by their authors, who are responsible for the licence and training data. Voxprint does not host or redistribute them.*
2. Choose an add-on from the list. Each add-on is shown by its trigger word, or by its id if it has none. **None** uses no add-on.
3. Set **Strength**. The field starts at the add-on's recommended value and stays within the range the add-on allows.
4. Click **Save**.

![The Add-ons window with an add-on selected.](screens/addons-dialog.png)

Canvas uses one add-on at a time. The choice is written into the Expert fields **Add-ons (LoRA)** (the add-on's folder) and **LoRA strength**, so you can also check or change it there.

If the models folder has no add-ons, the window says *No add-ons in the models folder.* A folder that cannot be used is listed under the add-ons with its name and the reason, for example *This add-on is for a different base model.* Export it again from Model Forge.

## The trigger word chip

When the selected add-on has a trigger word, a chip with that word appears next to **Generate**. Canvas never adds the word to your description by itself. Put the cursor where the word should go and click the chip (*Insert this word*).

![The trigger word chip next to Generate.](screens/lora-trigger-chip.png)

Example with a style add-on whose trigger word is `inkwash`: *a lighthouse on a rocky coast at dusk, inkwash*.

To stop using the add-on, open **More → Add-ons**, choose **None** and click **Save**.

# Where files are saved

| What | Windows | Linux |
|---|---|---|
| Program | `C:\Program Files\VOXPrint AI Suite\Canvas\` | `/opt/voxprint-ai-suite/canvas/` |
| Your pictures | `%LOCALAPPDATA%\VOXPrint AI Suite\state\canvas\pictures\` | `~/.local/share/voxprint-ai-suite/state/canvas/pictures/` |
| Photoshoot series | `...\pictures\series\<series>\` | `.../pictures/series/<series>/` |
| Models (shared) | `%LOCALAPPDATA%\VOXPrint AI Suite\models\` | `~/.local/share/voxprint-ai-suite/models/` |
| Add-ons from Model Forge | `<models folder>\user-models\lora\` | `<models folder>/user-models/lora/` |
| Installation logs | `C:\Program Files\VOXPrint AI Suite\Canvas\setup-work\canvas-install.log` and `%LOCALAPPDATA%\VOXPrint AI Suite\logs\canvas-install.log` | Printed in the terminal |

Pictures get a generated file name and the extension of the chosen file type (`.avif`, `.jxl` or `.webp`). **More → Open folder** and **Open folder** under a picture open the pictures folder; for a series, the series folder.

# Updating and uninstalling

## Check for an update

Open **More → Check this PC** and click **Check for an update**. Canvas says either *This is the current release.* or *A newer release is* followed by the version.

## Update on Windows

Download the new setup from the releases page and run it. The update removes the previous program folder and its shortcuts and installs the new version. Your pictures, models and settings in `%LOCALAPPDATA%\VOXPrint AI Suite` stay.

## Update on Linux

Download and unpack the new archive and run `sudo ./install.sh` again, as in chapter 3. Your data in `~/.local/share/voxprint-ai-suite` stays.

A newer `install.sh` installs into the same `/opt/voxprint-ai-suite/canvas`. It keeps a Python that is already there, installs the libraries again when the CUDA check fails, rewrites the menu entry and the command links, and runs the model download again. Files that are already complete are kept. It does not delete the program folder first. Your data under `~/.local/share/voxprint-ai-suite` stays.

## Uninstall on Windows

Use **Settings → Apps → Installed apps** in Windows and choose **VOXPrint AI Canvas**. The uninstaller removes the program and unregisters Canvas from the shared model list.

Your pictures and the shared models stay in `%LOCALAPPDATA%\VOXPrint AI Suite`, because other VOXPrint apps may use the models. To free the space:

- remove models you no longer need in **More → Manage models** before uninstalling (a model is only removed if no other app still uses it);
- delete the pictures folder yourself if you don't want to keep your pictures.

## Uninstall on Linux

There is no uninstall script. As your user, run `voxprint-canvas-cli --unregister-models`. Then remove `/opt/voxprint-ai-suite/canvas`, `~/.local/share/applications/voxprint-canvas.desktop`, `~/.local/bin/voxprint-canvas`, `~/.local/bin/voxprint-canvas-cli`, and the copies in `/usr/local/bin` when they are present. Your pictures and models under `~/.local/share/voxprint-ai-suite` stay.

# Troubleshooting

## Check this PC

**More → Check this PC** checks the system, the graphics card and the file types Canvas can write. Click **Run check**. Tick **Check every file** to check every model file again (slower). The result starts with *This PC is ready.* or *This PC is not ready.*, followed by the lines **System**, **Graphics** and **File types**.

![Check this PC with a result.](screens/check-this-pc.png)

The same check is available from the command line: `voxprint-canvas-cli --selftest` (add `--json` for a detailed report and `--deep` to check every file).

## Setup exit codes (Windows)

| Code | Meaning |
|---|---|
| 0 | Canvas was installed |
| 1 | An install step failed or could not start. Setup shows where the logs are. |
| 2 | Setup was cancelled before the installation started |
| 5 | Setup was cancelled during the installation |
| other | Standard Inno Setup exit codes; see the Inno Setup documentation |

When an install step fails, Setup says:

> VOXPrint AI Canvas setup failed. The machine log is `C:\Program Files\VOXPrint AI Suite\Canvas\setup-work\canvas-install.log`. The user log is `%LOCALAPPDATA%\VOXPrint AI Suite\logs\canvas-install.log`.

Open both logs; the last lines name the step that failed.

## Messages during installation

| Message | What to do |
|---|---|
| *uv could not install Python 3.14. Install Python 3.14 from … and run this setup again.* | Install Python 3.14 from the address in the message, then run Setup again |
| *voxprint-common 0.3.1 failed the SHA-256 check.* | The download was damaged. Check the connection and run Setup again |
| *CUDA 13.0 PyTorch is still missing.* | The graphics libraries did not install. Check the connection and the free space, then run Setup again |
| *Setup stopped because this path contains a link: …* | A folder Setup writes to is a link or junction. Replace it with a normal folder and run Setup again |
| *VOXPrint AI Canvas could not start its setup script.* | PowerShell could not be started. Install PowerShell 7 and run Setup again |
| *VOXPrint AI Canvas needs permission to create /opt/voxprint-ai-suite/canvas.* (Linux) | Run the installer with `sudo ./install.sh` |

## Canvas does not open

If the startup window closes and no main window appears, Canvas found a requirement that is not met. Run `voxprint-canvas-cli --selftest --json` to see which one. The messages are:

| Message | Meaning |
|---|---|
| *Windows 11 24H2 or newer is required.* | Update Windows |
| *A Linux distribution from 2025 or later is required.* | Use a newer distribution |
| *An NVIDIA RTX 40 series GPU or newer is required.* | The graphics card is not supported, or the NVIDIA driver is missing |
| *This GPU does not have enough memory.* | The card has less than 15.5 GB of video memory |

These messages are printed to the console only. A Start-menu launch does not show them. Run `voxprint-canvas-cli --selftest` to see which requirement failed. Add `--json` for the detailed report.

## Messages in the main window

| Message | What to do |
|---|---|
| *Models are not ready yet. Use Manage models.* | Open **More → Manage models** and finish the download |
| *This mode needs another image.* | Choose a picture with **Use a picture**; for **Paint over** and **Extend**, also make a mask with **Edit the mask** |
| *This mode is not available on this GPU.* | The mode needs the 24 GB tier, for example **Bolder** |
| *This picture does not fit on the GPU.* | Choose a smaller size, close other programs that use the graphics card, or set **VRAM limit** to **Automatic** |
| *The GPU is cooling down. Try again in a little while.* | Wait a moment and click **Generate** again |
| *The model could not be downloaded.* | Check the connection and try again in **More → Manage models** |
| *The model file could not be verified.* | Open **More → Manage models** and use **Repair** |
| *Install VOXPrint AI Enlarge to use this.* | Install Enlarge (chapter 10) or choose **Standard** |
| *Write a few words about the picture.* | The description is empty |
| *Image generation is unavailable.* | Restart Canvas. If it happens again, run **Check this PC** and, if needed, run Setup again |
| *This picture was stopped.* | If a picture is not produced, change the prompt or the input image |

## Messages from Enlarge

| Message | What to do |
|---|---|
| *The GPU is busy.* | Wait until the other job on the graphics card is finished |
| *Enlarge needs more free GPU memory.* / *Enlarge ran out of memory.* | Close other programs that use the graphics card and try again |
| *Download the upscale model in VOXPrint AI Enlarge, then try again.* | Open VOXPrint AI Enlarge and download its upscale model |
| *Enlarge is not installed.* | Install it (chapter 10) |

## Model downloads

The **Manage models** window (VOXPrint MDL) shows what is installed (**Installed**) and what this computer can add (**Available for this computer**). **Download** adds a model, **Remove** removes it, **Repair** downloads a damaged model again and **Check installed files** checks all files. A download can be paused and resumed; what is already downloaded is kept. If the first source is slow, MDL switches to another one automatically.

![The Manage models window.](screens/manage-models.png)

# Licences and third-party notices

**VOXPrint AI Canvas** is free software under the Apache License 2.0. © 2026 Aleksandr Mitroshenkov. The licence text (`LICENSE`) and the notices (`NOTICE`) are in the program folder.

The models that Canvas downloads keep their own licences. Z-Image-Turbo, Creative Z-Image, the ControlNet packs, and the scenario writer are published under Apache-2.0 by their authors. The **Manage models** window shows each model and where it comes from.

The libraries that Setup installs (for example PyTorch and Qt for Python) keep their own licences.

There is no separate third-party notices file in this release. `LICENSE` and `NOTICE` in the program folder are the notice files this release ships.

Third-party add-ons are provided by their authors, who are responsible for the licence and training data. Voxprint does not host or redistribute them.

Do not use the app to create content that is illegal or socially unacceptable; you are responsible for what you create.

**About VOXPrint AI Canvas** (**More → Settings**) shows the exact version and build of your installation.

![About VOXPrint AI Canvas.](screens/about-dialog.png)
