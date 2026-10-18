# Screenshots for the user guide

Each id below is referenced from `en.md` and `ru.md` as `![caption](screens/<id>.png)`. Capture every id once per
language and save it as `docs/guide/screens/<lang>/<id>.png` (`en`, `ru`); the build maps `screens/<id>.png` to the
folder of the language it renders (see `BUILD-NOTES.md`).

## Rules for every Qt capture

- Use the offscreen harness pattern of `tools/grab_shell_screens.py`: `QT_QPA_PLATFORM=offscreen`, a temporary
  `VOXPRINT_HOME`, `tests.fakes.World`, `set_language(<lang>)` plus `LANG`/`LC_ALL` (`en_US.UTF-8` or `ru_RU.UTF-8`),
  window size 1280 × 800, `Tier.GB16` unless noted. No weights are loaded and nothing is generated.
- Finished pictures, history thumbnails and source pictures are synthetic or come from Proving Ground's set of
  clothed, everyday test images (landscapes, objects, street scenes). No other content in any screenshot.
- The prompt text shown in a screenshot is the example prompt from the same chapter, in the language of the capture.
- No personal paths: `VOXPRINT_HOME` points at a neutral folder such as `/home/user/.local/share/voxprint-ai-suite`.
- `docs/ui/screens/*.png` are older UI-review shots. Do not reuse them: `addons-*.png` there still shows the
  pre-rc.11 folder field, and `photoshoot-ru.png` has untranslated buttons.

## Not reproducible in Qt (capture on a Windows 11 VM)

| Id | UI state |
|---|---|
| `setup-pwsh-offer` | `VoxprintCanvasSetup-<version>.exe` on a Windows 11 24H2 VM without PowerShell 7, interactive run, Setup language = capture language. The Yes/No message box with the `PwshAsk` text is on top of the wizard. |
| `setup-pwsh-fallback` | Same VM; answer **No** to the offer. The information box with the `PwshFallback` text is on top of the wizard. |

## Qt captures

| Id | UI state to reproduce |
|---|---|
| `splash` | `canvas.ui.splash.show_splash()` with the stage set to `text("splash.gpu")` (Checking the GPU / Проверка видеокарты). Capture the splash card alone. |
| `main-window-simple` | `ShellWindow` in simple mode, Expert off, empty prompt, phase EMPTY ("The picture will show up here."), default chips (Balanced, Photo, Square 1:1, Standard, 1), a history strip of 8–12 synthetic thumbnails. Same as the harness shot `simple`. |
| `more-menu` | `main-window-simple` with the **More** menu (`QPushButton#overflow`) open, showing Picture setup, Add-ons, Check this PC, Open folder, Manage models, Settings, Expert mode. Render the menu with `menu.popup()` next to the button and grab the window including the popup. |
| `history-item-menu` | `first-image-result` with the context menu of the selected history thumbnail open (`SimpleWidget.history_menu(item_id)`, not the confirm step). |
| `first-image-prompt` | Simple mode, Expert off, the chapter 5 example prompt in `promptBox`, chips at default, phase EMPTY, focus on Generate. |
| `first-image-progress` | Same prompt, `layout.generate()` then `layout.progress(1, 1, text("canvas.stage.draw"))`, so the busy view shows "Creating the picture...", the stage, the percentage and Cancel. |
| `first-image-result` | Same prompt, a finished synthetic landscape picture selected (phase READY), action bar visible (More like this, Save, Open folder, Use prompt again, Use as input picture, Delete). Same as the harness shot `simple-result`. |
| `size-ratio-menu` | Simple mode with **Landscape** chosen and the ratio combo (`QComboBox#sizeRatio`) popup open, listing 16:9, 21:9, 3:2, 4:3. |
| `expert-panel` | Expert on (`window.set_expert(True)`), default values, no picture. Same as the harness shot `expert`. |
| `expert-custom` | Expert on, a finished picture shown, **Steps** changed to 33 so the Quality chip reads "Balanced + changes" and the panel shows "Reset to Balanced". Same as the harness shot `expert-custom`. |
| `output-4k-detail` | Simple mode with an Enlarge stub that reports `ready()` true, **4K** selected in Output size, so the Detail row (Light / Medium / Strong) is visible. |
| `picture-setup` | `window.setup_wizard()` with default values (From text, Usual, Restyle, AVIF). Grab the dialog over the window. Same as the harness shot `setup`. |
| `use-picture` | `window.picture_wizard()` with `load_path(<sample landscape>)`, so the preview shows the picture. Same as the harness shot `picture`. |
| `mask-editor` | `window.use_picture(<sample interior photo>)`, then `window.mask_wizard()`; paint one area (for example where an armchair stands) on the mask board before grabbing. |
| `enlarge-confirm` | `window.enlarge_confirm()` with the size known, so the text reads "4K needs VOXPrint Enlarge (≈ …). Install it?" with Install and Cancel. |
| `enlarge-progress` | `window.enlarge_progress()` after `report(done, total)` at about 40 %. |
| `photoshoot-on` | Simple mode, the chapter 11 product prompt ("A yellow kettle with a blue handle" / «Жёлтый чайник с синей ручкой») in the prompt box, the **Photoshoot** chip (`photoshootMode`) checked, phase EMPTY. |
| `photoshoot-running` | Same prompt, Photoshoot on, `photoshoot.show_scenario(...)` with 6 shot lines from a stub scenario writer (kitchen table, close view, three-quarter view, from above, from the side, by the window), `mark(1, "drawing")`, phase BUSY, so "Shot 2 of 6" and Cancel are visible. |
| `photoshoot-done` | After a stubbed 6-frame series: the series entry selected in the history strip, its first frame shown, action bar visible, Photoshoot still checked. |
| `addons-dialog` | A temporary models folder with one valid Model Forge LoRA of kind `style` (id `inkwash`, trigger `inkwash`) under `user-models/lora/inkwash/`, built the way `tests/` builds LoRA fixtures. `window.addons_wizard()` with that add-on selected and Strength at its default. No other entries. |
| `lora-trigger-chip` | Same fixture saved through the Add-ons window, simple mode, prompt "a lighthouse on a rocky coast at dusk", so the `loraTrigger` chip reading `inkwash` is visible next to Generate. Hover tooltip not needed. |
| `check-this-pc` | `window.diagnostics_wizard()` with `show_report(...)` filled from a stub report that passes ("This PC is ready.", System, Graphics, File types). Same as the harness shot `check`, plus the report text. |
| `manage-models` | The VOXPrint MDL window (`voxprint_common.ui.mdl_window`) in its Manage models view for `--app canvas`, with a stub queue: the default pack under Installed, ControlNet lite under Available for this computer. TODO(check): confirm the MDL window can be driven offscreen with a stub; otherwise capture it on the Windows VM. |
| `about-dialog` | `canvas.ui.about.AboutDialog(window)` with the BUILD.json of the release being documented. |
