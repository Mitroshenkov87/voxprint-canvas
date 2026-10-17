# Photoshoot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a simple-mode Photoshoot («Фотосессия») that writes a scenario and batch-generates 6, 12, or 20 continuing pictures of one subject.

**Architecture:** The button opens a panel on the shared simple screen. Qwen3-4B, the Z-Image text encoder already used by Improve prompt, writes a JSON scenario. The app copies a fixed subject block into every shot, checks it, then generates the series one frame at a time. A hero frame plus moderate img2img is the consistency path. No new model pack, no InsightFace, no training, and no Qwen-Image-Edit.

**Tech Stack:** PySide6 shell on `ui/shell-migration`, `voxprint-common` simple screen (pin `f4e95cc`), diffusers 0.41.0, Z-Image-Turbo BF16 (`Tongyi-MAI/Z-Image-Turbo` `f332072`), Qwen3-4B causal LM via tied embeddings.

This document is the plan and the mockup. It does not ship the feature.

## Global Constraints

- Licence-clean components only. Apache-2.0 or MIT. No InsightFace. No training of any kind, and no LoRA training.
- Qwen-Image-Edit stays removed. Do not call it, re-enable it, or use it for consistency.
- The minors gate is unchanged and non-configurable. It covers the idea, the scenario, every shot prompt, any reference image, and every output frame.
- The Adult content switch only changes the prompt policy, as it does everywhere else.
- A reference photo of a real person is out of scope for v1.
- Simple mode is not a sparse form. Each Photoshoot screen shows 2–5 everyday preset rows. Fine technical parameters stay in Expert.
- New user-facing strings live in the Canvas i18n catalogs `canvas/ui/locales/en.json` and `canvas/ui/locales/ru.json`. Reused shell strings stay in `voxprint-common`.
- Z-Image weights stay BF16 on both tiers. The 16 GB FP8 generator is not a shipping path.

---

## 1. Simple-mode flow

Photoshoot sits in the action row, immediately after Improve prompt (`simple.improve`, «Улучшить запрос» / "Improve prompt"). The main prompt field stays. Photoshoot has its own idea field. Generate, Revert, size, and output stay visible. The shell's picture-count chips (`simple.count`, 1 / 2 / 4) hide while the panel is open, so the screen has one count.

The panel has three screens. Each one shows the same five everyday preset rows, which is the top of the 2–5 range:

| Row | Where | Choices | Default | Catalog |
| --- | --- | --- | --- | --- |
| Size | Shell | Square, Landscape, Portrait | Square | existing `simple.size` and its group labels |
| Output | Shell | Standard, 4K | Standard | existing `simple.output`, `simple.output.standard`, `simple.output.4k` |
| How many shots | Panel | 6, 12, 20 | 6 | `photoshoot.count`, `photoshoot.count.6`, `.12`, `.20` |
| Style | Panel | Photo, Cinematic, Portrait | Photo | existing `simple.style`, `simple.style.photo`, `.cinematic`, `.portrait` |
| Place | Panel | Same place, Different places | Different places | `photoshoot.place`, `.same`, `.different` |

4K uses the existing Enlarge rule. A series does not invent a second output control. The shell picture-count chips stay hidden, so 6 / 12 / 20 is the only count.

The other looks (illustration, anime, 3D, painting, sketch) stay on the main style row when Photoshoot is closed. While the panel is open, the style row is these three everyday looks, so there is one style row and it still fits. They write `canvas.style` through the same path as the main row.

### Idea

«Фотосессия» opens the panel on the idea screen. The idea is one plain field, separate from the prompt. The five rows above are already filled. «Составить сценарий» sends the idea, the shot count, the style, and the place choice to the scenario model. The button shows «Составляем сценарий…» and is busy until the list appears. Cancel (`simple.cancel`) is visible during that wait and discards the reply.

### Shots

The next screen is an editable list of one line per shot. The line count starts at the selected chip (6, 12, or 20). Each line can be edited, moved up, moved down, deleted, or a new line can be added. Add is disabled at 20. Delete is disabled at 6. «Создать серию» is disabled outside 6–20, and the hint `photoshoot.range` shows.

Changing 6 / 12 / 20 sets the length. Growing asks the model only for the new lines and appends them. Shrinking drops lines from the end. Style, place, and size stay visible. Changing style or place does not rewrite the list by itself. «Обновить сценарий» appears and, when pressed, replaces the lines. It does not run unless the user presses it.

### Create and progress

«Создать серию» starts the run. The progress screen keeps the same five rows, frozen, so the choices stay readable. A grid fills one cell at a time. The line is «Кадр {current} из {total}». «Отмена» stays visible. Cancel stops before the next frame. The frame already in flight is allowed to finish and is kept if the gate allows it. Frames already saved stay in the series folder.

These actions stay visible on the progress screen, not behind Expert:

- Cancel (`simple.cancel`)
- Open the series folder (`photoshoot.folder`)
- Contact sheet (`photoshoot.export.sheet`)
- ZIP (`photoshoot.export.zip`)

The sheet and the ZIP include whatever has been kept. They are enabled after the first kept frame, not only at the end.

### Results

Kept frames are history items and files in their own folder:

`state/canvas/pictures/series/<series_id>/`

Files are `01.avif` (or the chosen type) and `series.json` (idea, subject block, place mode, style, lines, seeds, blocked indexes). History items gain `series_id` and `series_index` when the shared history writer accepts extra fields. Until that hook exists in `voxprint-common`, `series.json` is the grouping, and the strip still lists the frames. Open folder on any item in the series opens that folder, not the flat pictures directory. The strip label for the group uses `photoshoot.series.history`.

A blocked frame has no image file. Its grid cell and its history slot use `photoshoot.frame.blocked`.

### Strings

New keys, both catalogs. `{current}` and `{total}` are the only placeholders.

| Key | English | Russian |
| --- | --- | --- |
| `photoshoot.button` | Photoshoot | Фотосессия |
| `photoshoot.tooltip` | A series of the same subject | Серия кадров одного предмета |
| `photoshoot.title` | Photoshoot | Фотосессия |
| `photoshoot.idea.label` | Idea | Идея |
| `photoshoot.idea.placeholder` | Describe the idea. The same subject, different shots. | Опишите идею. Один и тот же предмет или человек, разные кадры. |
| `photoshoot.idea.write` | Write the scenario | Составить сценарий |
| `photoshoot.idea.writing` | Writing the scenario… | Составляем сценарий… |
| `photoshoot.idea.update` | Update the scenario | Обновить сценарий |
| `photoshoot.count` | How many shots | Сколько кадров |
| `photoshoot.count.6` | 6 | 6 |
| `photoshoot.count.12` | 12 | 12 |
| `photoshoot.count.20` | 20 | 20 |
| `photoshoot.place` | Place | Место |
| `photoshoot.place.same` | Same place | Одно место |
| `photoshoot.place.different` | Different places | Разные места |
| `photoshoot.shots.hint` | Each line can be edited, moved, deleted, or added. | Каждую строку можно изменить, передвинуть, удалить или добавить. |
| `photoshoot.shots.add` | Add a shot | Добавить кадр |
| `photoshoot.shots.up` | Up | Выше |
| `photoshoot.shots.down` | Down | Ниже |
| `photoshoot.shots.create` | Create the series | Создать серию |
| `photoshoot.range` | A series needs 6 to 20 shots. | Нужно от 6 до 20 кадров. |
| `photoshoot.progress.frame` | Shot {current} of {total} | Кадр {current} из {total} |
| `photoshoot.frame.blocked` | Shot blocked | Кадр заблокирован |
| `photoshoot.frame.waiting` | Waiting | Ждёт |
| `photoshoot.progress.done` | Series ready | Серия готова |
| `photoshoot.scenario.stopped` | The scenario cannot be used. The series has stopped. | Сценарий нельзя использовать. Серия остановлена. |
| `photoshoot.subject.missing` | This line is missing the subject. This shot will be skipped. | В строке нет описания предмета. Этот кадр будет пропущен. |
| `photoshoot.shot.dropped` | This shot was left out of the scenario. | Этот кадр не вошёл в сценарий. |
| `photoshoot.folder` | Open the series folder | Открыть папку серии |
| `photoshoot.export.sheet` | Contact sheet | Лист контактов |
| `photoshoot.export.zip` | ZIP archive | Архив ZIP |
| `photoshoot.series.history` | Series | Серия |
| `photoshoot.expert.seed` | Seed | Зерно |
| `photoshoot.expert.seed.series` | In order | По порядку |
| `photoshoot.expert.seed.locked` | Same seed | Одно и то же |
| `photoshoot.expert.seed.random` | Random | Случайное |
| `photoshoot.expert.strength` | How close | Насколько похоже |
| `photoshoot.expert.strength.low` | Looser | Слабее |
| `photoshoot.expert.strength.medium` | Usual | Обычно |
| `photoshoot.expert.strength.high` | Closer | Ближе |
| `photoshoot.expert.aspect` | Shot shape | Формат кадра |
| `photoshoot.expert.style_override` | Style for this shot | Свой стиль кадра |

Reused, not duplicated: `simple.improve`, `simple.revert` (Canvas override «Вернуть мой запрос» / "Revert to my prompt"), `simple.generate`, `simple.cancel`, `simple.history.delete`, `simple.style`, `simple.style.photo`, `simple.style.cinematic`, `simple.style.portrait`, `simple.size`, `simple.size.square`, `simple.size.landscape`, `simple.size.portrait`, `simple.output`, `simple.output.standard`, `simple.output.4k`, `canvas.stage.draw`, `gate.blocked`, `gate.unavailable`.

Row delete uses `simple.history.delete` («Удалить» / "Delete").

---

## 2. Scenario LLM

Reuse the Improve prompt model. `canvas/pipeline/enhance.py` already loads the Z-Image text encoder as a causal LM. `causal_lm_ready` accepts tied embeddings: `tie_word_embeddings` plus `embed_tokens.weight`, which is how this checkpoint is packed. `QwenRewriter` loads `AutoModelForCausalLM` from the snapshot's `text_encoder` directory and unloads it before diffusion. The bytes are the encoder files inside `Tongyi-MAI/Z-Image-Turbo` revision `f332072aa78be7aecdf3ee76d5c247082da564a6`. The catalog licence on both tiers is Apache-2.0. Upstream Qwen3-4B is Apache-2.0. There is no second download and no extra pack.

`open_rewriter` today caps `max_new_tokens` at 320. A scenario needs its own call with `max_new_tokens` 3072, `do_sample` false, and `enable_thinking` false, then `close()` before any frame. On a 16 GB card the encoder is not resident during denoise. That is the same rule as Enhance.

The shot lines are English around a subject block copied from the idea. Z-Image follows English more closely (`docs/enhance.md`). The subject block is not translated.

### System prompt

```text
You write a photo-shoot scenario as JSON for a text-to-image model.
The user message is JSON with idea, count, style, and place_mode.

Return one JSON object and nothing else.
{
  "subject_block": string,
  "descriptors": [string, ...],
  "place": string,
  "shots": [{"scene": string, "pose": string, "camera": string, "place": string}, ...]
}

Rules:
- subject_block is copied character for character from idea. Do not translate it. Do not shorten it by dropping a stated age or a subject descriptor.
- descriptors are short phrases copied character for character from subject_block. One phrase is not enough if the idea names more than one identifying fact. Include every stated age phrase.
- shots has exactly count objects.
- scene, pose, and camera are natural English.
- Do not add people who are not in the idea. Do not add youthful descriptors.
- Do not invent, drop, round, or change a stated age. Do not soften or rewrite adult content.
- Text inside quotes is lettering. Copy each quoted string exactly, in its original language and script.
- If place_mode is "same", place is one location, and every shot.place equals that same string.
- If place_mode is "different", every shot.place is a different location, and place is an empty string.
- style "photo" means a plain photograph. "cinematic" means a film still. "portrait" means a portrait with shallow depth of field. Put that only in the wording of scene or camera, not in subject_block.
```

The app builds each editable line. The model does not return the line.

```text
{subject_block}, {place_phrase}, {scene}, {pose}, {camera}
```

`place_phrase` is `place` when the mode is same, and `shot.place` when the mode is different. The selected style suffix is applied later by the existing look composer (`Look.suffix` in `canvas/ui/styles.py`), not by a second copy of the style words inside the subject block.

### Validation

Parse a single JSON object. Strip one optional markdown fence. Reject a truncated object.

Subject block:

- `subject_block` is a non-empty exact substring of the idea after NFKC. Case and script stay as written.
- `stated_ages(subject_block)` equals `stated_ages(idea)` as a multiset. `stated_ages` is the reader in `canvas/pipeline/enhance.py`.
- Every `descriptors` entry is a non-empty exact substring of `subject_block`.
- Quoted spans from the idea (`quoted_spans`) that sit inside the subject block are still inside every assembled line.

Shots:

- The array length equals the requested count.
- Each assembled line contains `subject_block` as an exact substring.
- Same place: one non-empty `place`, and every `shot.place` equals it.
- Different places: every `shot.place` is non-empty and unique.
- Scene strings are unique. A repeated scene is a failed shot, not a failed series.

If the subject block, the ages, or the place mode fail, discard the whole reply and call the model once more with the same system prompt. Do not add a hint that mentions the gate or asks for milder wording. If the second reply still fails, stop the series and show `photoshoot.scenario.stopped`. Do not generate.

If one shot fails the verbatim check or repeats a scene, discard that shot and regenerate it, up to two more times, with a user message that contains only the subject block, the place mode, and "write one different shot". If it still fails, drop that shot, show `photoshoot.shot.dropped` on the row, and do not backfill it with a reworded prompt. «Создать серию» stays disabled while fewer than 6 lines remain.

User edits are kept as typed. Before a line is sent to the generator, the subject block must still be an exact substring. If the user removed it, skip that line, show `photoshoot.subject.missing`, and do not put the block back.

Adult markers follow the Enhance rule: keep them verbatim, and do not soften them. The same check that Enhance uses for a dropped or changed stated age applies to the subject block.

---

## 3. Subject consistency

The series is one subject in varied scenes and poses. These are the realistic options in this stack.

### Fixed subject block

The block from section 2 is copied into every prompt. This is required. It holds a distinctive object (a yellow teapot with a blue handle) better than it holds a face. Faces drift as soon as the scene and the pose change. Level if used alone: low for people, moderate for a distinctive object.

### Seed

Z-Image-Turbo is an 8-step distilled model, guidance 0. The same seed repeats a picture only when the prompt is the same. `EmbeddingCache` in `canvas/pipeline/bf16_schedule.py` keys on the prompt, not the seed. A shared seed does not lock identity across different shot lines. Use seeds for reproducibility: simple mode uses `series` (hero seed S, shot i uses S+i). Expert can choose locked or random. Locked is still not identity. Do not describe it as likeness.

### Hero frame, then img2img

Generate shot 1 as text-to-image. That kept frame is the hero. Later shots use `ZImageImg2ImgPipeline`, which is already on both tier plans (`canvas/models/runtime.py` calls it for restyle at strength 0.6).

In diffusers, strength is how much of the hero is noised away. Lower strength stays closer to the hero and also keeps its scene. Higher strength lets the new scene in and loses the face.

| Expert label | Strength | What to expect |
| --- | --- | --- |
| Closer | 0.30 | Scene barely changes. Identity is higher. |
| Usual | 0.45 | The simple-mode default. Scene can change. Face is only a family resemblance. |
| Looser | 0.65 | Scene changes more. Identity is weak. Slightly looser than today's restyle strength of 0.6. |

Simple mode always uses 0.45 and does not show the slider.

If the hero output is blocked, do not use it as an image prompt. The next unblocked shot is tried as a text-to-image hero. If none is kept, the rest of the series is text-to-image with the subject block only, and the progress line says the reference frame was blocked. Prompts are not rewritten.

### Structure guidance

Pose, depth, and edge already exist through `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` (Apache-2.0, both the lite and the full weights, quality gate passed). They copy structure from a guide image. Copying the hero's pose or edges fights "varied poses". Do not use them as the default consistency path. v1 does not add an annotator. OpenPose's licence is not the Apache/MIT bar this app uses, and a new annotator is out of scope.

### Identity or reference adapters

None that we can ship.

Checked on 17 October 2026:

- Tencent IP-Adapter (`tencent-ailab/IP-Adapter`) is Apache-2.0. The published weights target Stable Diffusion 1.5 and SDXL UNets, not Z-Image's S3-DiT. They do not load on `ZImagePipeline`.
- IP-Adapter-FaceID, InstantID, and PuLID need InsightFace. InsightFace is non-commercial. They are out.
- Tongyi's public model zoo lists Z-Image-Turbo and Z-Image. It does not list an identity adapter. Z-Image-Edit is marked unreleased. It is not a stand-in, and Qwen-Image-Edit must not return.
- BoyoNodes (`DragonDiffusionbyBoyo/Boyonodes`, MIT licence file) has ComfyUI nodes that patch Z-Image with converted SD3 IP-Adapter weights. The InvokeAI request that points at those nodes (issue 8743, January 2026) records the author's own note: conversion results were limited, and a Z-Image IP-Adapter was still being trained. Later comments through March 2026 have no published checkpoint. The MIT licence covers the node source. It does not make an unpublished SD3 weight conversion a pinned Apache Z-Image adapter. Do not ship it.

### Recommendation

Use the subject block on every shot, simple-mode seed `series`, and a hero frame followed by img2img at strength 0.45.

Expected consistency, stated plainly: a distinctive object often keeps its color and shape. A person keeps clothes and a rough likeness. The face will change between shots. This is not a synthetic identity dataset. It will not pass a face-matching test, and it should not be described as one.

---

## 4. Time and VRAM

Frames run one at a time. Peak memory is one frame, not the series. The gate and Cancel need that checkpoint anyway.

### What the repo actually measures

`bakeoff/README.md` planning estimate, used only to size a run: Z-Image-Turbo about 3 seconds per image at 1024×1024, 8 network evaluations, on an A100, BF16, weights on the GPU. The same page says A100 seconds and VRAM are not 4090 or 4080 numbers, and that the FP8 arm is dequantized to BF16 on the A100, so it does not measure FP8 speed. `results/` is gitignored. There is no measured consumer seconds-per-image in the tree.

The shipping 16 GB tier is BF16, not FP8. Catalog id `z-image-turbo-bf16-16gb` has `"quant": "bf16"` and `"quality_gate": "passed"`. The bakeoff FP8 arm was rejected (LPIPS 0.14–0.22 against BF16). The 16 GB product path group-offloads BF16. There is no 16 GB FP8 time and no 16 GB FP8 VRAM figure to put in this table.

### VRAM, from the code

| Stage | 16 GB BF16, shipping | 24 GB BF16, shipping | 16 GB FP8 |
| --- | --- | --- | --- |
| Text encoder | 8.0 GB (`ENCODER_GB`), then freed. CPU if the budget is short. | Loaded with the pipeline. `text_encoder_offload` is empty. `vram_hint_gb` is 16. | Not shipping |
| Transformer | 12.3 GB stays in system RAM. One 1.6 GB block group is on the GPU (`GROUP_BLOCK_GB`). | 12.3 GB resident | Not shipping |
| Planning peak | 14.5 GB at 2048×1152 and 2048×2048 (`docs/ARCHITECTURE.md`) | No separate 1024 peak is published. The hint is 16 GB of weights-class use. | None |
| ControlNet | Not loaded for this feature | Not loaded for this feature | None |

A different prompt on every shot misses `EmbeddingCache`. On 16 GB the schedule loads the 8 GB encoder and frees it before every denoise. The scenario is one extra encoder session before the first frame, then the encoder is unloaded.

### Time

Diffusion only, 1024×1024, 8 steps, guidance 0. The 24 GB column is the A100 planning estimate of about 3 seconds per image. It excludes encoder load, the scenario, the VAE, and the safety classifiers. The 16 GB column is not estimated: group offload plus a per-shot encoder load has no seconds figure in the repo, and multiplying 3 seconds would be a guess.

| Frames | 24 GB BF16 | 16 GB BF16 | 16 GB FP8 |
| --- | --- | --- | --- |
| 6 | about 18 s of diffusion, A100 planning estimate | not measured | not a shipping path |
| 12 | about 36 s, same estimate | not measured | not a shipping path |
| 20 | about 60 s, same estimate | not measured | not a shipping path |

Img2img at strength 0.45 runs fewer than 8 denoising steps. That speedup is not timed here. Do not subtract it from the table.

The scenario call is one completion of up to 3072 new tokens on the 8 GB encoder. Enhance's 320-token cap is the only generation length in the code. The longer call is not timed. Count it as one encoder session before the table, on both tiers.

Default the chip to 6 because 12 and 20 multiply an unmeasured 16 GB frame. 12 and 20 stay one click away.

---

## 5. Safety

The minors gate does not gain a setting. `AgeGate.review_prompt` and `review_image` run as they do now. Nothing in the series reads a flag that weakens them.

Order:

1. Gate the idea text. A block stops the series before the language model runs. Show `gate.blocked`. Do not rewrite the idea.
2. Gate the scenario text (subject block, places, and every assembled line) as one prompt. A block stops the series. Do not call the model again. A validation retry from section 2 happens only when the JSON is well formed and the gate has not blocked. The retry prompt must not mention the gate.
3. On Create, gate each shot line. A blocked line is dropped, shown as `photoshoot.frame.blocked`, and the run continues. The line is not reworded.
4. The hero, and every later frame, is an output image. Gate it before it is kept, with the shot line as the stated-age support, the same way a single picture is gated. A blocked frame is not written and is not used as the next hero. The cell shows blocked. The run continues.
5. v1 has no user reference photo. If a later version adds one, that image is an input and goes through `review_image` before any shot. A block stops the series.

The Adult content switch stays the Expert prompt policy (`layout.prompt_policy()` in `canvas/ui/shell_layout.py`). It adds the audience template and negatives. It does not change the gate. Each shot is composed the same way as `composed_prompt`.

A reference photo of a real person is out of scope for v1. **NEEDS OWNER DECISION:** consent and likeness. The idea field is text. "Use a picture" is not an input to Photoshoot in this version.

---

## 6. Expert extras

Expert keeps the fine controls. The everyday ones moved to the simple rows in section 1 because a simple screen needs 2–5 preset rows and those choices are plain words.

Stays in Expert:

| Control | Values | Notes |
| --- | --- | --- |
| Seed | In order, Same seed, Random | Default In order. Same seed does not lock identity. See section 3. |
| How close | Looser 0.65, Usual 0.45, Closer 0.30 | Img2img strength. Hidden in simple mode, which uses 0.45. |
| Shot shape | Any catalog aspect | Default is the simple size row. One row can override it. |
| Style for this shot | Any of the three everyday looks, or the series default | Overrides `canvas.style` for that line only. |

Moved out of Expert, on every Photoshoot screen:

- How many shots: 6 / 12 / 20, plus add and delete inside 6–20
- Style: Photo / Cinematic / Portrait
- Place: Same place / Different places
- Contact sheet and ZIP, on the progress screen

The per-shot style override is the expert form of style. The series-wide style is the simple chip.

---

## 7. Effort and recommendation

Ship this after the first release. Do not put it in the current shell migration.

The minimal version is not cheap enough to pull forward. The shell is still mid-migration (`pending Yitro` in `docs/ui/FEATURE-MAP.md` for the image surface and the live expert settings). History has no series fields. The safety wiring has to be right before a batch of pictures of a person can run. The expensive parts are the scenario validator, the batch loop with Cancel, the series folder, and the gate on every text and every frame. The five preset rows are part of that minimal screen, not a later polish, because a sparse form would fail the suite rule.

### Minimal version (first post-release slice)

- The button, the idea field, the five preset rows, the editable list, Create, and Cancel
- Scenario JSON, verbatim subject block, place-mode check, and the gate order in section 5
- Text-to-image only, with the subject block on every line. No hero frame and no img2img
- Series folder, `series.json`, and history items. Strip grouping waits on `voxprint-common` if the writer cannot store `series_id`
- Contact sheet and ZIP of the kept frames
- No Expert seed, strength, per-shot shape, or per-shot style override

Consistency of the minimal version is the subject block alone: low for faces, moderate for a distinctive object. The UI must not call it a matched identity.

### Second slice

Hero frame plus img2img at 0.45, then the Expert rows in section 6. Still no adapter, no InsightFace, no training, and no Qwen-Image-Edit.

### Later files

- Create `canvas/pipeline/photoshoot.py` for the prompt, the JSON types, and the validator
- Create `canvas/pipeline/series.py` for the one-frame loop, Cancel, and the folder
- Create `canvas/ui/photoshoot_panel.py` for the three screens
- Modify `canvas/ui/shell_screen.py` to host the button and the panel
- Modify `canvas/ui/locales/en.json` and `ru.json` with the table in section 1
- Test `tests/test_photoshoot.py`: ages copied, a dropped age discards the shot, a blocked idea stops the series, a blocked frame does not rewrite the line, place mode, and the 6–20 bounds

---

## Mockups

Static Qt mock, 1280×800, offscreen, shared dark tokens from `voxprint-common` `tokens.json` (`voxprint-dark`). The pictures are geometric still lifes of a teapot. They are not people.

- `docs/plans/photoshoot/photoshoot-idea-ru.png`
- `docs/plans/photoshoot/photoshoot-idea-en.png`
- `docs/plans/photoshoot/photoshoot-shots-ru.png`
- `docs/plans/photoshoot/photoshoot-shots-en.png`
- `docs/plans/photoshoot/photoshoot-progress-ru.png`
- `docs/plans/photoshoot/photoshoot-progress-en.png`

The idea screen shows the idea field and «Составить сценарий» / "Write the scenario". The shots screen shows six editable lines and «Создать серию» / "Create the series". The progress screen shows the grid filling in, «Отмена» / "Cancel", and the folder, contact sheet, and ZIP actions. All three show size, output, shot count, style, and place.
