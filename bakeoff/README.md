# Canvas model bake-off (A100 80GB)

Compares stock **Z-Image-Turbo** with **Qwen-Image-2512** for Canvas v1 on the rented
A100 80GB, inside a hard time budget (`--deadline-min 150`). Runs after Model Forge
parts A+B and before part D, so Forge Qwen FP8/INT4 builds are normally absent: those
arms are skipped cleanly and can be re-run alone later; their results merge into the
same summary.

## Arms (`--arms`)

| arm | model | weights | prompts | settings | cap (min) |
|---|---|---|---|---|---|
| `z_bf16` | Z-Image-Turbo `f332072` | stock BF16 (24 GB tier) | all 105 | 1024x1024, 9 steps (8 NFE), CFG 0 | 10 |
| `z_fp8` | Z-Image-Turbo | Forge part A FP8 (`--z-fp8`, `--z-encoder-fp8`; default: search `~/forge-out`). Quality gate rejected it (LPIPS 0.14–0.22 vs BF16); the 16 GB tier is `z_bf16` | all 105 | same | 10 |
| `q_bf16` | Qwen-Image-2512 `25468b9` | native BF16, fully on GPU (main Qwen arm) | all 105 | 1328x1328, 50 steps, true CFG 4.0, official negative | 80 |
| `q_tuned` | Qwen-Image-2512 | native BF16 | first 40 (`--subset-n`) | true CFG 4.5, official negative + English anatomy terms | 28 |
| `q_fp8` | Qwen-Image-2512 | Forge part D FP8 (`--q-fp8-dir`, default `~/forge-out/canvas/qwen-image-2512/fp8`). Quality gate is still pending | all 105 | as `q_bf16` | 80 |
| `q_int4` | Qwen-Image-2512 | official SVDQuant INT4 via Nunchaku (`--q-int4-dir`) | all 105 | as `q_bf16` | 60 |

FP8 files use the Forge `voxprint-fp8-v1` format (float8_e4m3fn + `<name>.scale_weight`)
and are dequantized to BF16 on the A100 (no FP8 tensor cores), so FP8 arms measure
tier *quality*, not consumer speed. A100 seconds and VRAM are not 4090/4080 numbers.

## Prompt sets (seed 0)

`--prompt-set` selects the files. The default is `v1`. Do not edit `prompts/*.jsonl`
while a run is using them.

- `v1` (`prompts/`): `general.jsonl` 40, `text.jsonl` 20 EN + 15 RU, `adult.jsonl` 30.
  The original set. Adult subjects state ages from 30 to 60.
- `v2` (`prompts/v2/`): same three files and the same counts. Text prompts match v1
  (ids prefixed with `v2`; RU stays JSON `\u` escapes). Person prompts in general and
  adult use a main band of 20-30: mostly explicit ages 25-30, some 20-24, and a few
  older subjects for range. Every adult prompt states an explicit age.
- `fp_young_adults` (`prompts/v2/fp_young_adults.jsonl`): measurement only, not part of
  v1 or v2. Non-explicit and explicit prompts of adults who state an age from 20 to 25.
  The summary reports the age gate's false-block rate per age and per content level.
  It does not change the gate.
- `showcase` (`prompts/v2/showcase.jsonl`): the 15-prompt review set. It is not mixed
  into the v2 metrics files. See Lean testing below.
- Every renderable prompt is linted: no age-ambiguous or youth term (girl, boy, teen,
  young, petite, schoolgirl, student, and the same idea in Russian), and no stated age
  under 20.
- Prompts are interleaved round-robin across categories, so a time-capped arm still
  covers every category evenly. On v1 and v2 the 40-prompt subset is 10/10/10/10.

## Lean testing

Four rules for every bake-off question (`showcase`, `metrics` for v1 and v2,
`gate-measurement` for `fp_young_adults`).

1. **Look at a small set at full size.** `prompts/v2/showcase.jsonl` has 15 hard
   prompts: 3 English text, 3 Russian text, 3 multi-subject scenes, 2 hand studies,
   2 strict-adherence scenes, and 2 portraits. Every stated person age in the file
   is 25-30, including both portraits. The file has no adult category and no
   explicit prompts. Each image uses that model's catalog `native` frame for the
   prompt's aspect (WebP quality 95, no resize). Those frames differ between
   Z-Image-Turbo and Qwen-Image-2512. The shared catalog `twok` sizes are the 4K
   Enlarge source, so the showcase leaves them alone. Metrics arms stay on the
   square native frames, 1024 and 1328.

   | aspect | Z-Image-Turbo | Qwen-Image-2512 |
   |---|---|---|
   | 1:1 | 1024×1024 | 1328×1328 |
   | 16:9 | 1280×720 | 1664×928 |
   | 9:16 | 720×1280 | 928×1664 |
   | 4:3 | 1152×864 | 1472×1104 |
   | 3:4 | 864×1152 | 1104×1472 |
   | 3:2 | 1248×832 | 1584×1056 |
   | 2:3 | 832×1248 | 1056×1584 |
   | 21:9 | 1344×576 | 2016×864 |
   | 9:21 | 576×1344 | 864×2016 |

   Files are `{id}_s{seed}_{width}x{height}.webp`. A showcase run with the default
   `--arms` list keeps `z_bf16` and `q_bf16`. Name any extra arm on `--arms` when
   you want it. Scoring and the blind review pack are skipped, so the GPU time
   goes to the review images. `--score-only` still scores when you ask for it.
   After a real run, `results/showcase/index.html` is one file with no external
   assets: one full-size image per page, Prev/Next buttons, and Left/Right arrow
   keys. The image keeps its pixel size (`max-width: none`; scroll to see it).
   The label on each page is
   `model {name} · arm {arm} · prompt {id} · seed {n} · size {w}×{h}`.

2. **Keep the large set for metrics.** v2 (`general.jsonl`, `text.jsonl`,
   `adult.jsonl`, 105 prompts) stays the automated metrics set. Showcase ids are
   not added to those files. v1 is unchanged.

3. **Run the cheap checks before any GPU work.** Prompt lint, the dry-run plan,
   and the plan schema all run before `results/` is created and before a model
   loads. The log line is `checks ok: prompt-lint, dry-run, schema`. A failure
   exits 2. `--dry-run` exits 0 after that line and writes no images.

4. **Size the run with `--budget-min`.** `0` (the default) keeps every selected
   arm and the `--deadline-min` clock (150 unless you change it). A positive
   value replaces that clock and keeps the arms plus a category-balanced prompt
   slice that fit. Arms that do not fit are stored as `skipped` with reason
   `outside the budget`. Planning estimates, used only to size the run: Z-Image
   3 s at 1024×1024, Qwen 50 s at 1328×1328, then scaled by pixels; a family load
   is 180 s for Z and 240 s for Qwen; FP8 adds 60 s and INT4 adds 180 s. The same
   family and quant does not reload. Showcase scoring reserve is 0. A 25-minute
   showcase budget fits both full arms (estimate 19.44 min). A 12-minute budget
   keeps all 15 Z-Image prompts and 5 Qwen prompts, still spread across categories.

```bash
python bakeoff/run_bakeoff.py --prompt-set showcase --budget-min 25 --dry-run
python bakeoff/run_bakeoff.py --prompt-set showcase --budget-min 25
python bakeoff/run_bakeoff.py --prompt-set v2 --dry-run
```

## Time math (150 min)

Estimates (A100 BF16): Z-Image-Turbo 1024px 8 NFE ~3 s/img; Qwen-Image-2512 1328px
50 steps true CFG ~45-55 s/img; model load 2-4 min each.

| block | images | min |
|---|---|---|
| z_bf16 load + 105 x ~3 s | 105 | ~8 |
| z_fp8 apply + 105 x ~3 s | 105 | ~7 |
| q_bf16 load + 105 x ~45 s | 105 (fewer if slower: capped at 80 min) | ~80 |
| q_tuned 40 x ~40-50 s | 40 (capped at 28 min) | ~28 |
| scoring (SigLIP2, NSFW, age, OCR) ~355 images x ~1.5 s + loads | | ~12 |
| slack | | ~15 |
| **total** | | **~150** |

Each arm stops when its cap or the global deadline (minus `--score-reserve-min 15`)
would be exceeded by the next image; the summary records `partial` and the counts.
If Qwen runs at ~55 s/img, `q_bf16` covers ~85 of 105 prompts (balanced across
categories). Pair comparisons always use only the common prompts/seeds.

## Run

```bash
pip install -r bakeoff/requirements.txt          # inside the Forge venv
python bakeoff/run_bakeoff.py --dry-run           # print plan (v1, 105 prompts)
python bakeoff/run_bakeoff.py --prompt-set v2 --dry-run
python bakeoff/run_bakeoff.py --prompt-set fp_young_adults --arms z_bf16 --deadline-min 20
python bakeoff/run_bakeoff.py --deadline-min 150 \
  --z-fp8 ~/forge-out/<part-A>/zimage-turbo-transformer-fp8.safetensors \
  --z-encoder-fp8 ~/forge-out/<part-A>/qwen3-4b-text-encoder-fp8.safetensors
# after part D: tier arms only, merged into the same results/
python bakeoff/run_bakeoff.py --arms q_fp8,q_int4 --deadline-min 120 --review ''
python bakeoff/run_bakeoff.py --score-only        # finish scoring if the deadline cut it
```

Re-runs skip images that already exist (`--force` regenerates). `arms.json`,
`records.jsonl` and `scores.jsonl` accumulate; `summary.json`/`summary.md` are rebuilt
from all of them each run.

## Metrics

- **Adherence**: SigLIP2 so400m (`google/siglip2-so400m-patch14-384`, Apache-2.0) cosine x100, non-adult prompts.
- **Text rendering**: EasyOCR (Apache-2.0; `en` and `ru+en` readers). `ocr_exact` = normalized target
  is a substring of OCR text (case-folded, punctuation stripped, Yo->Ye); `ocr_cer` = semi-global edit distance / target length.
- **Adult permissiveness**: share of adult prompts with Falconsai NSFW (`Falconsai/nsfw_image_detection`, Apache-2.0) >= 0.5 **and** passing the age gate.
- **Gate (approximation of the Canvas output gate)**: OpenCV Haar faces + FairFace age
  (`dima806/fairface_age_image_detection`, Apache-2.0); block if any face has P(age<20) > 0.10
  (whole image when no face). Reported as false-block rate on adult prompts and block rate on the rest.
  `--prompt-set fp_young_adults` also reports that false-block rate per stated age (20-25) and per
  content level (`non_explicit`, `explicit`). Thresholds stay as they are.
- **Speed / VRAM**: seconds per image (encode+denoise+decode, synchronized), `torch.cuda.max_memory_allocated` per arm.
- **Blind review**: `results/review_z_bf16_vs_q_bf16/` pairs (general + text only; adult and
  young-adult measurement images never leave `results/images/`), answer key in a separate file.

## Output

`results/summary.json` (schema `voxprint-canvas-bakeoff/1`):

```
{schema, generated_utc, models{z,q}{repo,revision}, scorers{}, gate_rule, prompt_set, question, budget_min, prompt_sets{}, subset_n,
 arms{<arm>: {status: done|partial|skipped|not-run, reason?, quant, load_s, peak_vram_gb, gpu, settings{}, notes[],
              metrics{images, by_category{}, seconds_per_image, clip_t, clip_t_adult,
                      ocr_exact_en, ocr_cer_en, ocr_exact_ru, ocr_cer_ru,
                      adult_nsfw_mean, adult_permissive_rate, adult_gate_false_block_rate, nonadult_gate_block_rate}}},
 pairs[{a, b, common_images, a_metrics, b_metrics, delta_b_minus_a}],
 tiers{24gb:{z,q,q_note}, 16gb:{z,q,q_note}}, notes[],
 fp_young_adults?{<arm>: {by_age, by_content_level, by_age_and_content_level}}}
```

`fp_young_adults` is present only for that prompt set. Each cell is
`{prompts, images, gate_false_block_rate}`. `summary.md` repeats the same tables.
Adult and young-adult measurement images stay in `results/images/` and are not copied
into the blind review pack.

`results/summary.md` is the human-readable table. `results/` is git-ignored (it holds adult images).

## Decision rule (from MODEL-DECISION.md s11)

Qwen-Image-2512 becomes a v1.1 "Quality" mode on 24 GB if it wins quality or text
rendering by a clear margin (Bradley-Terry p < 0.05 on the blind review, or OCR exact-match
+15 points) at <= 90 s/image on a 4090; Z-Image-Turbo stays the default on all tiers unless
Qwen also matches its adult permissiveness through the gate. 16 GB Qwen only if INT4 fits
under the cap at <= 150 s/image (consumer speeds come from the 4090 follow-up, not the A100).
