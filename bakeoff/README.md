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
| `z_fp8` | Z-Image-Turbo | Forge part A FP8 (`--z-fp8`, `--z-encoder-fp8`; default: search `~/forge-out`) (16 GB tier) | all 105 | same | 10 |
| `q_bf16` | Qwen-Image-2512 `25468b9` | native BF16, fully on GPU (main Qwen arm) | all 105 | 1328x1328, 50 steps, true CFG 4.0, official negative | 80 |
| `q_tuned` | Qwen-Image-2512 | native BF16 | first 40 (`--subset-n`) | true CFG 4.5, official negative + English anatomy terms | 28 |
| `q_fp8` | Qwen-Image-2512 | Forge part D FP8 (`--q-fp8-dir`, default `~/forge-out/canvas/qwen-image-2512/fp8`) | all 105 | as `q_bf16` | 80 |
| `q_int4` | Qwen-Image-2512 | official SVDQuant INT4 via Nunchaku (`--q-int4-dir`) | all 105 | as `q_bf16` | 60 |

FP8 files use the Forge `voxprint-fp8-v1` format (float8_e4m3fn + `<name>.scale_weight`)
and are dequantized to BF16 on the A100 (no FP8 tensor cores), so FP8 arms measure
tier *quality*, not consumer speed. A100 seconds and VRAM are not 4090/4080 numbers.

## Prompt sets (`prompts/`, seed 0)

- `general.jsonl`: 40 (photo, painting, product, people explicitly adult).
- `text.jsonl`: 20 EN + 15 RU with exact target strings (RU stored as JSON `\u` escapes).
- `adult.jsonl`: 30 artistic/erotic prompts, every subject an explicitly stated adult age
  (30-60). No prompt in any set refers to minors.
- Prompts are interleaved round-robin across categories, so a time-capped arm still
  covers every category evenly and the 40-prompt subset is 10/10/10/10.

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
python bakeoff/run_bakeoff.py --dry-run           # print plan
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
- **Speed / VRAM**: seconds per image (encode+denoise+decode, synchronized), `torch.cuda.max_memory_allocated` per arm.
- **Blind review**: `results/review_z_bf16_vs_q_bf16/` pairs (general + text only; adult images never leave
  `results/images/`), answer key in a separate file.

## Output

`results/summary.json` (schema `voxprint-canvas-bakeoff/1`):

```
{schema, generated_utc, models{z,q}{repo,revision}, scorers{}, gate_rule, prompt_sets{}, subset_n,
 arms{<arm>: {status: done|partial|skipped|not-run, reason?, quant, load_s, peak_vram_gb, gpu, settings{}, notes[],
              metrics{images, by_category{}, seconds_per_image, clip_t, clip_t_adult,
                      ocr_exact_en, ocr_cer_en, ocr_exact_ru, ocr_cer_ru,
                      adult_nsfw_mean, adult_permissive_rate, adult_gate_false_block_rate, nonadult_gate_block_rate}}},
 pairs[{a, b, common_images, a_metrics, b_metrics, delta_b_minus_a}],
 tiers{24gb:{z,q,q_note}, 16gb:{z,q,q_note}}, notes[]}
```

`results/summary.md` is the human-readable table. `results/` is git-ignored (it holds adult images).

## Decision rule (from MODEL-DECISION.md s11)

Qwen-Image-2512 becomes a v1.1 "Quality" mode on 24 GB if it wins quality or text
rendering by a clear margin (Bradley-Terry p < 0.05 on the blind review, or OCR exact-match
+15 points) at <= 90 s/image on a 4090; Z-Image-Turbo stays the default on all tiers unless
Qwen also matches its adult permissiveness through the gate. 16 GB Qwen only if INT4 fits
under the cap at <= 150 s/image (consumer speeds come from the 4090 follow-up, not the A100).
