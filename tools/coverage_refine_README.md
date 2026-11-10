# Region-refine compare

`tools/coverage_refine.py` renders one prompt and one fixed seed up to four ways:

| Variant | What it is |
|---|---|
| A | Baseline LoRA render at `--strength` |
| B | Same render, LoRA at `--strength` for the first `--early-steps`, then `--late-strength` |
| C | A, then a base-model inpaint (LoRA off) over the feathered, face-locked mask at `--denoise` |
| D | B, then the same base-model inpaint |

Any zone can be refined (hands, back, profile, feet, and so on). A weak angle has no box: variant B applies the late-strength schedule to the whole render. Face-lock is on by default and never repaints the face unless `--zone face`.

Default variants are **A,C,D** when `--refine-region` is set or a coverage document marks a region weak. Coverage is read from `<lora stem>.coverage.json` next to the LoRA if present, else `--coverage PATH`, else the embedded `voxprint.coverage` key. The document is `voxprint-coverage/1` schema_rev 3 (`bins.angles`, `bins.regions`, qualified `weak` ids). Rev 2 flat bins are migrated (`back` becomes `angles.back`). With no coverage, a drawn `--box` / `--mask` and the toggle are enough.

18+ OFF/ON (`--adult-content`) has the same meaning as in Canvas and selects `score_nude` or `score_clothed`. Every prompt, input image and output still goes through the content gate. The tool never downloads weights.

Default output folder: `~/.local/share/voxprint-ai-suite/state/canvas/pictures/<YYYYMMDD-HHMMSS>/` (`variant_*.png`, `compare.png`, `log.json`).

A one-line example:

    python tools/coverage_refine.py --prompt "<TRIGGER>, a 30-year-old adult woman, full body, studio light" \
      --lora /path/to/vxpkswswl-254c5409.safetensors --seed 4242 --adult-content on \
      --refine-region --zone hands --box X,Y,W,H --late-strength 0.3 --denoise 0.45 --device cuda:0
