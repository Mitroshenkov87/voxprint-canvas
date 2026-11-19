## Unreleased (0.1.0-rc.46)

- Prompt rule (owner rule 2026-11-16): a prompt is refused only when it describes someone under 18 AND hints at 18+ content. Removed: the school-context and stylised prompt refusals and the guard's non-consent markers. A stated adult age now overrides the soft word "teen". The 18+ lexicon covers undressing, underwear, sexual acts, sexualised poses, fetish terms (EN/RU/DE) and the Expert adult-direction wording; the 18+ switch counts as 18+.
- Input photos (owner rule 2026-11-16): one age check per uploaded photo, only for an 18+ edit, refusing only a minor. School and stylised photo cues no longer refuse. A `--count`/`--seeds`/`--prompts` run, a series and a batch check each photo once.
- Generation is prompt-check-only (owner decision 2026-11-16, supersedes the 2026-11-15 verified-LoRA rule). Every generation path (window, worker, `generate` including `--count`/`--seeds`/`--prompts`, Photoshoot, `batch`, `render_batch`, and the 4K Enlarge output) runs the prompt check and, when the request has uploaded images, the input-image checks. No classifier runs on generated pictures any more, with any LoRA or none. Generated pictures leave through `canvas.safety.gate.release_generated`.
- Generation warms only the prompt guard (`AgeGate.warm_prompt`). The image check models are no longer pre-loaded for generation or series; they load the first time an input image is screened. All six safety models stay required on disk.
- The prompt check, its thresholds and wording, the input-image checks and the fail-closed rules are unchanged.
- Forge-verified LoRA records are kept as metadata and no longer change the gate. `canvas.lora_verify.review_or_skip` is a deprecated alias of `release_generated`.
- The series timing stage `output_check` is now `release`, and the single-picture `render_output_check` timer is gone.

## Earlier

Removed Qwen-Image-Edit.
