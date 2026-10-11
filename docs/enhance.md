# Enhance and translate

Z-Image-Turbo follows English prompts more closely than prompts in other languages. **Enhance & translate prompt** rewrites the text you typed into a concrete English description before the picture is made.

The rewrite names the subject, composition, lighting, lens, and style. It keeps every fact you stated. Ages stay as you wrote them. Adult markers stay as you wrote them. Quoted lettering stays in its original language and script, so a sign still reads the way you wrote it.

**Enhance**, beside the prompt, rewrites that text in the same field. The button shows a busy state while the rewrite runs. You can edit the result. **Revert** is an undo icon, enabled only after a rewrite is placed in the field. It puts back the exact text from before that enhancement, including when you have edited the field since. Ctrl+Z in the field undoes the last change there as well.

A rewrite that drops or changes a stated age is discarded. The field keeps the text you had, and a short note appears under the prompt. The same happens when quoted lettering is missing from the rewrite.

The same content check runs on the original prompt and on the rewrite. If either one is refused, nothing is generated.

The rewriter is the Qwen3-4B text encoder that already ships inside Z-Image-Turbo. Those weights are a causal language model with tied input and output embeddings, so there is no second download and no separate language-model pack. The encoder is loaded for the rewrite and unloaded before the picture is generated. On a 16 GB card that leaves the GPU free for the diffusion pipeline.

From the command line:

```text
voxprint-canvas generate --prompt "красная лодка" --out picture.avif --enhance --json
```

`--json` includes `prompt` (what you typed) and `enhanced_prompt` (the rewrite). `used_prompt` is the text that was sent to the picture. `fallback` is `age`, `quote`, or `empty` when the rewrite was discarded, and empty when the rewrite was used.
