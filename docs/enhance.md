# Enhance and translate

Z-Image-Turbo follows English prompts more closely than prompts in other languages. **Enhance & translate prompt** rewrites the text you typed into a concrete English description before the picture is made.

The rewrite names the subject, composition, lighting, lens, and style. It keeps every fact you stated. Ages stay as you wrote them. Adult markers stay as you wrote them. Quoted lettering stays in its original language and script, so a sign still reads the way you wrote it.

The rewritten prompt appears in a box beside the original. **Use this prompt** sends that text to the picture. **Revert** goes back to the original. You can edit the rewrite before you use it.

A rewrite that drops or changes a stated age is discarded, and the original prompt is used. The same happens when quoted lettering is missing from the rewrite.

The same content check runs on the original prompt and on the rewrite. If either one is refused, nothing is generated.

The rewriter is the Qwen3-4B text encoder that already ships inside Z-Image-Turbo. Those weights are a causal language model with tied input and output embeddings, so there is no second download and no separate language-model pack. The encoder is loaded for the rewrite and unloaded before the picture is generated. On a 16 GB card that leaves the GPU free for the diffusion pipeline.

From the command line:

```text
voxprint-canvas generate --prompt "красная лодка" --out picture.avif --enhance --json
```

`--json` includes `prompt` (what you typed) and `enhanced_prompt` (the rewrite). `used_prompt` is the text that was sent to the picture. `fallback` is `age`, `quote`, or `empty` when the rewrite was discarded, and empty when the rewrite was used.
