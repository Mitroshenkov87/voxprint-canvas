# Enhance

Z-Image-Turbo follows English prompts more closely than prompts in other languages. **Enhance** improves the prompt you typed into a concrete English description before the picture is made. An English prompt is improved in English. A prompt in another language is improved and written in English.

The rewrite names the subject, composition, lighting, lens, and style. It keeps every fact you stated. Ages stay as you wrote them. Adult markers stay as you wrote them. Quoted lettering stays in its original language and script, so a sign still reads the way you wrote it.

**Enhance**, beside the prompt, improves that text in the same field. Its tooltip is "Improve the prompt". The button shows a busy state while the rewrite runs. You can edit the result. **Revert** is an undo icon, enabled only after a rewrite is placed in the field. It puts back the exact text from before that enhancement, including when you have edited the field since. Ctrl+Z in the field undoes the last change there as well.

A rewrite that drops or changes a stated age is discarded. The field keeps the text you had, and a short note appears under the prompt. The same happens when quoted lettering is missing from the rewrite.

The same content check runs on the original prompt and on the rewrite. If either one is refused, nothing is generated.

The scenario writer is a separate text model of about 8 GB. It is downloaded the first time a scenario or an improved prompt needs it, then unloaded before the picture is generated. The Z-Image text encoder stays the picture model's encoder. On a 16 GB card the writer is gone before diffusion starts.

From the command line:

```text
voxprint-canvas generate --prompt "красная лодка" --out picture.avif --enhance --json
```

`--json` includes `prompt` (what you typed) and `enhanced_prompt` (the rewrite). `used_prompt` is the text that was sent to the picture. `fallback` is `age`, `quote`, or `empty` when the rewrite was discarded, and empty when the rewrite was used.
