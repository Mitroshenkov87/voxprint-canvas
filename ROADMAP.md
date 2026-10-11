# Roadmap

The next version of Voxprint AI Canvas is listed below. Items under Later are plans, not promises.

## Next version

0.1 RC, "first release".

- Text to image, and image to image: restyle, inpaint and outpaint, and pose, depth, and edge guidance.
- Grouped aspect ratios.
- 4K output through Voxprint AI Enlarge.
- Automatic model choice from GPU memory: 16 GB, and 24 GB and above.
- Smart memory placement on 16 GB cards. Components load one at a time, so full-quality BF16 models fit.
- Modern output formats: AVIF, JPEG XL, and WebP.
- Opt-in third-party add-ons.
- Enhance and translate a prompt into English before generating, for the release candidate after 0.1.0-rc.3. The button reuses the Qwen3-4B text encoder that ships with Z-Image-Turbo. This change is not a release tag.

## Later

Plans, not promises.

- Instruction-based editing on 16 GB cards.
- Faster engines on your own GPU, with TensorRT.
- A full command-line interface with JSON output.
- More languages in the UI.
