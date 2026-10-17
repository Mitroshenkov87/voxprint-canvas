# Shared resources

Suite apps share one folder and one manifest.

| Platform | Folder |
| --- | --- |
| Windows | `%LOCALAPPDATA%\Voxprint\shared` |
| Linux | `~/.local/share/voxprint/shared` |

On Linux, `XDG_DATA_HOME/voxprint/shared` is used when `XDG_DATA_HOME` is set. `VOXPRINT_HOME` replaces the Voxprint directory on either platform. The program itself is installed under `C:\Program Files\VOXPrint AI Suite\Canvas\` or `/opt/voxprint-ai-suite/canvas/`, not in this data folder.

The suite runtime is Python 3.14 at `shared/runtime/py314-torch2.11-cu130`. Every pinned dependency publishes a wheel that installs on Python 3.14, so Canvas uses that runtime. The runtime contains PyTorch 2.11.0 for CUDA 13.0, the matching torchvision 0.26.0, and the GPU ONNX Runtime 1.31.0. Setup resolves numpy, torchao, and the other packages from PyPI, and takes torch and torchvision from the CUDA 13.0 index. torchao is pinned with `==` and is not taken from that index. The CPU ONNX Runtime build is not installed into it, because that build would shadow the GPU build. TensorRT 10.16.1.11 is an optional component, `tensorrt-10`. Python 3.14 has no TensorRT wheel, so setup skips it. The self-test records that note and does not fail.

The setup installs anything that is missing before the first picture: the suite runtime when it is not already there, the CUDA 13.0 wheels, the default Z-Image-Turbo pack, and the models that ship with that pack. Progress is printed while those files are fetched. Wheels are cached in `shared/wheels` so another suite app can reuse them. A component that another app already refs is left in place. A different version is installed beside it, under a new path.

Optional packs stay out of the setup. ControlNet lite and Creative Z-Image download only when a picture needs them. The window shows the size and an estimated time first. Instruction editing is coming soon, so setup and first launch do not download Qwen-Image-Edit weights, the Nunchaku runtime, or an INT4 pack.

`voxprint-canvas --selftest` reads the GPU, the model files, and the output encoders. It does not download models or wheels, and its JSON report is the only text on stdout. A file whose path, size, and modification time match a record in the shared manifest is not hashed again. `--selftest --deep` hashes every file and refreshes that record. Files that have not been downloaded are reported as not downloaded, which is not a failure. Instruction editing is reported as unavailable, which is not a failure. A hash that does not match, or a check that cannot load after its files are present, is a failure. Launch repairs a recorded file only when it is missing or corrupt. Progress for that repair is printed to stderr.

## Manifest

`shared/manifest.json` is schema 1. Another suite app can read and write the same file. On Linux the path is `~/.local/share/voxprint/shared/manifest.json`. On Windows it is `%LOCALAPPDATA%\Voxprint\shared\manifest.json`.

```json
{
  "schema": 1,
  "components": {
    "torch-2.11.0-cu130": {
      "kind": "wheels",
      "version": "2.11.0+cu130",
      "path": "runtime/py314-torch2.11-cu130",
      "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
      "installed_at": "2026-10-11T12:00:00+00:00",
      "refs": ["enlarge", "canvas"]
    }
  }
}
```

- The top-level keys Canvas writes are `schema` (the integer 1) and `components`. Any other top-level key is kept when Canvas writes the file.
- A component id is lowercase and matches `[a-z0-9][a-z0-9._-]*`. Examples: `python-3.14`, `torch-2.11.0-cu130`, `onnxruntime-gpu-1.31`, `tensorrt-10`, and `model-` plus a SHA-256 prefix.
- `kind` is `runtime`, `wheels`, `model`, or `tool`.
- `version` is the component version, such as `2.11.0+cu130`.
- `path` is relative to the shared directory. It must not be absolute and must not contain `..`.
- `sha256` is the hex digest of the archive or of the lock file.
- `installed_at` is an ISO-8601 timestamp with a numeric offset.
- `refs` lists the suite apps that use the component. Installing Canvas adds `canvas`. Uninstalling removes `canvas`.

A component is reused when the id, version, and sha256 all match. Otherwise Canvas installs beside it, under a new path, and does not overwrite a component another app still refs. The directory is deleted only when `refs` is empty and no other component still uses that path.

Writes are atomic. Canvas locks `manifest.lock` (30 seconds), writes `manifest.json.tmp`, flushes it, and renames it onto `manifest.json`.

An older Canvas file that has `version` and `resources` is read once, converted to components, and written back as schema 1. A pack becomes a model. A suite entry becomes a tool.
