# Shared resources

Suite apps share one folder and one manifest.

| Platform | Folder |
| --- | --- |
| Windows | `%LOCALAPPDATA%\Voxprint\shared` |
| Linux | `~/.local/share/voxprint/shared` |

On Linux, `XDG_DATA_HOME/voxprint/shared` is used when `XDG_DATA_HOME` is set. `VOXPRINT_HOME` replaces the Voxprint directory on either platform.

The setup installs anything that is missing before the first picture: a Python 3.12 interpreter when the machine does not have one, the CUDA PyTorch wheels, the default Z-Image-Turbo pack, and the models that ship with that pack. Progress is printed while those files are fetched. The wheels are cached in `shared/wheels` so another suite app can reuse them. The interpreter, when the suite had to install one, lives in `shared/python`.

Optional packs stay out of the setup. ControlNet lite, Creative Z-Image, and Qwen-Image-Edit-2511 download only when a picture needs them. The window shows the size and an estimated time first.

The first launch and `voxprint-canvas --selftest` install the default pack when it is not on disk yet. Files that have not been downloaded are reported as not downloaded, which is not a failure. A hash that does not match, or a check that cannot load after its files are present, is a failure. Every launch also reads the manifest and replaces a missing or corrupt recorded file. Progress is printed while a file is repaired.

## Manifest

`shared/manifest.json` is version 1. Another suite app can read and write the same file.

```json
{
  "version": 1,
  "resources": {
    "example-pack": {
      "kind": "pack",
      "path": "models/Example--Pack",
      "bytes": 128,
      "files": {
        "weights.bin": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
      },
      "apps": ["canvas", "other-app"],
      "refcount": 2,
      "source": ""
    }
  }
}
```

- `version` is `1`.
- `path` is relative to the shared directory. It must not be absolute and must not contain `..`.
- `bytes` is the total size of the resource.
- `files` maps a file name to its SHA-256 hex digest.
- `apps` lists the suite apps that use the resource. Registering an app that is already listed does not add it again.
- `refcount` is the number of names in `apps`.
- `kind` is `pack` for model files, `wheels` for the CUDA wheel cache, `runtime` for a suite-installed interpreter, or `suite` for an empty registration.
- `source` is optional. An empty string means the model registry knows where to fetch the pack. `pytorch-cu128` means the CUDA wheel index. `python` means the suite interpreter archive.

An app unregisters by removing its name. Files are deleted only when `apps` becomes empty and `refcount` drops to zero. A resource another app still uses is left in place.
