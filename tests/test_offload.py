"""CPU-only smoke: the 16 GB offload hooks are installed, with a sequential fallback."""
from __future__ import annotations

from canvas.pipeline.offload import OffloadInstall, install_low_vram_offload


class _Data:
    def __init__(self) -> None:
        self.pinned = False

    def pin_memory(self) -> _Data:
        self.pinned = True
        return self


class _Param:
    def __init__(self) -> None:
        self.data = _Data()


class _Block:
    def __init__(self) -> None:
        self.weight = _Param()

    def parameters(self):
        return [self.weight]


class _Transformer:
    def __init__(self) -> None:
        self.transformer_blocks = [_Block(), _Block()]


class _Pipe:
    def __init__(self, *, group: bool = True, sequential: bool = True, fail_group: bool = False) -> None:
        self.transformer = _Transformer()
        self.group = group
        self.sequential_present = sequential
        self.fail_group = fail_group
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.tiled = False
        self.sliced = False

    def enable_group_offload(self, **kwargs: object) -> None:
        if self.fail_group:
            raise RuntimeError("no cuda stream")
        self.calls.append(("group", kwargs))

    def enable_sequential_cpu_offload(self) -> None:
        self.calls.append(("sequential", {}))

    def enable_vae_tiling(self) -> None:
        self.tiled = True
        self.calls.append(("tile", {}))

    def enable_vae_slicing(self) -> None:
        self.sliced = True
        self.calls.append(("slice", {}))


def _device_name(value: object) -> str:
    kind = getattr(value, "type", None)
    if isinstance(kind, str):
        return kind
    return str(value)


def test_group_offload_is_installed_with_streams_and_pinned_blocks():
    pipe = _Pipe()
    installed = install_low_vram_offload(pipe)
    assert isinstance(installed, OffloadInstall)
    assert installed.kind == "group-offload"
    assert installed.use_stream is True
    assert installed.record_stream is True
    assert installed.pinned is True
    assert installed.vae_tiling is True
    assert installed.vae_slicing is True
    kind, kwargs = pipe.calls[0]
    assert kind == "group"
    assert kwargs["use_stream"] is True
    assert kwargs["record_stream"] is True
    assert kwargs["offload_type"] == "block_level"
    assert kwargs["num_blocks_per_group"] == 1
    assert _device_name(kwargs["onload_device"]) == "cuda"
    assert _device_name(kwargs["offload_device"]) == "cpu"
    assert all(block.weight.data.pinned for block in pipe.transformer.transformer_blocks)
    assert pipe.calls[1][0] == "tile"
    assert pipe.calls[2][0] == "slice"


def test_missing_group_offload_falls_back_to_sequential_cpu_offload():
    pipe = _Pipe()
    pipe.enable_group_offload = None  # type: ignore[method-assign]
    installed = install_low_vram_offload(pipe)
    assert installed.kind == "sequential-cpu-offload"
    assert installed.use_stream is False
    assert installed.pinned is False
    assert pipe.calls[0][0] == "sequential"
    assert pipe.tiled is True and pipe.sliced is True


def test_a_failing_group_offload_falls_back_to_sequential_cpu_offload():
    pipe = _Pipe(fail_group=True)
    installed = install_low_vram_offload(pipe)
    assert installed.kind == "sequential-cpu-offload"
    assert pipe.calls[0][0] == "sequential"
