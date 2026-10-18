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


class _Registry:
    """Stands in for a diffusers HookRegistry on one module."""

    def __init__(self) -> None:
        self.hooks: dict[str, object] = {}

    def get_hook(self, name: str) -> object | None:
        return self.hooks.get(name)

    def remove_hook(self, name: str, recurse: bool = True) -> None:
        self.hooks.pop(name, None)


class _Component:
    def __init__(self, *, removable: bool = True) -> None:
        self._diffusers_hook = _Registry() if removable else None
        self.stuck = False

    def modules(self):
        return [self]

    @property
    def offloaded(self) -> bool:
        return self.stuck or (self._diffusers_hook is not None and bool(self._diffusers_hook.hooks))

    def hook(self) -> None:
        if self._diffusers_hook is None:
            self.stuck = True
        else:
            self._diffusers_hook.hooks["group_offloading"] = object()


class _PartialGroupPipe:
    """Hooks components in order like diffusers and fails on the second one.

    ``fail_low_ram`` makes the per-transfer pinning retry fail the same way.
    ``enable_sequential_cpu_offload`` refuses while any component is hooked,
    as diffusers does.
    """

    def __init__(self, *, fail_low_ram: bool = False, removable: bool = True) -> None:
        self.components = {"text_encoder": _Component(removable=removable), "transformer": _Component(removable=removable)}
        self.fail_low_ram = fail_low_ram
        self.calls: list[tuple[str, dict[str, object]]] = []

    def _maybe_raise_error_if_group_offload_active(self, raise_error: bool = False) -> bool:
        active = any(item.offloaded for item in self.components.values())
        if active and raise_error:
            raise ValueError("You are trying to apply model/sequential CPU offloading to a pipeline that contains "
                             "components with group offloading enabled.")
        return active

    def enable_group_offload(self, onload_device=None, offload_device=None, offload_type="block_level",
                             num_blocks_per_group=None, non_blocking=False, use_stream=False,
                             record_stream=False, low_cpu_mem_usage=False) -> None:
        self.calls.append(("group", {"low_cpu_mem_usage": low_cpu_mem_usage}))
        encoder, transformer = self.components.values()
        encoder.hook()
        if low_cpu_mem_usage and not self.fail_low_ram:
            transformer.hook()
            return
        raise RuntimeError("CUDA error: out of memory")

    def enable_sequential_cpu_offload(self) -> None:
        self._maybe_raise_error_if_group_offload_active(raise_error=True)
        self.calls.append(("sequential", {}))


def test_a_partial_group_offload_is_never_followed_by_sequential_offload():
    # rc.12, 24 GB card capped at 16 GB: group offload failed after the first
    # component was hooked, and the sequential fallback then raised in diffusers.
    pipe = _PartialGroupPipe()
    installed = install_low_vram_offload(pipe)
    assert installed.kind == "group-offload"
    assert installed.use_stream is True
    assert installed.pinned is False
    assert [kind for kind, _ in pipe.calls] == ["group", "group"]
    assert pipe.calls[0][1]["low_cpu_mem_usage"] is False
    assert pipe.calls[1][1]["low_cpu_mem_usage"] is True
    assert all(item.offloaded for item in pipe.components.values())


def test_sequential_runs_alone_after_partial_group_hooks_are_removed():
    pipe = _PartialGroupPipe(fail_low_ram=True)
    installed = install_low_vram_offload(pipe)
    assert installed.kind == "sequential-cpu-offload"
    assert [kind for kind, _ in pipe.calls] == ["group", "group", "sequential"]
    assert not any(item.offloaded for item in pipe.components.values())


def test_group_hooks_that_cannot_be_removed_stop_without_a_second_strategy():
    import pytest

    from canvas.pipeline.offload import OffloadUnavailable

    pipe = _PartialGroupPipe(removable=False)
    with pytest.raises(OffloadUnavailable) as caught:
        install_low_vram_offload(pipe)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert [kind for kind, _ in pipe.calls] == ["group"]
