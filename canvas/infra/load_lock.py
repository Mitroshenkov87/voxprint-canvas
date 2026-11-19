"""One model build at a time in this process.

transformers builds a model inside ``local_torch_dtype``, which calls
``torch.set_default_dtype``. That default is process-wide, not per thread.
The picture checks load on a side thread while the writer loads, so a check
loaded as fp32 flipped the default in the middle of the writer's build. Every
layer created in that window was fp32, and the bf16 checkpoint was upcast into
it. The writer and every check build their modules under this lock.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_LOCK = threading.RLock()


@contextmanager
def pretrained_lock() -> Iterator[None]:
    """Hold the process build lock. A nested call on the same thread does not block."""
    with _LOCK:
        yield
