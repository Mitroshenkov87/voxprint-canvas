"""Print the VRAM tier for the installer. ``none`` when this PC cannot run Canvas."""
from __future__ import annotations

from canvas.infra.gpu import probe_gpu
from canvas.models.modes import select_tier


def main() -> int:
    info = probe_gpu()
    if info is None or not info.supported:
        print("none")
        return 0
    tier = select_tier(info.total_gb)
    print("none" if tier is None else tier.value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
