"""Install the default shared packs. The setup scripts call this module."""
from __future__ import annotations

import sys

from canvas.models.provision import install_defaults, record_runtime, record_wheels
from canvas.models.registry import load_catalog


def main(argv: list[str] | None = None) -> int:
    """Download the default packs and record the shared interpreter and wheels."""
    del argv
    install_defaults(load_catalog(), lambda message: print(message, file=sys.stderr))
    record_runtime()
    record_wheels()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
