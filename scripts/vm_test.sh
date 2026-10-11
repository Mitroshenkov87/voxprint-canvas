#!/bin/sh
# Non-interactive Canvas check. Exit 0 pass, 1 check failed, 2 setup, 3 no GPU.
# --smoke is the short run. --full is the longer picture set. --json prints the report.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed." >&2
  exit 2
fi
exec python3 -m tools.vm_test "$@"
