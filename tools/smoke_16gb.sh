#!/bin/sh
# 16 GB smoke run: install Canvas from this checkout, fetch the models through
# VOXPrint MDL (mdl.json), then generate five 1024x1024 pictures with
# Z-Image-Turbo BF16 on the 16 GB group-offload path. Every output goes through
# the content gate. Writes <out>/result.json.
# Exit 0 pass, 1 a picture failed, 2 setup, 3 no GPU, 4 gate unavailable,
# 5 a manifest entry failed its download or hash check.
# Usage: tools/smoke_16gb.sh [out-folder]   (default: smoke-16gb)
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${1:-smoke-16gb}
PY=${PYTHON:-python3.14}
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
COMMON_URL="https://github.com/Mitroshenkov87/voxprint-common/releases/download/v0.2.4/voxprint_common-0.2.4-py3-none-any.whl"
COMMON_SHA="3051bc64700964705bf5cee5182829d1311560a38a9f7436195f26481e4fa239"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "$PY is not installed. Set PYTHON to a Python 3.14 interpreter." >&2
  exit 2
fi
VENV="$ROOT/.smoke-venv"
if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV" || exit 2
fi
VPY="$VENV/bin/python"
mkdir -p "$VENV/wheels"
WHEEL="$VENV/wheels/voxprint_common-0.2.4-py3-none-any.whl"
curl -fsSL -o "$WHEEL" "$COMMON_URL" || exit 2
echo "$COMMON_SHA  $WHEEL" | sha256sum -c - || exit 2
"$VPY" -m pip install -q --upgrade pip || exit 2
"$VPY" -m pip install -q "$WHEEL" || exit 2
"$VPY" -m pip install -q --extra-index-url "$TORCH_INDEX" \
  -c constraints.txt -c constraints-cuda.txt -e ".[gen]" || exit 2
# A failed download or a hash mismatch stops the run here with exit 5.
if ! "$VPY" -m voxprint_common.mdl download --app canvas --manifest "$ROOT/mdl.json"; then
  echo "MDL download or hash check failed." >&2
  exit 5
fi
exec "$VPY" -m tools.smoke_16gb --python "$VPY" --out "$OUT"
