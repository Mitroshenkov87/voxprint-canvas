#!/bin/sh
# Install VOXPrint AI Canvas into /opt/voxprint-ai-suite/canvas.
# The archive does not contain GPU libraries or model files.
# Missing dependencies are installed here, before the first picture.
# Per-user data stays in ~/.local/share/voxprint.
set -eu

SRC=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
if [ -n "${VOXPRINT_HOME:-}" ]; then
  DATA="$VOXPRINT_HOME"
elif [ -n "${XDG_DATA_HOME:-}" ]; then
  DATA="$XDG_DATA_HOME/voxprint"
else
  DATA="$HOME/.local/share/voxprint"
fi
APP="/opt/voxprint-ai-suite/canvas"
VENV="$APP/venv"
SHARED="$DATA/shared"
RUNTIME_REL="runtime/py314-torch2.11-cu130"
RUNTIME="$SHARED/$RUNTIME_REL"
APP_DIR="$HOME/.local/share/applications"
DIR_DIR="$HOME/.local/share/desktop-directories"
BIN_DIR="$HOME/.local/bin"
LINUX_PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261009/cpython-3.14.8%2B20261009-x86_64-unknown-linux-gnu-install_only.tar.gz"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
UV_CONFIG="$SRC/canvas/infra/uv.cu130.toml"

echo "Checking Python..."
FRESH=0
if [ -x "$RUNTIME/bin/python3" ]; then
  PY="$RUNTIME/bin/python3"
elif [ -d "$RUNTIME" ] && [ -n "$(find "$RUNTIME" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "The suite runtime is already installed. Canvas will not replace it." >&2
  exit 1
else
  echo "Installing Python..."
  mkdir -p "$RUNTIME"
  archive=$(mktemp)
  curl -fsSL -o "$archive" "$LINUX_PYTHON_URL"
  tar -xzf "$archive" -C "$RUNTIME" --strip-components=1
  rm -f "$archive"
  PY="$RUNTIME/bin/python3"
  FRESH=1
fi

if ! mkdir -p "$APP"; then
  echo "VOXPrint AI Canvas needs permission to create $APP." >&2
  exit 1
fi
mkdir -p "$APP_DIR" "$DIR_DIR" "$BIN_DIR" "$SHARED/wheels" "$SHARED/tensorrt-10"
"$PY" -m venv "$VENV"
if [ ! -f "$SRC/constraints.txt" ]; then
  echo "constraints.txt is missing from the setup. The install cannot continue." >&2
  exit 1
fi
"$VENV/bin/python" -m pip install --upgrade pip
# PyPI is the main index. uv pins torch and torchvision to the CUDA 13.0 index.
# torchao is installed from PyPI with an exact pin.
"$VENV/bin/python" -m pip install --index-url https://pypi.org/simple "uv==0.13.0"
UV="$VENV/bin/uv"
if [ ! -f "$UV_CONFIG" ]; then
  echo "The CUDA index configuration is missing from the setup. The install cannot continue." >&2
  exit 1
fi
if [ "$FRESH" -eq 1 ]; then
  "$PY" -m pip install --upgrade pip
  "$UV" pip install --python "$PY" --config-file "$UV_CONFIG" -c "$SRC/constraints.txt" \
    torch torchvision onnxruntime-gpu
fi
echo "Installing PyTorch..."
"$VENV/bin/python" -m pip download -c "$SRC/constraints.txt" \
  "torch==2.11.0+cu130" "torchvision==0.26.0+cu130" \
  --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple -d "$SHARED/wheels"
"$VENV/bin/python" -m pip download -c "$SRC/constraints.txt" \
  "onnxruntime-gpu==1.31.0" "torchao==0.18.0" \
  --index-url https://pypi.org/simple -d "$SHARED/wheels"
"$UV" pip install --python "$VENV/bin/python" --index-url https://pypi.org/simple "torchao==0.18.0"
"$UV" pip install --python "$VENV/bin/python" --config-file "$UV_CONFIG" -c "$SRC/constraints.txt" \
  "$SRC[gen]"
if "$VENV/bin/python" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 14) else 1)"; then
  echo "TensorRT is skipped on Python 3.14." >&2
elif [ -z "$(find "$SHARED/tensorrt-10" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  "$VENV/bin/python" -m pip install -c "$SRC/constraints.txt" --target "$SHARED/tensorrt-10" \
    --upgrade-strategy only-if-needed "tensorrt-cu13==10.16.1.11"
fi
"$VENV/bin/python" -m canvas.infra.layout migrate --install-dir "$APP"
echo "Installing the default image pack..."
"$VENV/bin/python" -m canvas.app.provision
mkdir -p "$DATA/canvas"
"$VENV/bin/python" -m canvas.models.tiers > "$DATA/canvas/tier.txt"
"$VENV/bin/python" -m canvas.app.main --register-models
ln -sfn "$VENV/bin/voxprint-canvas" "$BIN_DIR/voxprint-canvas"
ln -sfn "$VENV/bin/voxprint-canvas-cli" "$BIN_DIR/voxprint-canvas-cli"
if [ -w /usr/local/bin ] || [ "$(id -u)" -eq 0 ]; then
  ln -sfn "$VENV/bin/voxprint-canvas" /usr/local/bin/voxprint-canvas
  ln -sfn "$VENV/bin/voxprint-canvas-cli" /usr/local/bin/voxprint-canvas-cli
fi

cat > "$DIR_DIR/voxprint.directory" << 'EOF'
[Desktop Entry]
Version=1.0
Type=Directory
Name=VOXPrint AI Suite
EOF

cat > "$APP_DIR/voxprint-canvas.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=VOXPrint AI Canvas
GenericName=Image canvas
Comment=Local image generation
Exec=$VENV/bin/voxprint-canvas
Terminal=false
Categories=Graphics;
StartupWMClass=voxprint-canvas
EOF

echo "Installed."
