#!/bin/sh
# Install VOXPrint AI Canvas into /opt/voxprint-ai-suite/canvas.
# The archive does not contain GPU libraries or model files.
# Missing dependencies are installed here, before the first picture.
# Per-user data stays in ~/.local/share/voxprint.
# Wheels used by a privileged install live under /opt, not the user profile.
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
WHEELS="$APP/setup-work/wheels"
APP_DIR="$HOME/.local/share/applications"
DIR_DIR="$HOME/.local/share/desktop-directories"
BIN_DIR="$HOME/.local/bin"
LINUX_PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261009/cpython-3.14.8%2B20261009-x86_64-unknown-linux-gnu-install_only.tar.gz"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

scrub_environment() {
  # Drop inherited index, mirror, proxy, and interpreter overrides before any download.
  names=$(env | sed 's/=.*//')
  for name in $names; do
    case "$name" in
      UV_*|PIP_*|PYTHON*|VIRTUAL_ENV|CONDA_*|HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|NO_PROXY|FTP_PROXY|http_proxy|https_proxy|all_proxy|no_proxy|ftp_proxy|REQUESTS_CA_BUNDLE|CURL_CA_BUNDLE|SSL_CERT_FILE|SSL_CERT_DIR|GIT_SSL_CAINFO)
        unset "$name" || true
        ;;
    esac
  done
  export UV_NO_CONFIG=1
}

scrub_environment
echo "Checking Python..."
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
fi

if ! mkdir -p "$APP"; then
  echo "VOXPrint AI Canvas needs permission to create $APP." >&2
  exit 1
fi
cd "$APP"
mkdir -p "$APP_DIR" "$DIR_DIR" "$BIN_DIR" "$WHEELS" "$SHARED/tensorrt-10"
"$PY" -m venv "$VENV"
if [ ! -f "$SRC/constraints.txt" ]; then
  echo "constraints.txt is missing from the setup. The install cannot continue." >&2
  exit 1
fi
if [ ! -f "$SRC/constraints-cuda.txt" ]; then
  echo "constraints-cuda.txt is missing from the setup. The install cannot continue." >&2
  exit 1
fi
"$VENV/bin/python" -m pip --isolated install --upgrade pip
# PyPI is the main index. torch and torchvision come from the CUDA 13.0 index.
# torchao is installed from PyPI with an exact pin.
"$VENV/bin/python" -m pip --isolated install --index-url https://pypi.org/simple "uv==0.13.0"
UV="$VENV/bin/uv"

gpu_ready() {
  ort=$(printf '%s%s' onnx runtime)
  "$1" -c "
import importlib
import importlib.metadata as meta
import sys
try:
    import torch
except Exception:
    sys.exit(1)
cuda = getattr(getattr(torch, 'version', None), 'cuda', None) or ''
if not str(cuda).startswith('13.0'):
    sys.exit(1)
name = '${ort}'
gpu = name + '-gpu'
try:
    importlib.import_module(name)
except Exception:
    sys.exit(1)
found = {dist.metadata['Name'].lower() for dist in meta.distributions()}
if gpu not in found or name in found:
    sys.exit(1)
sys.exit(0)
"
}

install_gpu() {
  "$UV" pip install --no-config --python "$1" --index "$TORCH_INDEX" --index-strategy unsafe-best-match \
    --find-links "$WHEELS" -c "$SRC/constraints.txt" -c "$SRC/constraints-cuda.txt" \
    "torch==2.11.0+cu130" "torchvision==0.26.0+cu130"
  "$UV" pip install --no-config --python "$1" --index-url https://pypi.org/simple \
    -c "$SRC/constraints.txt" -c "$SRC/constraints-cuda.txt" \
    "onnxruntime-gpu==1.31.0"
  cpu=$(printf '%s%s' onnx runtime)
  "$UV" pip uninstall --no-config --python "$1" "$cpu"
}

runtime_ready=0
app_ready=0
if gpu_ready "$PY"; then
  runtime_ready=1
fi
if gpu_ready "$VENV/bin/python"; then
  app_ready=1
fi
if [ "$runtime_ready" -eq 1 ] && [ "$app_ready" -eq 1 ]; then
  echo "PyTorch is already installed."
elif find "$WHEELS" -name 'torch-2.11.0*.whl' -print -quit 2>/dev/null | grep -q .; then
  echo "PyTorch wheels are already downloaded."
else
  echo "Installing PyTorch..."
  "$VENV/bin/python" -m pip --isolated download -c "$SRC/constraints.txt" -c "$SRC/constraints-cuda.txt" \
    "torch==2.11.0+cu130" "torchvision==0.26.0+cu130" \
    --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple -d "$WHEELS"
fi
"$VENV/bin/python" -m pip --isolated download -c "$SRC/constraints.txt" -c "$SRC/constraints-cuda.txt" \
  "onnxruntime-gpu==1.31.0" "torchao==0.18.0" \
  --index-url https://pypi.org/simple -d "$WHEELS"
if [ "$runtime_ready" -eq 0 ]; then
  echo "Repairing the suite runtime..."
  "$PY" -m pip --isolated install --upgrade pip
  install_gpu "$PY"
  if ! gpu_ready "$PY"; then
    echo "CUDA 13.0 PyTorch is still missing." >&2
    exit 1
  fi
fi
if [ "$app_ready" -eq 0 ]; then
  install_gpu "$VENV/bin/python"
  if ! gpu_ready "$VENV/bin/python"; then
    echo "CUDA 13.0 PyTorch is still missing." >&2
    exit 1
  fi
fi
"$UV" pip install --no-config --python "$VENV/bin/python" --index-url https://pypi.org/simple "torchao==0.18.0"
"$UV" pip install --no-config --python "$VENV/bin/python" \
  --index "$TORCH_INDEX" --index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  -c "$SRC/constraints.txt" -c "$SRC/constraints-cuda.txt" \
  "$SRC[gen]"
if "$VENV/bin/python" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 14) else 1)"; then
  echo "TensorRT is skipped on Python 3.14." >&2
elif [ -z "$(find "$SHARED/tensorrt-10" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  "$VENV/bin/python" -m pip --isolated install -c "$SRC/constraints.txt" --target "$SHARED/tensorrt-10" \
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
