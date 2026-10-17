#!/bin/sh
# Install VOXPrint AI Canvas into /opt/voxprint-ai-suite/canvas.
# The archive does not contain GPU libraries or model files.
# Missing dependencies are installed here, before the first picture.
# Per-user data stays in ~/.local/share/voxprint.
# Wheels used by a privileged install live under /opt, not the user profile.
set -eu

SRC=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
APP="/opt/voxprint-ai-suite/canvas"
VENV="$APP/venv"
RUNTIME_REL="runtime/py314-torch2.11-cu130"
WHEELS="$APP/setup-work/wheels"
APP_PYTHON="$APP/python"
BASE="$APP_PYTHON/bin/python3"
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

refuse_symlink() {
  cursor=$1
  while [ -n "$cursor" ] && [ "$cursor" != "/" ]; do
    if [ -L "$cursor" ]; then
      echo "Setup stopped because this path contains a link: $cursor" >&2
      exit 1
    fi
    cursor=$(dirname "$cursor")
  done
}

installing_user_home() {
  # sudo's default HOME is /root. Profile paths must use the installing user's home.
  if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    home=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    if [ -z "$home" ] || [ ! -d "$home" ]; then
      echo "Setup stopped because the installing user's home could not be found." >&2
      exit 1
    fi
    printf '%s\n' "$home"
    return
  fi
  if [ -z "${HOME:-}" ] || [ ! -d "$HOME" ]; then
    echo "Setup stopped because HOME is not set." >&2
    exit 1
  fi
  printf '%s\n' "$HOME"
}

profile_xdg() {
  fallback="$USER_HOME/.local/share"
  if [ -z "${XDG_DATA_HOME:-}" ] || [ "${XDG_DATA_HOME#/}" = "$XDG_DATA_HOME" ]; then
    printf '%s\n' "$fallback"
    return
  fi
  case "$XDG_DATA_HOME" in
    "$USER_HOME"|"$USER_HOME"/*)
      printf '%s\n' "$XDG_DATA_HOME"
      ;;
    *)
      if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
        printf '%s\n' "$fallback"
      else
        printf '%s\n' "$XDG_DATA_HOME"
      fi
      ;;
  esac
}

data_root() {
  if [ -n "${VOXPRINT_HOME:-}" ]; then
    printf '%s\n' "$VOXPRINT_HOME"
    return
  fi
  printf '%s\n' "$PROFILE_XDG/voxprint"
}

under_user_home() {
  case "$1" in
    "$USER_HOME"|"$USER_HOME"/*) return 0 ;;
    *) return 1 ;;
  esac
}

as_user() {
  # A root install must not run or write the profile as root when sudo recorded the user.
  if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    sudo -u "$SUDO_USER" -- env \
      HOME="$USER_HOME" \
      USER="$SUDO_USER" \
      LOGNAME="$SUDO_USER" \
      UV_NO_CONFIG=1 \
      VOXPRINT_HOME="${VOXPRINT_HOME:-}" \
      XDG_DATA_HOME="$PROFILE_XDG" \
      "$@"
  else
    "$@"
  fi
}

handoff_dir() {
  # A previous root install may own this directory. Give it back before the user renames it.
  if [ "$(id -u)" -ne 0 ] || [ -z "${SUDO_USER:-}" ]; then
    return 0
  fi
  if ! under_user_home "$1"; then
    return 0
  fi
  if [ -d "$1" ]; then
    refuse_symlink "$1"
    chown "$SUDO_USER" "$1"
  fi
}

handoff_parents() {
  cursor=$1
  while [ -n "$cursor" ] && [ "$cursor" != "/" ] && [ "$cursor" != "$USER_HOME" ]; do
    handoff_dir "$cursor"
    cursor=$(dirname "$cursor")
  done
}

move_occupied_runtime_aside() {
  # A standalone tree is user data. Rename it and build a new runtime beside the backup.
  refuse_symlink "$SHARED"
  refuse_symlink "$RUNTIME"
  stamp=$(date +%Y%m%d%H%M%S)
  backup="$RUNTIME.bak-$stamp"
  suffix=0
  while [ -e "$backup" ]; do
    suffix=$((suffix + 1))
    backup="$RUNTIME.bak-$stamp-$suffix"
  done
  echo "Moving the existing suite runtime aside: $backup"
  if under_user_home "$RUNTIME" && [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    chown -R "$SUDO_USER" "$RUNTIME"
  fi
  handoff_parents "$(dirname "$RUNTIME")"
  as_user mv -- "$RUNTIME" "$backup"
}

if [ "${VOXPRINT_INSTALL_LIB:-}" = "1" ]; then
  return 0 2>/dev/null || exit 0
fi

USER_HOME=$(installing_user_home)
PROFILE_XDG=$(profile_xdg)
DATA=$(data_root)
SHARED="$DATA/shared"
RUNTIME="$SHARED/$RUNTIME_REL"
RUNTIME_SITE="$RUNTIME/lib/python3.14/site-packages"
APP_DIR="$USER_HOME/.local/share/applications"
DIR_DIR="$USER_HOME/.local/share/desktop-directories"
BIN_DIR="$USER_HOME/.local/bin"

scrub_environment
if ! mkdir -p "$APP"; then
  echo "VOXPrint AI Canvas needs permission to create $APP." >&2
  exit 1
fi
cd "$APP"
mkdir -p "$WHEELS"
refuse_symlink "$APP"
echo "Checking Python..."
if [ ! -x "$BASE" ]; then
  echo "Installing Python..."
  mkdir -p "$APP_PYTHON"
  archive="$APP/setup-work/python.tar.gz"
  curl -fsSL -o "$archive" "$LINUX_PYTHON_URL"
  tar -xzf "$archive" -C "$APP_PYTHON" --strip-components=1
  rm -f "$archive"
fi
"$BASE" -m venv "$VENV"
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

runtime_files_ready() {
  version="$RUNTIME_SITE/torch/version.py"
  if [ ! -f "$version" ] || ! grep -q "13.0" "$version"; then
    return 1
  fi
  meta_ok() {
    prefix=$1
    wanted=$2
    for dir in "$RUNTIME_SITE"/${prefix}*.dist-info; do
      if [ -f "$dir/METADATA" ] && grep -q "Version: $wanted" "$dir/METADATA"; then
        return 0
      fi
    done
    return 1
  }
  meta_ok "torch-" "2.11.0+cu130" || return 1
  meta_ok "torchvision-" "0.26.0+cu130" || return 1
  gpu=$(printf '%s%s' onnx runtime)
  meta_ok "${gpu}_gpu-" "1.31.0" || return 1
  for dir in "$RUNTIME_SITE"/${gpu}-*.dist-info; do
    if [ -d "$dir" ]; then
      return 1
    fi
  done
  return 0
}

install_runtime() {
  refuse_symlink "$RUNTIME"
  refuse_symlink "$RUNTIME_SITE"
  as_user "$UV" pip install --no-config --python "$BASE" --target "$RUNTIME_SITE" \
    --index "$TORCH_INDEX" --index-strategy unsafe-best-match \
    --find-links "$WHEELS" -c "$SRC/constraints.txt" -c "$SRC/constraints-cuda.txt" \
    "torch==2.11.0+cu130" "torchvision==0.26.0+cu130"
  as_user "$UV" pip install --no-config --python "$BASE" --target "$RUNTIME_SITE" \
    --index-url https://pypi.org/simple -c "$SRC/constraints.txt" -c "$SRC/constraints-cuda.txt" \
    "onnxruntime-gpu==1.31.0"
  cpu=$(printf '%s%s' onnx runtime)
  as_user "$UV" pip uninstall --no-config --python "$BASE" --target "$RUNTIME_SITE" "$cpu"
}

runtime_ready=0
app_ready=0
if runtime_files_ready; then
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
  if [ ! -f "$RUNTIME/pyvenv.cfg" ]; then
    if [ -d "$RUNTIME" ] && [ -n "$(find "$RUNTIME" -mindepth 1 -print -quit 2>/dev/null)" ]; then
      move_occupied_runtime_aside
    fi
    refuse_symlink "$SHARED"
    refuse_symlink "$RUNTIME"
    handoff_parents "$SHARED"
    as_user mkdir -p "$SHARED"
    as_user "$BASE" -m venv --without-pip "$RUNTIME"
  fi
  install_runtime
  if ! runtime_files_ready; then
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
  refuse_symlink "$SHARED/tensorrt-10"
  handoff_parents "$SHARED/tensorrt-10"
  as_user mkdir -p "$SHARED/tensorrt-10"
  as_user "$VENV/bin/python" -m pip --isolated install -c "$SRC/constraints.txt" --target "$SHARED/tensorrt-10" \
    --upgrade-strategy only-if-needed "tensorrt-cu13==10.16.1.11"
fi
"$VENV/bin/python" -m canvas.infra.layout migrate --install-dir "$APP"
echo "Installing the default image pack..."
refuse_symlink "$DATA"
refuse_symlink "$DATA/canvas/tier.txt"
handoff_parents "$DATA"
as_user "$VENV/bin/python" -m canvas.app.provision
as_user mkdir -p "$DATA/canvas"
as_user sh -c '"$1" -m canvas.models.tiers > "$2"' sh "$VENV/bin/python" "$DATA/canvas/tier.txt"
as_user "$VENV/bin/python" -m canvas.app.main --register-models
refuse_symlink "$BIN_DIR"
refuse_symlink "$APP_DIR"
refuse_symlink "$DIR_DIR"
handoff_parents "$BIN_DIR"
handoff_parents "$APP_DIR"
handoff_parents "$DIR_DIR"
as_user mkdir -p "$APP_DIR" "$DIR_DIR" "$BIN_DIR"
menu_file="$APP/setup-work/voxprint.directory"
desktop_file="$APP/setup-work/voxprint-canvas.desktop"
cat > "$menu_file" << 'EOF'
[Desktop Entry]
Version=1.0
Type=Directory
Name=VOXPrint AI Suite
EOF
cat > "$desktop_file" << EOF
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
as_user cp "$menu_file" "$DIR_DIR/voxprint.directory"
as_user cp "$desktop_file" "$APP_DIR/voxprint-canvas.desktop"
as_user ln -sfn "$VENV/bin/voxprint-canvas" "$BIN_DIR/voxprint-canvas"
as_user ln -sfn "$VENV/bin/voxprint-canvas-cli" "$BIN_DIR/voxprint-canvas-cli"
if [ -w /usr/local/bin ] || [ "$(id -u)" -eq 0 ]; then
  ln -sfn "$VENV/bin/voxprint-canvas" /usr/local/bin/voxprint-canvas
  ln -sfn "$VENV/bin/voxprint-canvas-cli" /usr/local/bin/voxprint-canvas-cli
fi

echo "Installed."
