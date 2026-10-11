#!/bin/sh
# Install Voxprint AI Canvas into a local environment.
# The archive does not contain GPU libraries or model files.
# Missing dependencies are installed here, before the first picture.
set -eu

SRC=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
if [ -n "${VOXPRINT_HOME:-}" ]; then
  ROOT="$VOXPRINT_HOME"
elif [ -n "${XDG_DATA_HOME:-}" ]; then
  ROOT="$XDG_DATA_HOME/voxprint"
else
  ROOT="$HOME/.local/share/voxprint"
fi
APP="$ROOT/canvas"
VENV="$APP/venv"
SHARED="$ROOT/shared"
APP_DIR="$HOME/.local/share/applications"
DIR_DIR="$HOME/.local/share/desktop-directories"
BIN_DIR="$HOME/.local/bin"
LINUX_PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261009/cpython-3.12.15%2B20261009-x86_64-unknown-linux-gnu-install_only.tar.gz"

echo "Checking Python..."
if python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  PY=python3
else
  echo "Installing Python..."
  mkdir -p "$SHARED/python"
  archive=$(mktemp)
  curl -fsSL -o "$archive" "$LINUX_PYTHON_URL"
  tar -xzf "$archive" -C "$SHARED/python" --strip-components=1
  rm -f "$archive"
  PY="$SHARED/python/bin/python3"
fi

mkdir -p "$APP" "$APP_DIR" "$DIR_DIR" "$BIN_DIR" "$SHARED/wheels"
"$PY" -m venv "$VENV"
if [ ! -f "$SRC/constraints.txt" ]; then
  echo "constraints.txt is missing from the setup. The install cannot continue." >&2
  exit 1
fi
"$VENV/bin/python" -m pip install --upgrade pip
echo "Installing PyTorch..."
"$VENV/bin/python" -m pip download -c "$SRC/constraints.txt" torch torchvision --index-url https://download.pytorch.org/whl/cu128 -d "$SHARED/wheels"
"$VENV/bin/python" -m pip install -c "$SRC/constraints.txt" --upgrade-strategy only-if-needed --find-links "$SHARED/wheels" torch torchvision
"$VENV/bin/python" -m pip install -c "$SRC/constraints.txt" --upgrade-strategy only-if-needed "$SRC[gen]"
echo "Installing the default image pack..."
"$VENV/bin/python" -m canvas.app.provision
"$VENV/bin/python" -m canvas.models.tiers > "$APP/tier.txt"
"$VENV/bin/python" -m canvas.app.main --register-models
ln -sfn "$VENV/bin/voxprint-canvas" "$BIN_DIR/voxprint-canvas"

cat > "$DIR_DIR/voxprint.directory" << 'EOF'
[Desktop Entry]
Version=1.0
Type=Directory
Name=Voxprint
EOF

cat > "$APP_DIR/voxprint-canvas.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Voxprint AI Canvas
GenericName=Image canvas
Comment=Local image generation
Exec=$VENV/bin/voxprint-canvas
Terminal=false
Categories=Graphics;
StartupWMClass=voxprint-canvas
EOF

echo "Installed."
