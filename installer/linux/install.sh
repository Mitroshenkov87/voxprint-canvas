#!/bin/sh
# Install Voxprint AI Canvas into a local environment.
# The archive does not contain GPU libraries or model files.
# It does not download model weights.
# The first picture downloads the pinned models.
set -eu

SRC=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
ROOT="${VOXPRINT_HOME:-$HOME/.local/share/voxprint}"
APP="$ROOT/canvas"
VENV="$APP/venv"
APP_DIR="$HOME/.local/share/applications"
DIR_DIR="$HOME/.local/share/desktop-directories"
BIN_DIR="$HOME/.local/bin"

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "Python 3.12 or newer is required." >&2
  exit 1
fi

mkdir -p "$APP" "$APP_DIR" "$DIR_DIR" "$BIN_DIR"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/python" -m pip install --upgrade-strategy only-if-needed "$SRC[gen]"
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

echo "Installed. The first picture downloads the models."
