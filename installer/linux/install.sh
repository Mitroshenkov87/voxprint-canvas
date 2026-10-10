#!/bin/sh
# Linux packaging stub for Voxprint AI Canvas.
# Installs a desktop entry in the shared Voxprint menu and records the VRAM tier.
# It does not download model weights.
set -eu

ROOT="${VOXPRINT_HOME:-$HOME/.local/share/voxprint}"
APP_DIR="$HOME/.local/share/applications"
DIR_DIR="$HOME/.local/share/desktop-directories"
mkdir -p "$ROOT/canvas" "$APP_DIR" "$DIR_DIR"

python -m canvas.models.tiers > "$ROOT/canvas/tier.txt"
python -m canvas.app.main --register-models

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
Exec=python -m canvas.app.main
Terminal=false
Categories=Graphics;
StartupWMClass=voxprint-canvas
EOF
