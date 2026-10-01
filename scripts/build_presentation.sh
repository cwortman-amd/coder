#!/usr/bin/env bash
# Build publication-quality presentation slides in HTML, PDF, and PPTX formats using Marp.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PRESENTATION_DIR="$ROOT_DIR/docs/presentation"
SLIDES_MD="$PRESENTATION_DIR/slides.md"

if [[ ! -f "$SLIDES_MD" ]]; then
  echo "Error: Slides source file not found at $SLIDES_MD" >&2
  exit 1
fi

echo "=========================================================="
echo "Compiling Presentation Slides from: $SLIDES_MD"
echo "=========================================================="

echo "[1/3] Generating standalone interactive HTML..."
npx -y @marp-team/marp-cli --allow-local-files "$SLIDES_MD" -o "$PRESENTATION_DIR/slides.html"

echo "[2/3] Generating presentation PDF..."
npx -y @marp-team/marp-cli --allow-local-files "$SLIDES_MD" -o "$PRESENTATION_DIR/slides.pdf"

echo "[3/3] Generating PowerPoint presentation (PPTX)..."
npx -y @marp-team/marp-cli --allow-local-files "$SLIDES_MD" -o "$PRESENTATION_DIR/slides.pptx"

echo "=========================================================="
echo "Successfully generated all presentation formats in $PRESENTATION_DIR:"
ls -lh "$PRESENTATION_DIR"/slides.*
echo "=========================================================="
