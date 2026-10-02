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

# PDF and PPTX export need Chrome, Edge, or Firefox. Ubuntu's chromium and
# firefox packages are snaps and are often absent on this host.
ensure_marp_browser() {
  if [[ -n "${CHROME_PATH:-}" && -x "${CHROME_PATH}" ]]; then
    return 0
  fi
  local candidate
  for candidate in google-chrome-stable google-chrome chromium-browser chromium firefox; do
    if command -v "$candidate" >/dev/null 2>&1; then
      CHROME_PATH="$(command -v "$candidate")"
      export CHROME_PATH
      return 0
    fi
  done

  local cache="${MARP_BROWSER_DIR:-${HOME}/.cache/marp-chrome}"
  mkdir -p "$cache"
  local existing
  existing="$(find "$cache" -type f -path '*/chrome-linux64/chrome' -print 2>/dev/null | head -1 || true)"
  if [[ -n "$existing" && -x "$existing" ]]; then
    CHROME_PATH="$existing"
    export CHROME_PATH
    return 0
  fi
  echo "No Chrome, Edge, or Firefox on PATH. Installing Chrome for Testing..."
  # Pin a release that runs on this host's Node 18. Newer @puppeteer/browsers needs Node 22.
  local install_log
  install_log="$(npx -y @puppeteer/browsers@2.10.2 install chrome --path "$cache")"
  echo "$install_log"
  CHROME_PATH="$(awk '/^chrome@/ { print $2 }' <<<"$install_log" | tail -1)"
  if [[ -z "${CHROME_PATH}" || ! -x "${CHROME_PATH}" ]]; then
    echo "Chrome for Testing installed, but its binary was not found under $cache" >&2
    exit 1
  fi
  export CHROME_PATH
}

ensure_marp_browser
missing_libs="$(ldd "$CHROME_PATH" 2>/dev/null | awk '/not found/ { print $1 }' || true)"
if [[ "$missing_libs" == *libasound.so.2* ]]; then
  echo "Installing libasound2t64 so Chrome can start..."
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libasound2t64
elif [[ -n "$missing_libs" ]]; then
  echo "Chrome is missing libraries: $missing_libs" >&2
  exit 1
fi
# Ubuntu's AppArmor profile blocks Chrome's user-namespace sandbox.
export CHROME_NO_SANDBOX=1
echo "Marp browser: ${CHROME_PATH}"

echo "=========================================================="
echo "Compiling Presentation Slides from: $SLIDES_MD"
echo "=========================================================="

echo "[1/3] Generating standalone interactive HTML..."
npx -y @marp-team/marp-cli --html --allow-local-files "$SLIDES_MD" -o "$PRESENTATION_DIR/slides.html"

echo "[2/3] Generating presentation PDF..."
npx -y @marp-team/marp-cli --html --allow-local-files "$SLIDES_MD" -o "$PRESENTATION_DIR/slides.pdf"

echo "[3/3] Generating PowerPoint presentation (PPTX)..."
npx -y @marp-team/marp-cli --html --allow-local-files "$SLIDES_MD" -o "$PRESENTATION_DIR/slides.pptx"

echo "=========================================================="
echo "Successfully generated all presentation formats in $PRESENTATION_DIR:"
ls -lh "$PRESENTATION_DIR"/slides.*
echo "=========================================================="
