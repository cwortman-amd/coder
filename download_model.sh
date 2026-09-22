#!/bin/bash
# ==============================================================================
# download_model.sh - Download Qwen3.8-27B-Q4_K_M.gguf to ./models/
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
mkdir -p "$MODELS_DIR"

MODEL_NAME="Qwen3.8-27B-Q4_K_M.gguf"
TARGET_FILE="${MODELS_DIR}/${MODEL_NAME}"

HF_REPO="Fancp/Qwen3.8-27B-Q4_K_M-GGUF"
HF_FILENAME="qwen3.8-27b-q4_k_m.gguf"
HF_URL="https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILENAME}"

# ANSI Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}${BOLD}============================================================${NC}"
echo -e "${BLUE}${BOLD}   Download Model: ${MODEL_NAME} (~16.8 GB)               ${NC}"
echo -e "${BLUE}${BOLD}============================================================${NC}"

if [ -f "$TARGET_FILE" ] && [ -s "$TARGET_FILE" ]; then
    SIZE=$(du -h "$TARGET_FILE" | cut -f1)
    echo -e "${GREEN}${BOLD}✓ Model file already exists:${NC} ${TARGET_FILE} (${SIZE})"
    echo "Ready to serve with: ./setup.sh"
    exit 0
fi

echo -e "Destination: ${BOLD}${TARGET_FILE}${NC}"
echo -e "Source Repo: ${BOLD}${HF_REPO}${NC}"
echo ""

# Prefer huggingface-cli if available
if command -v huggingface-cli &>/dev/null; then
    echo -e "${GREEN}Using huggingface-cli to download...${NC}"
    huggingface-cli download "$HF_REPO" "$HF_FILENAME" --local-dir "$MODELS_DIR" --local-dir-use-symlinks False
    if [ -f "${MODELS_DIR}/${HF_FILENAME}" ] && [ ! -f "$TARGET_FILE" ]; then
        mv "${MODELS_DIR}/${HF_FILENAME}" "$TARGET_FILE"
    fi
else
    echo -e "${YELLOW}huggingface-cli not found. Falling back to curl with resume support...${NC}"
    curl -L --progress-bar -C - "$HF_URL" -o "$TARGET_FILE"
fi

echo ""
echo -e "${GREEN}${BOLD}✓ Model download complete:${NC} ${TARGET_FILE}"
echo -e "Launch the server with: ${BOLD}./setup.sh${NC}"
