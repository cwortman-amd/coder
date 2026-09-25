#!/bin/bash
# ==============================================================================
# download_model.sh - Download Qwen3.8-27B-Q4_K_M.gguf to ./models/
# ==============================================================================
# Only enable strict error-exit mode if the script is being executed directly,
# not when sourced into an interactive shell.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    set -euo pipefail
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 1. Load user environment (~/.env) if present to pull in HF_TOKEN and user secrets
if [ -f "$HOME/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$HOME/.env"
    set +a
fi

# 2. Load repository environment configuration (.env)
if [ -f "${ROOT_DIR}/.env" ]; then
    PREV_HF_TOKEN="${HF_TOKEN:-}"
    set -a
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.env"
    set +a
    if [ -z "${HF_TOKEN:-}" ] && [ -n "${PREV_HF_TOKEN}" ]; then
        export HF_TOKEN="${PREV_HF_TOKEN}"
    fi
fi

export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="${HF_HOME:-${HF_CACHE_DIR:-$HOME/.cache/huggingface}}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$HF_HOME}"

MODELS_DIR="${MODELS_DIR:-${ROOT_DIR}/models}"
mkdir -p "$MODELS_DIR" "$HF_HOME"

MODEL_NAME="${MODEL_FILE:-Qwen3.8-27B-Q4_K_M.gguf}"
TARGET_FILE="${MODELS_DIR}/${MODEL_NAME}"

HF_REPO="Fancp/Qwen3.8-27B-Q4_K_M-GGUF"
HF_FILENAME="qwen3.8-27b-q4_k_m.gguf"
HF_URL="https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILENAME}"

# ANSI Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}${BOLD}============================================================${NC}"
echo -e "${BLUE}${BOLD}   Download Model: ${MODEL_NAME} (~16.8 GB)               ${NC}"
echo -e "${BLUE}${BOLD}============================================================${NC}"
echo -e "HF_HOME Cache      : ${BOLD}${HF_HOME}${NC}"
echo -e "Target Destination : ${BOLD}${TARGET_FILE}${NC}"
echo -e "Source Repository  : ${BOLD}${HF_REPO}${NC}"
if [ -n "${HF_TOKEN:-}" ]; then
    echo -e "Hugging Face Token : ${GREEN}Loaded (HF_TOKEN configured)${NC}"
else
    echo -e "Hugging Face Token : ${YELLOW}None (unauthenticated / public models only)${NC}"
fi
echo ""

# Locate Python with huggingface_hub: prefer uv venv, then auto-create with uv, then system python
PYTHON_BIN=""
if [ -f "${SCRIPT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
elif command -v uv &>/dev/null; then
    echo -e "${GREEN}Creating isolated virtual environment using uv at ${SCRIPT_DIR}/.venv...${NC}"
    uv venv "${SCRIPT_DIR}/.venv"
    uv pip install --python "${SCRIPT_DIR}/.venv/bin/python" huggingface_hub
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
elif python3 -c "import huggingface_hub" &>/dev/null; then
    PYTHON_BIN="python3"
fi

# Check if model already exists in HF_HOME
CACHED_FILE=""
if [ -n "$PYTHON_BIN" ]; then
    CACHED_FILE=$("$PYTHON_BIN" -c "
import os
from huggingface_hub import try_to_load_from_cache, constants
token = os.environ.get('HF_TOKEN')
p = try_to_load_from_cache('$HF_REPO', '$HF_FILENAME', cache_dir=constants.HF_HUB_CACHE, token=token)
if p and isinstance(p, str) and os.path.exists(p):
    print(p)
" 2>/dev/null || true)
fi

if [ -z "$CACHED_FILE" ]; then
    CACHED_FILE=$(find "$HF_HOME" -name "$HF_FILENAME" 2>/dev/null | head -n 1 || true)
fi

if [ -n "$CACHED_FILE" ] && [ -e "$CACHED_FILE" ]; then
    REAL_CACHED=$(readlink -f "$CACHED_FILE" || realpath "$CACHED_FILE" || echo "$CACHED_FILE")
    if [ ! -f "$TARGET_FILE" ] || [ ! -s "$TARGET_FILE" ]; then
        mkdir -p "$(dirname "$TARGET_FILE")"
        ln -f "$REAL_CACHED" "$TARGET_FILE" 2>/dev/null || ln -sf "$REAL_CACHED" "$TARGET_FILE" 2>/dev/null || cp "$REAL_CACHED" "$TARGET_FILE"
    fi
    SIZE=$(du -h "$TARGET_FILE" | cut -f1)
    echo -e "${GREEN}${BOLD}✓ Model already verified in HF_HOME:${NC} ${CACHED_FILE}"
    echo -e "${GREEN}${BOLD}✓ Model available at:${NC} ${TARGET_FILE} (${SIZE})"
    echo "Ready to serve with: ./setup.sh"
    (return 0 2>/dev/null) && return 0 || exit 0
fi

if [ -f "$TARGET_FILE" ] && [ -s "$TARGET_FILE" ]; then
    SIZE=$(du -h "$TARGET_FILE" | cut -f1)
    echo -e "${GREEN}${BOLD}✓ Model file already exists:${NC} ${TARGET_FILE} (${SIZE})"
    echo "Ready to serve with: ./setup.sh"
    (return 0 2>/dev/null) && return 0 || exit 0
fi

DOWNLOAD_SUCCESS=false

# Method 1: Download using uv-managed virtual environment on host
if [ -n "$PYTHON_BIN" ]; then
    echo -e "${GREEN}Using uv venv Python (${PYTHON_BIN}) to download into HF_HOME...${NC}"
    if "$PYTHON_BIN" -c "
import os, sys
from huggingface_hub import hf_hub_download
target = '${TARGET_FILE}'
token = os.environ.get('HF_TOKEN')
downloaded = hf_hub_download(repo_id='${HF_REPO}', filename='${HF_FILENAME}', token=token)
print(f'HF_CACHE_FILE={downloaded}')
try:
    if os.path.exists(target):
        os.remove(target)
    os.link(downloaded, target)
except Exception:
    import shutil
    shutil.copy2(downloaded, target)
"; then
        DOWNLOAD_SUCCESS=true
    fi
# Method 2: Download from inside Docker container with persistent HF_HOME mount
elif command -v docker &>/dev/null; then
    echo -e "${GREEN}Using Docker container with persistent HF_HOME mount to download...${NC}"
    DOCKER_IMAGE="${VLLM_IMAGE:-rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0}"
    if docker run --rm \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e HF_HOME=/root/.cache/huggingface \
        -v "${HF_HOME}:/root/.cache/huggingface" \
        --entrypoint python3 \
        "$DOCKER_IMAGE" \
        -c "import os; from huggingface_hub import hf_hub_download; p = hf_hub_download(repo_id='${HF_REPO}', filename='${HF_FILENAME}', token=os.environ.get('HF_TOKEN')); print('HF_CACHE_FILE=' + p)"; then
        CACHED_FILE=$(find "$HF_HOME" -name "$HF_FILENAME" 2>/dev/null | head -n 1 || true)
        if [ -n "$CACHED_FILE" ] && [ -f "$CACHED_FILE" ]; then
            ln -f "$CACHED_FILE" "$TARGET_FILE" 2>/dev/null || cp "$CACHED_FILE" "$TARGET_FILE"
            DOWNLOAD_SUCCESS=true
        fi
    fi
elif command -v hf &>/dev/null; then
    echo -e "${GREEN}Using hf CLI to download into HF_HOME...${NC}"
    if hf download "$HF_REPO" "$HF_FILENAME"; then
        CACHED_FILE=$(find "$HF_HOME" -name "$HF_FILENAME" 2>/dev/null | head -n 1 || true)
        if [ -n "$CACHED_FILE" ] && [ -f "$CACHED_FILE" ]; then
            ln -f "$CACHED_FILE" "$TARGET_FILE" 2>/dev/null || cp "$CACHED_FILE" "$TARGET_FILE"
            DOWNLOAD_SUCCESS=true
        fi
    fi
fi

if [ "$DOWNLOAD_SUCCESS" = false ]; then
    echo -e "${YELLOW}huggingface_hub download unavailable. Falling back to curl with resume support...${NC}"
    CURL_AUTH_HEADER=()
    if [ -n "${HF_TOKEN:-}" ]; then
        CURL_AUTH_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")
    fi
    if curl -L --progress-bar -C - "${CURL_AUTH_HEADER[@]}" "$HF_URL" -o "$TARGET_FILE"; then
        DOWNLOAD_SUCCESS=true
    fi
fi

if [ "$DOWNLOAD_SUCCESS" = true ] && [ -f "$TARGET_FILE" ] && [ -s "$TARGET_FILE" ]; then
    SIZE=$(du -h "$TARGET_FILE" | cut -f1)
    echo ""
    echo -e "${GREEN}${BOLD}✓ Model download complete:${NC} ${TARGET_FILE} (${SIZE})"
    echo -e "Launch the server with: ${BOLD}./setup.sh${NC}"
    (return 0 2>/dev/null) && return 0 || exit 0
else
    echo ""
    echo -e "${RED}${BOLD}✗ Failed to download model.${NC}"
    (return 0 2>/dev/null) && return 1 || exit 1
fi
