#!/bin/bash
# ==============================================================================
# setup.sh - Launch ROCm Inference Server & OpenCode Web Client
# ==============================================================================
# Only enable strict error-exit mode if the script is being executed directly,
# not when sourced into an interactive shell.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    set -euo pipefail
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ANSI Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 1. Load user environment (~/.env) if present to pull in HF_TOKEN and user secrets
if [ -f "$HOME/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$HOME/.env"
    set +a
fi

# 2. Load repository environment configuration (.env)
if [ -f "${SCRIPT_DIR}/.env" ]; then
    PREV_HF_TOKEN="${HF_TOKEN:-}"
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^[[:space:]]*#' "${SCRIPT_DIR}/.env" | grep -v '^[[:space:]]*$')
    set +a
    # Retain HF_TOKEN from ~/.env if .env did not define it or had it empty
    if [ -z "${HF_TOKEN:-}" ] && [ -n "${PREV_HF_TOKEN}" ]; then
        export HF_TOKEN="${PREV_HF_TOKEN}"
    fi
elif [ -f "${SCRIPT_DIR}/.env.example" ]; then
    echo -e "${YELLOW}Notice: .env not found. Initializing from .env.example...${NC}"
    cp "${SCRIPT_DIR}/.env.example" "${SCRIPT_DIR}/.env"
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^[[:space:]]*#' "${SCRIPT_DIR}/.env" | grep -v '^[[:space:]]*$')
    set +a
fi

# 3. Synchronize HF_TOKEN into .env so Docker Compose picks it up natively
if [ -n "${HF_TOKEN:-}" ] && [ -f "${SCRIPT_DIR}/.env" ]; then
    if grep -q '^[[:space:]]*HF_TOKEN=' "${SCRIPT_DIR}/.env"; then
        if grep -q '^[[:space:]]*HF_TOKEN=[[:space:]]*$' "${SCRIPT_DIR}/.env"; then
            sed -i "s|^[[:space:]]*HF_TOKEN=.*|HF_TOKEN=${HF_TOKEN}|" "${SCRIPT_DIR}/.env"
        fi
    else
        printf "\n# Hugging Face Access Token (auto-loaded from ~/.env during setup)\nHF_TOKEN=%s\n" "${HF_TOKEN}" >> "${SCRIPT_DIR}/.env"
    fi
fi

export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="${HF_HOME:-${HF_CACHE_DIR:-$HOME/.cache/huggingface}}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$HF_HOME}"

OPENCODE_PORT="${OPENCODE_PORT:-4096}"
INFERENCE_PORT="${INFERENCE_PORT:-8000}"
ENGINE="${INFERENCE_ENGINE:-vllm}"
MODEL_OVERRIDE=""

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--engine)
            ENGINE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_OVERRIDE="$2"
            shift 2
            ;;
        -p|--port)
            INFERENCE_PORT="$2"
            shift 2
            ;;
        --opencode-port)
            OPENCODE_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -e, --engine <vllm|llama.cpp|sglang>  Inference engine (default: vllm)"
            echo "  -m, --model <name>                   Model name or HF repository ID"
            echo "  -p, --port <port>                    Inference API port (default: 8000)"
            echo "  --opencode-port <port>               OpenCode Web UI port (default: 4096)"
            echo "  -h, --help                           Show this help message"
            (return 0 2>/dev/null) && return 0 || exit 0
            ;;
        *)
            shift
            ;;
    esac
done

ENGINE="${ENGINE,,}" # Lowercase
if [ "$ENGINE" = "llama.cpp" ] || [ "$ENGINE" = "gguf" ]; then
    ENGINE="llama.cpp"
    COMPOSE_FILE="docker-compose.gguf.yml"
    ENGINE_LABEL="llama.cpp ROCm Server (GGUF)"
    MODEL="${MODEL_OVERRIDE:-${MODEL_NAME:-Qwen3.8-27B}}"
elif [ "$ENGINE" = "sglang" ]; then
    ENGINE="sglang"
    COMPOSE_FILE="docker-compose.sglang.yml"
    ENGINE_LABEL="SGLang ROCm Server"
    MODEL="${MODEL_OVERRIDE:-${MODEL_NAME:-Qwen/Qwen3.8-27B-FP8}}"
else
    # Default: vLLM
    ENGINE="vllm"
    COMPOSE_FILE="docker-compose.yml"
    ENGINE_LABEL="vLLM ROCm Server (Default)"
    MODEL="${MODEL_OVERRIDE:-${MODEL_NAME:-Qwen/Qwen3.8-27B-FP8}}"
    # Normalize model alias to full repo ID for vLLM
    if [ "$MODEL" = "Qwen3.8-27B" ]; then
        MODEL="Qwen/Qwen3.8-27B-FP8"
    fi
fi
export MODEL_NAME="$MODEL"
export INFERENCE_ENGINE="$ENGINE"

echo -e "${BLUE}${BOLD}============================================================${NC}"
echo -e "${BLUE}${BOLD}   Starting AMD ROCm Inference & OpenCode Stack             ${NC}"
echo -e "${BLUE}${BOLD}============================================================${NC}"
echo -e "Inference Engine   : ${GREEN}${BOLD}${ENGINE_LABEL}${NC}"
echo -e "GPU Compute Target : ${BOLD}AMD Radeon™ AI PRO R9700 (gfx1201)${NC}"
echo -e "Configured Model   : ${BOLD}${MODEL}${NC}"
echo -e "Inference API Port : ${BOLD}http://localhost:${INFERENCE_PORT}/v1${NC}"
echo -e "OpenCode Web Port  : ${BOLD}http://localhost:${OPENCODE_PORT}${NC}"
if [ -n "${HF_TOKEN:-}" ]; then
    echo -e "Hugging Face Token : ${GREEN}Loaded (HF_TOKEN configured)${NC}"
else
    echo -e "Hugging Face Token : ${YELLOW}None (unauthenticated / public models only)${NC}"
fi
echo ""

# Check for GGUF model handling (llama.cpp or sglang with GGUF)
IS_GGUF=false
if [ "$ENGINE" = "llama.cpp" ] || [[ "$MODEL" == *.gguf ]] || ([ "$MODEL" = "Qwen3.8-27B" ] && [ "$ENGINE" != "vllm" ]); then
    IS_GGUF=true
fi

if [ "$IS_GGUF" = true ]; then
    TARGET_GGUF="${SCRIPT_DIR}/models/${MODEL_FILE:-Qwen3.8-27B-Q4_K_M.gguf}"
    if [ ! -f "$TARGET_GGUF" ] || [ ! -s "$TARGET_GGUF" ]; then
        # Check if model already exists in HF_HOME
        HF_GGUF_MATCH=""
        PYTHON_BIN=""
        if [ -f "${SCRIPT_DIR}/.venv/bin/python" ]; then
            PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
        elif python3 -c "import huggingface_hub" &>/dev/null; then
            PYTHON_BIN="python3"
        fi
        if [ -n "$PYTHON_BIN" ]; then
            HF_GGUF_MATCH=$("$PYTHON_BIN" -c "
import os
from huggingface_hub import try_to_load_from_cache, constants
repo = '${HF_REPO:-Fancp/Qwen3.8-27B-Q4_K_M-GGUF}'
fname = '${MODEL_FILE:-Qwen3.8-27B-Q4_K_M.gguf}'
token = os.environ.get('HF_TOKEN')
p = try_to_load_from_cache(repo, fname.lower(), cache_dir=constants.HF_HUB_CACHE, token=token)
if not p:
    p = try_to_load_from_cache(repo, fname, cache_dir=constants.HF_HUB_CACHE, token=token)
if p and isinstance(p, str) and os.path.exists(p):
    print(p)
" 2>/dev/null || true)
        fi
        if [ -z "$HF_GGUF_MATCH" ]; then
            HF_GGUF_MATCH=$(find "${HF_HOME:-$HOME/.cache/huggingface}" -name "${MODEL_FILE:-Qwen3.8-27B-Q4_K_M.gguf}" -o -name "qwen3.8-27b-q4_k_m.gguf" 2>/dev/null | head -n 1 || true)
        fi
        if [ -n "$HF_GGUF_MATCH" ] && [ -e "$HF_GGUF_MATCH" ]; then
            REAL_GGUF=$(readlink -f "$HF_GGUF_MATCH" || realpath "$HF_GGUF_MATCH" || echo "$HF_GGUF_MATCH")
            if [ -f "$REAL_GGUF" ] && [ -s "$REAL_GGUF" ]; then
                echo -e "${GREEN}Found model in HF_HOME: ${REAL_GGUF}${NC}"
                echo -e "${GREEN}Linking to: ${TARGET_GGUF}${NC}"
                mkdir -p "$(dirname "$TARGET_GGUF")"
                ln -f "$REAL_GGUF" "$TARGET_GGUF" 2>/dev/null || ln -sf "$REAL_GGUF" "$TARGET_GGUF" 2>/dev/null || cp "$REAL_GGUF" "$TARGET_GGUF"
            fi
        fi
    fi

    if [ ! -f "$TARGET_GGUF" ] || [ ! -s "$TARGET_GGUF" ]; then
        echo -e "${YELLOW}[Notice] GGUF model file not found at: ${TARGET_GGUF}${NC}"
        echo -e "${YELLOW}Run './download_model.sh' to download ${MODEL} (~16.8 GB) before starting.${NC}"
        echo ""
        read -r -p "Would you like to download it now? [y/N]: " CHOICE || CHOICE="n"
        if [[ "$CHOICE" =~ ^[Yy]$ ]]; then
            if ! "${SCRIPT_DIR}/download_model.sh"; then
                echo -e "${RED}Model download failed. Please check your connection or HF_HOME and try again.${NC}"
                (return 0 2>/dev/null) && return 1 || exit 1
            fi
        else
            echo -e "${YELLOW}Download skipped. Run ./download_model.sh when ready and re-run setup.sh${NC}"
            (return 0 2>/dev/null) && return 0 || exit 0
        fi
    fi

    if [ "$ENGINE" = "sglang" ]; then
        GGUF_BASENAME=$(basename "$TARGET_GGUF")
        export SGLANG_MODEL_PATH="/models/${GGUF_BASENAME}"
        export SGLANG_TOKENIZER_PATH="${SGLANG_TOKENIZER_PATH:-Qwen/Qwen3.8-27B}"
        export SGLANG_LOAD_FORMAT="gguf"
        echo -e "SGLang Model Path  : ${GREEN}${SGLANG_MODEL_PATH}${NC}"
        echo -e "SGLang Tokenizer   : ${GREEN}${SGLANG_TOKENIZER_PATH}${NC}"
        echo -e "SGLang Load Format : ${GREEN}${SGLANG_LOAD_FORMAT}${NC}"
    fi
elif [ "$ENGINE" = "sglang" ]; then
    export SGLANG_MODEL_PATH="$MODEL"
    export SGLANG_TOKENIZER_PATH="${SGLANG_TOKENIZER_PATH:-}"
    export SGLANG_LOAD_FORMAT="auto"
fi

if [[ "$MODEL" == *"FP8"* ]] || [[ "$MODEL" == *"fp8"* ]]; then
    echo -e "${BLUE}${BOLD}[INFO] Qwen3.8-27B FP8 precision consumes ~27.5 GB VRAM for model weights.${NC}"
    echo -e "${BLUE}Configured for single 32GB R9700 with MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}.${NC}"
    echo ""
fi

# Ensure persistent results and logs directory exists
mkdir -p "${SCRIPT_DIR}/_results"

# Launch docker compose services
docker compose -f "$COMPOSE_FILE" up -d

echo ""
echo -e "${GREEN}${BOLD}✓ Containers started in detached mode!${NC}"
echo ""
echo -e "${BOLD}Web Interface Access:${NC}"
echo -e "  👉 ${GREEN}${BOLD}http://localhost:${OPENCODE_PORT}${NC}"
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "")
if [ -n "$LOCAL_IP" ] && [ "$LOCAL_IP" != "127.0.0.1" ]; then
    echo -e "  👉 ${GREEN}${BOLD}http://${LOCAL_IP}:${OPENCODE_PORT}${NC} (from other machines on local network)"
fi

echo ""
echo -e "${BOLD}Next Steps:${NC}"
echo -e "  1. ${BOLD}Verify server readiness & live response:${NC}"
echo -e "     ${YELLOW}./check.sh --wait 60${NC}"
echo ""
echo -e "  2. ${BOLD}Monitor model loading & kernel compilation logs:${NC}"
echo -e "     ${YELLOW}docker compose logs -f inference${NC}"
echo ""
echo -e "  3. ${BOLD}Interact with OpenCode:${NC}"
echo -e "     - Web UI:      Navigate to ${GREEN}http://localhost:${OPENCODE_PORT}${NC} in your browser"
echo -e "     - Terminal TUI: Run ${YELLOW}docker compose exec -it opencode opencode${NC}"
echo -e "     - Host Shell:   Run ${YELLOW}opencode${NC} (if installed locally on host)"
echo ""
echo -e "  4. ${BOLD}Run Benchmarks & Generate Summary Report:${NC}"
echo -e "     - Run both tests & summary: ${YELLOW}./test.sh${NC} (Supports: -s sample, -d diamond, -m main, -a all)"
echo -e "     - SWE-bench only:           ${YELLOW}docker compose run --rm --no-deps benchmark${NC}"
echo -e "     - GPQA only:                ${YELLOW}python3 benchmark/run_gpqa.py --dataset sample${NC}"
echo -e "${BLUE}============================================================${NC}"
