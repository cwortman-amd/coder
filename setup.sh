#!/bin/bash
# ==============================================================================
# setup.sh - Launch ROCm Inference Server & OpenCode Web Client
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ANSI Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Load environment configuration if present
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^[[:space:]]*#' "${SCRIPT_DIR}/.env" | grep -v '^[[:space:]]*$')
    set +a
elif [ -f "${SCRIPT_DIR}/.env.example" ]; then
    echo -e "${YELLOW}Notice: .env not found. Initializing from .env.example...${NC}"
    cp "${SCRIPT_DIR}/.env.example" "${SCRIPT_DIR}/.env"
fi

OPENCODE_PORT="${OPENCODE_PORT:-4096}"
INFERENCE_PORT="${INFERENCE_PORT:-8000}"
MODEL="${MODEL_NAME:-Qwen3.8-27B}"

echo -e "${BLUE}${BOLD}============================================================${NC}"
echo -e "${BLUE}${BOLD}   Starting AMD ROCm Inference & OpenCode Stack             ${NC}"
echo -e "${BLUE}${BOLD}============================================================${NC}"
echo -e "GPU Compute Target : ${BOLD}AMD Radeon™ AI PRO R9700 (gfx1201)${NC}"
echo -e "Configured Model   : ${BOLD}${MODEL}${NC}"
echo -e "Inference API Port : ${BOLD}http://localhost:${INFERENCE_PORT}/v1${NC}"
echo -e "OpenCode Web Port  : ${BOLD}http://localhost:${OPENCODE_PORT}${NC}"
echo ""

COMPOSE_FILE="docker-compose.yml"
if [[ "$MODEL" != *"FP8"* ]] && [[ "$MODEL" != *"/"* ]] && ([[ "$MODEL" == *.gguf ]] || [[ "${MODEL,,}" == *"qwen3.8-27b"* ]]); then
    COMPOSE_FILE="docker-compose.gguf.yml"
    TARGET_GGUF="${SCRIPT_DIR}/models/${MODEL_FILE:-Qwen3.8-27B-Q4_K_M.gguf}"
    if [ ! -f "$TARGET_GGUF" ] || [ ! -s "$TARGET_GGUF" ]; then
        echo -e "${YELLOW}[Notice] GGUF model file not found at: ${TARGET_GGUF}${NC}"
        echo -e "${YELLOW}Run './download_model.sh' to download ${MODEL} (~16.8 GB) before starting.${NC}"
        echo ""
        read -r -p "Would you like to download it now? [y/N]: " CHOICE || CHOICE="n"
        if [[ "$CHOICE" =~ ^[Yy]$ ]]; then
            "${SCRIPT_DIR}/download_model.sh"
        else
            echo -e "${RED}Aborted. Download the model with ./download_model.sh and re-run ./setup.sh${NC}"
            exit 1
        fi
    fi
fi

if [[ "$MODEL" == *"FP8"* ]] || [[ "$MODEL" == *"fp8"* ]]; then
    echo -e "${YELLOW}${BOLD}[NOTICE] Qwen3.8-27B in FP8 precision consumes ~27 GB weights and causes Out-Of-Memory (OOM) on a single 32GB R9700!${NC}"
    echo -e "${YELLOW}Dual R9700 (64GB VRAM) is required for FP8. For a single R9700, Q4_K_M GGUF (MODEL_NAME=Qwen3.8-27B) is recommended.${NC}"
    echo ""
fi

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
echo -e "     - Run both tests & summary: ${YELLOW}./test.sh${NC}"
echo -e "     - SWE-bench only:           ${YELLOW}docker compose run --rm benchmark${NC}"
echo -e "     - GPQA only:                ${YELLOW}python3 benchmark/run_gpqa.py --dataset sample${NC}"
echo -e "${BLUE}============================================================${NC}"
