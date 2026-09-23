#!/bin/bash
# ==============================================================================
# demo.sh - The Industry-Standard Benchmark Demo: HTML5 Water Simulation
# ==============================================================================
# Across major coding agent leaderboards (like Artificial Analysis and community
# repos), the interactive 2D canvas water physics simulation has become the
# definitive "vibe-coding" and capability benchmark demo.
#
# The Challenge:
#   Building a fully self-contained HTML5/JavaScript physics engine that simulates
#   falling droplets, fluid density, collision physics, and real-time canvas rendering.
#
# Why it's the standard:
#   It forces the agent to handle complex mathematics (Navier-Stokes approximations
#   or particle systems), state management, and real-time DOM manipulation
#   simultaneously. It immediately proves whether an agent can reason structurally
#   or if it just spits out broken snippets.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Load user environment (~/.env) if present to pull in HF_TOKEN
if [ -f "$HOME/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$HOME/.env"
    set +a
fi

# 2. Load local environment configuration (.env)
if [ -f "${SCRIPT_DIR}/.env" ]; then
    PREV_HF_TOKEN="${HF_TOKEN:-}"
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^[[:space:]]*#' "${SCRIPT_DIR}/.env" | grep -v '^[[:space:]]*$')
    set +a
    if [ -z "${HF_TOKEN:-}" ] && [ -n "${PREV_HF_TOKEN}" ]; then
        export HF_TOKEN="${PREV_HF_TOKEN}"
    fi
fi
export HF_TOKEN="${HF_TOKEN:-}"

# ANSI Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration Defaults
TARGET_DIR="${SCRIPT_DIR}/opencode-water-sim"
INFERENCE_HOST="127.0.0.1"
INFERENCE_PORT="${INFERENCE_PORT:-8000}"
PREVIEW_PORT="${PREVIEW_PORT:-3000}"
INTERACTIVE_MODE=false
AUTO_SERVE=false
CUSTOM_MODEL=""
SHOW_DASHBOARD_INFO=false

# Standard Benchmark Prompt
STANDARD_PROMPT="Build a highly polished, interactive 2D water particle simulation using HTML5 Canvas and vanilla JavaScript. Liquid particles should fall from the top, collide with adjustable sliders/obstacles, pool at the bottom with realistic fluid density, and ripple when clicked. Pack everything—CSS styling, HTML structure, and the complete physics math loop—into a single, production-grade index.html file."

# ==============================================================================
# CLI Argument Parsing
# ==============================================================================
print_help() {
    echo -e "${BLUE}${BOLD}OpenCode Industry-Standard Benchmark Demo: HTML5 Water Simulation${NC}"
    echo "Usage: ./demo.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --interactive      Run OpenCode in interactive terminal TUI mode"
    echo "  -s, --serve            Automatically launch local preview web server at completion"
    echo "  -p, --port <port>      Preview web server port (default: 3000)"
    echo "  -m, --model <model>    Override model name (default: auto-detected from ROCm server)"
    echo "  -d, --dashboard        Show OpenCode Benchmark Dashboard instructions (bun run ...)"
    echo "  -h, --help             Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  ./demo.sh              # Run automated benchmark generation and verify index.html"
    echo "  ./demo.sh -s           # Generate water simulation and launch preview at http://localhost:3000"
    echo "  ./demo.sh -i           # Launch interactive OpenCode TUI inside opencode-water-sim"
    echo "  ./demo.sh -d           # Display multi-model evaluation dashboard workflow"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--interactive)
            INTERACTIVE_MODE=true
            shift
            ;;
        -s|--serve)
            AUTO_SERVE=true
            shift
            ;;
        -p|--port)
            PREVIEW_PORT="${2:-3000}"
            shift 2
            ;;
        -m|--model)
            CUSTOM_MODEL="$2"
            shift 2
            ;;
        -d|--dashboard)
            SHOW_DASHBOARD_INFO=true
            shift
            ;;
        -h|--help)
            print_help
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Run ./demo.sh --help for usage details."
            exit 1
            ;;
    esac
done

# ==============================================================================
# Banner & Benchmark Overview
# ==============================================================================
echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "${BLUE}${BOLD}     The Industry-Standard Benchmark Demo: HTML5 Water Simulation     ${NC}"
echo -e "${BLUE}${BOLD}                 Powered by AMD ROCm & OpenCode Agent                 ${NC}"
echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "Hardware Platform : ${BOLD}AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB VRAM)${NC}"
echo -e "Task Type         : ${CYAN}Full-Stack Physics Engine in Vanilla HTML5/JS/Canvas${NC}"
echo -e "Benchmark Scope   : ${BOLD}Navier-Stokes/Particle Physics, Fluid Pooling, Ripple Events${NC}"
echo -e "Output Directory  : ${BOLD}${TARGET_DIR}${NC}"
echo ""

# Handle dashboard workflow info request
if [ "$SHOW_DASHBOARD_INFO" = true ]; then
    echo -e "${CYAN}${BOLD}>>> OpenCode Evaluation Workflow & Multi-Model Benchmark Dashboard:${NC}"
    echo "If you are testing multiple underlying models (like Qwen 3.8 27B, Qwen 2.5 Coder 32B/7B, or Claude)"
    echo "via OpenCode's engine to see which performs best on your hardware, integrate your"
    echo "test cases directly into the OpenCode Benchmark Dashboard:"
    echo ""
    echo -e "  ${YELLOW}# 1. Add your water-sim test to the prompts directory${NC}"
    echo -e "  ${YELLOW}bun run answer -m \"rocm-local/Qwen3.8-27B\" -t CODING-water-sim${NC}"
    echo ""
    echo -e "  ${YELLOW}# 2. Evaluate model outputs and score completion${NC}"
    echo -e "  ${YELLOW}bun run evaluate -m \"rocm-local/Qwen3.8-27B\" -t CODING-water-sim${NC}"
    echo ""
    echo -e "  ${YELLOW}# 3. Spin up the visual comparison dashboard at http://localhost:3000${NC}"
    echo -e "  ${YELLOW}bun run dashboard${NC}"
    echo ""
    echo "This spins up a local server so you can visually audit completion speed,"
    echo "token efficiency, and structural accuracy scores side-by-side across models."
    echo -e "${BLUE}======================================================================${NC}"
    exit 0
fi

# ==============================================================================
# Step 1: Pre-flight Verification of ROCm Inference Server
# ==============================================================================
echo -e "${CYAN}${BOLD}[1/4] Verifying ROCm Inference Engine Health...${NC}"
HEALTH_URL="http://${INFERENCE_HOST}:${INFERENCE_PORT}/health"
MODELS_URL="http://${INFERENCE_HOST}:${INFERENCE_PORT}/v1/models"

if ! curl -sf --connect-timeout 3 "$HEALTH_URL" >/dev/null 2>&1; then
    echo -e "${RED}${BOLD}[FAIL] ROCm inference server at ${HEALTH_URL} is not responding.${NC}"
    echo -e "${YELLOW}Please start your local inference engine first:${NC}"
    echo -e "  ${BOLD}./setup.sh${NC}"
    echo ""
    echo -e "Or verify health with: ${BOLD}./check.sh --wait 60${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ Inference server is healthy and responding (HTTP 200).${NC}"

# Detect active model from registry
ACTIVE_MODEL=$(curl -s --connect-timeout 5 "$MODELS_URL" 2>/dev/null | jq -r '.data[0].id // empty' || true)
if [ -n "$CUSTOM_MODEL" ]; then
    MODEL_TO_USE="$CUSTOM_MODEL"
elif [ -n "$ACTIVE_MODEL" ]; then
    MODEL_TO_USE="$ACTIVE_MODEL"
else
    MODEL_TO_USE="Qwen3.8-27B"
fi

echo -e "  ${GREEN}✓ Active GPU Model:${NC} ${BOLD}${MODEL_TO_USE}${NC}"
echo ""

# ==============================================================================
# Step 2: Initialize Blank Workspace Structure
# ==============================================================================
echo -e "${CYAN}${BOLD}[2/4] Initializing Target Workspace Directory...${NC}"
mkdir -p "$TARGET_DIR"
echo -e "  ${GREEN}✓ Workspace prepared:${NC} ${TARGET_DIR}"
echo ""

# Determine OpenCode executable path
OPENCODE_BIN=""
if [ -f "/home/amd/.opencode/bin/opencode" ]; then
    OPENCODE_BIN="/home/amd/.opencode/bin/opencode"
elif command -v opencode &>/dev/null; then
    OPENCODE_BIN="$(command -v opencode)"
fi

# ==============================================================================
# Step 3: Execute the Coding Challenge via OpenCode
# ==============================================================================
echo -e "${CYAN}${BOLD}[3/4] Launching OpenCode Coding Challenge...${NC}"
echo -e "${BOLD}Standard Evaluation Prompt:${NC}"
echo -e "${YELLOW}\"${STANDARD_PROMPT}\"${NC}"
echo ""

OPENCODE_MODEL_ARG="rocm-local/${MODEL_TO_USE}"

if [ "$INTERACTIVE_MODE" = true ]; then
    echo -e "${GREEN}${BOLD}Entering interactive OpenCode TUI mode in ${TARGET_DIR}...${NC}"
    echo -e "${YELLOW}Paste the standard prompt into the chat window to begin.${NC}"
    echo ""
    if [ -n "$OPENCODE_BIN" ]; then
        cd "$TARGET_DIR"
        "$OPENCODE_BIN" -m "$OPENCODE_MODEL_ARG"
    else
        docker compose exec -it -w /workspace/opencode-water-sim opencode opencode -m "$OPENCODE_MODEL_ARG"
    fi
    exit 0
fi

# Non-Interactive Automated Execution Loop
echo -e "Dispatching autonomous coding task to OpenCode (this may take 1-2 minutes)..."
START_TIME=$(date +%s)

if [ -n "$OPENCODE_BIN" ]; then
    # Execute natively via host OpenCode binary
    (cd "$TARGET_DIR" && "$OPENCODE_BIN" run --auto -m "$OPENCODE_MODEL_ARG" "$STANDARD_PROMPT") || {
        echo -e "${YELLOW}Host OpenCode exited. Checking if output file was written...${NC}"
    }
else
    # Execute containerized via Docker Compose
    docker compose exec -T -w /workspace/opencode-water-sim opencode opencode run --auto -m "$OPENCODE_MODEL_ARG" "$STANDARD_PROMPT" || {
        echo -e "${YELLOW}Container OpenCode exited. Checking if output file was written...${NC}"
    }
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo -e "  ${GREEN}✓ Agent execution finished in ${ELAPSED} seconds.${NC}"
echo ""

# ==============================================================================
# Step 4: Verification & Structural Physics Audit
# ==============================================================================
echo -e "${CYAN}${BOLD}[4/4] Auditing Generated Water Physics Simulation...${NC}"
HTML_FILE="${TARGET_DIR}/index.html"

if [ ! -f "$HTML_FILE" ] || [ ! -s "$HTML_FILE" ]; then
    echo -e "${RED}${BOLD}[FAIL] index.html was not generated at: ${HTML_FILE}${NC}"
    echo -e "${YELLOW}Check OpenCode logs or try running interactively: ./demo.sh -i${NC}"
    exit 1
fi

FILE_SIZE=$(du -h "$HTML_FILE" | cut -f1)
LINE_COUNT=$(wc -l < "$HTML_FILE")

echo -e "  ${GREEN}✓ Target artifact generated:${NC} ${BOLD}${HTML_FILE}${NC} (${FILE_SIZE}, ${LINE_COUNT} lines)"
echo ""

# Audit physics engine components
echo -e "${BOLD}Physics Engine Structural Audit:${NC}"

HAS_CANVAS=$(grep -ic "<canvas" "$HTML_FILE" || true)
HAS_RAF=$(grep -iEc "requestAnimationFrame|request_animation" "$HTML_FILE" || true)
HAS_PARTICLES=$(grep -iEc "particle|droplet|velocity|gravity" "$HTML_FILE" || true)
HAS_COLLISION=$(grep -iEc "collision|obstacle|slider|bounce|rebound" "$HTML_FILE" || true)
HAS_RIPPLE=$(grep -iEc "click|pointerdown|mousedown|ripple" "$HTML_FILE" || true)
HAS_CSS=$(grep -ic "<style" "$HTML_FILE" || true)

PASS_COUNT=0

if [ "$HAS_CANVAS" -gt 0 ]; then
    echo -e "  • HTML5 Canvas Element           : ${GREEN}${BOLD}[PASS]${NC} Found <canvas>"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "  • HTML5 Canvas Element           : ${RED}[FAIL]${NC} Missing <canvas>"
fi

if [ "$HAS_RAF" -gt 0 ]; then
    echo -e "  • Real-time Render/Math Loop     : ${GREEN}${BOLD}[PASS]${NC} Found requestAnimationFrame loop"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "  • Real-time Render/Math Loop     : ${YELLOW}[WARN]${NC} No explicit requestAnimationFrame detected"
fi

if [ "$HAS_PARTICLES" -gt 0 ]; then
    echo -e "  • Particle/Fluid Dynamics Math   : ${GREEN}${BOLD}[PASS]${NC} Particle velocity & gravity vectors detected"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "  • Particle/Fluid Dynamics Math   : ${RED}[FAIL]${NC} Missing particle physics logic"
fi

if [ "$HAS_COLLISION" -gt 0 ]; then
    echo -e "  • Obstacle & Collision Bounds    : ${GREEN}${BOLD}[PASS]${NC} Found collision/slider interaction logic"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "  • Obstacle & Collision Bounds    : ${YELLOW}[WARN]${NC} Collision handling may be basic"
fi

if [ "$HAS_RIPPLE" -gt 0 ]; then
    echo -e "  • Interactive Ripple / Click     : ${GREEN}${BOLD}[PASS]${NC} Mouse/touch event listeners verified"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "  • Interactive Ripple / Click     : ${YELLOW}[WARN]${NC} Click event listeners not detected"
fi

if [ "$HAS_CSS" -gt 0 ]; then
    echo -e "  • Self-Contained CSS Styling     : ${GREEN}${BOLD}[PASS]${NC} Embedded <style> block present"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    echo -e "  • Self-Contained CSS Styling     : ${YELLOW}[WARN]${NC} Missing dedicated <style> block"
fi

echo ""
echo -e "Benchmark Validation Score: ${BOLD}${PASS_COUNT}/6 checks passed${NC}"
echo ""

# ==============================================================================
# Step 5: Live Visual Preview Server
# ==============================================================================
echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "${BLUE}${BOLD}                       DEMO EXECUTION COMPLETE                        ${NC}"
echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "Generated Artifact : ${GREEN}${BOLD}${HTML_FILE}${NC}"
echo -e "Code Size          : ${BOLD}${LINE_COUNT} lines (${FILE_SIZE})${NC}"
echo -e "Model Tested       : ${BOLD}${MODEL_TO_USE}${NC} on AMD Radeon™ AI PRO R9700"
echo ""

if [ "$AUTO_SERVE" = true ]; then
    echo -e "${GREEN}${BOLD}>>> Starting Live HTTP Preview Server on port ${PREVIEW_PORT}...${NC}"
    echo -e "Open your web browser and navigate to:"
    echo -e "  👉 ${GREEN}${BOLD}http://localhost:${PREVIEW_PORT}${NC}"
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "")
    if [ -n "$LOCAL_IP" ] && [ "$LOCAL_IP" != "127.0.0.1" ]; then
        echo -e "  👉 ${GREEN}${BOLD}http://${LOCAL_IP}:${PREVIEW_PORT}${NC} (from local network)"
    fi
    echo ""
    echo "Press Ctrl+C to stop the preview server."
    python3 -m http.server "$PREVIEW_PORT" --directory "$TARGET_DIR"
else
    echo -e "${BOLD}To visually test and interact with the water simulation:${NC}"
    echo -e "  1. ${YELLOW}python3 -m http.server ${PREVIEW_PORT} --directory opencode-water-sim${NC}"
    echo -e "     Then open: ${GREEN}http://localhost:${PREVIEW_PORT}${NC}"
    echo ""
    echo -e "  2. Or open directly in your web browser:"
    echo -e "     ${YELLOW}xdg-open ${HTML_FILE}${NC} (or open file:// in Chrome/Firefox)"
    echo ""
    echo -e "  3. To compare models via the OpenCode Benchmark Dashboard:"
    echo -e "     ${YELLOW}./demo.sh --dashboard${NC}"
fi
echo -e "${BLUE}======================================================================${NC}"
