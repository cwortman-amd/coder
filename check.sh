#!/bin/bash
# ==============================================================================
# check.sh - Health & Verification Check for Local ROCm Inference Server
# ==============================================================================
set -euo pipefail

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Determine project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load user environment (~/.env) if present
if [ -f "$HOME/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$HOME/.env"
    set +a
fi

# Load .env if present
if [ -f "${SCRIPT_DIR}/.env" ]; then
    PREV_HF_TOKEN="${HF_TOKEN:-}"
    # Export non-comment variables
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^[[:space:]]*#' "${SCRIPT_DIR}/.env" | grep -v '^[[:space:]]*$')
    set +a
    if [ -z "${HF_TOKEN:-}" ] && [ -n "${PREV_HF_TOKEN}" ]; then
        export HF_TOKEN="${PREV_HF_TOKEN}"
    fi
fi

PORT="${INFERENCE_PORT:-8000}"
HOST="${INFERENCE_HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}"
WAIT_TIMEOUT=0

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)
            PORT="$2"
            BASE_URL="http://${HOST}:${PORT}"
            shift 2
            ;;
        -w|--wait)
            WAIT_TIMEOUT="${2:-60}"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./check.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -p, --port <port>     Inference server port (default: 8000 or from .env)"
            echo "  -w, --wait <seconds>  Wait up to N seconds for server to become healthy (default: 0)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Run ./check.sh --help for usage."
            exit 1
            ;;
    esac
done

CHECKS_DIR="${SCRIPT_DIR}/_results/checks"
mkdir -p "$CHECKS_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CHECK_LOG="${CHECKS_DIR}/check_${TIMESTAMP}.log"
CHECK_REPORT="${CHECKS_DIR}/check_summary_${TIMESTAMP}.md"

# Redirect execution log for reviewing while displaying to terminal
exec > >(tee -a "$CHECK_LOG") 2>&1

echo -e "${BLUE}${BOLD}============================================================${NC}"
echo -e "${BLUE}${BOLD} ROCm Inference Server Health & Response Verification Check ${NC}"
echo -e "${BLUE}${BOLD}============================================================${NC}"
echo -e "Target Server: ${BOLD}${BASE_URL}${NC}"
echo -e "Review Log   : ${BOLD}${CHECK_LOG}${NC}"
echo ""

# Check dependencies
if ! command -v curl &>/dev/null; then
    echo -e "${RED}[FAIL] 'curl' is required but not installed.${NC}"
    exit 1
fi
if ! command -v jq &>/dev/null; then
    echo -e "${RED}[FAIL] 'jq' is required but not installed.${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 1. Health Check
# ------------------------------------------------------------------------------
echo -n "1. Checking server health status... "
START_TIME=$(date +%s)
HEALTH_HTTP_CODE="000"

while true; do
    HEALTH_HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "${BASE_URL}/health" 2>/dev/null || echo "000")
    if [ "$HEALTH_HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}${BOLD}[HEALTHY (HTTP 200)]${NC}"
        break
    fi

    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ "$ELAPSED" -ge "$WAIT_TIMEOUT" ]; then
        echo -e "${RED}${BOLD}[UNHEALTHY / UNREACHABLE]${NC}"
        echo -e "${RED}Server at ${BASE_URL}/health responded with HTTP ${HEALTH_HTTP_CODE}.${NC}"
        echo -e "${YELLOW}Tip: If the container was recently started, models may still be downloading/compiling.${NC}"
        echo -e "${YELLOW}     Check logs: docker compose logs -f inference${NC}"
        echo -e "${YELLOW}     Or retry with wait: ./check.sh --wait 60${NC}"
        exit 1
    fi

    sleep 2
done

# ------------------------------------------------------------------------------
# 2. Query Loaded Model Registry
# ------------------------------------------------------------------------------
echo -n "2. Querying loaded model registry... "
MODELS_RESPONSE=$(curl -s --connect-timeout 5 "${BASE_URL}/v1/models" 2>/dev/null || echo "{}")
ACTIVE_MODEL=$(echo "$MODELS_RESPONSE" | jq -r '.data[0].id // empty' 2>/dev/null || true)

if [ -z "$ACTIVE_MODEL" ] || [ "$ACTIVE_MODEL" = "null" ]; then
    echo -e "${RED}${BOLD}[FAILED]${NC}"
    echo -e "${RED}No active models returned by ${BASE_URL}/v1/models.${NC}"
    echo "$MODELS_RESPONSE"
    exit 1
fi

MODEL_COUNT=$(echo "$MODELS_RESPONSE" | jq -r '.data | length' 2>/dev/null || echo "1")
echo -e "${GREEN}${BOLD}[OK]${NC}"
echo -e "   - Model ID:       ${GREEN}${BOLD}${ACTIVE_MODEL}${NC}"
echo -e "   - Total Models:   ${MODEL_COUNT}"

# ------------------------------------------------------------------------------
# 3. Hello World Prompt Verification
# ------------------------------------------------------------------------------
echo -n "3. Sending 'Hello World' verification prompt... "

PROMPT_PAYLOAD=$(jq -n \
  --arg model "$ACTIVE_MODEL" \
  --arg prompt "Hello world! Respond with a brief greeting." \
  '{
    model: $model,
    messages: [
      {role: "user", content: $prompt}
    ],
    max_tokens: 150,
    temperature: 0.7
  }')

COMPLETION_RESPONSE=$(curl -s --connect-timeout 30 \
  -H "Content-Type: application/json" \
  -X POST "${BASE_URL}/v1/chat/completions" \
  -d "$PROMPT_PAYLOAD" 2>/dev/null || echo "{}")

# Extract content and token usage
CONTENT=$(echo "$COMPLETION_RESPONSE" | jq -r '.choices[0].message.content // empty' 2>/dev/null || true)
PROMPT_TOKENS=$(echo "$COMPLETION_RESPONSE" | jq -r '.usage.prompt_tokens // "?"' 2>/dev/null || echo "?")
COMPL_TOKENS=$(echo "$COMPLETION_RESPONSE" | jq -r '.usage.completion_tokens // "?"' 2>/dev/null || echo "?")
TOTAL_TOKENS=$(echo "$COMPLETION_RESPONSE" | jq -r '.usage.total_tokens // "?"' 2>/dev/null || echo "?")

if [ -z "$CONTENT" ]; then
    echo -e "${RED}${BOLD}[FAILED]${NC}"
    echo -e "${RED}Did not receive a valid completion response from server.${NC}"
    echo "$COMPLETION_RESPONSE" | jq . 2>/dev/null || echo "$COMPLETION_RESPONSE"
    exit 1
fi

echo -e "${GREEN}${BOLD}[RESPONSE RECEIVED]${NC}"
echo ""
echo -e "${BOLD}Model Response:${NC}"
echo -e "${BLUE}------------------------------------------------------------${NC}"
# Clean thinking tags if reasoning model produced them
CLEAN_CONTENT=$(echo "$CONTENT" | python3 -c '
import sys, re
raw = sys.stdin.read()
cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
print(cleaned if cleaned else raw.strip())
')
echo "$CLEAN_CONTENT"
echo -e "${BLUE}------------------------------------------------------------${NC}"
echo -e "Usage: ${BOLD}${PROMPT_TOKENS}${NC} prompt tokens + ${BOLD}${COMPL_TOKENS}${NC} completion tokens = ${BOLD}${TOTAL_TOKENS}${NC} total"
echo ""

# Write reviewable summary report
cat << EOF > "$CHECK_REPORT"
# Inference Health & Verification Check Report

- **Date**: $(date)
- **Target Server**: \`${BASE_URL}\`
- **Active Model**: \`${ACTIVE_MODEL}\`
- **Health Status**: \`HTTP 200 (HEALTHY)\`
- **Verification Prompt**: \`Hello world! Respond with a brief greeting.\`
- **Usage**: ${PROMPT_TOKENS} prompt tokens + ${COMPL_TOKENS} completion tokens = ${TOTAL_TOKENS} total
- **Execution Log**: \`${CHECK_LOG}\`

## Model Response
\`\`\`text
${CLEAN_CONTENT}
\`\`\`
EOF

echo -e "${GREEN}${BOLD}✓ Review Report Saved:${NC} ${CHECK_REPORT}"
echo -e "${GREEN}${BOLD}✓ Verification Complete: Inference server is healthy, serving '${ACTIVE_MODEL}', and responding to queries!${NC}"
exit 0
