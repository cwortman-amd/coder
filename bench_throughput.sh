#!/bin/bash
# ==============================================================================
# bench_throughput.sh - Multi-Length Token Throughput Benchmarking Suite
# ==============================================================================
# Benchmarks vLLM model throughput across requested Input/Output length matrix:
#   1. 2048:512   (2048 input, 512 output)
#   2. 2048:2048  (2048 input, 2048 output)
#   3. 128:2048   (128 input, 2048 output)
#   4. 1024:1024  (1024 input, 1024 output)
#   5. 8192:1024  (8192 input, 1024 output)
#   6. 1024:8192  (1024 input, 8192 output)
# Reference: https://docs.vllm.ai/en/latest/cli/bench/throughput/
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ANSI Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${SCRIPT_DIR}/benchmark_results/throughput/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

NUM_PROMPTS="${NUM_PROMPTS:-2}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8000}"

echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "${BLUE}${BOLD}   vLLM ROCm Model Throughput Benchmark Suite (Radeon AI PRO R9700)   ${NC}"
echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "Target Server     : ${BOLD}${SERVER_URL}${NC}"
echo -e "Hardware Platform : ${BOLD}AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB VRAM)${NC}"
echo -e "Prompts / Test    : ${BOLD}${NUM_PROMPTS}${NC}"
echo -e "Reference Docs    : ${CYAN}https://docs.vllm.ai/en/latest/cli/bench/throughput/${NC}"
echo ""

# Pre-flight check
if ! curl -sf --connect-timeout 3 "${SERVER_URL}/health" >/dev/null 2>&1; then
    echo -e "${RED}[FAIL] Inference server at ${SERVER_URL} is not responding.${NC}"
    echo -e "${YELLOW}Please start it first using: ./setup.sh${NC}"
    exit 1
fi

MODEL_NAME=$(curl -s "${SERVER_URL}/v1/models" | jq -r '.data[0].id // "Unknown"')
echo -e "Active Model      : ${GREEN}${BOLD}${MODEL_NAME}${NC}"
echo ""

# Define requested test cases: "INPUT_LEN:OUTPUT_LEN"
TEST_CASES=(
  "2048:512"
  "2048:2048"
  "128:2048"
  "1024:1024"
  "8192:1024"
  "1024:8192"
)

TOTAL_CASES=${#TEST_CASES[@]}
CURRENT_CASE=0

# Summary tracking arrays
declare -a TABLE_ROWS

for TC in "${TEST_CASES[@]}"; do
    CURRENT_CASE=$((CURRENT_CASE + 1))
    IFS=":" read -r IN_LEN OUT_LEN <<< "$TC"
    
    echo -e "${CYAN}${BOLD}>>> [${CURRENT_CASE}/${TOTAL_CASES}] Testing Configuration: Input=${IN_LEN} tokens | Output=${OUT_LEN} tokens...${NC}"
    
    RESULT_FILE="bench_${IN_LEN}_${OUT_LEN}.json"
    CONTAINER_RES_DIR="/results"

    # Run vllm bench serve: use docker exec against running server container if available, else docker run
    if docker ps --format '{{.Names}}' | grep -q "^rocm-inference-server$"; then
        docker exec rocm-inference-server vllm bench serve \
          --backend openai-chat \
          --model "$MODEL_NAME" \
          --endpoint /v1/chat/completions \
          --host 127.0.0.1 \
          --port 8000 \
          --dataset-name random \
          --random-input-len "$IN_LEN" \
          --random-output-len "$OUT_LEN" \
          --num-prompts "$NUM_PROMPTS" \
          --request-rate inf \
          --save-result \
          --result-dir "/workspace/benchmark_results/throughput/${TIMESTAMP}" \
          --result-filename "$RESULT_FILE" > "${RESULTS_DIR}/bench_${IN_LEN}_${OUT_LEN}.log" 2>&1 || true
    else
        docker run --rm --network host \
          --device /dev/kfd --device /dev/dri \
          --group-add video --group-add render \
          --security-opt seccomp=unconfined \
          -e HIP_VISIBLE_DEVICES=0 \
          -v "${RESULTS_DIR}:${CONTAINER_RES_DIR}" \
          --entrypoint python3 vllm/vllm-openai-rocm:latest -m vllm.entrypoints.cli.main bench serve \
          --backend openai-chat \
          --model "$MODEL_NAME" \
          --endpoint /v1/chat/completions \
          --host 127.0.0.1 \
          --port 8000 \
          --dataset-name random \
          --random-input-len "$IN_LEN" \
          --random-output-len "$OUT_LEN" \
          --num-prompts "$NUM_PROMPTS" \
          --request-rate inf \
          --save-result \
          --result-dir "$CONTAINER_RES_DIR" \
          --result-filename "$RESULT_FILE" >/dev/null 2>&1 || true
    fi

    # Parse and report metrics
    FULL_PATH="${RESULTS_DIR}/${RESULT_FILE}"
    if [ -f "$FULL_PATH" ]; then
        OUT_TOK_S=$(jq -r '.output_throughput // 0' "$FULL_PATH" | awk '{printf "%.2f", $1}')
        TOTAL_TOK_S=$(jq -r '.total_token_throughput // 0' "$FULL_PATH" | awk '{printf "%.2f", $1}')
        TTFT_MS=$(jq -r '.mean_ttft_ms // 0' "$FULL_PATH" | awk '{printf "%.2f", $1}')
        TPOT_MS=$(jq -r '.mean_tpot_ms // 0' "$FULL_PATH" | awk '{printf "%.2f", $1}')
        DURATION_S=$(jq -r '.duration // 0' "$FULL_PATH" | awk '{printf "%.2f", $1}')
        
        echo -e "    ${GREEN}✓ Done in ${DURATION_S}s${NC} | Output: ${BOLD}${OUT_TOK_S} tok/s${NC} | Total: ${BOLD}${TOTAL_TOK_S} tok/s${NC} | TTFT: ${TTFT_MS} ms | TPOT: ${TPOT_MS} ms"
        TABLE_ROWS+=("${IN_LEN}|${OUT_LEN}|${OUT_TOK_S}|${TOTAL_TOK_S}|${TTFT_MS}|${TPOT_MS}|${DURATION_S}")
    else
        echo -e "    ${RED}✗ Benchmark failed for ${IN_LEN}:${OUT_LEN}${NC}"
        TABLE_ROWS+=("${IN_LEN}|${OUT_LEN}|ERROR|ERROR|ERROR|ERROR|ERROR")
    fi
    echo ""
done

# Ensure results directory permissions
chmod -R ugo+rwX "$RESULTS_DIR" 2>/dev/null || true

# ------------------------------------------------------------------------------
# Terminal Formatted Results Table
# ------------------------------------------------------------------------------
echo -e "${BLUE}${BOLD}========================================================================================${NC}"
echo -e "${BLUE}${BOLD}                   THROUGHPUT BENCHMARK RESULTS SUMMARY MATRIX                         ${NC}"
echo -e "${BLUE}${BOLD}========================================================================================${NC}"
printf "%-10s | %-11s | %-16s | %-15s | %-12s | %-12s\n" \
  "Input (tok)" "Output (tok)" "Output Throughput" "Total Throughput" "Mean TTFT" "Mean TPOT"
echo "----------------------------------------------------------------------------------------"

for ROW in "${TABLE_ROWS[@]}"; do
    IFS="|" read -r R_IN R_OUT R_OUT_S R_TOT_S R_TTFT R_TPOT R_DUR <<< "$ROW"
    printf "%-10s | %-11s | %-16s | %-15s | %-12s | %-12s\n" \
      "${R_IN}" "${R_OUT}" "${R_OUT_S} tok/s" "${R_TOT_S} tok/s" "${R_TTFT} ms" "${R_TPOT} ms"
done

echo "----------------------------------------------------------------------------------------"
echo -e "Tested Model    : ${BOLD}${MODEL_NAME}${NC}"
echo -e "Hardware Device : AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6)"
echo -e "Metrics Folder  : ${RESULTS_DIR}"

# ------------------------------------------------------------------------------
# Generate Markdown Report File
# ------------------------------------------------------------------------------
REPORT_FILE="${RESULTS_DIR}/throughput_benchmark_report.md"
cat << EOF > "$REPORT_FILE"
# Throughput Benchmark Performance Report

- **Model Evaluated**: \`${MODEL_NAME}\`
- **Target Hardware**: AMD Radeon™ AI PRO R9700 (\`gfx1201\`, 32 GB GDDR6 VRAM)
- **Benchmark Suite**: vLLM Throughput Matrix (\`vllm bench\`)
- **Reference**: [vLLM Throughput CLI Docs](https://docs.vllm.ai/en/latest/cli/bench/throughput/)
- **Timestamp**: $(date)
- **Prompts per Configuration**: ${NUM_PROMPTS}

## Throughput & Latency Matrix

| Input Tokens (ISL) | Output Tokens (OSL) | Total Tokens | Output Throughput | Total Throughput | Mean TTFT | Mean TPOT | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
EOF

for ROW in "${TABLE_ROWS[@]}"; do
    IFS="|" read -r R_IN R_OUT R_OUT_S R_TOT_S R_TTFT R_TPOT R_DUR <<< "$ROW"
    TOTAL_TOK=$(( R_IN + R_OUT )) 2>/dev/null || TOTAL_TOK="N/A"
    cat << EOF >> "$REPORT_FILE"
| **${R_IN}** | **${R_OUT}** | ${TOTAL_TOK} | **${R_OUT_S} tok/s** | **${R_TOT_S} tok/s** | ${R_TTFT} ms | ${R_TPOT} ms | ${R_DUR} s |
EOF
done

cat << EOF >> "$REPORT_FILE"

## Metric Definitions
- **Input Tokens (ISL)**: Number of prompt context tokens fed into the model.
- **Output Tokens (OSL)**: Number of generative completion tokens sampled.
- **Output Throughput**: Speed of generated tokens (\`completion_tokens / duration\`).
- **Total Throughput**: Combined prefill and decode token processing speed (\`(input_tokens + output_tokens) / duration\`).
- **Mean TTFT (Time to First Token)**: Prefill latency before the first token is emitted.
- **Mean TPOT (Time per Output Token)**: Average decode step time per subsequent token.
EOF

echo ""
echo -e "${GREEN}${BOLD}✓ Archival Markdown Report Saved:${NC} ${REPORT_FILE}"
echo -e "${BLUE}========================================================================================${NC}"
