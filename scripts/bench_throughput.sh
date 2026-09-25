#!/bin/bash
# ==============================================================================
# bench_throughput.sh - Multi-Engine Token Throughput Benchmarking Suite
# ==============================================================================
# Benchmarks throughput across requested Input/Output length matrix:
#   1. 8192:1024  (8192 input, 1024 output)
#   2. 1024:8192  (1024 input, 8192 output)
#   3. 1024:1024  (1024 input, 1024 output)
# Supports:
#   - vLLM (Default ROCm PagedAttention engine)
#   - llama.cpp (Alternative GGML GGUF engine)
#   - SGLang (Alternative RadixAttention engine)
# Reference: https://docs.vllm.ai/en/latest/cli/bench/throughput/
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "${ROOT_DIR}/lib/gpu_profile.sh"

# 1. Load user environment (~/.env) if present to pull in HF_TOKEN
if [ -f "$HOME/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$HOME/.env"
    set +a
fi

# 2. Load local environment configuration (.env)
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
# The public throughput.sh entrypoint sets this to auto unless explicitly
# overridden. Reapply it after .env so stale host-specific settings do not win.
GPU_PROFILE="${GPU_PROFILE_OVERRIDE:-${GPU_PROFILE:-auto}}"
apply_gpu_profile

# ANSI Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${ROOT_DIR}/_results/throughput/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

CONC="${CONC:-1}"
NUM_PROMPTS_OVERRIDE="${NUM_PROMPTS:-}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8000}"
ENGINE="${INFERENCE_ENGINE:-vllm}"
QUICK_MODE=false
CUSTOM_CASES=""

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

vLLM, vLLM-MXFP4, llama.cpp & SGLang Multi-Length Token Throughput Benchmarking Suite

Options:
  -e, --engine <engine>    Inference engine to benchmark ('vllm', 'mxfp4', 'llama.cpp', 'sglang', or 'all') (default: vllm)
  --engines <list>         Comma-separated list of engines (e.g. 'vllm,mxfp4,llama.cpp,sglang' or 'all')
  -c, --concurrency <N>    Concurrency level (default: 1)
  -n, --num-prompts <N>    Override number of test prompts per slice (default: dynamic by OSL & CONC)
  --test-cases <cases>     Comma-separated I:O token pairs (default: '8192:1024')
  -q, --quick              Quick smoke test (1 prompt, 128:64) for rapid multi-engine validation
  -u, --server-url <URL>   Target server URL (default: http://127.0.0.1:8000)
  --compare-engines        Compare throughput across all engines (equivalent to -e all)
  -h, --help               Show this help message

Dynamic Prompt Sizing:
  • If OSL == 8192: NUM_PROMPTS = CONC * 20 (e.g. 20 prompts at CONC=1)
  • If OSL != 8192: NUM_PROMPTS = CONC * 50 (e.g. 50 prompts at CONC=1)
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--conc|--concurrency)
            CONC="$2"
            shift 2
            ;;
        -e|--engine)
            ENGINE="$2"
            shift 2
            ;;
        --engines)
            ENGINE="$2"
            shift 2
            ;;
        --compare-engines)
            ENGINE="all"
            shift
            ;;
        -n|--num-prompts)
            NUM_PROMPTS_OVERRIDE="$2"
            shift 2
            ;;
        --test-cases)
            CUSTOM_CASES="$2"
            shift 2
            ;;
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        -u|--url|--server-url)
            SERVER_URL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage." >&2
            exit 1
            ;;
    esac
done

# Resolve list of engines to benchmark
declare -a ENGINE_LIST=()
if [ "$ENGINE" = "all" ] || [ "$ENGINE" = "compare" ]; then
    ENGINE_LIST=("vllm" "mxfp4" "llama.cpp" "sglang")
elif [[ "$ENGINE" == *","* ]]; then
    IFS="," read -ra SPLIT_ENGINES <<< "$ENGINE"
    for E in "${SPLIT_ENGINES[@]}"; do
        E_TRIM=$(echo "$E" | xargs)
        [ -n "$E_TRIM" ] && ENGINE_LIST+=("$E_TRIM")
    done
else
    ENGINE_LIST=("$ENGINE")
fi

# Define test cases
declare -a TEST_CASES=()
if [ "$QUICK_MODE" = true ]; then
    TEST_CASES=("128:64")
    [ -z "$NUM_PROMPTS_OVERRIDE" ] && NUM_PROMPTS_OVERRIDE="1"
elif [ -n "$CUSTOM_CASES" ]; then
    IFS="," read -ra SPLIT_CASES <<< "$CUSTOM_CASES"
    for C in "${SPLIT_CASES[@]}"; do
        C_TRIM=$(echo "$C" | xargs)
        [ -n "$C_TRIM" ] && TEST_CASES+=("$C_TRIM")
    done
else
    # Default comparison slice: 8192:1024
    TEST_CASES=(
      "8192:1024"
    )
fi

echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "${BLUE}${BOLD}            AMD GPU Inference Throughput Benchmark Suite              ${NC}"
echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "Hardware Platform : ${BOLD}${GPU_LABEL}${NC}"
echo -e "Target Engines    : ${CYAN}${BOLD}${ENGINE_LIST[*]}${NC}"
echo -e "Concurrency (CONC): ${BOLD}${CONC}${NC}"
if [ -n "${NUM_PROMPTS_OVERRIDE:-}" ]; then
    echo -e "Prompts / Test    : ${BOLD}${NUM_PROMPTS_OVERRIDE} (User Override)${NC}"
else
    echo -e "Prompts / Test    : ${BOLD}Dynamic by OSL (OSL=8192 -> $(( CONC * 20 )), OSL!=8192 -> $(( CONC * 50 )))${NC}"
fi
echo -e "Test Cases Matrix : ${BOLD}${TEST_CASES[*]}${NC}"
echo -e "Results Directory : ${RESULTS_DIR}"
echo ""

# Helper to wait for server health
wait_for_server() {
    local port="${1:-8000}"
    local max_wait="${2:-90}"
    local elapsed=0
    echo -n "Waiting for inference server on port ${port}..."
    while ! (curl -s -f "http://127.0.0.1:${port}/health" >/dev/null 2>&1 || curl -s -f "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1); do
        sleep 2
        elapsed=$((elapsed + 2))
        echo -n "."
        if [ "$elapsed" -ge "$max_wait" ]; then
            echo " TIMEOUT"
            return 1
        fi
    done
    echo " READY (${elapsed}s)"
    return 0
}

# Helper to stop a container reliably across snap AppArmor environments
stop_container() {
    local name="$1"
    if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
        docker update --restart=no "$name" >/dev/null 2>&1 || true
        local pid
        pid=$(docker inspect -f '{{.State.Pid}}' "$name" 2>/dev/null || echo "0")
        if ! docker stop -t 3 "$name" >/dev/null 2>&1; then
            if [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null; then
                sudo -n kill -TERM "$pid" >/dev/null 2>&1 || true
                local count=0
                while sudo -n kill -0 "$pid" >/dev/null 2>&1 && [ "$count" -lt 6 ]; do
                    sleep 0.5
                    count=$((count + 1))
                done
                if sudo -n kill -0 "$pid" >/dev/null 2>&1; then
                    sudo -n kill -9 "$pid" >/dev/null 2>&1 || true
                    sleep 1
                fi
            fi
        fi
        docker rm -f "$name" >/dev/null 2>&1 || sudo -n docker rm -f "$name" >/dev/null 2>&1 || true
    fi
}

# Helper to start an engine
start_engine() {
    local target="$1"
    echo -e "${CYAN}Switching active server to: ${BOLD}${target}${NC}..."
    case "$target" in
        vllm)
            if (docker ps --format '{{.Names}}' | grep -E -q "^(rocm-inference-server|vllm-rocm10-test)$") && curl -s -f "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
                echo "Using currently active vLLM container."
                return 0
            fi
            stop_container "rocm-llama-server"
            stop_container "rocm-sglang-server"
            stop_container "rocm-mxfp4-server"
            docker compose -p coder-vllm -f "${ROOT_DIR}/docker/docker-compose.yml" up -d inference
            wait_for_server 8000 90
            ;;
        mxfp4|vllm-mxfp4|radiance)
            if docker ps --format '{{.Names}}' | grep -q "^rocm-mxfp4-server$" && curl -s -f "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
                echo "Using currently active vLLM MXFP4 container."
                return 0
            fi
            stop_container "rocm-inference-server"
            stop_container "rocm-llama-server"
            stop_container "rocm-sglang-server"
            local mxfp4_model="/models/Qwen3.8-27B-Quark-AWQ-MXFP4"
            if [ -n "${MXFP4_MODEL_PATH:-}" ]; then
                mxfp4_model="${MXFP4_MODEL_PATH}"
            elif [ -d "${ROOT_DIR}/models/Qwen3.8-27B-Quark-AWQ-MXFP4" ]; then
                mxfp4_model="/models/Qwen3.8-27B-Quark-AWQ-MXFP4"
            elif [ -d "${ROOT_DIR}/models/Qwen3.8-27B-MXFP4-mtpfp8" ]; then
                mxfp4_model="/models/Qwen3.8-27B-MXFP4-mtpfp8"
            elif [ -d "${ROOT_DIR}/models/amd/Qwen3.8-27B-Quark-AWQ-MXFP4" ]; then
                mxfp4_model="/models/amd/Qwen3.8-27B-Quark-AWQ-MXFP4"
            else
                echo -e "${RED}MXFP4 weights not found under ${ROOT_DIR}/models. Refusing to start FP8 with --quantization quark.${NC}" >&2
                echo "Set MXFP4_MODEL_PATH or place Qwen3.8-27B-Quark-AWQ-MXFP4 in MODELS_DIR." >&2
                return 1
            fi
            MODEL_PATH="${mxfp4_model}" docker compose -p coder-mxfp4 -f "${ROOT_DIR}/docker/docker-compose.mxfp4.yml" up -d inference
            wait_for_server 8000 120
            ;;
        llama.cpp|llamacpp|gguf)
            local gguf_model="Qwen3.8-27B-Q4_K_M.gguf"
            if [ -n "${MODEL_FILE:-}" ] && [ -f "${ROOT_DIR}/models/${MODEL_FILE}" ]; then
                gguf_model="${MODEL_FILE}"
            elif [ -f "${ROOT_DIR}/models/Qwen3.8-27B-Q4_K_M.gguf" ]; then
                gguf_model="Qwen3.8-27B-Q4_K_M.gguf"
            elif [ -f "${ROOT_DIR}/models/qwen2.5-0.5b-instruct-q4_k_m.gguf" ]; then
                gguf_model="qwen2.5-0.5b-instruct-q4_k_m.gguf"
            fi
            if [ ! -f "${ROOT_DIR}/models/${gguf_model}" ]; then
                echo -e "${YELLOW}Notice: GGUF weights not found at ./models/${gguf_model}.${NC}"
                return 1
            fi
            if docker ps --format '{{.Names}}' | grep -q "^rocm-llama-server$" && curl -s -f "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
                echo "Using currently active llama.cpp container."
                return 0
            fi
            stop_container "rocm-inference-server"
            stop_container "rocm-sglang-server"
            stop_container "rocm-mxfp4-server"
            MODEL_FILE="${gguf_model}" MODEL_ALIAS="${gguf_model}" docker compose -p coder-llama -f "${ROOT_DIR}/docker/docker-compose.gguf.yml" up -d inference
            wait_for_server 8000 60
            ;;
        sglang)
            if docker ps --format '{{.Names}}' | grep -q "^rocm-sglang-server$" && (curl -s -f "http://127.0.0.1:8000/health" >/dev/null 2>&1 || curl -s -f "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1); then
                echo "Using currently active SGLang container."
                return 0
            fi
            stop_container "rocm-inference-server"
            stop_container "rocm-llama-server"
            stop_container "rocm-mxfp4-server"
            local sglang_model="${SGLANG_MODEL_PATH:-}"
            local sglang_tok="${SGLANG_TOKENIZER_PATH:-}"
            local sglang_fmt="${SGLANG_LOAD_FORMAT:-auto}"
            if [ -z "$sglang_model" ] && [ -d "${ROOT_DIR}/models/Qwen3.8-27B-Quark-AWQ-MXFP4" ]; then
                sglang_model="/models/Qwen3.8-27B-Quark-AWQ-MXFP4"
                sglang_fmt="auto"
            elif [ -z "$sglang_model" ] && [ -f "${ROOT_DIR}/models/Qwen3.8-27B-Q4_K_M.gguf" ]; then
                sglang_model="/models/Qwen3.8-27B-Q4_K_M.gguf"
                sglang_tok="${sglang_tok:-Qwen/Qwen3.8-27B-FP8}"
                sglang_fmt="gguf"
            elif [ -z "$sglang_model" ] && [ -f "${ROOT_DIR}/models/qwen2.5-0.5b-instruct-q4_k_m.gguf" ]; then
                sglang_model="/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
                sglang_tok="Qwen/Qwen2.5-0.5B-Instruct"
                sglang_fmt="gguf"
            elif [ -z "$sglang_model" ]; then
                sglang_model="${MODEL_NAME:-Qwen/Qwen3.8-27B-FP8}"
                sglang_fmt="auto"
            fi
            echo "SGLang model=${sglang_model} load-format=${sglang_fmt}"
            SGLANG_MODEL_PATH="${sglang_model}" \
              SGLANG_TOKENIZER_PATH="${sglang_tok}" \
              SGLANG_LOAD_FORMAT="${sglang_fmt}" \
              SGLANG_SERVED_MODEL_NAME="${SGLANG_SERVED_MODEL_NAME:-$(basename "${sglang_model}")}" \
              docker compose -p coder-sglang -f "${ROOT_DIR}/docker/docker-compose.sglang.yml" up -d inference
            wait_for_server 8000 180
            ;;
        *)
            echo -e "${RED}Unknown engine target: ${target}${NC}"
            return 1
            ;;
    esac
}

# Summary tracking table
declare -a TABLE_ROWS=()
BENCH_FAILURES=0

for CURR_ENGINE in "${ENGINE_LIST[@]}"; do
    echo ""
    echo -e "${BLUE}${BOLD}======================================================================${NC}"
    echo -e "${BLUE}${BOLD}>>> BENCHMARKING ENGINE: ${CURR_ENGINE^^}${NC}"
    echo -e "${BLUE}${BOLD}======================================================================${NC}"

    if ! start_engine "$CURR_ENGINE"; then
        echo -e "${YELLOW}Skipping throughput run for ${CURR_ENGINE} (engine unavailable).${NC}"
        BENCH_FAILURES=$((BENCH_FAILURES + 1))
        for TC in "${TEST_CASES[@]}"; do
            IFS=":" read -r IN_LEN OUT_LEN <<< "$TC"
            TABLE_ROWS+=("${CURR_ENGINE}|${IN_LEN}|${OUT_LEN}|N/A|UNAVAILABLE|UNAVAILABLE|UNAVAILABLE|UNAVAILABLE|UNAVAILABLE")
        done
        continue
    fi

    # Query active model ID
    MODEL_NAME=$(curl -s "http://127.0.0.1:8000/v1/models" 2>/dev/null | jq -r '.data[0].id // "Unknown"' || echo "Unknown")
    echo -e "Active Model ID   : ${GREEN}${BOLD}${MODEL_NAME}${NC}"
    echo ""

    TOTAL_CASES=${#TEST_CASES[@]}
    CURRENT_CASE=0

    for TC in "${TEST_CASES[@]}"; do
        CURRENT_CASE=$((CURRENT_CASE + 1))
        IFS=":" read -r IN_LEN OUT_LEN <<< "$TC"
        ISL="$IN_LEN"
        OSL="$OUT_LEN"

        # Dynamic prompt sizing based on OSL and CONC
        if [ -n "${NUM_PROMPTS_OVERRIDE:-}" ]; then
            export NUM_PROMPTS="$NUM_PROMPTS_OVERRIDE"
        else
            if [[ "$OSL" == "8192" ]]; then
                export NUM_PROMPTS=$(( CONC * 20 ))
            else
                export NUM_PROMPTS=$(( CONC * 50 ))
            fi
        fi

        echo -e "${CYAN}${BOLD}[${CURR_ENGINE}] [${CURRENT_CASE}/${TOTAL_CASES}] Testing: Input=${ISL} | Output=${OSL} | Prompts=${NUM_PROMPTS} | Concurrency=${CONC}...${NC}"

        RESULT_FILE="bench_${CURR_ENGINE}_${ISL}_${OSL}.json"
        FULL_PATH="${RESULTS_DIR}/${RESULT_FILE}"
        CONTAINER_RES_DIR="/results"
        # Determine matching HF tokenizer for client-side token counting
        tok_args=()
        if [[ "$MODEL_NAME" == *".gguf"* ]] || [[ "$MODEL_NAME" != *"/"* ]] || [[ "$MODEL_NAME" == *"MXFP4"* ]]; then
            if [[ "$MODEL_NAME" == *"0.5b"* ]] || [[ "$MODEL_NAME" == *"0.5B"* ]]; then
                tok_args=(--tokenizer "Qwen/Qwen2.5-0.5B-Instruct")
            else
                tok_args=(--tokenizer "Qwen/Qwen3.8-27B-FP8")
            fi
        fi

        # Execute vllm bench serve
        ACTIVE_VLLM=$(docker ps --format '{{.Names}}' | grep -E "^(rocm-inference-server|vllm-rocm10-test|rocm-mxfp4-server|qwen38-r9700-radiance)$" | head -n1 || true)
        if { [ "$CURR_ENGINE" = "vllm" ] || [ "$CURR_ENGINE" = "mxfp4" ]; } && [ -n "$ACTIVE_VLLM" ]; then
            docker exec "$ACTIVE_VLLM" vllm bench serve \
              --backend openai-chat \
              --model "$MODEL_NAME" \
              "${tok_args[@]}" \
              --endpoint /v1/chat/completions \
              --host 127.0.0.1 \
              --port 8000 \
              --dataset-name random \
              --random-input-len "$ISL" \
              --random-output-len "$OSL" \
              --num-prompts "$NUM_PROMPTS" \
              --max-concurrency "$CONC" \
              --request-rate inf \
              --save-result \
              --result-dir "/results" \
              --result-filename "$RESULT_FILE" > "${RESULTS_DIR}/${CURR_ENGINE}_bench_${ISL}_${OSL}.log" 2>&1 || true
            # Copy result if placed inside /results
            [ -f "${SCRIPT_DIR}/_results/${RESULT_FILE}" ] && cp -f "${SCRIPT_DIR}/_results/${RESULT_FILE}" "${FULL_PATH}" 2>/dev/null || true
        else
            # Standalone containerized benchmark client targeting http://127.0.0.1:8000
            docker run --rm --network host \
              --device /dev/kfd --device /dev/dri \
              --group-add "${VIDEO_GID:-44}" --group-add "${RENDER_GID:-109}" \
              --security-opt seccomp=unconfined \
              --security-opt apparmor=unconfined \
              -e HIP_VISIBLE_DEVICES=0 \
              -e HF_TOKEN="${HF_TOKEN:-}" \
              -e HF_HOME=/root/.cache/huggingface \
              -v "${HF_HOME}:/root/.cache/huggingface" \
              -v "${RESULTS_DIR}:${CONTAINER_RES_DIR}" \
              --entrypoint python ${VLLM_BENCH_IMAGE:-${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}} -m vllm.entrypoints.cli.main bench serve \
              --backend openai-chat \
              --model "$MODEL_NAME" \
              "${tok_args[@]}" \
              --endpoint /v1/chat/completions \
              --host 127.0.0.1 \
              --port 8000 \
              --dataset-name random \
              --random-input-len "$ISL" \
              --random-output-len "$OSL" \
              --num-prompts "$NUM_PROMPTS" \
              --max-concurrency "$CONC" \
              --request-rate inf \
              --save-result \
              --result-dir "$CONTAINER_RES_DIR" \
              --result-filename "$RESULT_FILE" > "${RESULTS_DIR}/${CURR_ENGINE}_bench_${ISL}_${OSL}.log" 2>&1 || true
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
            TABLE_ROWS+=("${CURR_ENGINE}|${ISL}|${OSL}|${NUM_PROMPTS}|${OUT_TOK_S}|${TOTAL_TOK_S}|${TTFT_MS}|${TPOT_MS}|${DURATION_S}")
        else
            echo -e "    ${RED}✗ Benchmark failed for ${CURR_ENGINE} on ${ISL}:${OSL}${NC}"
            TABLE_ROWS+=("${CURR_ENGINE}|${ISL}|${OSL}|${NUM_PROMPTS}|ERROR|ERROR|ERROR|ERROR|ERROR")
            BENCH_FAILURES=$((BENCH_FAILURES + 1))
        fi
        echo ""
    done
done

# Restore default vLLM engine if multiple engines were benchmarked
if [ "${#ENGINE_LIST[@]}" -gt 1 ] || [ "${ENGINE_LIST[0]}" != "vllm" ]; then
    echo -e "${CYAN}Restoring default inference engine: vLLM...${NC}"
    stop_container "rocm-llama-server"
    stop_container "rocm-sglang-server"
    docker compose -p coder-vllm -f "${ROOT_DIR}/docker/docker-compose.yml" up -d inference >/dev/null 2>&1 || true
fi

# Ensure results directory permissions
chmod -R u+rwX,g+rwX "$RESULTS_DIR" 2>/dev/null || true

# ------------------------------------------------------------------------------
# Terminal Formatted Results Table
# ------------------------------------------------------------------------------
echo -e "${BLUE}${BOLD}===============================================================================================================${NC}"
echo -e "${BLUE}${BOLD}                           THROUGHPUT BENCHMARK RESULTS SUMMARY MATRIX                                         ${NC}"
echo -e "${BLUE}${BOLD}===============================================================================================================${NC}"
printf "%-11s | %-10s | %-11s | %-8s | %-16s | %-15s | %-12s | %-12s\n" \
  "Engine" "Input (tok)" "Output (tok)" "Prompts" "Output Throughput" "Total Throughput" "Mean TTFT" "Mean TPOT"
echo "---------------------------------------------------------------------------------------------------------------"

for ROW in "${TABLE_ROWS[@]}"; do
    IFS="|" read -r R_ENG R_IN R_OUT R_NUM R_OUT_S R_TOT_S R_TTFT R_TPOT R_DUR <<< "$ROW"
    printf "%-11s | %-10s | %-11s | %-8s | %-16s | %-15s | %-12s | %-12s\n" \
      "${R_ENG}" "${R_IN}" "${R_OUT}" "${R_NUM}" "${R_OUT_S} tok/s" "${R_TOT_S} tok/s" "${R_TTFT} ms" "${R_TPOT} ms"
done

echo "---------------------------------------------------------------------------------------------------------------"
echo -e "Hardware Device : ${GPU_LABEL}"
echo -e "Concurrency     : ${BOLD}${CONC}${NC}"
echo -e "Metrics Folder  : ${RESULTS_DIR}"

# ------------------------------------------------------------------------------
# Generate Markdown Report File
# ------------------------------------------------------------------------------
REPORT_FILE="${RESULTS_DIR}/throughput_benchmark_report.md"
cat << EOF > "$REPORT_FILE"
# Multi-Engine Throughput Benchmark Performance Report

- **Target Hardware**: ${GPU_LABEL}
- **GPU Profile**: \`${GPU_PROFILE}\`
- **ROCm ISA**: \`${PYTORCH_ROCM_ARCH}\`
- **Engines Tested**: \`${ENGINE_LIST[*]}\`
- **Benchmark Suite**: vLLM Throughput Matrix (\`vllm bench serve\`)
- **Reference**: [vLLM Throughput CLI Docs](https://docs.vllm.ai/en/latest/cli/bench/throughput/)
- **Timestamp**: $(date)
- **Concurrency (CONC)**: ${CONC}
- **Prompt Sizing Strategy**: Dynamic by OSL (OSL=8192 -> $(( CONC * 20 )), OSL!=8192 -> $(( CONC * 50 )))

## Comparative Throughput & Latency Matrix

| Engine | Input Tokens (ISL) | Output Tokens (OSL) | Prompts | Total Tokens | Output Throughput | Total Throughput | Mean TTFT | Mean TPOT | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
EOF

for ROW in "${TABLE_ROWS[@]}"; do
    IFS="|" read -r R_ENG R_IN R_OUT R_NUM R_OUT_S R_TOT_S R_TTFT R_TPOT R_DUR <<< "$ROW"
    TOTAL_TOK="N/A"
    if [[ "${R_NUM}" =~ ^[0-9]+$ ]]; then
        TOTAL_TOK=$(( (R_IN + R_OUT) * R_NUM ))
    fi
    cat << EOF >> "$REPORT_FILE"
| **${R_ENG}** | **${R_IN}** | **${R_OUT}** | ${R_NUM} | ${TOTAL_TOK} | **${R_OUT_S} tok/s** | **${R_TOT_S} tok/s** | ${R_TTFT} ms | ${R_TPOT} ms | ${R_DUR} s |
EOF
done

cat << EOF >> "$REPORT_FILE"

## Metric Definitions
- **Input Tokens (ISL)**: Number of prompt context tokens fed into the model.
- **Output Tokens (OSL)**: Number of generative completion tokens sampled.
- **Prompts**: Number of requests executed in this test slice.
- **Output Throughput**: Speed of generated tokens (\`completion_tokens / duration\`).
- **Total Throughput**: Combined prefill and decode token processing speed (\`(input_tokens + output_tokens) / duration\`).
- **Mean TTFT (Time to First Token)**: Prefill latency before the first token is emitted.
- **Mean TPOT (Time per Output Token)**: Average decode step time per subsequent token.
EOF

echo ""
echo -e "${GREEN}${BOLD}✓ Archival Markdown Report Saved:${NC} ${REPORT_FILE}"
echo -e "${BLUE}===============================================================================================================${NC}"

if [ "$BENCH_FAILURES" -gt 0 ]; then
    echo -e "${RED}${BENCH_FAILURES} throughput benchmark slice(s) failed or were unavailable.${NC}" >&2
    exit 1
fi
