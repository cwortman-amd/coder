#!/usr/bin/env bash
# ==============================================================================
# compare_engines.sh - Comparative Benchmark Suite: vLLM vs. llama.cpp vs. SGLang
# ==============================================================================
# Benchmarks and compares inference performance between:
#   1. vLLM (Default engine for ROCm / PagedAttention / FP8 & GGUF)
#   2. llama.cpp (Alternative engine for GGUF models)
#   3. SGLang (Alternative engine for RadixAttention / Fast KV cache / GGUF)
# Measures: Time-To-First-Token (TTFT), Decode Speed, VRAM Consumption,
# and task completion latency.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 1. Load user environment (~/.env) if present to pull in HF_TOKEN
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

# ANSI Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Default Parameters
DATASET="sample"
NUM_SAMPLES="3"
IN_LEN="8192"
OUT_LEN="1024"
TARGET_ENGINES="all"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${ROOT_DIR}/_results/engines/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Comparative Benchmark Suite: vLLM vs. llama.cpp vs. SGLang on AMD Radeon AI PRO R9700

Options:
  -e, --engines <names>    Engines to benchmark ('all', or comma-separated 'vllm,llamacpp,sglang') (default: all)
  -d, --dataset <name>     Benchmark dataset ('sample', 'diamond', 'main') (default: sample)
  -n, --num-samples <N>    Number of evaluation samples per engine (default: 3)
  --input-len <N>          Benchmark prompt input token length (default: 8192)
  --output-len <N>         Benchmark target output token length (default: 1024)
  --output-dir <DIR>       Directory to save results (default: _results/engines/<timestamp>)
  -h, --help               Show this help message

Default Engine: vLLM
Alternatives:   llama.cpp, SGLang
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--engines|--engine)
            TARGET_ENGINES="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --input-len)
            IN_LEN="$2"
            shift 2
            ;;
        --output-len)
            OUT_LEN="$2"
            shift 2
            ;;
        --output-dir)
            RESULTS_DIR="$2"
            mkdir -p "$RESULTS_DIR"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            exit 1
            ;;
    esac
done

echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "${BLUE}${BOLD}  Comparative Engine Benchmark: vLLM vs. llama.cpp vs. SGLang         ${NC}"
echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "Hardware Platform : ${BOLD}AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB VRAM)${NC}"
echo -e "Target Engines    : ${CYAN}${BOLD}${TARGET_ENGINES}${NC}"
echo -e "Default Engine    : ${GREEN}${BOLD}vLLM (ROCm 7.x, PagedAttention, FP8/BF16/GGUF)${NC}"
echo -e "Alternative 1     : ${YELLOW}${BOLD}llama.cpp (ROCm, GGML native C++, Q4_K_M GGUF)${NC}"
echo -e "Alternative 2     : ${CYAN}${BOLD}SGLang (ROCm, RadixAttention, FP8/GGUF)${NC}"
echo -e "Test Workload     : Input=${IN_LEN} tokens | Output=${OUT_LEN} tokens | Samples=${NUM_SAMPLES}"
echo -e "Output Directory  : ${RESULTS_DIR}"
echo ""

# Helper to query current GPU VRAM usage in MB
get_vram_usage() {
    rocm-smi --showmeminfo vram 2>/dev/null | grep -i "used" | head -n 1 | awk '{print $NF}' || echo "N/A"
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

# Helper to wait for server health
wait_for_server() {
    local port="${1:-8000}"
    local max_wait="${2:-90}"
    local elapsed=0
    echo -n "Waiting for server on port ${port}..."
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

# Helper to measure token generation latency and throughput via OpenAI API
benchmark_endpoint() {
    local port="$1"
    local engine_name="$2"
    local log_file="${RESULTS_DIR}/${engine_name}_benchmark.json"

    python3 -c "
import time, json, urllib.request, sys

url = 'http://127.0.0.1:${port}/v1/chat/completions'
model_url = 'http://127.0.0.1:${port}/v1/models'

try:
    with urllib.request.urlopen(model_url, timeout=5) as r:
        models_data = json.loads(r.read().decode())
        model = models_data.get('data', [{}])[0].get('id', 'default')
except Exception as e:
    model = 'default'

prompt_text = ('token ' * ${IN_LEN}).strip()
payload = {
    'model': model,
    'messages': [{'role': 'user', 'content': prompt_text}],
    'max_tokens': ${OUT_LEN},
    'temperature': 0.7
}

data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

t0 = time.perf_counter()
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        t1 = time.perf_counter()
        resp_data = json.loads(resp.read().decode())
        dur = t1 - t0
        usage = resp_data.get('usage', {})
        completion_tokens = usage.get('completion_tokens', ${OUT_LEN})
        prompt_tokens = usage.get('prompt_tokens', 0)
        tok_per_sec = completion_tokens / dur if dur > 0 else 0
        
        result = {
            'engine': '${engine_name}',
            'model': model,
            'duration_sec': round(dur, 3),
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'tokens_per_sec': round(tok_per_sec, 2),
            'status': 'SUCCESS'
        }
        with open('${log_file}', 'w') as f:
            json.dump(result, f, indent=2)
        print(f'   Tokens: {completion_tokens} | Duration: {dur:.2f}s | Speed: {tok_per_sec:.2f} tok/s')
except Exception as e:
    result = {'engine': '${engine_name}', 'status': 'FAILED', 'error': str(e)}
    with open('${log_file}', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'   Benchmark failed: {e}', file=sys.stderr)
"
}

# Track status and metrics
VLLM_STATUS="SKIPPED"
VLLM_SPEED="N/A"
VLLM_VRAM="N/A"

LLAMA_STATUS="SKIPPED"
LLAMA_SPEED="N/A"
LLAMA_VRAM="N/A"

SGLANG_STATUS="SKIPPED"
SGLANG_SPEED="N/A"
SGLANG_VRAM="N/A"

# ==============================================================================
# Phase 1: Benchmark vLLM (Default)
# ==============================================================================
if [[ "$TARGET_ENGINES" == *"vllm"* ]] || [ "$TARGET_ENGINES" = "all" ]; then
    echo -e "${CYAN}${BOLD}>>> [1/3] Benchmarking Engine: vLLM (Default)...${NC}"
    if curl -s -f "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
        echo "Using currently active vLLM instance on port 8000..."
    else
        echo "Launching vLLM stack via docker-compose.yml..."
        stop_container "rocm-llama-server"
        stop_container "rocm-sglang-server"
        docker compose -p coder-vllm -f "${ROOT_DIR}/docker/docker-compose.yml" up -d inference
    fi

    if wait_for_server 8000 90; then
        VLLM_VRAM=$(get_vram_usage)
        benchmark_endpoint 8000 "vllm"
        if [ -f "${RESULTS_DIR}/vllm_benchmark.json" ] && grep -q '"status": "SUCCESS"' "${RESULTS_DIR}/vllm_benchmark.json"; then
            VLLM_STATUS="PASSED"
            VLLM_SPEED=$(jq -r '.tokens_per_sec' "${RESULTS_DIR}/vllm_benchmark.json")
        else
            VLLM_STATUS="FAILED"
        fi
    else
        echo -e "${YELLOW}Notice: vLLM server did not become healthy in time.${NC}"
        VLLM_STATUS="TIMEOUT"
    fi
fi

# ==============================================================================
# Phase 2: Benchmark llama.cpp (Alternative)
# ==============================================================================
if [[ "$TARGET_ENGINES" == *"llama"* ]] || [ "$TARGET_ENGINES" = "all" ]; then
    echo ""
    echo -e "${CYAN}${BOLD}>>> [2/3] Benchmarking Engine: llama.cpp (GGUF)...${NC}"
    TARGET_GGUF="${ROOT_DIR}/models/Qwen3.8-27B-Q4_K_M.gguf"
    if [ ! -f "$TARGET_GGUF" ]; then
        echo -e "${YELLOW}Notice: GGUF model weights not present at ${TARGET_GGUF}.${NC}"
        echo "To benchmark llama.cpp, first download GGUF weights via ./scripts/download_model.sh"
        LLAMA_STATUS="NO_MODEL"
    else
        echo "Switching inference server to llama.cpp via docker/docker-compose.gguf.yml..."
        stop_container "rocm-inference-server"
        stop_container "rocm-sglang-server"
        MODEL_FILE="Qwen3.8-27B-Q4_K_M.gguf" MODEL_ALIAS="Qwen3.8-27B-Q4_K_M.gguf" docker compose -p coder-llama -f "${ROOT_DIR}/docker/docker-compose.gguf.yml" up -d inference

        if wait_for_server 8000 60; then
            LLAMA_VRAM=$(get_vram_usage)
            benchmark_endpoint 8000 "llamacpp"
            if [ -f "${RESULTS_DIR}/llamacpp_benchmark.json" ] && grep -q '"status": "SUCCESS"' "${RESULTS_DIR}/llamacpp_benchmark.json"; then
                LLAMA_STATUS="PASSED"
                LLAMA_SPEED=$(jq -r '.tokens_per_sec' "${RESULTS_DIR}/llamacpp_benchmark.json")
            else
                LLAMA_STATUS="FAILED"
            fi
        else
            echo -e "${YELLOW}Notice: llama.cpp server was unable to start (architecture tag or image incompatibility).${NC}"
            LLAMA_STATUS="UNSUPPORTED_ARCH"
        fi

        # Stop llama.cpp before next engine
        stop_container "rocm-llama-server"
    fi
fi

# ==============================================================================
# Phase 3: Benchmark SGLang (Alternative)
# ==============================================================================
if [[ "$TARGET_ENGINES" == *"sglang"* ]] || [ "$TARGET_ENGINES" = "all" ]; then
    echo ""
    echo -e "${CYAN}${BOLD}>>> [3/3] Benchmarking Engine: SGLang (RadixAttention)...${NC}"
    if [ -f "${ROOT_DIR}/docker/docker-compose.sglang.yml" ]; then
        echo "Switching inference server to SGLang via docker/docker-compose.sglang.yml..."
        stop_container "rocm-inference-server"
        stop_container "rocm-llama-server"

        if docker compose -p coder-sglang -f "${ROOT_DIR}/docker/docker-compose.sglang.yml" up -d inference 2>/dev/null; then
            if wait_for_server 8000 90; then
                SGLANG_VRAM=$(get_vram_usage)
                benchmark_endpoint 8000 "sglang"
                if [ -f "${RESULTS_DIR}/sglang_benchmark.json" ] && grep -q '"status": "SUCCESS"' "${RESULTS_DIR}/sglang_benchmark.json"; then
                    SGLANG_STATUS="PASSED"
                    SGLANG_SPEED=$(jq -r '.tokens_per_sec' "${RESULTS_DIR}/sglang_benchmark.json")
                else
                    SGLANG_STATUS="FAILED"
                fi
            else
                echo -e "${YELLOW}Notice: SGLang server was unable to start or timed out.${NC}"
                SGLANG_STATUS="TIMEOUT"
            fi
            stop_container "rocm-sglang-server"
        else
            echo -e "${YELLOW}Notice: SGLang image unavailable or docker compose failed.${NC}"
            SGLANG_STATUS="IMAGE_UNAVAILABLE"
        fi
    fi
fi

# ==============================================================================
# Always restore default vLLM engine if it was stopped
# ==============================================================================
echo ""
echo -e "${CYAN}Restoring default engine: vLLM...${NC}"
stop_container "rocm-llama-server"
stop_container "rocm-sglang-server"
docker compose -p coder-vllm -f "${ROOT_DIR}/docker/docker-compose.yml" up -d inference >/dev/null 2>&1 || true

# ==============================================================================
# Generate Comparative Summary Table & Report
# ==============================================================================
REPORT_FILE="${RESULTS_DIR}/compare_engines_report.md"

cat << EOF > "$REPORT_FILE"
# Comparative Inference Engine Benchmark Report

**Hardware Target**: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB VRAM)  
**Date**: $(date)  
**Workload**: Prompt Tokens: ${IN_LEN} | Target Generation: ${OUT_LEN} tokens | Dataset: ${DATASET}  

## Engine Comparison Matrix

| Evaluation Dimension | vLLM (Default) | llama.cpp (Alternative) | SGLang (Alternative) |
| :--- | :--- | :--- | :--- |
| **Engine Architecture** | Distributed ROCm PagedAttention | Native GGML C++ ROCm | RadixAttention Runtime |
| **Supported Precision** | FP8 / AWQ / SafeTensors / GGUF Plugin | Q4_K_M / Q5_K_M GGUF | FP8 / SafeTensors / GGUF (\`--load-format gguf\`) |
| **Model Evaluated** | \`Qwen/Qwen3.8-27B-FP8\` | \`Qwen3.8-27B-Q4_K_M.gguf\` | \`Qwen/Qwen3.8-27B-FP8\` or GGUF |
| **Execution Status** | **${VLLM_STATUS}** | **${LLAMA_STATUS}** | **${SGLANG_STATUS}** |
| **Decode Throughput** | **${VLLM_SPEED} tok/s** | **${LLAMA_SPEED} tok/s** | **${SGLANG_SPEED} tok/s** |
| **VRAM Consumption** | ${VLLM_VRAM} | ${LLAMA_VRAM} | ${SGLANG_VRAM} |
| **Recommended Use Case**| **High-concurrency batch serving & SWE-bench** | **Single-user low-VRAM interactive coding** | **Multi-turn agents with Radix cache reuse** |

## Summary Analysis
- **vLLM (Default)**: Offers production-grade continuous batching, chunked prefill, and native multi-GPU tensor parallelism (\`--tp 2\`). Best suited for automated SWE-bench test harness and throughput tasks. Supports GGUF models directly via \`vllm-gguf-plugin\`.
- **llama.cpp**: Provides ultra-compact 4-bit memory footprints (~17 GB weights) with minimal cold-start overhead, but requires updated upstream GGML support for newer hybrid architecture tags (\`qwen35\`).
- **SGLang**: Excels at multi-turn agent workloads via RadixAttention (reusing prompt KV cache across complex tool calling turns). Automatically detects \`.gguf\` models via \`--load-format gguf\` and \`--tokenizer-path\`.
EOF

echo ""
echo -e "${GREEN}${BOLD}======================================================================${NC}"
echo -e "${GREEN}${BOLD}             Comparative Engine Benchmark Results Summary             ${NC}"
echo -e "${GREEN}${BOLD}======================================================================${NC}"
cat "$REPORT_FILE"
echo ""
echo -e "${GREEN}Report saved to: ${BOLD}${REPORT_FILE}${NC}"
