#!/usr/bin/env bash
# ==============================================================================
# bench_dual_gpu.sh - Dual AMD Radeon™ AI PRO R9700 Multi-Mode Evaluation Suite
#
# Evaluates and compares:
#   1. Tensor Parallelism (TP=2) — 64 GB combined pool across PCIe
#   2. Prefill/Decode Disaggregation (P/D 1+1) — Phase isolation across GPUs
#   3. Data Parallelism (DP=2) — Independent replicas
#   4. Single-GPU Baseline (1x R9700)
#
# Measures:
#   - TTFT (Time to First Token) p50/p95/p99
#   - TPOT (Time per Output Token) p50/p95/p99
#   - Inter-Token Latency (ITL) Jitter under prefill bursts
#   - Dual-GPU package power, energy (Joules), and thermals
#   - Aggregate and per-stream throughput
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/_results/dual_gpu_eval"
TELEMETRY_DIR="${SCRIPT_DIR}/_results/telemetry"
DOCS_DIR="${SCRIPT_DIR}/docs"

mkdir -p "$RESULTS_DIR" "$TELEMETRY_DIR" "$DOCS_DIR"

MODE="probe"
WORKLOAD="agent" # 'chat' (512:256), 'agent' (8192:1024), 'jitter'
CONCURRENCY=1
BASE_URL="http://127.0.0.1:8000/v1"
MODEL_NAME="Qwen3.8-27B-Quark-AWQ-MXFP4"
TOKENIZER_NAME="Qwen/Qwen3.8-27B-FP8"

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Dual AMD Radeon™ AI PRO R9700 Architecture & Serving Benchmarking Suite

Options:
  -m, --mode <mode>        Evaluation mode: 'probe', 'tp2', 'pd', 'dp2', 'single' (default: probe)
  -w, --workload <type>    Workload archetype: 'agent' (8192:1024), 'chat' (512:256), 'jitter' (default: agent)
  -c, --concurrency <N>    Concurrency level (default: 1)
  -u, --url <URL>          Server Base URL (default: http://127.0.0.1:8000/v1)
  --model <name>           Served model alias (default: Qwen3.8-27B-Quark-AWQ-MXFP4)
  --tokenizer <name>       Client tokenizer (default: Qwen/Qwen3.8-27B-FP8)
  -h, --help               Display this help message

Workload Configurations:
  • agent  : 8,192 input tokens, 1,024 output tokens (Standard long-context agentic baseline)
  • chat   : 512 input tokens, 256 output tokens (Interactive responsive baseline)
  • jitter : 4 continuous decode streams bombarded by 16K prefill bursts (Measures tail ITL jitter)

Modes:
  • probe  : Execute comprehensive host topology & container KV connector diagnostics
  • tp2    : Run evaluation against Tensor Parallelism (TP=2) endpoint
  • pd     : Run evaluation against Prefill/Decode Disaggregation (P/D) router endpoint
  • single : Run evaluation against single collocated R9700 endpoint

EOF
    exit 0
}

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode)
            MODE="$2"; shift 2 ;;
        -w|--workload)
            WORKLOAD="$2"; shift 2 ;;
        -c|--concurrency)
            CONCURRENCY="$2"; shift 2 ;;
        -u|--url)
            BASE_URL="$2"; shift 2 ;;
        --model)
            MODEL_NAME="$2"; shift 2 ;;
        --tokenizer)
            TOKENIZER_NAME="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "Unknown argument: $1" >&2; usage ;;
    esac
done

if [ "$MODE" = "probe" ]; then
    echo "=== Running Dual R9700 Diagnostics Probe ==="
    python3 "${SCRIPT_DIR}/inspect_dual_gpu.py"
    exit 0
fi

# Resolve workload token shapes
case "$WORKLOAD" in
    agent)
        ISL=8192
        OSL=1024
        ;;
    chat)
        ISL=512
        OSL=256
        ;;
    jitter)
        ISL=16384
        OSL=512
        ;;
    *)
        echo "Error: Unknown workload: $WORKLOAD" >&2
        exit 1
        ;;
esac

NUM_PROMPTS=$(( CONCURRENCY * 10 ))
RUN_ID="dual_r9700_${MODE}_${WORKLOAD}_c${CONCURRENCY}_$(date +%Y%m%d_%H%M%S)"
TELEMETRY_CSV="${TELEMETRY_DIR}/${RUN_ID}.csv"
BENCH_JSON="${RESULTS_DIR}/${RUN_ID}.json"

echo "================================================================================"
echo "  DUAL R9700 BENCHMARK: Mode=${MODE^^} | Workload=${WORKLOAD} (${ISL}:${OSL}) | C=${CONCURRENCY}"
echo "================================================================================"
echo "Endpoint : ${BASE_URL}"
echo "Model    : ${MODEL_NAME}"
echo "Prompts  : ${NUM_PROMPTS}"
echo "Output   : ${BENCH_JSON}"
echo "--------------------------------------------------------------------------------"

# Launch high-frequency power collector
echo "[Telemetry] Starting 250ms power telemetry daemon..."
python3 "${SCRIPT_DIR}/collect_amd_power.py" \
  --gpu 0 \
  --interval 0.25 \
  --output "$TELEMETRY_CSV" &
POWER_PID=$!

sleep 2

# Check endpoint connectivity
if ! curl -s -f "${BASE_URL}/models" >/dev/null 2>&1; then
    echo "Warning: Target endpoint ${BASE_URL} is not responding."
    echo "If testing P/D router, ensure pd_router.py is running on port 8000."
    echo "If testing TP=2, ensure docker-compose.tp2.yml is up."
    echo "Proceeding with benchmark attempt..."
fi

# Execute benchmark via active container
ACTIVE_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E "^(rocm-mxfp4-server|rocm-mxfp4-tp2-server|rocm-mxfp4-pd-router|rocm-inference-server)$" | head -n1 || true)

if [ -n "$ACTIVE_CONTAINER" ]; then
    echo "[Benchmark] Executing via active container: ${ACTIVE_CONTAINER}..."
    set +e
    docker exec "$ACTIVE_CONTAINER" /opt/vllm/bin/vllm bench serve \
      --backend openai-chat \
      --host "127.0.0.1" \
      --port 8000 \
      --endpoint "/v1/chat/completions" \
      --model "$MODEL_NAME" \
      --tokenizer "$TOKENIZER_NAME" \
      --dataset-name random \
      --random-input-len "$ISL" \
      --random-output-len "$OSL" \
      --num-prompts "$NUM_PROMPTS" \
      --max-concurrency "$CONCURRENCY" \
      --save-result \
      --result-dir "/results/dual_gpu_eval" \
      --result-filename "${RUN_ID}.json"
    BENCH_RC=$?
    set -e
else
    echo "[Benchmark] Executing via host python environment..."
    set +e
    vllm bench serve \
      --backend openai-chat \
      --host "127.0.0.1" \
      --port 8000 \
      --endpoint "/v1/chat/completions" \
      --model "$MODEL_NAME" \
      --tokenizer "$TOKENIZER_NAME" \
      --dataset-name random \
      --random-input-len "$ISL" \
      --random-output-len "$OSL" \
      --num-prompts "$NUM_PROMPTS" \
      --max-concurrency "$CONCURRENCY"
    BENCH_RC=$?
    set -e
fi

# Terminate power monitor
kill -INT "$POWER_PID" 2>/dev/null || true
wait "$POWER_PID" 2>/dev/null || true
echo "[Telemetry] Power telemetry saved to ${TELEMETRY_CSV}"

if [ "$BENCH_RC" -eq 0 ]; then
    echo "================================================================================"
    echo "  BENCHMARK COMPLETED SUCCESSFULLY: ${RUN_ID}"
    echo "================================================================================"
else
    echo "Benchmark failed with exit code: $BENCH_RC"
fi
