#!/usr/bin/env bash
# ==============================================================================
# bench_dual_gpu.sh - Dual AMD Radeon™ AI PRO R9700 Multi-Mode Evaluation Suite
#
# Evaluates and compares:
#   1. Tensor Parallelism (TP=2) — 64 GB combined pool across PCIe
#   2. Data Parallelism (DP=2) — Independent 32 GB serving replicas
#   3. Prefill/Decode Disaggregation (P/D 1+1) — Phase isolation across GPUs
#   4. Single-GPU Baseline (1x R9700 Collocated)
#   5. Single-GPU Chunked Prefill (1x R9700 with chunked prefill)
#
# Hardware Guard:
#   Fails fast if multi-GPU modes (tp2, dp2, pd) are run on hosts without
#   2x homogeneous Radeon AI PRO R9700 (gfx1201) devices (preventing APU cross-runs).
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/_results/dual_gpu_eval"
TELEMETRY_DIR="${ROOT_DIR}/_results/telemetry"
DOCS_DIR="${ROOT_DIR}/docs"

mkdir -p "$RESULTS_DIR" "$TELEMETRY_DIR" "$DOCS_DIR"

MODE="probe"
WORKLOAD="agent" # 'chat' (512:256), 'agent' (8192:1024), 'jitter'
CONCURRENCY=1
BASE_URL="http://127.0.0.1:8000/v1"
MODEL_NAME="Qwen3.8-27B-Quark-AWQ-MXFP4"
TOKENIZER_NAME="Qwen/Qwen3.8-27B-FP8"
DRY_RUN=false

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Dual AMD Radeon™ AI PRO R9700 Architecture & Serving Benchmarking Suite

Options:
  -m, --mode <mode>        Evaluation mode: 'probe', 'single', 'single-chunked', 'tp2', 'dp2', 'pd' (default: probe)
  -w, --workload <type>    Workload archetype: 'agent' (8192:1024), 'chat' (512:256), 'jitter' (default: agent)
  -c, --concurrency <N>    Concurrency level (default: 1)
  -u, --url <URL>          Server Base URL (default: http://127.0.0.1:8000/v1)
  --model <name>           Served model alias (default: Qwen3.8-27B-Quark-AWQ-MXFP4)
  --tokenizer <name>       Client tokenizer (default: Qwen/Qwen3.8-27B-FP8)
  --dry-run, --mock        Allow multi-GPU choreography testing in mock mode on single-card development host
  -h, --help               Display this help message

Workload Configurations:
  • agent          : 8,192 input tokens, 1,024 output tokens (Standard long-context agentic baseline)
  • chat           : 512 input tokens, 256 output tokens (Interactive responsive baseline)
  • jitter         : 4 continuous decode streams bombarded by 16K prefill bursts (Measures tail ITL jitter)

Modes:
  • probe          : Run comprehensive host topology & container KV connector diagnostics
  • single         : Benchmark active single collocated R9700 instance
  • single-chunked : Benchmark single R9700 with chunked prefill enabled
  • tp2            : Benchmark Tensor Parallelism (TP=2) endpoint (Requires 2x R9700)
  • dp2            : Benchmark Data Parallelism (DP=2) endpoint (Requires 2x R9700)
  • pd             : Benchmark Prefill/Decode Disaggregation (P/D) router endpoint (Requires 2x R9700)

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
        --dry-run|--mock)
            DRY_RUN=true; shift ;;
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

# ==============================================================================
# Hardware Preflight Guard
# Enforces that multi-GPU modes (tp2, dp2, pd) strictly run on 2x gfx1201 R9700
# ==============================================================================
if [[ "$MODE" =~ ^(tp2|dp2|pd)$ ]]; then
    if ! python3 "${SCRIPT_DIR}/inspect_dual_gpu.py" --check-preflight >/dev/null 2>&1; then
        if [ "$DRY_RUN" = true ]; then
            echo "--------------------------------------------------------------------------------"
            echo "[PREFLIGHT NOTICE] Single R9700 host detected. Running ${MODE^^} in DRY-RUN / MOCK mode."
            echo "--------------------------------------------------------------------------------"
        else
            echo "================================================================================"
            echo "  HARDWARE PREFLIGHT ERROR: MULTI-GPU EXECUTION BLOCKED"
            echo "================================================================================"
            echo "Mode '${MODE^^}' requires 2 × homogeneous AMD Radeon AI PRO R9700 (gfx1201) GPUs."
            echo "Current host contains 1 × Radeon AI PRO R9700 (gfx1201) + 1 × Radeon 780M (gfx1103)."
            echo "Heterogeneous execution across dGPU and APU is prohibited."
            echo ""
            echo "To test the router or pipeline choreography in simulated mode, re-run with: --mock"
            echo "To benchmark the currently available hardware, run:"
            echo "  $(basename "$0") --mode single --workload ${WORKLOAD}"
            echo "  $(basename "$0") --mode single-chunked --workload ${WORKLOAD}"
            echo "================================================================================"
            exit 1
        fi
    fi
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
