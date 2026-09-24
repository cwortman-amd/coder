#!/bin/bash
# Accuracy benchmark runner: SWE-bench and GPQA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/gpu_profile.sh"

if [ -f "$HOME/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$HOME/.env"
    set +a
fi
if [ -f "${SCRIPT_DIR}/.env" ]; then
    PREV_HF_TOKEN="${HF_TOKEN:-}"
    set -a
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.env"
    set +a
    if [ -z "${HF_TOKEN:-}" ] && [ -n "$PREV_HF_TOKEN" ]; then
        export HF_TOKEN="$PREV_HF_TOKEN"
    fi
fi

# Always detect the installed card unless the caller explicitly overrides it.
GPU_PROFILE="${GPU_PROFILE_OVERRIDE:-auto}"
apply_gpu_profile

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

PROFILE="sample"
SWE_DATASET="sample"
GPQA_SUBSET="sample"
NUM_SAMPLES=""
RUN_SWE=true
RUN_GPQA=true
RUN_EVAL=false
ENGINE="${INFERENCE_ENGINE:-vllm}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run model-accuracy evaluations (SWE-bench and GPQA).

Accuracy tiers:
  -s, --sample                 Small offline samples (default)
  -d, --diamond, -l, --lite   SWE-bench Lite and GPQA Diamond
  -m, --main                  SWE-bench Verified and GPQA Main
  -a, --all                   Full SWE-bench and GPQA Extended

Options:
  -n, --limit <N>             Limit each benchmark to N samples
  --swe-only                  Run only SWE-bench
  --gpqa-only                 Run only GPQA
  --eval                      Run the SWE-bench evaluation harness
  -e, --engine <engine>       vllm | mxfp4 | llama.cpp | sglang
  -g, --gpu-profile <profile> auto | r9700 | mi350p (default: auto)
  -h, --help                  Show help

Examples:
  ./accuracy.sh
  ./accuracy.sh -d -n 5
  ./accuracy.sh --gpqa-only -d -g auto
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--sample)
            PROFILE="sample"; SWE_DATASET="sample"; GPQA_SUBSET="sample"; shift ;;
        -d|--diamond|-l|--lite)
            PROFILE="diamond"; SWE_DATASET="princeton-nlp/SWE-bench_Lite"; GPQA_SUBSET="diamond"; shift ;;
        -m|--main)
            PROFILE="main"; SWE_DATASET="princeton-nlp/SWE-bench_Verified"; GPQA_SUBSET="main"; shift ;;
        -a|--all)
            PROFILE="all"; SWE_DATASET="princeton-nlp/SWE-bench"; GPQA_SUBSET="extended"; shift ;;
        -n|--limit|--num-samples)
            [[ "${2:-}" =~ ^[0-9]+$ ]] || { echo "Error: $1 requires an integer." >&2; exit 1; }
            NUM_SAMPLES="$2"; shift 2 ;;
        --swe-only)
            RUN_SWE=true; RUN_GPQA=false; shift ;;
        --gpqa-only)
            RUN_SWE=false; RUN_GPQA=true; shift ;;
        --eval|--run-eval|--run-evaluation)
            RUN_EVAL=true; shift ;;
        -e|--engine)
            ENGINE="${2:-}"; shift 2 ;;
        -g|--gpu-profile)
            GPU_PROFILE="${2:-auto}"; apply_gpu_profile; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown accuracy option: $1" >&2
            usage >&2
            exit 1 ;;
    esac
done

ENGINE="${ENGINE,,}"
case "$ENGINE" in
    vllm|mxfp4|sglang) ;;
    llama.cpp|llamacpp|gguf) ENGINE="llama.cpp" ;;
    *) echo "Unsupported engine: $ENGINE" >&2; exit 1 ;;
esac

COMPOSE_FILE="$(compose_file_for_engine "$ENGINE")"
INFERENCE_PORT="${INFERENCE_PORT:-8000}"
BASE_URL="http://127.0.0.1:${INFERENCE_PORT}"
export GPU_PROFILE PYTORCH_ROCM_ARCH GPU_LABEL COMPOSE_FILE INFERENCE_PORT

echo -e "${BLUE}${BOLD}Accuracy Benchmark Suite${NC}"
echo -e "Device  : ${BOLD}${GPU_LABEL}${NC}"
echo -e "Engine  : ${BOLD}${ENGINE}${NC}"
echo -e "Profile : ${BOLD}${PROFILE}${NC}"

if ! curl -sf --connect-timeout 3 "${BASE_URL}/health" >/dev/null 2>&1 &&
   ! curl -sf --connect-timeout 3 "${BASE_URL}/v1/models" >/dev/null 2>&1; then
    echo -e "${RED}Inference server is not healthy at ${BASE_URL}.${NC}" >&2
    echo "Start it with: ./setup.sh -g ${GPU_PROFILE} -e ${ENGINE}" >&2
    exit 1
fi

ACTIVE_MODEL="$(
    curl -sf "${BASE_URL}/v1/models" |
        jq -r '.data[0].id // "Unknown"' 2>/dev/null || echo "Unknown"
)"
echo -e "Model   : ${BOLD}${ACTIVE_MODEL}${NC}"

START_TIME="$(date +%s)"
SWE_STATUS="SKIPPED"
GPQA_STATUS="SKIPPED"

if [ "$RUN_SWE" = true ]; then
    SWE_ARGS=(
        "run_benchmark.py"
        "--base-url" "${BASE_URL}/v1"
        "--dataset" "$SWE_DATASET"
    )
    [ -n "$NUM_SAMPLES" ] && SWE_ARGS+=("--num-samples" "$NUM_SAMPLES")
    [ "$RUN_EVAL" = true ] && SWE_ARGS+=("--run-evaluation")

    echo -e "\n${BOLD}Running SWE-bench...${NC}"
    if docker compose -f "$COMPOSE_FILE" run --rm --no-deps benchmark "${SWE_ARGS[@]}"; then
        SWE_STATUS="PASSED"
    else
        echo -e "${YELLOW}Container run failed; trying the host Python environment.${NC}"
        if python3 "${SCRIPT_DIR}/benchmark/run_benchmark.py" "${SWE_ARGS[@]:1}"; then
            SWE_STATUS="PASSED"
        else
            SWE_STATUS="FAILED"
        fi
    fi
fi

if [ "$RUN_GPQA" = true ]; then
    GPQA_ARGS=(
        "--base-url" "${BASE_URL}/v1"
        "--subset" "$GPQA_SUBSET"
    )
    [ -n "$NUM_SAMPLES" ] && GPQA_ARGS+=("--num-samples" "$NUM_SAMPLES")

    echo -e "\n${BOLD}Running GPQA...${NC}"
    if python3 "${SCRIPT_DIR}/benchmark/run_gpqa.py" "${GPQA_ARGS[@]}"; then
        GPQA_STATUS="PASSED"
    else
        GPQA_STATUS="FAILED"
    fi
fi

END_TIME="$(date +%s)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="${SCRIPT_DIR}/_results/accuracy_summary_${TIMESTAMP}.md"
mkdir -p "${SCRIPT_DIR}/_results"
cat >"$REPORT" <<EOF
# Accuracy Benchmark Summary

- **Device:** ${GPU_LABEL}
- **GPU profile:** ${GPU_PROFILE}
- **ROCm ISA:** ${PYTORCH_ROCM_ARCH}
- **Engine:** ${ENGINE}
- **Model:** ${ACTIVE_MODEL}
- **Accuracy tier:** ${PROFILE}
- **SWE-bench dataset:** ${SWE_DATASET}
- **GPQA subset:** ${GPQA_SUBSET}
- **SWE-bench status:** ${SWE_STATUS}
- **GPQA status:** ${GPQA_STATUS}
- **Runtime:** $((END_TIME - START_TIME)) seconds
EOF

echo -e "\nSWE-bench: ${BOLD}${SWE_STATUS}${NC}"
echo -e "GPQA:     ${BOLD}${GPQA_STATUS}${NC}"
echo -e "Report:   ${GREEN}${REPORT}${NC}"

if [ "$SWE_STATUS" = "FAILED" ] || [ "$GPQA_STATUS" = "FAILED" ]; then
    exit 1
fi
