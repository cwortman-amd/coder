#!/bin/bash
# Unified benchmark dispatcher for accuracy.sh and throughput.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ACCURACY=true
RUN_THROUGHPUT=true
GPU_PROFILE="auto"
ENGINE=""
QUICK=false
ACCURACY_ARGS=()
THROUGHPUT_ARGS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [SUITES] [SHARED OPTIONS] [SUITE OPTIONS]

Run both accuracy and throughput benchmarks by default.

Suites:
  --both                 Run accuracy then throughput (default)
  --accuracy             Run only accuracy.sh
  --throughput           Run only throughput.sh

Shared options:
  -g, --gpu-profile <p>  auto | r9700 | mi350p (default: auto)
  -e, --engine <engine>  vllm | mxfp4 | llama.cpp | sglang
  --models <list>        Compare profiles, one GPU each when the host has enough cards
  --kernel-variant <k>   Stock rocm_attn plus aiter or mxfp4 on the next GPU
  -q, --quick            Small accuracy sample and 128:64 throughput smoke test
  -h, --help             Show this help

Accuracy options (forwarded to accuracy.sh):
  -s, --sample
  -d, --diamond, -l, --lite
  -m, --main
  -a, --all
  --accuracy-limit <N>
  --swe-only | --gpqa-only
  --eval

Throughput options (forwarded to throughput.sh):
  -c, --concurrency <N>
  --num-prompts <N>
  --test-cases <I:O,...>
  --engines <list>
  --compare-engines
  -u, --server-url <URL>

Examples:
  ./test.sh -q
  ./test.sh --accuracy -d --accuracy-limit 5
  ./test.sh --throughput -e vllm -c 8 --test-cases 8192:1024
  ./test.sh --both -g auto -e mxfp4 -d --accuracy-limit 5 -c 8
  ./test.sh --models qwen3.8,gpt-oss-20b -q
  ./test.sh --models gpt-oss-20b
  ./scripts/bench_gpt_oss_20b.sh
  ./test.sh --model qwen3.8 --kernel-variant aiter --throughput -q
EOF
}

MODEL_LIST=""
KERNEL_VARIANT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --both)
            RUN_ACCURACY=true; RUN_THROUGHPUT=true; shift ;;
        --accuracy|--accuracy-only)
            RUN_ACCURACY=true; RUN_THROUGHPUT=false; shift ;;
        --throughput|--throughput-only)
            RUN_ACCURACY=false; RUN_THROUGHPUT=true; shift ;;
        -g|--gpu-profile)
            [ -n "${2:-}" ] || { echo "Error: $1 requires a profile." >&2; exit 1; }
            GPU_PROFILE="$2"; shift 2 ;;
        -e|--engine)
            [ -n "${2:-}" ] || { echo "Error: $1 requires an engine." >&2; exit 1; }
            ENGINE="$2"; shift 2 ;;
        -q|--quick)
            QUICK=true; shift ;;
        --model|--models)
            [ -n "${2:-}" ] || { echo "Error: $1 requires a profile list." >&2; exit 1; }
            if [ -n "$MODEL_LIST" ]; then
                echo "Pass --models once." >&2
                exit 1
            fi
            MODEL_LIST="$2"; shift 2 ;;
        --kernel-variant)
            [ -n "${2:-}" ] || { echo "Error: --kernel-variant requires rocm_attn, aiter, or mxfp4." >&2; exit 1; }
            KERNEL_VARIANT="$2"; shift 2 ;;

        -s|--sample|-d|--diamond|-l|--lite|-m|--main|-a|--all|--swe-only|--gpqa-only|--eval|--run-eval|--run-evaluation)
            ACCURACY_ARGS+=("$1"); shift ;;
        --accuracy-limit)
            [[ "${2:-}" =~ ^[0-9]+$ ]] || { echo "Error: --accuracy-limit requires an integer." >&2; exit 1; }
            ACCURACY_ARGS+=("--limit" "$2"); shift 2 ;;

        -c|--concurrency|-n|--num-prompts|--test-cases|--engines|-u|--url|--server-url)
            [ -n "${2:-}" ] || { echo "Error: $1 requires a value." >&2; exit 1; }
            THROUGHPUT_ARGS+=("$1" "$2"); shift 2 ;;
        --compare-engines)
            THROUGHPUT_ARGS+=("$1"); shift ;;

        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1 ;;
    esac
done

case "${GPU_PROFILE,,}" in
    auto|r9700|gfx1201|radeon|mi350p|mi350|gfx950|instinct) ;;
    *) echo "Unsupported GPU profile: $GPU_PROFILE" >&2; exit 1 ;;
esac

if [ -n "$ENGINE" ]; then
    ACCURACY_ARGS+=("--engine" "$ENGINE")
    THROUGHPUT_ARGS+=("--engine" "$ENGINE")
fi
ACCURACY_ARGS+=("--gpu-profile" "$GPU_PROFILE")
THROUGHPUT_ARGS+=("--gpu-profile" "$GPU_PROFILE")

if [ "$QUICK" = true ]; then
    # accuracy.sh defaults to the sample tier.
    THROUGHPUT_ARGS+=("--quick")
fi

# gpt-oss-20b native MXFP4 on a 32 GB R9700S: 1024/1024, 10 prompts, concurrency 1.
gpt_oss_only=0
if [ -n "$MODEL_LIST" ]; then
    gpt_oss_only=1
    IFS=',' read -ra _model_specs <<< "$MODEL_LIST"
    for spec in "${_model_specs[@]}"; do
        spec="${spec//@*/}"
        spec="${spec// /}"
        case "$spec" in
            gpt-oss|gpt-oss-20b|openai/gpt-oss-20b) ;;
            *) gpt_oss_only=0 ;;
        esac
    done
fi
has_cases=0
has_prompts=0
for arg in "${THROUGHPUT_ARGS[@]}"; do
    [ "$arg" = "--test-cases" ] && has_cases=1
    [ "$arg" = "--num-prompts" ] && has_prompts=1
done
if [ "$gpt_oss_only" -eq 1 ] && [ "$QUICK" = false ] && [ "$has_cases" -eq 0 ]; then
    THROUGHPUT_ARGS+=(--test-cases "1024:1024")
fi
if [ "$gpt_oss_only" -eq 1 ] && [ "$QUICK" = false ] && [ "$has_prompts" -eq 0 ]; then
    THROUGHPUT_ARGS+=(--num-prompts 10)
fi

FAILED=0

run_suites() {
    local failed=0
    local port="$1"
    local served="$2"
    local tokenizer="$3"
    local tag="$4"
    local url="http://127.0.0.1:${port}"
    if [ "$RUN_ACCURACY" = true ]; then
        echo "========================================================================"
        echo "Accuracy  ${tag}  ${url}"
        echo "========================================================================"
        local acc=("${ACCURACY_ARGS[@]}" --port "$port")
        [ -n "$served" ] && acc+=(--model "$served")
        if ! "${SCRIPT_DIR}/accuracy.sh" "${acc[@]}"; then
            echo "Accuracy benchmarks failed for ${tag}." >&2
            failed=1
        fi
    fi
    if [ "$RUN_THROUGHPUT" = true ]; then
        echo "========================================================================"
        echo "Throughput  ${tag}  ${url}"
        echo "========================================================================"
        if ! ATTACH_ONLY=1 \
            TOKENIZER_NAME_OVERRIDE="$tokenizer" \
            RESULTS_TAG="$tag" \
            "${SCRIPT_DIR}/throughput.sh" "${THROUGHPUT_ARGS[@]}" --server-url "$url"; then
            echo "Throughput benchmarks failed for ${tag}." >&2
            failed=1
        fi
    fi
    return "$failed"
}

SPECS=""
if [ -n "$KERNEL_VARIANT" ]; then
    if [[ "$MODEL_LIST" == *,* ]]; then
        echo "--kernel-variant benchmarks one model. The optimized kernel is placed on another GPU." >&2
        exit 1
    fi
    base="${MODEL_LIST:-qwen3.8}"
    SPECS="${base}@rocm_attn,${base}@${KERNEL_VARIANT}"
elif [ -n "$MODEL_LIST" ]; then
    SPECS="$MODEL_LIST"
fi

if [ -z "$SPECS" ]; then
    run_suites "${INFERENCE_PORT:-8000}" "" "" "active" || FAILED=1
    exit "$FAILED"
fi

slot_count=0
IFS=',' read -ra SPEC_ARR <<< "$SPECS"
for spec in "${SPEC_ARR[@]}"; do
    [ -n "${spec// /}" ] && slot_count=$((slot_count + 1))
done

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/gpu_profile.sh"
gpu_ids=()
mapfile -t gpu_ids < <(list_target_gpu_ids "$GPU_PROFILE" || true)
gpu_count="${#gpu_ids[@]}"
if [ "$gpu_count" -ge "$slot_count" ]; then
    echo "Starting ${slot_count} servers across ${gpu_count} GPUs."
    "${SCRIPT_DIR}/scripts/parallel_serve.sh" --gpu-profile "$GPU_PROFILE" --slots "$SPECS"
    pids=()
    while IFS=$'\t' read -r gpu port profile kernel url served tokenizer; do
        [ -n "${port:-}" ] || continue
        tag="${profile}_${kernel}_g${gpu}"
        run_suites "$port" "$served" "$tokenizer" "$tag" &
        pids+=("$!")
    done < "${SCRIPT_DIR}/_results/parallel/slots.tsv"
    for pid in "${pids[@]}"; do
        wait "$pid" || FAILED=1
    done
else
    echo "Host shows ${gpu_count} GPU(s) and ${slot_count} slots. Each slot is served and scored in turn on one GPU." >&2
    echo "With one card per slot, the servers stay up together and kernel variants do not share a GPU." >&2
    for spec in "${SPEC_ARR[@]}"; do
        spec="${spec// /}"
        [ -n "$spec" ] || continue
        "${SCRIPT_DIR}/scripts/parallel_serve.sh" --gpu-profile "$GPU_PROFILE" --slots "$spec"
        IFS=$'\t' read -r gpu port profile kernel url served tokenizer < "${SCRIPT_DIR}/_results/parallel/slots.tsv"
        run_suites "$port" "$served" "$tokenizer" "${profile}_${kernel}_g${gpu}" || FAILED=1
    done
fi

exit "$FAILED"
