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
EOF
}

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

FAILED=0

if [ "$RUN_ACCURACY" = true ]; then
    echo "========================================================================"
    echo "Running accuracy benchmarks"
    echo "========================================================================"
    if ! "${SCRIPT_DIR}/accuracy.sh" "${ACCURACY_ARGS[@]}"; then
        echo "Accuracy benchmarks failed." >&2
        FAILED=1
    fi
fi

if [ "$RUN_THROUGHPUT" = true ]; then
    echo "========================================================================"
    echo "Running throughput benchmarks"
    echo "========================================================================"
    if ! "${SCRIPT_DIR}/throughput.sh" "${THROUGHPUT_ARGS[@]}"; then
        echo "Throughput benchmarks failed." >&2
        FAILED=1
    fi
fi

exit "$FAILED"
