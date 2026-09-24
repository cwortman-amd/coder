#!/bin/bash
# Primary throughput benchmark entrypoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect the available card by default, even if .env contains an old profile.
GPU_PROFILE_OVERRIDE="auto"
FORWARD_ARGS=()
HAS_TEST_CASES=false
QUICK=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpu-profile)
            [ -n "${2:-}" ] || { echo "Error: $1 requires auto, r9700, or mi350p." >&2; exit 1; }
            GPU_PROFILE_OVERRIDE="$2"
            shift 2
            ;;
        --test-cases)
            [ -n "${2:-}" ] || { echo "Error: --test-cases requires I:O pairs." >&2; exit 1; }
            HAS_TEST_CASES=true
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        -q|--quick)
            QUICK=true
            FORWARD_ARGS+=("$1")
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run live-serving throughput, latency, and multi-engine benchmarks.

Device:
  -g, --gpu-profile <profile> auto | r9700 | mi350p (default: auto)

Throughput:
  -e, --engine <engine>       vllm | mxfp4 | llama.cpp | sglang | all
  --engines <list>            Comma-separated engines
  -c, --concurrency <N>       Concurrent requests (default: 1)
  -n, --num-prompts <N>       Prompts per workload
  --test-cases <I:O,...>      Input:output token shapes
  -q, --quick                 One 128:64 smoke test
  -u, --server-url <URL>      Target server URL
  --compare-engines           Benchmark every engine
  -h, --help                  Show help

Examples:
  ./throughput.sh -q
  ./throughput.sh -e vllm -c 8 --test-cases 8192:1024
  ./throughput.sh --compare-engines -q
EOF
            exit 0
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$HAS_TEST_CASES" = false ] && [ "$QUICK" = false ]; then
    FORWARD_ARGS+=("--test-cases" "8192:1024,1024:8192,1024:1024")
fi

export GPU_PROFILE_OVERRIDE
export GPU_PROFILE="$GPU_PROFILE_OVERRIDE"
exec "${SCRIPT_DIR}/bench_throughput.sh" "${FORWARD_ARGS[@]}"
