#!/usr/bin/env bash
# Locate the concurrency knees for the qualified, non-speculative
# MI350P Qwen3.8-27B MXFP4 profile (ROCM_ATTN, VLLM_ROCM_USE_AITER unset).
#
# Does not restart the server and does not change --max-num-seqs, graph
# capture, or KV dtype. Two results are reported separately:
#
#   Throughput knee — first concurrency beyond which a doubling adds under
#   10% aggregate output tok/s. C24, C48, or C96 is filled in around the
#   first such doubling. Still rising at C128 means the knee is not located.
#
#   SLO knee — highest concurrency at which every replicate still passes
#   the token-arrival and TTFT gates. Default token-arrival gate is p95
#   ITL <= 11 ms (GATES.md). TTFT is unset until --ttft-p95-max-ms is given.
#
# Burst TTFT is timed from HTTP send. num-prompts equals concurrency, so
# the client does not queue. The paced policy ramps request rate at the
# SLO concurrency and records both HTTP-send TTFT and scheduled-arrival TTFT.
#
# Usage:
#   ./scripts/concurrency.sh
#   ./scripts/concurrency.sh --policy burst
#   ./scripts/concurrency.sh --itl-p95-max-ms 13.36
#   ./scripts/concurrency.sh --quick
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE_URL:-http://127.0.0.1:8000}"
CONTAINER="${CONTAINER:-rocm-inference-server}"
SKIP_PROFILE_CHECK=0
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-profile-check)
      SKIP_PROFILE_CHECK=1
      shift
      ;;
    -h|--help)
      sed -n '2,28p' "$0"
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

health="${BASE%/}"
health="${health%/v1}"
if ! curl -sf -m 5 "${health}/health" >/dev/null; then
  echo "Refusing: ${health}/health did not answer. Start the qualified MXFP4 server first." >&2
  exit 2
fi

if [[ "${SKIP_PROFILE_CHECK}" -eq 0 ]]; then
  if ! aiter="$(docker exec "${CONTAINER}" python3 -c 'import os; print(os.environ.get("VLLM_ROCM_USE_AITER") or "")')"; then
    echo "Refusing: could not read VLLM_ROCM_USE_AITER from ${CONTAINER}." >&2
    echo "Pass --skip-profile-check only if you have already confirmed the qualified profile." >&2
    exit 2
  fi
  if [[ -n "${aiter}" ]]; then
    echo "Refusing: VLLM_ROCM_USE_AITER=${aiter}. The qualified profile leaves it unset." >&2
    exit 2
  fi
fi

exec python3 "${ROOT}/scripts/concurrency_knee.py" --base-url "${health}/v1" "${ARGS[@]}"
