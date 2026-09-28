#!/usr/bin/env bash
# gpt-oss-20b native MXFP4 bench. The checkpoint fits a 32 GB R9700S.
#
# Server:
#   VLLM_ROCM_USE_AITER=1 vllm serve openai/gpt-oss-20b \
#     --dtype auto -tp 1 --no-enable-prefix-caching --disable-uvicorn-access-log
#
# Client:
#   vllm bench serve --model openai/gpt-oss-20b \
#     --percentile-metrics tpot,ttft,itl,e2el \
#     --dataset-name random --ignore-eos --temperature 0 \
#     --max-concurrency 1 --num-prompts 10 \
#     --random-input-len 1024 --random-output-len 1024
#
# ./scripts/bench_gpt_oss_20b.sh              # launch, then bench
# ./scripts/bench_gpt_oss_20b.sh --bench-only # server already healthy
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="openai/gpt-oss-20b"
PORT="${VLLM_PORT:-8000}"
NAME="${VLLM_CONTAINER_NAME:-rocm-gpt-oss-server}"
OUT_DIR="${ROOT}/_results/gpt_oss_20b"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT="gpt_oss_20b_1024_1024_${STAMP}.json"
BENCH_ONLY=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--bench-only]

Launch openai/gpt-oss-20b (native MXFP4, AITER, TP=1, prefix caching off)
and run the 1024/1024 concurrency-1 bench. The 20B model fits a 32 GB R9700S.

  --bench-only   Skip launch. The server on port ${PORT} must already be healthy.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench-only) BENCH_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"

if [ "$BENCH_ONLY" -eq 0 ]; then
  "${ROOT}/scripts/launch_vllm_gpt_oss_20b.sh"
elif ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "No healthy server on port ${PORT}. Run without --bench-only." >&2
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "Container ${NAME} is not running. The bench client runs inside it." >&2
  exit 1
fi

echo "bench ${MODEL} 1024/1024 concurrency 1, 10 prompts"
docker exec "${NAME}" vllm bench serve \
  --model "${MODEL}" \
  --percentile-metrics tpot,ttft,itl,e2el \
  --dataset-name random \
  --ignore-eos \
  --temperature 0 \
  --max-concurrency 1 \
  --num-prompts 10 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --save-result \
  --result-dir /results/gpt_oss_20b \
  --result-filename "${RESULT}" | tee "${OUT_DIR}/bench_${STAMP}.log"

echo "result ${OUT_DIR}/${RESULT}"
