#!/usr/bin/env bash
# Matched DFlash depth sweep vs existing crossover protocol.
# Usage: ./scripts/dflash_depth_sweep.sh 2 3 1 4 5
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
DEPTHS=("${@:-2}")
OUT="_results/priority_eval/dflash_depth"
mkdir -p "${OUT}"
MODEL="awq-mxfp4-dflash2"

bench() {
  local tag=$1 conc=$2 n=$3
  python3 scripts/bench_openai_chat.py \
    --base-url http://127.0.0.1:8000/v1 --model "${MODEL}" \
    --input-len 1024 --output-len 1024 \
    --num-prompts "${n}" --concurrency "${conc}" --timeout 900 \
    --out "${OUT}/${tag}.json"
}

stream_c1() {
  local tag=$1
  python3 scripts/bench_openai_stream.py \
    --base-url http://127.0.0.1:8000/v1 --model "${MODEL}" \
    --input-len 1024 --output-len 256 \
    --num-prompts 8 --concurrency 1 --timeout 900 \
    --out "${OUT}/${tag}.json"
}

metrics() {
  curl -sf http://127.0.0.1:8000/metrics > "$1"
}

for d in "${DEPTHS[@]}"; do
  echo "===== DFLASH-${d} ====="
  VLLM_SERVED_NAME="${MODEL}" ./scripts/launch_vllm_mxfp4.sh \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"/models/Qwen3.8-27B-DFlash2-sharded\",\"num_speculative_tokens\":${d}}" \
    | tee "${OUT}/dflash${d}_launch.log"
  docker logs rocm-inference-server > "${OUT}/dflash${d}_startup.log" 2>&1 || true
  bench "dflash${d}_warmup" 8 8
  metrics "${OUT}/dflash${d}_metrics_before.prom"
  bench "dflash${d}_c1" 1 4
  bench "dflash${d}_c5" 5 5
  bench "dflash${d}_c6" 6 6
  bench "dflash${d}_c8" 8 8
  stream_c1 "dflash${d}_stream_c1"
  metrics "${OUT}/dflash${d}_metrics_after.prom"
  python3 scripts/summarize_vllm_metrics.py \
    --before "${OUT}/dflash${d}_metrics_before.prom" \
    --after "${OUT}/dflash${d}_metrics_after.prom" \
    --out "${OUT}/dflash${d}_metrics.json" || true
done
echo "depth sweep done"
