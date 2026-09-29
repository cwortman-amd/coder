#!/bin/bash
# One pass of the MI350P GPT-OSS ISL/OSL/concurrency sheet.
# Skips cells whose JSON already exists. Stops if the server is unhealthy.
set -u
HOST_OUT=/home/amd/workspace/coder/_results/aim_mi350p
LOG=/home/amd/workspace/coder/_results/aim_logs/matrix.log
MODEL=openai/gpt-oss-120b

run_one() {
  local isl="$1" osl="$2" c="$3"
  local fn="gptoss_isl${isl}_osl${osl}_c${c}.json"
  if [[ -f "$HOST_OUT/$fn" ]]; then
    echo "SKIP $(date -Is) $fn" | tee -a "$LOG"
    return 0
  fi
  if ! curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null; then
    echo "UNHEALTHY $(date -Is) before $fn" | tee -a "$LOG"
    return 2
  fi
  echo "START $(date -Is) isl=$isl osl=$osl c=$c" | tee -a "$LOG"
  timeout 10800 docker exec aim-gptoss vllm bench serve \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --port 8000 \
    --dataset-name random \
    --random-input-len "$isl" \
    --random-output-len "$osl" \
    --max-concurrency "$c" \
    --num-prompts $((10 * c)) \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 75,90,99 \
    --trust-remote-code \
    --num-warmups $((2 * c)) \
    --save-result \
    --result-dir /results \
    --result-filename "$fn" \
    > "/home/amd/workspace/coder/_results/aim_logs/${fn%.json}.log" 2>&1
  local rc=$?
  if [[ -f "$HOST_OUT/$fn" ]]; then
    python3 - "$HOST_OUT/$fn" "$isl" "$osl" <<'PY' | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
print(
    f"DONE isl={sys.argv[2]} osl={sys.argv[3]} "
    f"c={d.get('max_concurrency')} tok/s={d.get('output_throughput'):.2f} "
    f"median_ttft_ms={d.get('median_ttft_ms'):.2f} "
    f"completed={d.get('completed')} failed={d.get('failed')}"
)
PY
  else
    echo "FAIL $(date -Is) $fn rc=$rc" | tee -a "$LOG"
    if ! curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null; then
      echo "UNHEALTHY $(date -Is) after $fn" | tee -a "$LOG"
      return 2
    fi
  fi
  return 0
}

CELLS=(
  "128 2048 1,4,8,16,32,64,128,256"
  "256 256 1,4,8,16,32,64,128,256"
  "1024 1024 1,4,8,16,32,64,128,256,512"
  "2048 128 1,4,8,16,32,64,128,256"
  "8192 1024 1,4,8,16,32,64,128"
  "8192 3072 1,4,8,16,32,64,128,256"
  "32768 1024 1,4,8,16,32"
  "65536 1024 1,4,8,16"
  "122880 1024 1,4"
)

echo "MATRIX $(date -Is) begin" | tee -a "$LOG"
for spec in "${CELLS[@]}"; do
  read -r isl osl concs <<<"$spec"
  IFS=',' read -r -a cs <<<"$concs"
  for c in "${cs[@]}"; do
    run_one "$isl" "$osl" "$c" || exit $?
  done
done
echo "MATRIX $(date -Is) complete" | tee -a "$LOG"
