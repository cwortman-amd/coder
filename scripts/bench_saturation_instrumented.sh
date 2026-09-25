#!/usr/bin/env bash
# Instrumented C32–C128 on the frozen HF MXFP4 control (graphs on, AITER unset).
# Records power (amd-smi metric), Prometheus ITL/TTFT, and KV usage.
# Does not change server flags. Streaming ITL is sampled at C32 only (cost).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-awq}"
ILEN="${INPUT_LEN:-1024}"
OLEN="${OUTPUT_LEN:-256}"
OUT="${1:-${ROOT}/_results/priority_eval/saturation_c32_c128}"
shift || true
CONCS="${CONCS:-32,48,64,96,128}"
TIMEOUT="${TIMEOUT:-1800}"

mkdir -p "${OUT}/telemetry"
{
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "base=${BASE} model=${MODEL} input=${ILEN} output=${OLEN}"
  echo "graph_note=HF control O2 FULL_AND_PIECEWISE; VLLM_ROCM_USE_AITER must stay unset"
  curl -sf "${BASE}/health" >/dev/null && echo "health=ok"
  docker exec rocm-inference-server python3 -c 'import os; print("VLLM_ROCM_USE_AITER", os.environ.get("VLLM_ROCM_USE_AITER"))' 2>/dev/null || true
} | tee "${OUT}/meta.txt"

telemetry() {
  local tag=$1
  timeout 15 sudo -n amd-smi metric -g 0 >"${OUT}/telemetry/metric_${tag}.txt" 2>&1 || true
  curl -s "${BASE}/metrics" >"${OUT}/metrics_${tag}.prom" || true
}

IFS=',' read -r -a LIST <<<"${CONCS}"
SUMMARY="${OUT}/summary.jsonl"
: >"${SUMMARY}"

for C in "${LIST[@]}"; do
  echo "===== C${C} =====" | tee -a "${OUT}/run.log"
  telemetry "c${C}_before"
  python3 "${ROOT}/scripts/bench_openai_chat.py" \
    --base-url "${BASE}/v1" --model "${MODEL}" \
    --input-len "${ILEN}" --output-len "${OLEN}" \
    --num-prompts "${C}" --concurrency "${C}" \
    --timeout "${TIMEOUT}" \
    --out "${OUT}/bench_c${C}.json" | tee -a "${OUT}/run.log"
  telemetry "c${C}_after"
  python3 "${ROOT}/scripts/summarize_vllm_metrics.py" \
    --before "${OUT}/metrics_c${C}_before.prom" \
    --after "${OUT}/metrics_c${C}_after.prom" \
    --out "${OUT}/metrics_c${C}_delta.json"
  python3 - "${OUT}" "${C}" <<'PY' | tee -a "${SUMMARY}"
import json, re, sys
from pathlib import Path
out, c = Path(sys.argv[1]), sys.argv[2]
bench = json.loads((out / f"bench_c{c}.json").read_text())
delta = json.loads((out / f"metrics_c{c}_delta.json").read_text())
metric = (out / "telemetry" / f"metric_c{c}_after.txt").read_text(errors="replace")
def grab(pat):
    m = re.search(pat, metric, re.I)
    return m.group(1) if m else None
row = {
    "concurrency": int(c),
    "output_tok_s": bench.get("output_throughput"),
    "mean_latency_s": bench.get("mean_latency_s"),
    "successful": bench.get("successful"),
    "failed": bench.get("failed"),
    "prom_mean_itl_s": (delta.get("derived") or {}).get("mean_itl_s"),
    "prom_mean_ttft_s": (delta.get("derived") or {}).get("mean_ttft_s"),
    "kv_cache_usage_perc_after": ((delta.get("gauges_end") or {}).get("vllm:kv_cache_usage_perc") or {}).get("after"),
    "power_w": grab(r"POWER[:\s]+([0-9.]+)"),
    "gfx_clk": grab(r"GFX_CLK[:\s]+([0-9.]+)"),
    "mem_clk": grab(r"MEM_CLK[:\s]+([0-9.]+)"),
}
print(json.dumps(row))
(out / f"row_c{c}.json").write_text(json.dumps(row, indent=2) + "\n")
PY
done

if [[ "${STREAM_C32:-1}" == "1" ]]; then
  python3 "${ROOT}/scripts/bench_openai_stream.py" \
    --base-url "${BASE}/v1" --model "${MODEL}" \
    --input-len "${ILEN}" --output-len 64 \
    --num-prompts 8 --concurrency 1 \
    --timeout 600 \
    --out "${OUT}/stream_c1_sample.json" || true
fi

echo "artifacts: ${OUT}"
