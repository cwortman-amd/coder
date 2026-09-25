#!/usr/bin/env bash
# Capture one profiling package: vLLM metrics, host telemetry, rocprofv3 attach window.
# Assumes the MXFP4 server is already healthy with ROCP_TOOL_ATTACH=1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: profile_capture.sh <output_dir> [--model NAME] [--concurrency N] [--num-prompts N] [--input-len N] [--output-len N]}"
shift

MODEL="awq"
CONC=1
NPROM=4
ILEN=1024
OLEN=1024
ATTACH_MSEC=0
SKIP_ATTACH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --concurrency) CONC="$2"; shift 2 ;;
    --num-prompts) NPROM="$2"; shift 2 ;;
    --input-len) ILEN="$2"; shift 2 ;;
    --output-len) OLEN="$2"; shift 2 ;;
    --attach-msec) ATTACH_MSEC="$2"; shift 2 ;;
    --skip-attach) SKIP_ATTACH=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${OUT}/telemetry" "${OUT}/rocprof"
{
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model=${MODEL} concurrency=${CONC} num_prompts=${NPROM} input=${ILEN} output=${OLEN}"
  echo "container=$(docker inspect rocm-inference-server --format '{{.Id}} {{.Config.Image}}' 2>/dev/null || true)"
  docker inspect rocm-inference-server --format '{{json .Config.Cmd}}' 2>/dev/null || true
  echo "rocm=$(cat /opt/rocm/.info/version 2>/dev/null || true)"
  rocprofv3 --version 2>&1 | head -6
  sudo -n amd-smi version 2>&1 | head -8 || true
  lspci -nn | grep -iE 'amd|display|vga' || true
} > "${OUT}/version_manifest.txt"

python3 - <<PY
import json, time
from urllib.request import Request, urlopen
payload={"model":"${MODEL}","messages":[{"role":"user","content":"Say ping."}],"max_tokens":8,"temperature":0,"ignore_eos":True,"chat_template_kwargs":{"enable_thinking":False}}
req=Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"}, method="POST")
t0=time.perf_counter()
with urlopen(req, timeout=180) as resp: json.loads(resp.read().decode())
print("tiny_warmup", round(time.perf_counter()-t0, 2))
PY
python3 "${ROOT}/scripts/bench_openai_chat.py" \
  --base-url http://127.0.0.1:8000/v1 --model "${MODEL}" \
  --input-len "${ILEN}" --output-len 64 --num-prompts "${CONC}" --concurrency "${CONC}" \
  --timeout 300 --out "${OUT}/shape_warmup.json" >/dev/null

ENGINE_PID="$(docker top rocm-inference-server -eo pid,comm | awk '$2 ~ /EngineCor|EngineCore/ {print $1; exit}')"
if [[ -z "${ENGINE_PID}" ]]; then
  echo "could not find EngineCore host PID" >&2
  docker top rocm-inference-server -eo pid,comm,args
  exit 1
fi
echo "engine_host_pid=${ENGINE_PID}" | tee "${OUT}/engine_pid.txt"

curl -s http://127.0.0.1:8000/metrics > "${OUT}/metrics_before.prom"
docker logs rocm-inference-server > "${OUT}/server_before.log" 2>&1 || true

sudo -n amd-smi monitor -g 0 -w 1 -p -u -m -t > "${OUT}/telemetry/amd-smi.log" &
echo $! > "${OUT}/telemetry/amd-smi.pid"
pidstat -durh -p "${ENGINE_PID}" 1 > "${OUT}/telemetry/pidstat.log" 2>/dev/null &
echo $! > "${OUT}/telemetry/pidstat.pid"
vmstat 1 > "${OUT}/telemetry/vmstat.log" &
echo $! > "${OUT}/telemetry/vmstat.pid"
iostat -x 1 > "${OUT}/telemetry/iostat.log" &
echo $! > "${OUT}/telemetry/iostat.pid"
(
  while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ
    curl -fsS http://127.0.0.1:8000/metrics || true
    echo '---SCRAPE---'
    sleep 2
  done
) > "${OUT}/metrics_timeseries.prom" &
echo $! > "${OUT}/telemetry/metrics_loop.pid"

if [[ "${SKIP_ATTACH}" -eq 0 ]]; then
  if [[ "${ATTACH_MSEC}" -eq 0 ]]; then
    ATTACH_MSEC=$(( (NPROM / CONC + 2) * (OLEN / 60 + 12) * 1000 + 20000 ))
  fi
  echo "attach_msec=${ATTACH_MSEC}" | tee -a "${OUT}/engine_pid.txt"
  sudo -n /opt/rocm/bin/rocprofv3 \
    --pid "${ENGINE_PID}" \
    --attach-children=true \
    --attach-duration-msec "${ATTACH_MSEC}" \
    --attach-sync-output \
    --kernel-trace --hip-runtime-trace --memory-copy-trace \
    --stats --summary \
    --output-format csv pftrace \
    --output-directory "${OUT}/rocprof" \
    --output-file timed \
    > "${OUT}/rocprofv3_attach.log" 2>&1 &
  echo $! > "${OUT}/telemetry/rocprof.pid"
  sleep 2
else
  echo "skip_attach=1" | tee -a "${OUT}/engine_pid.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ | tee "${OUT}/timed_window_start.txt"
fi

python3 "${ROOT}/scripts/bench_openai_chat.py" \
  --base-url http://127.0.0.1:8000/v1 --model "${MODEL}" \
  --input-len "${ILEN}" --output-len "${OLEN}" \
  --num-prompts "${NPROM}" --concurrency "${CONC}" \
  --timeout 900 --out "${OUT}/bench.json"

date -u +%Y-%m-%dT%H:%M:%SZ | tee "${OUT}/timed_window_end.txt"
if [[ "${SKIP_ATTACH}" -eq 0 && -f "${OUT}/telemetry/rocprof.pid" ]]; then
  wait "$(cat "${OUT}/telemetry/rocprof.pid")" || true
fi

curl -s http://127.0.0.1:8000/metrics > "${OUT}/metrics_after.prom"
docker logs rocm-inference-server > "${OUT}/server_after.log" 2>&1 || true
sudo -n amd-smi static -g 0 > "${OUT}/telemetry/amd-smi-static.txt" 2>&1 || true
numastat -p "$(docker inspect -f '{{.State.Pid}}' rocm-inference-server)" > "${OUT}/telemetry/numastat.txt" 2>&1 || true
{
  echo '=== numactl --hardware ==='
  numactl --hardware || true
  echo '=== lspci amd ==='
  lspci -nn | grep -iE 'amd|display|vga' || true
} > "${OUT}/telemetry/numa_once.txt"

for pidf in amd-smi.pid pidstat.pid vmstat.pid iostat.pid metrics_loop.pid; do
  if [[ -f "${OUT}/telemetry/${pidf}" ]]; then
    kill "$(cat "${OUT}/telemetry/${pidf}")" >/dev/null 2>&1 || true
  fi
done
sleep 1

python3 "${ROOT}/scripts/summarize_vllm_metrics.py" \
  --before "${OUT}/metrics_before.prom" \
  --after "${OUT}/metrics_after.prom" \
  --out "${OUT}/metrics_delta.json" || true
python3 "${ROOT}/scripts/aggregate_kernel_csv.py" \
  --input-dir "${OUT}/rocprof" \
  --out "${OUT}/kernel_top.json" || true

python3 - <<PY
import json
d=json.load(open("${OUT}/bench.json"))
print("BENCH", "C${CONC}", round(d["output_throughput"],2), "gen", d["total_generated_tokens"], "fail", d["failed"])
PY
