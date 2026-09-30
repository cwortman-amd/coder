#!/usr/bin/env bash
# Matched C1 check or gated decode trace for the MI350P TP comparison.
#
#   scripts/profile_tp_matched_decode.sh <1|2> <confirm|trace> [custom|nccl]
#
# confirm: same serve flags as docs/MI350P-TP.md, 8 prompts, no profiler.
# trace:   rocprofv3 wraps each multiprocessing worker from spawn. ROCTx
#          keeps the trace paused through model load, then records two
#          prompts. rocprofv3 --pid cannot see these workers.
# nccl:    pass --disable-custom-all-reduce and write under tpN_nccl so the
#          custom-allreduce gate traces are left in place.
set -euo pipefail

TP="${1:?usage: profile_tp_matched_decode.sh 1|2 confirm|trace [custom|nccl]}"
MODE="${2:?usage: profile_tp_matched_decode.sh 1|2 confirm|trace [custom|nccl]}"
COLLECTIVE="${3:-custom}"
if [[ "${TP}" != "1" && "${TP}" != "2" ]]; then
  echo "TP must be 1 or 2" >&2
  exit 2
fi
if [[ "${MODE}" != "confirm" && "${MODE}" != "trace" ]]; then
  echo "mode must be confirm or trace" >&2
  exit 2
fi
if [[ "${COLLECTIVE}" != "custom" && "${COLLECTIVE}" != "nccl" ]]; then
  echo "collective must be custom or nccl" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="rocm-tp-profile"
if [[ "${COLLECTIVE}" == "nccl" ]]; then
  OUT_TAG="tp${TP}_nccl"
else
  OUT_TAG="tp${TP}${TP_OUT_TAG:-}"
fi
OUT_CONT="/results/tp_profile_mi350p/${OUT_TAG}"
OUT_HOST="${ROOT}/_results/tp_profile_mi350p/${OUT_TAG}"
mkdir -p "${OUT_HOST}/trace"
rm -f "${OUT_HOST}/trace/go" "${OUT_HOST}/trace/gate.log" "${OUT_HOST}/trace/spawn.log"
docker stop rocm-tp-compare "${NAME}" >/dev/null 2>&1 || true

export VLLM_TP_SIZE="${TP}"
export VLLM_CONTAINER_NAME="${NAME}"
export VLLM_KV_CACHE_MEMORY_BYTES=34359738368
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT
unset ENGINECORE_ROCP_EXEC ENGINECORE_ROCPROF_DIR ENGINECORE_ROCPROF_GATE_FILE || true

if [[ "${MODE}" == "trace" ]]; then
  export ENGINECORE_ROCP_EXEC=1
  export ENGINECORE_ROCPROF_DIR="${OUT_CONT}/trace"
  export ENGINECORE_ROCPROF_GATE_FILE="${OUT_CONT}/trace/go"
fi

SERVE_EXTRA=()
if [[ "${COLLECTIVE}" == "nccl" ]]; then
  SERVE_EXTRA+=(--disable-custom-all-reduce)
fi

"${ROOT}/scripts/launch_vllm_mxfp4.sh" \
  --no-enable-prefix-caching \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  "${SERVE_EXTRA[@]}"

bench() {
  local prompts="$1"
  local filename="$2"
  docker exec "${NAME}" vllm bench serve \
    --backend openai \
    --endpoint /v1/completions \
    --model awq \
    --tokenizer /models/Qwen3.8-27B-Quark-AWQ-MXFP4-sharded \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts "${prompts}" \
    --max-concurrency 1 \
    --request-rate inf \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result \
    --result-dir "${OUT_CONT}" \
    --result-filename "${filename}"
}

save_server_log() {
  docker logs "${NAME}" >"${OUT_HOST}/server.log" 2>&1 || true
}

if [[ "${MODE}" == "confirm" ]]; then
  echo "unprofiled C1 confirm, 8 prompts, collective=${COLLECTIVE}"
  if [[ -n "${TP_OUT_TAG:-}" ]]; then
    RESULT="tp${TP}_unprofiled_c1${TP_OUT_TAG}.json"
  elif [[ "${COLLECTIVE}" == "custom" && "${TP}" == "2" ]]; then
    RESULT="tp${TP}_unprofiled_c1_repeat.json"
  elif [[ "${COLLECTIVE}" == "nccl" ]]; then
    RESULT="tp${TP}_unprofiled_c1_nccl.json"
  else
    RESULT="tp${TP}_unprofiled_c1.json"
  fi
  bench 8 "${RESULT}"
  save_server_log
  exit 0
fi

echo "waiting for rocprof gate"
paused=0
for _ in $(seq 1 60); do
  if [[ -f "${OUT_HOST}/trace/gate.log" ]] && grep -q "paused status=" "${OUT_HOST}/trace/gate.log"; then
    paused=1
    break
  fi
  sleep 1
done
echo "--- gate log ---"
cat "${OUT_HOST}/trace/gate.log" 2>/dev/null || true
echo "--- spawn log ---"
cat "${OUT_HOST}/trace/spawn.log" 2>/dev/null || true
if [[ "${paused}" -ne 1 ]]; then
  echo "worker did not pause tracing" >&2
  exit 1
fi

echo "opening trace for 2 prompts"
touch "${OUT_HOST}/trace/go"
bench 2 "tp${TP}_profiled_c1.json" || true
rm -f "${OUT_HOST}/trace/go"
sleep 2
echo "stopping traced workers so rocprof can flush"
mapfile -t ROCPROF_PIDS < <(awk '{for (i = 1; i <= NF; i++) if ($i ~ /^pid=/) {split($i, a, "="); print a[2]}}' "${OUT_HOST}/trace/spawn.log")
for roc_pid in "${ROCPROF_PIDS[@]}"; do
  docker exec "${NAME}" kill -TERM "${roc_pid}" >/dev/null 2>&1 || true
done
flushed=0
for _ in $(seq 1 90); do
  if compgen -G "${OUT_HOST}/trace/*_kernel_trace.csv" >/dev/null; then
    flushed=1
    break
  fi
  sleep 2
done
docker stop -t 30 "${NAME}" >/dev/null || true
save_server_log
if [[ "${flushed}" -ne 1 ]]; then
  echo "kernel trace was not written" >&2
  find "${OUT_HOST}/trace" -type f -printf '%s %p\n' | sort -n
  exit 1
fi
find "${OUT_HOST}/trace" -type f -printf '%s %p\n' | sort -n
