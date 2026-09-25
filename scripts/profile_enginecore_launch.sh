#!/usr/bin/env bash
# Unblock EngineCore GPU traces by launching vLLM *under* rocprofv3 (not attach).
#
# Attach after HIP/HSA init does not work on this image (no rocp-bg-attach).
# rocprofv3 is reliable when it is the parent of vllm serve from process creation.
#
# This ~32 GiB host cannot hold two 27B servers. Default: stop port 8000, profile
# on 8001, restore the HF control. Use --keep-baseline only if you have RAM.
#
# Usage:
#   ./scripts/profile_enginecore_launch.sh
#   ./scripts/profile_enginecore_launch.sh --dflash 3
#   ./scripts/profile_enginecore_launch.sh --concurrency 8 --keep-baseline
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
MODEL="${VLLM_MODEL:-/models/Qwen3.8-27B-Quark-AWQ-MXFP4-sharded}"
BASE_NAME="${VLLM_CONTAINER_NAME:-rocm-inference-server}"
PROF_NAME="${VLLM_PROFILE_CONTAINER:-rocm-inference-profile}"
PORT="${VLLM_PROFILE_PORT:-8001}"
SERVED="${VLLM_SERVED_NAME:-awq-profile-control}"
GPU=0
CONC=1
WARM=8
NPROM=8
ILEN=1024
OLEN=128
KEEP_BASELINE=0
RESTORE=1
RELAX_PTRACE=0
DFLASH=""
KV="${VLLM_KV_BYTES:-103223724237}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT=""

usage() {
  cat <<'EOF'
profile_enginecore_launch.sh — launch-under-rocprofv3 (do not attach to live EngineCore)

  --out DIR
  --port N              default 8001
  --gpu N
  --concurrency N       1 or 8
  --warmup N            shape-warmup requests (default 8)
  --num-prompts N       timed requests (default 8)
  --input-len / --output-len   default 1024 / 128 (short traces)
  --dflash {2,3,7}      optional speculative depth
  --keep-baseline       leave 8000 running (needs enough host RAM)
  --no-restore          do not relaunch HF control on 8000
  --relax-ptrace        lab-only: sysctl kernel.yama.ptrace_scope=0 for the run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --out) OUT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --concurrency) CONC="$2"; shift 2 ;;
    --warmup) WARM="$2"; shift 2 ;;
    --num-prompts) NPROM="$2"; shift 2 ;;
    --input-len) ILEN="$2"; shift 2 ;;
    --output-len) OLEN="$2"; shift 2 ;;
    --dflash) DFLASH="$2"; shift 2 ;;
    --keep-baseline) KEEP_BASELINE=1; shift ;;
    --no-restore) RESTORE=0; shift ;;
    --relax-ptrace) RELAX_PTRACE=1; shift ;;
    *) echo "unknown arg $1" >&2; usage; exit 2 ;;
  esac
done

OUT="${OUT:-${ROOT}/_results/profiling/launch_${STAMP}}"
mkdir -p "${OUT}/rocprof" "${OUT}/telemetry"
PTRACE_OLD=""

cleanup() {
  docker stop "${PROF_NAME}" >/dev/null 2>&1 || true
  docker rm "${PROF_NAME}" >/dev/null 2>&1 || true
  if [[ -n "${PTRACE_OLD}" ]]; then
    sudo -n sysctl "kernel.yama.ptrace_scope=${PTRACE_OLD}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${RELAX_PTRACE}" -eq 1 ]]; then
  PTRACE_OLD="$(cat /proc/sys/kernel/yama/ptrace_scope)"
  sudo -n sysctl kernel.yama.ptrace_scope=0
fi

STOPPED_BASE=0
if [[ "${KEEP_BASELINE}" -eq 0 ]]; then
  if docker ps --format '{{.Names}}' | grep -qx "${BASE_NAME}"; then
    echo "stopping ${BASE_NAME} (this host cannot run two 27B servers)" | tee "${OUT}/meta.txt"
    docker stop "${BASE_NAME}" >/dev/null
    STOPPED_BASE=1
  fi
else
  echo "WARNING: --keep-baseline with two 27B processes on ~32 GiB RAM will likely OOM" | tee "${OUT}/meta.txt"
fi

SPEC_ARGS=()
if [[ -n "${DFLASH}" ]]; then
  SERVED="awq-profile-dflash${DFLASH}"
  SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/models/Qwen3.8-27B-DFlash2-sharded\",\"num_speculative_tokens\":${DFLASH}}")
fi

{
  echo "utc=${STAMP} port=${PORT} conc=${CONC} isl=${ILEN} osl=${OLEN} dflash=${DFLASH:-none}"
  echo "ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope)"
  docker exec "${BASE_NAME}" rocprofv3 --version 2>/dev/null || true
} | tee -a "${OUT}/meta.txt"

docker rm -f "${PROF_NAME}" >/dev/null 2>&1 || true

# rocprofv3 as PID 1 still only traces the API parent unless EngineCore is
# in-process. V1 multiprocessing must be off for kernel CSV (child PID 383
# previously opened KFD but produced no kernel_trace). --process-sync waits
# for descendant flush when multiprocessing is left on; it is not sufficient
# on this image.
docker run -d \
  --name "${PROF_NAME}" --restart=no --network host --ipc host \
  --shm-size 64g \
  --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined --security-opt label=disable \
  -e HIP_VISIBLE_DEVICES="${GPU}" \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  -e GPU_ARCHS=gfx950 \
  -e HF_HOME=/root/.cache/huggingface \
  -e SAFETENSORS_FAST_GPU=1 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e ROCPROFILER_LIBRARY_CTOR=1 \
  -v /home/amd/.cache/huggingface:/root/.cache/huggingface \
  -v "${ROOT}/models:/models" \
  -v "${ROOT}/_results:/results" \
  --entrypoint /opt/rocm/bin/rocprofv3 \
  "${IMAGE}" \
  --kernel-trace --hip-runtime-trace --memory-copy-trace \
  --process-sync \
  --stats --summary \
  --output-format csv pftrace \
  --output-directory "/results/profiling/$(basename "${OUT}")/rocprof" \
  --output-file timed \
  -- \
  vllm serve "${MODEL}" \
    --served-model-name "${SERVED}" \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --kv-cache-memory-bytes "${KV}" \
    "${SPEC_ARGS[@]}"

echo "started ${PROF_NAME} on :${PORT}" | tee -a "${OUT}/meta.txt"
for i in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "healthy after ${i} iterations" | tee -a "${OUT}/meta.txt"
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${PROF_NAME}"; then
    echo "profile container exited" | tee -a "${OUT}/meta.txt"
    docker logs "${PROF_NAME}" 2>&1 | tail -100 | tee "${OUT}/server_fail.log"
    exit 1
  fi
  if [[ "${i}" -eq 360 ]]; then
    echo "timeout waiting for health" | tee -a "${OUT}/meta.txt"
    docker logs "${PROF_NAME}" 2>&1 | tail -100 | tee "${OUT}/server_fail.log"
    exit 1
  fi
  sleep 2
done

{
  echo "===== processes ====="
  docker exec "${PROF_NAME}" ps -eo pid,ppid,comm,args | grep -Ei 'vllm|Engine|rocprof' || true
  echo "===== kfd/dri fds ====="
  docker exec "${PROF_NAME}" bash -lc '
    for pid in $(pgrep -f "vllm|EngineCore" || true); do
      echo "===== PID ${pid} ====="
      ls -l /proc/${pid}/fd 2>/dev/null | grep -E "/dev/kfd|/dev/dri" || true
    done
  ' || true
} > "${OUT}/engine_identity.txt"

python3 "${ROOT}/scripts/bench_openai_chat.py" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "${SERVED}" \
  --input-len "${ILEN}" --output-len "${OLEN}" \
  --num-prompts "${WARM}" --concurrency "${CONC}" --timeout 900 \
  --out "${OUT}/warmup.json"

sleep 15

date -u +%Y-%m-%dT%H:%M:%S.%NZ | tee -a "${OUT}/markers.log"
echo "BENCHMARK_START" | tee -a "${OUT}/markers.log"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "${OUT}/timed_window_start.txt"

python3 "${ROOT}/scripts/bench_openai_chat.py" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "${SERVED}" \
  --input-len "${ILEN}" --output-len "${OLEN}" \
  --num-prompts "${NPROM}" --concurrency "${CONC}" --timeout 900 \
  --out "${OUT}/bench.json"

date -u +%Y-%m-%dT%H:%M:%SZ | tee "${OUT}/timed_window_end.txt"
echo "BENCHMARK_END" | tee -a "${OUT}/markers.log"
date -u +%Y-%m-%dT%H:%M:%S.%NZ | tee -a "${OUT}/markers.log"

curl -sf "http://127.0.0.1:${PORT}/metrics" > "${OUT}/metrics_after.prom" || true
docker logs "${PROF_NAME}" > "${OUT}/server.log" 2>&1 || true

# Stop so rocprofv3 can finalize descendant traces (--process-sync).
docker stop "${PROF_NAME}" >/dev/null || true
sleep 3
docker rm "${PROF_NAME}" >/dev/null 2>&1 || true

python3 "${ROOT}/scripts/aggregate_kernel_csv.py" \
  --input-dir "${OUT}/rocprof" \
  --out "${OUT}/kernel_top.json" || true

python3 - <<PY
import json
from pathlib import Path
p=Path("${OUT}/bench.json")
if p.exists():
    d=json.loads(p.read_text())
    print("BENCH C${CONC}", round(d.get("output_throughput") or 0, 2),
          "tok/s (profiler overhead; not a campaign headline)")
kt=Path("${OUT}/kernel_top.json")
print("kernel_top", "present" if kt.exists() and kt.stat().st_size>8 else "MISSING — check rocprof/")
print("artifacts ${OUT}")
print("Open rocprof/*.pftrace in https://ui.perfetto.dev and clip to BENCHMARK_START/END UTC")
PY

if [[ "${RESTORE}" -eq 1 ]]; then
  if curl -sf "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
    echo "HF control already healthy on 8000"
  else
    echo "restoring HF control on 8000"
    "${ROOT}/scripts/launch_vllm_mxfp4.sh"
  fi
fi
