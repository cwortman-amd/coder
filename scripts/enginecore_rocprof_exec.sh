#!/usr/bin/env bash
# Python multiprocessing executable used by the vLLM API parent.
#
# Resource-tracker and helper processes are passed through to Python. Only a
# multiprocessing worker (`--multiprocessing-fork`) is launched under
# rocprofv3. On TP=1 vLLM V1 this is the GPU-owning EngineCore process.
# On TP>1 this selector can also match MultiprocExecutor workers; the route is
# validated here only for TP=1.
set -euo pipefail

PYTHON="${ENGINECORE_PYTHON:-/usr/bin/python}"
OUT_DIR="${ENGINECORE_ROCPROF_DIR:-/results/profiling/enginecore_exec}"

IS_MP_WORKER=0
for arg in "$@"; do
  if [[ "${arg}" == "--multiprocessing-fork" ]]; then
    IS_MP_WORKER=1
    break
  fi
done

if [[ "${IS_MP_WORKER}" -eq 0 ]]; then
  exec "${PYTHON}" "$@"
fi

mkdir -p "${OUT_DIR}"
{
  printf 'utc=%s pid=%s ppid=%s argv=' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$PPID"
  printf '%q ' "$@"
  printf '\n'
} >>"${OUT_DIR}/spawn.log"

TRACE_ARGS=(--process-sync)
OUT_FILE="enginecore"
if [[ -n "${ENGINECORE_ROCPROF_GATE_FILE:-}" ]]; then
  # ROCTx pause/resume inside the worker. Startup stays untraced.
  # Skip --process-sync: TP=2 ranks do not exit together, and the wait
  # drops the trace when the container stops.
  export ROCPROF_GATED=1
  TRACE_ARGS=(--rccl-trace --selected-regions)
  OUT_FILE="decode-$$"
fi

exec /opt/rocm/bin/rocprofv3 \
  --kernel-trace --hip-runtime-trace --memory-copy-trace \
  "${TRACE_ARGS[@]}" \
  --stats --summary \
  --output-format csv \
  --output-directory "${OUT_DIR}" \
  --output-file "${OUT_FILE}" \
  -- \
  "${PYTHON}" "$@"
