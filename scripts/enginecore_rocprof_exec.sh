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

exec /opt/rocm/bin/rocprofv3 \
  --kernel-trace --hip-runtime-trace --memory-copy-trace \
  --process-sync --stats --summary \
  --output-format csv pftrace \
  --output-directory "${OUT_DIR}" \
  --output-file enginecore \
  -- \
  "${PYTHON}" "$@"
