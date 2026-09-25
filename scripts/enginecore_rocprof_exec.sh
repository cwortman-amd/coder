#!/usr/bin/env bash
# multiprocessing spawn executable: rocprofv3 is parent of EngineCore Python only.
# Do not set this as the docker --entrypoint (that wraps vllm serve).
# Usage (next launch, not live attach):
#   bind-mount this file and, in the API parent before EngineCore start,
#   multiprocessing.spawn.set_executable("/path/to/enginecore_rocprof_exec.sh")
set -euo pipefail
OUT_DIR="${ENGINECORE_ROCPROF_DIR:-/results/profiling/enginecore_exec}"
mkdir -p "${OUT_DIR}"
exec /opt/rocm/bin/rocprofv3 \
  --kernel-trace --hip-runtime-trace --memory-copy-trace \
  --process-sync --stats --summary \
  --output-format csv pftrace \
  --output-directory "${OUT_DIR}" \
  --output-file enginecore \
  -- \
  "$@"
