#!/usr/bin/env bash
# Launch the frozen HF MXFP4 vLLM recipe on MI350P GPU 0.
# Extra vllm serve args are forwarded (e.g. speculative-config).
#
# EngineCore-only rocprof (restart required):
#   ENGINECORE_ROCP_EXEC=1 \
#   ENGINECORE_ROCPROF_DIR=/results/profiling/enginecore_exec/<stamp> \
#     ./scripts/launch_vllm_mxfp4.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/lib/serve.sh"
serve_gpu_flags "${HIP_VISIBLE_DEVICES:-0}"
NAME="${VLLM_CONTAINER_NAME:-rocm-inference-server}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
MODEL="${VLLM_MODEL:-/models/Qwen3.8-27B-Quark-AWQ-MXFP4-sharded}"
SERVED="${VLLM_SERVED_NAME:-awq}"
PORT="${VLLM_PORT:-8000}"

docker stop "${NAME}" >/dev/null 2>&1 || true
docker rm "${NAME}" >/dev/null 2>&1 || true

COMMON=(
  --name "${NAME}" --restart=no --network host --ipc host
  --shm-size 64g
  "${SERVE_GPU_FLAGS[@]}"
  -e PYTORCH_ROCM_ARCH=gfx950
  -e GPU_ARCHS=gfx950
  -e HF_HOME=/root/.cache/huggingface
  -e SAFETENSORS_FAST_GPU=1
  -e PYTHONUNBUFFERED=1
  -e TOKENIZERS_PARALLELISM=false
  -e ROCP_TOOL_ATTACH=1
  -v /home/amd/.cache/huggingface:/root/.cache/huggingface
  -v "${ROOT}/models:/models"
  -v "${ROOT}/_results:/results"
)

if [[ -n "${VLLM_CPUSET_CPUS:-}" ]]; then
  COMMON+=(--cpuset-cpus "${VLLM_CPUSET_CPUS}")
fi
if [[ -n "${VLLM_CPUSET_MEMS:-}" ]]; then
  COMMON+=(--cpuset-mems "${VLLM_CPUSET_MEMS}")
fi
if [[ -n "${VLLM_ENABLE_V1_MULTIPROCESSING:-}" ]]; then
  COMMON+=(-e "VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING}")
fi

if [[ -n "${REAL_ATTN_CAPTURE_DIR:-}" ]]; then
  COMMON+=(
    -e "REAL_ATTN_CAPTURE_DIR=${REAL_ATTN_CAPTURE_DIR}"
    -e "REAL_ATTN_CAPTURE_TOKENS=${REAL_ATTN_CAPTURE_TOKENS:-44}"
    -e PYTHONPATH=/opt/vllm-attn-capture
    -v "${ROOT}/scripts/attn_capture_sitecustomize.py:/opt/vllm-attn-capture/sitecustomize.py:ro"
  )
fi

if [[ "${ENGINECORE_ROCP_ATTACH:-0}" == "1" || "${ENGINECORE_ROCP_EXEC:-0}" == "1" ]]; then
  # Python spawn starts EngineCore in a fresh interpreter. sitecustomize can
  # request attach for an attach-enabled register build or route only the
  # spawned EngineCore via rocprofv3.
  COMMON+=(
    --cap-add SYS_PTRACE
    -e PYTHONPATH=/opt/vllm-enginecore-attach
    -v "${ROOT}/scripts/rocprof_attach_sitecustomize.py:/opt/vllm-enginecore-attach/sitecustomize.py:ro"
  )
fi
if [[ "${ENGINECORE_ROCP_ATTACH:-0}" == "1" ]]; then
  COMMON+=(-e VLLM_ENGINECORE_ROCP_ATTACH=1)
fi
if [[ "${ENGINECORE_ROCP_EXEC:-0}" == "1" ]]; then
  mkdir -p "${ROOT}/_results/profiling/enginecore_exec"
  COMMON+=(
    -e VLLM_ENGINECORE_ROCP_EXEC=1
    -e VLLM_ENGINECORE_ROCP_EXECUTABLE=/opt/vllm-enginecore-attach/enginecore_rocprof_exec.sh
    -e "ENGINECORE_ROCPROF_DIR=${ENGINECORE_ROCPROF_DIR:-/results/profiling/enginecore_exec}"
    -v "${ROOT}/scripts/enginecore_rocprof_exec.sh:/opt/vllm-enginecore-attach/enginecore_rocprof_exec.sh:ro"
  )
fi

if [[ -n "${VLLM_ROCM_USE_AITER:-}" ]]; then
  COMMON+=(-e "VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER}")
fi
if [[ -n "${AITER_AFP4_DEFAULT_JSON:-}" ]]; then
  COMMON+=(-v "${AITER_AFP4_DEFAULT_JSON}:/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_afp4wfp4/DEFAULT.json:ro")
fi
# One exact-shape file. Basename must be GEMM-AFP4WFP4-N=<N>-K=<K>.json.
# This does not replace DEFAULT.json and does not set VLLM_ROCM_USE_AITER.
if [[ -n "${AITER_AFP4_SHAPE_JSON:-}" ]]; then
  shape_base="$(basename "${AITER_AFP4_SHAPE_JSON}")"
  COMMON+=(
    -v "${AITER_AFP4_SHAPE_JSON}:/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_afp4wfp4/${shape_base}:ro"
  )
fi

SERVE_ARGS=(
  "${MODEL}"
  --served-model-name "${SERVED}"
  --trust-remote-code
  --tensor-parallel-size 1
  --max-model-len 16384
  --host 127.0.0.1
  --port "${PORT}"
  --kv-cache-memory-bytes 103223724237
  "$@"
)

if [[ -n "${VLLM_ROCPROF_DIR:-}" ]]; then
  mkdir -p "${ROOT}/_results/${VLLM_ROCPROF_DIR}"
  # rocprofv3 attach is unavailable. EngineCore is a child process, so disable
  # V1 multiprocessing so HIP/kernels run in the wrapped PID.
  docker run -d "${COMMON[@]}" \
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    -e ROCPROFILER_LIBRARY_CTOR=1 \
    -e LD_PRELOAD=/opt/rocm/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so:/opt/rocm/lib/librocprofiler-sdk.so \
    -e ROCP_TOOL_LIBRARIES=/opt/rocm/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so \
    -e ROCPROF_OUTPUT_FILE_NAME=timed \
    -e ROCPROF_OUTPUT_PATH="/results/${VLLM_ROCPROF_DIR}" \
    -e ROCPROF_OUTPUT_FORMAT=csv,pftrace \
    -e ROCPROF_HIP_RUNTIME_API_TRACE=1 \
    -e ROCPROF_KERNEL_TRACE=1 \
    -e ROCPROF_MEMORY_COPY_TRACE=1 \
    --entrypoint /opt/rocm/bin/rocprofv3 "${IMAGE}" \
    --kernel-trace --hip-runtime-trace --memory-copy-trace \
    --stats --summary \
    --output-format csv pftrace \
    --output-directory "/results/${VLLM_ROCPROF_DIR}" \
    --output-file timed \
    -- \
    vllm serve "${SERVE_ARGS[@]}"
else
  docker run -d "${COMMON[@]}" "${IMAGE}" "${SERVE_ARGS[@]}"
fi

echo "started ${NAME}"
for i in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "healthy after ${i} iterations"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo "container exited"
    docker logs "${NAME}" 2>&1 | tail -80
    exit 1
  fi
  sleep 2
done
echo "timeout waiting for health"
docker logs "${NAME}" 2>&1 | tail -80
exit 1
