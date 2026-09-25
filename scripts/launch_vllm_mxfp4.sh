#!/usr/bin/env bash
# Launch the frozen HF MXFP4 vLLM recipe on MI350P GPU 0.
# Extra vllm serve args are forwarded (e.g. speculative-config).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
  --device /dev/kfd --device /dev/dri
  --group-add 44 --group-add 993
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined --security-opt label=disable
  -e HIP_VISIBLE_DEVICES=0
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

if [[ -n "${VLLM_ROCM_USE_AITER:-}" ]]; then
  COMMON+=(-e "VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER}")
fi
if [[ -n "${AITER_AFP4_DEFAULT_JSON:-}" ]]; then
  COMMON+=(-v "${AITER_AFP4_DEFAULT_JSON}:/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_afp4wfp4/DEFAULT.json:ro")
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
