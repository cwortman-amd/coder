#!/usr/bin/env bash
# Exclusive fused gate-up tile sweep. Stops the 27B server and does not restore it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/lib/serve.sh"
serve_gpu_flags 0
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
NAME="${VLLM_CONTAINER_NAME:-rocm-inference-server}"
BENCH_NAME="${VLLM_GEMM_CONTAINER:-rocm-mxfp4-gemm-bench}"
OUT="${ROOT}/_results/profiling/mxfp4_gateup"

mkdir -p "${OUT}"
docker stop "${NAME}" "${BENCH_NAME}" >/dev/null 2>&1 || true
docker rm "${BENCH_NAME}" >/dev/null 2>&1 || true

docker run --rm --name "${BENCH_NAME}" --network host --ipc host --shm-size 16g \
  "${SERVE_GPU_FLAGS[@]}" \
  --cap-add SYS_PTRACE \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  -e GPU_ARCHS=gfx950 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e PYTHONUNBUFFERED=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -v "${ROOT}/models:/models" \
  -v "${ROOT}/scripts:/workspace/scripts" \
  -v "${ROOT}/_results:/results" \
  --entrypoint python3 "${IMAGE}" \
  /workspace/scripts/bench_mxfp4_gateup_shape.py \
  --out /results/profiling/mxfp4_gateup/shape_sweep.json \
  | tee "${OUT}/shape_sweep.log"
