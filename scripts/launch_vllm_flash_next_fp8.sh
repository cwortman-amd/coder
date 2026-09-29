#!/usr/bin/env bash
# Experimental Flash-Next FP8 on two MI350P GPUs.
# Adapted from the vLLM ROCm recipe, which was published for 4x MI355X TP=4.
# This is not a validated MI350P recipe and is not the frozen 27B MXFP4 control.
#
# Text-only, 16K context, MTP off. N-gram weights stay on GPU: this image's
# Engram CPU-offload path requires CUDA, and the host has 30 GiB RAM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/lib/serve.sh"
serve_gpu_flags "${HIP_VISIBLE_DEVICES:-0,1}"
NAME=rocm-flash-next-server
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
MODEL="${FLASH_NEXT_MODEL:-/models/Qwen3.8-Flash-Next-FP8}"
PORT="${VLLM_PORT:-8000}"
HOST_MODEL="${ROOT}/models/Qwen3.8-Flash-Next-FP8"
LOG_DIR="${ROOT}/_results/flash_next"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${HOST_MODEL}/config.json" ]]; then
  echo "missing ${HOST_MODEL}/config.json" >&2
  echo "run ./scripts/download_flash_next_fp8.sh first" >&2
  exit 2
fi
shards="$(find "${HOST_MODEL}" -name 'model-*.safetensors' | wc -l)"
if [[ "${shards}" -lt 131 ]]; then
  echo "checkpoint has ${shards}/131 shards; download is incomplete" >&2
  exit 2
fi

avail_gib="$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
if [[ "${avail_gib}" -lt 6 && "${FLASH_NEXT_ALLOW_LOW_RAM:-0}" != "1" ]]; then
  echo "MemAvailable is ${avail_gib} GiB. Stop other servers before this load." >&2
  exit 2
fi

docker stop rocm-inference-server flash-next-download >/dev/null 2>&1 || true
docker rm -f "${NAME}" >/dev/null 2>&1 || true

{
  echo "launch $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  awk '/MemTotal|MemAvailable|SwapFree/ {print}' /proc/meminfo
} | tee "${LOG_DIR}/startup_mem.txt"

docker run -d \
  --name "${NAME}" --restart=no --network host --ipc host \
  --shm-size 8g \
  "${SERVE_GPU_FLAGS[@]}" \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  -e GPU_ARCHS=gfx950 \
  -e HF_HOME=/root/.cache/huggingface \
  -e SAFETENSORS_FAST_GPU=1 \
  -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_PLE_CPU_OFFLOAD=0 \
  -v /home/amd/.cache/huggingface:/root/.cache/huggingface \
  -v "${ROOT}/models:/models" \
  -v "${ROOT}/_results:/results" \
  "${IMAGE}" \
  "${MODEL}" \
  --served-model-name flash-next-fp8 \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --host 127.0.0.1 \
  --port "${PORT}" \
  "$@"

echo "started ${NAME}"
for i in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "healthy after ${i} iterations"
    {
      echo "healthy $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      awk '/MemAvailable|SwapFree/ {print}' /proc/meminfo
      amd-smi metric -m
    } | tee "${LOG_DIR}/healthy_mem.txt"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo "container exited" >&2
    docker logs "${NAME}" 2>&1 | tail -80
    exit 1
  fi
  if (( i % 15 == 0 )); then
    {
      echo "poll ${i} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      awk '/MemAvailable|SwapFree/ {print}' /proc/meminfo
    } >> "${LOG_DIR}/startup_mem.txt"
  fi
  sleep 2
done
echo "timed out waiting for health" >&2
docker logs "${NAME}" 2>&1 | tail -40
exit 1
