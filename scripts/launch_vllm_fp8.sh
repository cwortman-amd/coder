#!/usr/bin/env bash
# Exclusive-GPU FP8 comparison recipe. This intentionally never stops MXFP4.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${VLLM_PORT:-8000}"
NAME="${VLLM_CONTAINER_NAME:-rocm-inference-server-fp8}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
MODEL="${VLLM_MODEL:-Qwen/Qwen3.8-27B-FP8}"
SERVED="${VLLM_SERVED_NAME:-fp8}"

if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "port ${PORT} already serves a model; stop it manually before the exclusive-GPU FP8 run" >&2
  exit 2
fi
if docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "container ${NAME} already exists; refusing to replace it" >&2
  exit 2
fi

exec docker run --rm --name "${NAME}" --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri \
  --group-add "${VIDEO_GID:-44}" --group-add "${RENDER_GID:-993}" \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  -e GPU_ARCHS=gfx950 \
  -e HF_HOME=/root/.cache/huggingface \
  -e SAFETENSORS_FAST_GPU=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e PYTHONUNBUFFERED=1 \
  -v /home/amd/.cache/huggingface:/root/.cache/huggingface:ro \
  -v "${ROOT}/_results:/results" \
  "${IMAGE}" "${MODEL}" \
  --served-model-name "${SERVED}" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  "$@"
