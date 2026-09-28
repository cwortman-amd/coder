#!/usr/bin/env bash
# Serve openai/gpt-oss-20b native MXFP4 on one GPU.
# The 20B checkpoint fits a 32 GB Radeon AI PRO R9700S. gpt-oss-120b does not.
#
#   VLLM_ROCM_USE_AITER=1 vllm serve openai/gpt-oss-20b \
#     --dtype auto -tp 1 --no-enable-prefix-caching --disable-uvicorn-access-log
#
# Restore the previous Qwen server with ./setup.sh when this track is idle.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export ROOT_DIR="$ROOT"
export SCRIPT_DIR="$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/lib/gpu_profile.sh"

if [ -f "$HOME/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/.env"
  set +a
fi
if [ -f "${ROOT}/.env" ]; then
  prev_token="${HF_TOKEN:-}"
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
  if [ -z "${HF_TOKEN:-}" ] && [ -n "$prev_token" ]; then
    export HF_TOKEN="$prev_token"
  fi
fi

GPU_PROFILE="${GPU_PROFILE_OVERRIDE:-${GPU_PROFILE:-auto}}"
apply_gpu_profile

NAME="${VLLM_CONTAINER_NAME:-rocm-gpt-oss-server}"
IMAGE="${VLLM_IMAGE}"
MODEL="${GPT_OSS_MODEL:-openai/gpt-oss-20b}"
PORT="${VLLM_PORT:-8000}"

if [ "$MODEL" != "openai/gpt-oss-20b" ] && [[ "$MODEL" != *gpt-oss-20b* ]]; then
  echo "This launcher is the 20B MXFP4 recipe. Refusing model ${MODEL}." >&2
  exit 1
fi
if [ "$GPU_PROFILE" = "r9700" ]; then
  echo "GPU ${GPU_LABEL}: gpt-oss-20b native MXFP4 fits this 32 GB card."
fi

docker stop "${NAME}" >/dev/null 2>&1 || true
docker rm "${NAME}" >/dev/null 2>&1 || true
# Host networking: another server on this port will not show up as a published port.
for other in rocm-inference-server rocm-mxfp4-server rocm-ep-server; do
  if [ "$other" = "$NAME" ]; then
    continue
  fi
  if docker ps --format '{{.Names}}' | grep -qx "$other"; then
    echo "Stopping ${other} so port ${PORT} is free."
    docker stop "$other" >/dev/null
  fi
done

args=(
  --name "${NAME}" --restart=no --network host --ipc host
  --shm-size 64g
  --device /dev/kfd --device /dev/dri
  --group-add "${VIDEO_GID}" --group-add "${RENDER_GID}"
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined
  -e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
  -e "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
  -e "GPU_ARCHS=${PYTORCH_ROCM_ARCH}"
  -e "GPU_PROFILE=${GPU_PROFILE}"
  -e VLLM_ROCM_USE_AITER=1
  -e HF_HOME=/root/.cache/huggingface
  -e "HF_TOKEN=${HF_TOKEN:-}"
  -e SAFETENSORS_FAST_GPU=1
  -e PYTHONUNBUFFERED=1
  -e TOKENIZERS_PARALLELISM=false
  -v "${HF_HOME}:/root/.cache/huggingface"
  -v "${ROOT}/_results:/results"
  -v "${MODELS_DIR}:/models"
)
if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
  args+=(-e "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}")
fi

docker run -d "${args[@]}" --entrypoint vllm "${IMAGE}" \
  serve "${MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype auto \
  --tensor-parallel-size 1 \
  --no-enable-prefix-caching \
  --disable-uvicorn-access-log

echo "started ${NAME} (${MODEL}, VLLM_ROCM_USE_AITER=1)"
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
