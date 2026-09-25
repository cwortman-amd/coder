#!/usr/bin/env bash
# Preflight, or exclusively run, vLLM-native captured mini-decode profiling.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${VLLM_CONTAINER_NAME:-rocm-inference-server}"
RUN_NAME="${VLLM_CAPTURE_CONTAINER:-vllm-captured-decode}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
OUT_HOST="${ROOT}/_results/profiling/vllm_captured"
OUT_CONTAINER="/results/profiling/vllm_captured"
EXCLUSIVE=0

usage() {
  cat <<'EOF'
Usage:
  scripts/bench_vllm_captured_decode.sh
      Safe preflight inside the running container. Never loads another model.

  scripts/bench_vllm_captured_decode.sh --exclusive
      Stop :8000, run an in-process vLLM O2 FULL_AND_PIECEWISE experiment,
      then restore :8000 with scripts/launch_vllm_mxfp4.sh.

This host cannot safely hold the serving engine and a second 27B engine.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exclusive) EXCLUSIVE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "${OUT_HOST}"

if [[ "${EXCLUSIVE}" -eq 0 ]]; then
  if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo "${NAME} is not running; use --exclusive to run the experiment" >&2
    exit 2
  fi
  # The project is not mounted in the serving container, so stream the script
  # to Python while retaining /results and the live image environment.
  docker exec -i "${NAME}" python3 - \
    --preflight-only --out-dir "${OUT_CONTAINER}" \
    < "${ROOT}/scripts/bench_vllm_captured_decode.py"
  echo "preflight written to ${OUT_HOST}"
  exit 0
fi

STOPPED_BASE=0
restore() {
  docker rm -f "${RUN_NAME}" >/dev/null 2>&1 || true
  if [[ "${STOPPED_BASE}" -eq 1 ]]; then
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
      echo "frozen control already healthy on :8000"
    else
      echo "restoring frozen control on :8000"
      env -u VLLM_ROCM_USE_AITER -u AITER_AFP4_DEFAULT_JSON \
        "${ROOT}/scripts/launch_vllm_mxfp4.sh"
    fi
  fi
}
trap restore EXIT

if docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "stopping ${NAME}; exclusive GPU is required"
  docker stop "${NAME}" >/dev/null
  STOPPED_BASE=1
fi

docker rm -f "${RUN_NAME}" >/dev/null 2>&1 || true
docker run --rm \
  --name "${RUN_NAME}" --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 993 \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --security-opt label=disable \
  -e HIP_VISIBLE_DEVICES=0 \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  -e GPU_ARCHS=gfx950 \
  -e HF_HOME=/root/.cache/huggingface \
  -e SAFETENSORS_FAST_GPU=1 \
  -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -v /home/amd/.cache/huggingface:/root/.cache/huggingface \
  -v "${ROOT}/models:/models:ro" \
  -v "${ROOT}/scripts:/workspace/scripts:ro" \
  -v "${ROOT}/_results:/results" \
  --entrypoint python3 \
  "${IMAGE}" \
  /workspace/scripts/bench_vllm_captured_decode.py \
  --exclusive-confirmed --out-dir "${OUT_CONTAINER}"

echo "experiment written to ${OUT_HOST}"
