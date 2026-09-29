#!/usr/bin/env bash
# Download Qwen/Qwen3.8-Flash-Next-FP8 (about 173 GiB, 131 shards).
# Stops the dense 27B server first: this host has 30 GiB RAM and cannot
# keep that server resident while the checkpoint fills the page cache.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
DEST="${ROOT}/models/Qwen3.8-Flash-Next-FP8"
LOG="${ROOT}/_results/flash_next/download.log"
NAME=flash-next-download

mkdir -p "${DEST}" "${ROOT}/_results/flash_next"
docker stop rocm-inference-server >/dev/null 2>&1 || true
docker rm -f "${NAME}" >/dev/null 2>&1 || true

avail_gib="$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
{
  echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "MemAvailable_GiB ${avail_gib}"
  df -h "${ROOT}" | tail -1
} | tee "${LOG}"

if [[ "${avail_gib}" -lt 6 && "${FLASH_NEXT_ALLOW_LOW_RAM:-0}" != "1" ]]; then
  echo "MemAvailable is ${avail_gib} GiB after stopping the 27B server." | tee -a "${LOG}"
  echo "Refusing the download. Set FLASH_NEXT_ALLOW_LOW_RAM=1 to override." | tee -a "${LOG}"
  exit 2
fi

docker run -d --name "${NAME}" --network host \
  -e HF_HOME=/root/.cache/huggingface \
  -e PYTHONUNBUFFERED=1 \
  -e HF_HUB_DISABLE_XET=1 \
  -v /home/amd/.cache/huggingface:/root/.cache/huggingface \
  -v "${ROOT}/models:/models" \
  --entrypoint hf "${IMAGE}" \
  download Qwen/Qwen3.8-Flash-Next-FP8 \
  --local-dir /models/Qwen3.8-Flash-Next-FP8 \
  --max-workers 2

echo "download container ${NAME} started; log: ${LOG}"
echo "restore dense control later with ./scripts/launch_vllm_mxfp4.sh"
