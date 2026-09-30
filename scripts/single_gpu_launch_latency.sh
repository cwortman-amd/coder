#!/usr/bin/env bash
# Time a 1-block kernel on one visible GPU. HIP_VISIBLE_DEVICES selects the card.
# amd-smi index 0 is 0000:8b:00.0. Index 1 is 0001:c7:00.0.
#
#   bash scripts/single_gpu_launch_latency.sh 0
#   bash scripts/single_gpu_launch_latency.sh 1
#   bash scripts/single_gpu_launch_latency.sh 0 1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/scripts/single_gpu_launch_latency.cu"
BIN="${ROOT}/_results/single_gpu_launch_latency"
LAUNCHES="${LAUNCHES:-20000}"
GPUS=("$@")
if [[ ${#GPUS[@]} -eq 0 ]]; then
    GPUS=(0 1)
fi

mkdir -p "${ROOT}/_results"
hipcc -O3 -o "${BIN}" "${SRC}"

for gpu in "${GPUS[@]}"; do
    echo "HIP_VISIBLE_DEVICES=${gpu}"
    HIP_VISIBLE_DEVICES="${gpu}" "${BIN}" "${LAUNCHES}"
done
