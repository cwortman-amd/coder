#!/usr/bin/env bash
# Launch the validated MI350P MXFP4 control recipe as a dedicated GPU 1 decoder.
#
# By default, stop the prefill/router peers so host swap cannot contaminate the
# decode control. Set KEEP_PD_PEERS=1 only after the standalone control passes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${KEEP_PD_PEERS:-0}" != "1" ]]; then
  docker stop rocm-mxfp4-pd-prefill rocm-mxfp4-pd-router >/dev/null 2>&1 || true
fi

export HIP_VISIBLE_DEVICES="${DECODE_GPU:-0}"
export VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-rocm-mxfp4-pd-decode}"
export VLLM_PORT="${VLLM_PORT:-8200}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-awq}"
export VLLM_CPUSET_CPUS="${VLLM_CPUSET_CPUS:-0-7,16-23}"
export VLLM_CPUSET_MEMS="${VLLM_CPUSET_MEMS:-0}"
# Keep the decoder resident when a prefiller shares this 32 GiB host.
export VLLM_MEMORY_RESERVATION="${VLLM_MEMORY_RESERVATION:-8g}"

# Do not add max-num-batched-tokens or max-num-seqs here. The validated control
# uses vLLM's O2 defaults: an 8192-token compile range and graph capture to 512.
exec "${ROOT}/scripts/launch_vllm_mxfp4.sh" "$@"
