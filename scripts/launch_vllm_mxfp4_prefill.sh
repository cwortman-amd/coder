#!/usr/bin/env bash
# Launch the validated MI350P MXFP4 recipe as the dedicated GPU 1 prefiller.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HIP_VISIBLE_DEVICES="${PREFILL_GPU:-1}"
export VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-rocm-mxfp4-pd-prefill}"
export VLLM_PORT="${VLLM_PORT:-8100}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-awq}"
export VLLM_CPUSET_CPUS="${VLLM_CPUSET_CPUS:-8-15,24-31}"
export VLLM_CPUSET_MEMS="${VLLM_CPUSET_MEMS:-1}"

# Keep the validated O2 defaults and 8192-token compile range. Phase-specific
# limits can be tested only after this control matches the standalone server.
exec "${ROOT}/scripts/launch_vllm_mxfp4.sh" "$@"
