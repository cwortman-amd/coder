#!/usr/bin/env bash
# Performance candidate: frozen MXFP4 path plus explicit Triton attention.
# This changes greedy outputs versus ROCM_ATTN on 18/116 diagnostic prompts,
# so it is opt-in until the quality policy accepts that numerical difference.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT}/scripts/launch_vllm_mxfp4.sh" \
  --attention-backend TRITON_ATTN \
  "$@"
