# Wrap vllm serve with rocprofv3 from process start (host, no Docker).
# Prefer scripts/profile_enginecore_launch.sh for the MI350P container path.
set -euo pipefail
OUT="${1:?usage: profile_server.sh <output_dir> [vllm-serve-args...]}"
shift
mkdir -p "${OUT}"
exec rocprofv3 \
  --kernel-trace \
  --hip-runtime-trace \
  --memory-copy-trace \
  --process-sync \
  --output-format pftrace csv \
  --output-directory "${OUT}" \
  -- \
  vllm serve amd/Qwen3.8-27B-Quark-AWQ-MXFP4 \
    --served-model-name awq \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --host 127.0.0.1 \
    --port 8001 \
    "$@"
