#!/usr/bin/env bash
# PyTorch-profiler window for the documented MI350P TP comparison.
#
# vLLM 0.30 does not read VLLM_TORCH_PROFILER_DIR. The server has to be
# started with --profiler-config, then /start_profile and /stop_profile
# bracket a short request. delay_iterations=1 skips the first engine step
# and max_iterations=8 keeps the chrome trace to a few decode steps.
#
# Usage: scripts/profile_tp_torch.sh 1|2
set -euo pipefail

TP="${1:?usage: profile_tp_torch.sh 1|2}"
if [[ "${TP}" != "1" && "${TP}" != "2" ]]; then
  echo "TP must be 1 or 2" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="rocm-tp-torch"
OUT_CONT="/results/tp_torch_profile/tp${TP}"
OUT_HOST="${ROOT}/_results/tp_torch_profile/tp${TP}"
mkdir -p "${OUT_HOST}"
docker stop rocm-tp-compare rocm-tp-profile "${NAME}" >/dev/null 2>&1 || true

export VLLM_TP_SIZE="${TP}"
export VLLM_CONTAINER_NAME="${NAME}"
export VLLM_KV_CACHE_MEMORY_BYTES=34359738368
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT

"${ROOT}/scripts/launch_vllm_mxfp4.sh" \
  --no-enable-prefix-caching \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${OUT_CONT}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":false,\"torch_profiler_use_gzip\":true,\"ignore_frontend\":true,\"delay_iterations\":1,\"max_iterations\":8}"

docker exec -i "${NAME}" python3 - <<'PY'
from pathlib import Path
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    "/models/Qwen3.8-27B-Quark-AWQ-MXFP4-sharded", trust_remote_code=True
)
ids = [tok.eos_token_id or 1] * 1024
Path("/results/tp_torch_profile/prompt.txt").write_text(tok.decode(ids))
print("prompt_tokens", len(ids))
PY

complete() {
  local max_tokens="$1"
  docker exec -e MAX_TOKENS="${max_tokens}" "${NAME}" python3 -c '
import json, os, urllib.request
from pathlib import Path
body = json.dumps({
    "model": "awq",
    "prompt": Path("/results/tp_torch_profile/prompt.txt").read_text(),
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": 0,
    "ignore_eos": True,
}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as resp:
    payload = json.load(resp)
print(json.dumps({"usage": payload.get("usage", {})}))
'
}

echo "warmup request"
complete 8 | tee "${OUT_HOST}/warmup.json"

echo "start_profile"
curl -sf -X POST "http://127.0.0.1:8000/start_profile"
echo
echo "profiled request"
complete 16 | tee "${OUT_HOST}/profiled.json"
echo "stop_profile"
curl -sf -X POST "http://127.0.0.1:8000/stop_profile"
echo
sleep 3
find "${OUT_HOST}" -type f -printf '%s %p\n' | sort -n
