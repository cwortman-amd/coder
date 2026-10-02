#!/usr/bin/env bash
# Reproduce the MI355X vLLM scaling curves on this 2× MI350P host.
# Source: https://rocm.blogs.amd.com/artificial-intelligence/scaling-ai-inference/README.html
#
# The blog ran TP=8 on 8× MI355X (288 GB, 8 TB/s, 256 CU each) with vLLM 0.10.1.
# This host is 2× MI350P (144 GB, 4 TB/s, 128 CU each) and vllm/vllm-openai-rocm:latest.
# Two MI350P cards have the same aggregate HBM bandwidth as one MI355X, so a
# per-GPU 50% result is about (8-GPU chart throughput) / 8 on this box.
#
# Adaptations required by 32 GiB host RAM (the blog machine had 3072 GiB):
#   CUDA-graph capture is limited to batch sizes 1..128, not every size through 8192.
#   --max-seq-len-to-capture is omitted so startup does not capture a 10k graph.
#   gpu-memory-utilization is 0.90.
# The old VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4 and
# VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE flags are gone in vLLM 0.30.
# GPT-OSS uses AITER MXFP4-weight / BF16-activation MoE (the A16W4 successor).
# Llama sets VLLM_ROCM_USE_AITER_TRITON_ROPE=1 in place of the fused-RoPE flag.
#
#   ./scripts/bench_mi350p_scaling.sh                  # download, both models, full matrix
#   ./scripts/bench_mi350p_scaling.sh --model gpt-oss  # one model
#   ./scripts/bench_mi350p_scaling.sh --model llama --bench-only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/lib/serve.sh"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
NAME="rocm-scaling-server"
PORT="${VLLM_PORT:-8000}"
RESULTS="${ROOT}/_results/scaling_mi350p"
TP=2
CONCURRENCIES=(4 8 16 32 64 128)
# ISL OSL pairs from the blog.
SHAPES=("1024 1024" "1024 8192" "8192 1024")

MODEL_CHOICE="both"
BENCH_ONLY=0
SKIP_DOWNLOAD=0
KERNEL="aiter"
SHAPE_SET="all"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--model gpt-oss|llama|both] [--kernel aiter|triton] [--shapes all|1k] [--bench-only] [--skip-download]

Serves with the blog's AITER flags on vLLM 0.30, TP=${TP}, then runs
vllm bench serve at concurrencies ${CONCURRENCIES[*]} for ISL/OSL
1024/1024, 1024/8192, and 8192/1024.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_CHOICE="$2"; shift 2 ;;
    --kernel) KERNEL="$2"; shift 2 ;;
    --shapes) SHAPE_SET="$2"; shift 2 ;;
    --bench-only) BENCH_ONLY=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

case "${MODEL_CHOICE}" in
  gpt-oss|llama|both) ;;
  *) echo "--model must be gpt-oss, llama, or both" >&2; exit 2 ;;
esac
case "${KERNEL}" in
  aiter|triton) ;;
  *) echo "--kernel must be aiter or triton" >&2; exit 2 ;;
esac
case "${SHAPE_SET}" in
  all) ;;
  1k) SHAPES=("1024 1024") ;;
  *) echo "--shapes must be all or 1k" >&2; exit 2 ;;
esac

mkdir -p "${RESULTS}"

model_id() {
  case "$1" in
    gpt-oss) echo "openai/gpt-oss-120b" ;;
    llama) echo "amd/Llama-3.3-70B-Instruct-FP8-KV" ;;
  esac
}

local_dir() {
  case "$1" in
    gpt-oss) echo "${ROOT}/models/gpt-oss-120b" ;;
    llama) echo "${ROOT}/models/Llama-3.3-70B-Instruct-FP8-KV" ;;
  esac
}

checkpoint_ready() {
  local dir="$1"
  local index="${dir}/model.safetensors.index.json"
  [[ -f "${dir}/config.json" && -f "${index}" ]] || return 1
  python3 - "${dir}" "${index}" <<'PY'
import json, sys
from pathlib import Path
root, index = Path(sys.argv[1]), Path(sys.argv[2])
shards = set(json.loads(index.read_text())["weight_map"].values())
missing = [name for name in shards if not (root / name).is_file() or (root / name).stat().st_size < 1_000_000]
sys.exit(0 if not missing else 1)
PY
}

download_one() {
  local key="$1"
  local dir id
  dir="$(local_dir "${key}")"
  id="$(model_id "${key}")"
  if checkpoint_ready "${dir}"; then
    echo "checkpoint present: ${dir}"
    return 0
  fi
  local avail
  avail="$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
  echo "MemAvailable ${avail} GiB before ${id}"
  if [[ "${avail}" -lt 4 && "${SCALING_ALLOW_LOW_RAM:-0}" != "1" ]]; then
    echo "Refusing download below 4 GiB MemAvailable." >&2
    exit 2
  fi
  mkdir -p "${dir}"
  docker rm -f scaling-download >/dev/null 2>&1 || true
  docker run --name scaling-download --network host \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONUNBUFFERED=1 \
    -v /home/amd/.cache/huggingface:/root/.cache/huggingface \
    -v "${ROOT}/models:/models" \
    --entrypoint hf \
    "${IMAGE}" \
    download "${id}" \
    --local-dir "/models/$(basename "${dir}")" \
    --max-workers 2
  checkpoint_ready "${dir}" || { echo "download finished without weights: ${dir}" >&2; exit 2; }
  echo "downloaded ${dir}"
}

stop_others() {
  local other
  for other in rocm-ep-server rocm-inference-server rocm-mxfp4-server rocm-gpt-oss-server rocm-scaling-server; do
    if docker ps --format '{{.Names}}' | grep -qx "${other}"; then
      echo "stopping ${other}"
      docker rm -f "${other}" >/dev/null
    fi
  done
}

wait_healthy() {
  local i
  for i in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "healthy after $((i * 10)) s"
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
      echo "container ${NAME} exited" >&2
      docker logs "${NAME}" 2>&1 | tail -80 >&2
      return 1
    fi
    sleep 10
  done
  echo "health timeout" >&2
  docker logs "${NAME}" 2>&1 | tail -40 >&2
  return 1
}

launch_model() {
  local key="$1"
  local dir id
  dir="$(local_dir "${key}")"
  id="$(model_id "${key}")"
  local -a env_args=()
  local -a serve_args=(
    "/models/$(basename "${dir}")"
    --served-model-name "${id}"
    --tensor-parallel-size "${TP}"
    --gpu-memory-utilization 0.90
    --host 127.0.0.1
    --port "${PORT}"
    --no-enable-prefix-caching
    --async-scheduling
    --max-num-seqs 128
  )
  local graph='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128],"max_cudagraph_capture_size":128}'
  if [[ "${key}" == "gpt-oss" && "${KERNEL}" == "triton" ]]; then
    # GPT-OSS Triton experts, ROCm attention. AITER unified attention and
    # AITER_MXFP4_BF16 left the MI350P at ~120 W on the blog flags.
    env_args+=(
      -e VLLM_ROCM_USE_AITER=0
      -e VLLM_ROCM_USE_AITER_MOE=0
      -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
    )
    serve_args+=(
      --max-model-len 10368
      --block-size 64
      --moe-backend triton
      --compilation-config "${graph}"
    )
  elif [[ "${key}" == "gpt-oss" ]]; then
    # Blog: unified attention on, MHA off, fused MoE A16W4.
    # vLLM 0.30: AITER master switch selects MXFP4-weight BF16-activation MoE.
    env_args+=(
      -e VLLM_ROCM_USE_AITER=1
      -e VLLM_ROCM_USE_AITER_MOE=1
      -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
      -e VLLM_ROCM_USE_AITER_MHA=0
    )
    serve_args+=(
      --max-model-len 10368
      --block-size 64
      --compilation-config "${graph}"
    )
  else
    # Blog also sets QuickReduce INT4. On this cross-socket PCIe pair that
    # all-reduce path measured 17 tok/s at concurrency 4, so it stays off.
    # vLLM 0.30 still selects ROCM_ATTN when it is valid; AITER MHA is requested.
    env_args+=(
      -e VLLM_ROCM_USE_AITER=1
      -e VLLM_ROCM_USE_AITER_MHA=1
      -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
      -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE
      -e VLLM_ROCM_USE_AITER_TRITON_ROPE=1
    )
    local llama_graph='{"custom_ops":["-rms_norm","-quant_fp8","-silu_and_mul"],"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128],"max_cudagraph_capture_size":128}'
    serve_args+=(
      --max-model-len 10240
      --max-num-batched-tokens 8192
      --kv-cache-dtype fp8
      --compilation-config "${llama_graph}"
    )
  fi
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  serve_gpu_flags "0,1"
  docker run -d \
    --name "${NAME}" --restart=no --network host --ipc host \
    --shm-size 8g \
    "${SERVE_GPU_FLAGS[@]}" \
    -e PYTORCH_ROCM_ARCH=gfx950 \
    -e GPU_ARCHS=gfx950 \
    -e HF_HOME=/root/.cache/huggingface \
    -e SAFETENSORS_FAST_GPU=1 \
    -e PYTHONUNBUFFERED=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e OMP_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_P2P_DISABLE=0 \
    "${env_args[@]}" \
    -v /home/amd/.cache/huggingface:/root/.cache/huggingface \
    -v "${ROOT}/models:/models" \
    -v "${ROOT}/_results:/results" \
    "${IMAGE}" \
    "${serve_args[@]}"
  docker logs -f "${NAME}" > "${RESULTS}/${key}_${KERNEL}.server.log" 2>&1 &
  echo $! > "${RESULTS}/${key}_${KERNEL}.logpid"
  echo "===== launch ${key} ${KERNEL} ====="
  wait_healthy
}

prompt_count() {
  local conc="$1" osl="$2"
  if [[ "${osl}" -ge 8192 ]]; then
    echo "${conc}"
  elif [[ "${conc}" -le 16 ]]; then
    echo $((conc * 4))
  else
    echo $((conc * 2))
  fi
}

bench_model() {
  local key="$1"
  local id dir
  id="$(model_id "${key}")"
  dir="$(local_dir "${key}")"
  local shape isl osl conc n tag
  for shape in "${SHAPES[@]}"; do
    isl="${shape%% *}"
    osl="${shape##* }"
    for conc in "${CONCURRENCIES[@]}"; do
      n="$(prompt_count "${conc}" "${osl}")"
      tag="${key}_${KERNEL}_isl${isl}_osl${osl}_c${conc}"
      if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "server unhealthy before ${tag}" >&2
        return 1
      fi
      echo "===== ${tag} prompts=${n} ====="
      mapfile -t BENCH_ARGS < <(python3 "${ROOT}/scripts/bench_serve.py" \
        --model "${id}" \
        --tokenizer "/models/$(basename "${dir}")" \
        --input-len "${isl}" \
        --output-len "${osl}" \
        --num-prompts "${n}" \
        --max-concurrency "${conc}" \
        --backend openai \
        --endpoint /v1/completions \
        --port "${PORT}" \
        --request-rate 10000 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --no-metric-percentiles \
        --ignore-eos \
        --temperature 0 \
        --no-save-detailed \
        --result-dir /results/scaling_mi350p \
        --result-filename "${tag}.json")
      docker exec -w /results/scaling_mi350p "${NAME}" \
        vllm "${BENCH_ARGS[@]}" \
        | tee "${RESULTS}/${tag}.bench.txt"
    done
  done
}

summarize() {
  python3 - "${RESULTS}" <<'PY'
import json, sys
from pathlib import Path
results = Path(sys.argv[1])
rows = []
for path in sorted(results.glob("*.json")):
    if path.name == "SUMMARY.json":
        continue
    data = json.loads(path.read_text())
    rows.append({
        "file": path.name,
        "completed": data.get("completed"),
        "failed": data.get("failed"),
        "output_tok_s": data.get("output_throughput"),
        "total_tok_s": data.get("total_token_throughput"),
        "mean_e2el_ms": data.get("mean_e2el_ms"),
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "mean_tpot_ms": data.get("mean_tpot_ms"),
        "concurrency": data.get("max_concurrency"),
    })
out = results / "SUMMARY.json"
out.write_text(json.dumps(rows, indent=2) + "\n")
print(f"{'run':<42} {'out tok/s':>10} {'e2e s':>8} {'done':>6}")
for row in rows:
    e2e = row["mean_e2el_ms"]
    e2e_s = f"{e2e/1000:.2f}" if isinstance(e2e, (int, float)) else "—"
    rate = row["output_tok_s"]
    rate_s = f"{rate:.1f}" if isinstance(rate, (int, float)) else "—"
    print(f"{row['file']:<42} {rate_s:>10} {e2e_s:>8} {row['completed']!s:>6}")
print(out)
PY
}

keys=()
case "${MODEL_CHOICE}" in
  both) keys=(gpt-oss llama) ;;
  *) keys=("${MODEL_CHOICE}") ;;
esac

if [[ "${SKIP_DOWNLOAD}" -eq 0 && "${BENCH_ONLY}" -eq 0 ]]; then
  for key in "${keys[@]}"; do
    download_one "${key}"
  done
fi

for key in "${keys[@]}"; do
  if [[ "${BENCH_ONLY}" -eq 0 ]]; then
    stop_others
    launch_model "${key}"
  elif ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "No healthy server on port ${PORT}." >&2
    exit 1
  fi
  bench_model "${key}"
  if [[ "${BENCH_ONLY}" -eq 0 ]]; then
    docker rm -f "${NAME}" >/dev/null 2>&1 || true
  fi
done

summarize
echo "results ${RESULTS}"
