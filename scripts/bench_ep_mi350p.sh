#!/usr/bin/env bash
# Equivalent of the NVIDIA forum expert-parallel burst on 2x MI350P.
#
# Forum protocol (do not add flags that change the model or the burst):
#   vllm serve Qwen/Qwen3.5-35B-A3B \
#     --tensor-parallel-size 2 \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 32768
#   and the same line plus --enable-expert-parallel
#
#   vllm bench serve \
#     --model Qwen/Qwen3.5-35B-A3B \
#     --dataset-name random \
#     --random-input-len 1000 \
#     --random-output-len 1000 \
#     --request-rate 10000 \
#     --num-prompts 16 \
#     --ignore-eos
#
# https://forums.developer.nvidia.com/t/expert-parallelism-using-6000-pro-pcie-gen5-vs-b200-nvlink/378258
#
# This host has ~8 GiB MemAvailable with the GPUs idle. Inductor compile of
# this MoE swapped at ~250 MB/s and stopped writing kernels. The default arm
# passes --enforce-eager. --graphs drops that flag and writes graphs_*.json.
# It refuses to start below 24 GiB MemAvailable.
#
# --skip-mm-profiling does not keep the vision tower out of the process: the
# 28 September log still recorded a 12 s multi-modal warmup. --language-model-only
# sets every modality limit to 0, and Qwen3_5MoeForConditionalGeneration then
# builds the tower as a StageMissingLayer. That is the default. Those runs
# write text_*.json so tp.json and ep.json stay the eager pair that loaded
# the tower. --with-vision repeats that older serve line and reuses those names.
#
# --topology records PCIe width and NUMA and does not stop the dense server.
# --run stops rocm-inference-server, downloads the 67 GiB checkpoint if needed,
# and serves both arms. Restore the dense control afterwards with
# ./scripts/launch_vllm_mxfp4.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/lib/serve.sh"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
NAME=rocm-ep-server
WEIGHTS="bf16"
KERNEL="triton"
MODEL_ID="Qwen/Qwen3.5-35B-A3B"
HOST_MODEL="${ROOT}/models/Qwen3.5-35B-A3B"
RESULTS="${ROOT}/_results/ep_mi350p"
PORT="${VLLM_PORT:-8000}"
SHARD_COUNT=14

DO_TOPOLOGY=0
DO_DOWNLOAD=0
DO_RUN=0
DO_GRAPHS=0
TEXT_ONLY=1
ARM="both"
# Inductor on this MoE swapped when MemAvailable was ~8 GiB. Graphs stay off
# until the host can hold the compile without that path.
GRAPHS_MIN_GIB=24

usage() {
  cat <<EOF
Usage: $(basename "$0") [--topology] [--download] [--run] [--arm tp|ep|both]
       [--graphs] [--with-vision]

  --topology    Write PCIe Gen5 width, NUMA, and PCI domain. Leaves servers alone.
  --download    Stop the dense server and download ${MODEL_ID} (~67 GiB).
  --run         Download if needed, then serve and bench the selected arms.
  --arm         tp (no EP), ep (--enable-expert-parallel), or both. Default both.
  --graphs      Drop --enforce-eager. Requires MemAvailable >= ${GRAPHS_MIN_GIB} GiB
                unless EP_ALLOW_LOW_RAM=1. Result names start with graphs_.
  --with-vision Omit --language-model-only and reuse tp.json / ep.json.
                The default keeps the vision tower out and writes text_*.json.
  --weights     bf16 (Qwen/Qwen3.5-35B-A3B) or mxfp4 (amd/Qwen3.5-35B-A3B-MXFP4).
  --kernel      triton, aiter, or both. Used with --weights mxfp4.
                triton is the OCP MX Triton expert path (--moe-backend emulation,
                VLLM_ROCM_USE_AITER=0). --moe-backend triton rejects this
                checkpoint's MXFP4 activations.
                aiter is the CK W4A4 kernel (--moe-backend aiter_mxfp4_mxfp4,
                VLLM_ROCM_USE_AITER=1, VLLM_ROCM_USE_AITER_MOE=1). Unified
                attention stays off, matching the Qwen3.8 gate.

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topology) DO_TOPOLOGY=1; shift ;;
    --download) DO_DOWNLOAD=1; shift ;;
    --run) DO_RUN=1; shift ;;
    --graphs) DO_GRAPHS=1; shift ;;
    --with-vision) TEXT_ONLY=0; shift ;;
    --arm)
      ARM="$2"
      shift 2
      ;;
    --weights)
      WEIGHTS="$2"
      shift 2
      ;;
    --kernel)
      KERNEL="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "${DO_TOPOLOGY}" -eq 0 && "${DO_DOWNLOAD}" -eq 0 && "${DO_RUN}" -eq 0 ]]; then
  DO_TOPOLOGY=1
fi

case "${ARM}" in
  tp|ep|both) ;;
  *) echo "--arm must be tp, ep, or both" >&2; exit 2 ;;
esac
case "${WEIGHTS}" in
  bf16|mxfp4) ;;
  *) echo "--weights must be bf16 or mxfp4" >&2; exit 2 ;;
esac
case "${KERNEL}" in
  triton|aiter|both) ;;
  *) echo "--kernel must be triton, aiter, or both" >&2; exit 2 ;;
esac
if [[ "${WEIGHTS}" == "mxfp4" ]]; then
  MODEL_ID="amd/Qwen3.5-35B-A3B-MXFP4"
  HOST_MODEL="${ROOT}/models/Qwen3.5-35B-A3B-MXFP4"
fi

mkdir -p "${RESULTS}"

record_topology() {
  python3 - "${RESULTS}/topology.json" <<'PY'
import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])

def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None

bdfs = []
lspci = subprocess.run(["lspci", "-D", "-d", "1002:75a8"], capture_output=True, text=True)
for line in lspci.stdout.splitlines():
    bdf = line.split()[0]
    bdfs.append({
        "bdf": bdf,
        "current_link_speed": read(f"/sys/bus/pci/devices/{bdf}/current_link_speed"),
        "current_link_width": read(f"/sys/bus/pci/devices/{bdf}/current_link_width"),
        "max_link_speed": read(f"/sys/bus/pci/devices/{bdf}/max_link_speed"),
        "max_link_width": read(f"/sys/bus/pci/devices/{bdf}/max_link_width"),
        "numa_node": read(f"/sys/bus/pci/devices/{bdf}/numa_node"),
    })

lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
cpu = {}
for line in lscpu.splitlines():
    if ":" not in line:
        continue
    key, val = line.split(":", 1)
    key = key.strip()
    if key in {"Model name", "Socket(s)", "Core(s) per socket", "NUMA node(s)", "NUMA node0 CPU(s)", "NUMA node1 CPU(s)"}:
        cpu[key] = val.strip()

# PCIe 5.0 32 GT/s, 128b/130b: 32e9 * (128/130) / 8 bytes per lane.
per_lane_gbs = 32e9 * (128 / 130) / 8 / 1e9
payload = {
    "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "cpu": cpu,
    "gpus": bdfs,
    "pcie_gen5_unidirectional_GBps": {
        "x16": round(per_lane_gbs * 16, 2),
        "x8": round(per_lane_gbs * 8, 2),
    },
    "note": "Payload rate is the PCIe wire ceiling. Cross-socket copies can land below it.",
}
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
for gpu in bdfs:
    print(f"{gpu['bdf']}  {gpu['current_link_speed']} x{gpu['current_link_width']}  NUMA {gpu['numa_node']}")
PY
}

checkpoint_ready() {
  [[ -f "${HOST_MODEL}/config.json" ]] || return 1
  local n
  n="$(find "${HOST_MODEL}" -name 'model.safetensors-*.safetensors' | wc -l)"
  [[ "${n}" -ge "${SHARD_COUNT}" ]]
}

stop_dense() {
  docker stop rocm-inference-server >/dev/null 2>&1 || true
}

download_model() {
  if docker ps --format '{{.Names}}' | grep -qx qwen35-mxfp4-download; then
    echo "waiting for qwen35-mxfp4-download"
    docker wait qwen35-mxfp4-download >/dev/null
  fi
  if checkpoint_ready; then
    echo "checkpoint present: ${HOST_MODEL}"
    return 0
  fi
  stop_dense
  local avail
  avail="$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
  echo "MemAvailable ${avail} GiB"
  if [[ "${avail}" -lt 6 && "${EP_ALLOW_LOW_RAM:-0}" != "1" ]]; then
    echo "Refusing download below 6 GiB MemAvailable. Set EP_ALLOW_LOW_RAM=1 to override." >&2
    exit 2
  fi
  mkdir -p "${HOST_MODEL}"
  docker rm -f qwen35-ep-download >/dev/null 2>&1 || true
  docker run --name qwen35-ep-download --network host \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_HUB_DISABLE_XET=1 \
    -e PYTHONUNBUFFERED=1 \
    -v /home/amd/.cache/huggingface:/root/.cache/huggingface \
    -v "${ROOT}/models:/models" \
    --entrypoint hf \
    "${IMAGE}" \
    download "${MODEL_ID}" \
    --local-dir "/models/$(basename "${HOST_MODEL}")" \
    --max-workers 2
  checkpoint_ready || { echo "download finished short of ${SHARD_COUNT} shards" >&2; exit 2; }
}

wait_healthy() {
  local i
  for i in $(seq 1 540); do
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
  return 1
}

result_tag() {
  local arm="$1"
  local tag="${arm}"
  if [[ "${TEXT_ONLY}" -eq 1 ]]; then
    tag="text_${tag}"
  fi
  if [[ "${DO_GRAPHS}" -eq 1 ]]; then
    tag="graphs_${tag}"
  fi
  if [[ "${WEIGHTS}" == "mxfp4" ]]; then
    tag="mxfp4_${KERNEL}_${tag}"
  fi
  echo "${tag}"
}

mem_available_gib() {
  awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo
}

launch_arm() {
  local arm="$1"
  local tag
  tag="$(result_tag "${arm}")"
  local -a kernel_env=()
  local -a serve_args=(
    "/models/$(basename "${HOST_MODEL}")"
    --served-model-name "${MODEL_ID}"
    --tensor-parallel-size 2
    --gpu-memory-utilization 0.9
    --max-model-len 32768
    --skip-mm-profiling
    --host 127.0.0.1
    --port "${PORT}"
  )
  if [[ "${DO_GRAPHS}" -eq 0 ]]; then
    serve_args+=(--enforce-eager)
  fi
  if [[ "${TEXT_ONLY}" -eq 1 ]]; then
    # Modality limits of 0 make _mark_tower_model install StageMissingLayer
    # in place of Qwen3_VisionTransformer. --skip-mm-profiling does not.
    serve_args+=(--language-model-only)
  fi
  if [[ "${arm}" == "ep" ]]; then
    serve_args+=(--enable-expert-parallel)
  fi
  if [[ "${WEIGHTS}" == "mxfp4" && "${KERNEL}" == "triton" ]]; then
    # OCP MX Triton experts. moe-backend=triton is the GPT-OSS BF16-activation
    # kernel and rejects this checkpoint's dynamic MXFP4 activations.
    kernel_env+=(
      -e VLLM_ROCM_USE_AITER=0
      -e VLLM_ROCM_USE_AITER_MOE=0
      -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
    )
    serve_args+=(--moe-backend emulation)
  elif [[ "${WEIGHTS}" == "mxfp4" && "${KERNEL}" == "aiter" ]]; then
    # AITER CK W4A4. Unified attention stays off, same as the Qwen3.8 gate.
    kernel_env+=(
      -e VLLM_ROCM_USE_AITER=1
      -e VLLM_ROCM_USE_AITER_MOE=1
      -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
    )
    serve_args+=(--moe-backend aiter_mxfp4_mxfp4)
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
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_P2P_DISABLE=0 \
    -e NCCL_DEBUG=INFO \
    -e NCCL_DEBUG_SUBSYS=INIT \
    "${kernel_env[@]}" \
    -v /home/amd/.cache/huggingface:/root/.cache/huggingface \
    -v "${ROOT}/models:/models" \
    -v "${ROOT}/_results:/results" \
    "${IMAGE}" \
    "${serve_args[@]}"
  docker logs -f "${NAME}" > "${RESULTS}/${tag}.server.log" 2>&1 &
  echo $! > "${RESULTS}/${tag}.logpid"
  wait_healthy
  # RCCL init lines say whether the two cards used P2P or a host path.
  python3 - "${RESULTS}/${tag}.server.log" "${RESULTS}/${tag}.rccl.txt" "${RESULTS}/${tag}.kernels.txt" <<'PY'
import sys
from pathlib import Path
src, dst, kdst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
text = src.read_text(errors="replace").splitlines() if src.is_file() else []
rccl = [line for line in text if any(k in line for k in ("P2P", "SHM", "NET", "via", "Channel", "Bootstrap")) and ("NCCL" in line or "RCCL" in line)]
kern = [line for line in text if any(k in line for k in ("MoE backend", "GDN", "Attention", "AITER", "Triton", "moe_backend", "Mxfp4"))]
dst.write_text("\n".join(rccl) + ("\n" if rccl else ""))
kdst.write_text("\n".join(kern) + ("\n" if kern else ""))
print(f"{dst} ({len(rccl)} lines)")
print(f"{kdst} ({len(kern)} lines)")
for line in text:
    if "Multi-modal warmup completed in " in line:
        print(line)
PY
}

start_gpu_sampler() {
  local tag="$1"
  python3 "${ROOT}/scripts/telemetry.py" begin \
    --output "${RESULTS}/${tag}.power.json" \
    --all-gpus \
    --interval 2 \
    --profile "${GPU_PROFILE:-mi350p}" \
    --gpus-json "${RESULTS}/${tag}.gpus.json"
}

stop_gpu_sampler() {
  local tag="$1"
  python3 "${ROOT}/scripts/telemetry.py" end --output "${RESULTS}/${tag}.power.json"
}

_bench_arm_exec() {
  local arm="$1"
  local tag
  tag="$(result_tag "${arm}")"
  docker exec -w /results/ep_mi350p "${NAME}" \
    vllm bench serve \
      --backend openai \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --endpoint /v1/completions \
      --model "${MODEL_ID}" \
      --tokenizer "/models/$(basename "${HOST_MODEL}")" \
      --dataset-name random \
      --random-input-len 1000 \
      --random-output-len 1000 \
      --request-rate 10000 \
      --num-prompts 16 \
      --ignore-eos \
      --save-result \
      --result-dir /results/ep_mi350p \
      --result-filename "${tag}.json" \
      | tee "${RESULTS}/${tag}.bench.txt"
}

bench_arm() {
  local arm="$1"
  local tag rc
  tag="$(result_tag "${arm}")"
  start_gpu_sampler "${tag}"
  rc=0
  _bench_arm_exec "${arm}" || rc=$?
  stop_gpu_sampler "${tag}"
  return "${rc}"
}

run_arm() {
  local arm="$1"
  local tag
  tag="$(result_tag "${arm}")"
  echo "===== ${tag} ====="
  launch_arm "${arm}"
  bench_arm "${arm}"
  if [[ -f "${RESULTS}/${tag}.logpid" ]]; then
    kill "$(cat "${RESULTS}/${tag}.logpid")" >/dev/null 2>&1 || true
  fi
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
}

run_arm_kernels() {
  local arm="$1"
  if [[ "${WEIGHTS}" == "mxfp4" && "${KERNEL}" == "both" ]]; then
    local saved="${KERNEL}"
    KERNEL=triton
    run_arm "${arm}"
    KERNEL=aiter
    run_arm "${arm}"
    KERNEL="${saved}"
  else
    run_arm "${arm}"
  fi
}

if [[ "${DO_TOPOLOGY}" -eq 1 || "${DO_RUN}" -eq 1 ]]; then
  record_topology
fi

if [[ "${DO_DOWNLOAD}" -eq 1 || "${DO_RUN}" -eq 1 ]]; then
  download_model
fi

if [[ "${DO_RUN}" -eq 1 && "${DO_GRAPHS}" -eq 1 ]]; then
  avail="$(mem_available_gib)"
  echo "MemAvailable ${avail} GiB"
  if [[ "${avail}" -lt "${GRAPHS_MIN_GIB}" && "${EP_ALLOW_LOW_RAM:-0}" != "1" ]]; then
    echo "Refusing --graphs below ${GRAPHS_MIN_GIB} GiB MemAvailable. Inductor compile of this MoE swapped at ~250 MB/s when about 8 GiB was free. Eager stays the baseline. Set EP_ALLOW_LOW_RAM=1 to override." >&2
    exit 2
  fi
fi

if [[ "${DO_RUN}" -eq 1 ]]; then
  stop_dense
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  if [[ "${ARM}" == "both" || "${ARM}" == "tp" ]]; then
    run_arm_kernels tp
  fi
  if [[ "${ARM}" == "both" || "${ARM}" == "ep" ]]; then
    run_arm_kernels ep
  fi
  python3 "${ROOT}/scripts/ep_compare.py" \
    --results-dir "${RESULTS}" \
    --out "${RESULTS}/COMPARE.md"
  echo "scorecard ${RESULTS}/COMPARE.md"
  echo "dense control is stopped. restore with ./scripts/launch_vllm_mxfp4.sh"
else
  python3 "${ROOT}/scripts/ep_compare.py" \
    --results-dir "${RESULTS}" \
    --out "${RESULTS}/COMPARE.md"
fi
