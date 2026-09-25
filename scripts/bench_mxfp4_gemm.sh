#!/usr/bin/env bash
# Isolated AITER MXFP4 GEMM (same process as rocprofv3 — no EngineCore).
# Stops the 27B server so the microbench owns GPU 0, then restores HF control.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
NAME="${VLLM_CONTAINER_NAME:-rocm-inference-server}"
BENCH_NAME="${VLLM_GEMM_CONTAINER:-rocm-mxfp4-gemm-bench}"
OUT="${ROOT}/_results/profiling/mxfp4_gemm"
RESTORE=1
PROFILE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --no-restore) RESTORE=0; shift ;;
    --no-profile) PROFILE=0; shift ;;
    *) echo "unknown $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${OUT}/rocprof"
docker stop "${NAME}" "${BENCH_NAME}" >/dev/null 2>&1 || true
docker rm "${BENCH_NAME}" >/dev/null 2>&1 || true

COMMON=(
  --rm --name "${BENCH_NAME}" --network host --ipc host --shm-size 16g
  --device /dev/kfd --device /dev/dri
  --group-add 44 --group-add 993
  --cap-add SYS_PTRACE
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined --security-opt label=disable
  -e HIP_VISIBLE_DEVICES=0
  -e PYTORCH_ROCM_ARCH=gfx950
  -e GPU_ARCHS=gfx950
  -e HIP_FORCE_DEV_KERNARG=1
  -e PYTHONUNBUFFERED=1
  -e SAFETENSORS_FAST_GPU=1
  -v "${ROOT}/models:/models"
  -v "${ROOT}/scripts:/workspace/scripts"
  -v "${ROOT}/_results:/results"
)

echo "unprofiled GEMM sweep (image default; VLLM_ROCM_USE_AITER unset)"
docker run "${COMMON[@]}" --entrypoint python3 "${IMAGE}" \
  /workspace/scripts/bench_mxfp4_gemm.py \
  --out /results/profiling/mxfp4_gemm/summary.json \
  --ms 1,2,4,8,16,32,64 \
  --warmup 50 --iters 200 | tee "${OUT}/bench.log"

echo "unprofiled GEMM sweep with VLLM_ROCM_USE_AITER=1 (ASM path if gfx950)"
docker run "${COMMON[@]}" -e VLLM_ROCM_USE_AITER=1 --entrypoint python3 "${IMAGE}" \
  /workspace/scripts/bench_mxfp4_gemm.py \
  --out /results/profiling/mxfp4_gemm/summary_aiter1.json \
  --layers mlp.gate_proj,mlp.down_proj \
  --ms 1,8,64 \
  --warmup 50 --iters 200 | tee "${OUT}/bench_aiter1.log"

if [[ "${PROFILE}" -eq 1 ]]; then
  echo "rocprofv3 M=1 gate_proj (same-process kernels)"
  docker run "${COMMON[@]}" \
    --entrypoint /opt/rocm/bin/rocprofv3 "${IMAGE}" \
    --kernel-trace --hip-runtime-trace --memory-copy-trace \
    --stats --summary \
    --output-format csv pftrace \
    --output-directory /results/profiling/mxfp4_gemm/rocprof \
    --output-file gemm_m1 \
    -- \
    python3 /workspace/scripts/bench_mxfp4_gemm.py \
      --out /results/profiling/mxfp4_gemm/summary_rocprof_m1.json \
      --layers mlp.gate_proj --ms 1 --warmup 10 --iters 40 \
    | tee "${OUT}/rocprof_bench.log" || true
  python3 "${ROOT}/scripts/aggregate_kernel_csv.py" \
    --input-dir "${OUT}/rocprof" \
    --out "${OUT}/kernel_top.json" || true
fi

if [[ "${RESTORE}" -eq 1 ]]; then
  echo "restoring HF control on 8000"
  "${ROOT}/scripts/launch_vllm_mxfp4.sh"
fi
