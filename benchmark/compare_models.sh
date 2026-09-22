#!/usr/bin/env bash
# ==============================================================================
# Multi-Model Comparative Benchmark Suite for AMD Radeon AI PRO R9700
# Supports SWE-bench (Software Engineering) and GPQA (Scientific Reasoning)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/benchmark_results"

BENCHMARK_TYPE="${1:-all}"  # "swebench", "gpqa", or "all"
DATASET="${2:-sample}"
NUM_SAMPLES="${3:-3}"

MODELS=(
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "Qwen/Qwen2.5-Coder-14B-Instruct"
  "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
)

mkdir -p "${RESULTS_DIR}"

echo "=========================================================================="
echo " Starting Multi-Model Comparative Benchmark on AMD Radeon AI PRO R9700"
echo " Benchmark   : ${BENCHMARK_TYPE}"
echo " Dataset     : ${DATASET}"
echo " Samples/Run : ${NUM_SAMPLES}"
echo " Models      : ${#MODELS[@]}"
echo "=========================================================================="

wait_for_server() {
  local port="${1:-8000}"
  local max_wait=300
  local elapsed=0
  echo -n "Waiting for inference engine to become healthy on port ${port}..."
  while ! curl -s -f "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
    sleep 5
    elapsed=$((elapsed + 5))
    echo -n "."
    if [ "${elapsed}" -ge "${max_wait}" ]; then
      echo " TIMEOUT"
      echo "Error: Server failed to become healthy within ${max_wait} seconds."
      docker compose logs --tail 30 inference
      return 1
    fi
  done
  echo " READY (${elapsed}s)"
}

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "--------------------------------------------------------------------------"
  echo " Deploying Model: ${MODEL}"
  echo "--------------------------------------------------------------------------"

  # Determine tool call parser based on model family
  TOOL_PARSER="hermes"
  if [[ "${MODEL}" =~ "deepseek" ]]; then
    TOOL_PARSER="deepseek"
  elif [[ "${MODEL}" =~ "mistral" ]] || [[ "${MODEL}" =~ "Codestral" ]]; then
    TOOL_PARSER="mistral"
  fi

  # Start/recreate inference container with target model
  MODEL_NAME="${MODEL}" TOOL_PARSER="${TOOL_PARSER}" docker compose -f "${ROOT_DIR}/docker-compose.yml" up -d --force-recreate inference

  # Wait for server to load weights and report healthy
  wait_for_server 8000

  # 1. Run SWE-bench if requested
  if [ "${BENCHMARK_TYPE}" = "swebench" ] || [ "${BENCHMARK_TYPE}" = "all" ]; then
    echo "Running SWE-bench benchmark against ${MODEL}..."
    python3 "${SCRIPT_DIR}/run_benchmark.py" \
      --base-url "http://127.0.0.1:8000/v1" \
      --model "${MODEL}" \
      --dataset "${DATASET}" \
      --num-samples "${NUM_SAMPLES}" \
      --output-dir "${RESULTS_DIR}/swebench"
  fi

  # 2. Run GPQA if requested
  if [ "${BENCHMARK_TYPE}" = "gpqa" ] || [ "${BENCHMARK_TYPE}" = "all" ]; then
    echo "Running GPQA reasoning benchmark against ${MODEL}..."
    python3 "${SCRIPT_DIR}/run_gpqa.py" \
      --base-url "http://127.0.0.1:8000/v1" \
      --model "${MODEL}" \
      --subset "${DATASET}" \
      --num-samples "${NUM_SAMPLES}" \
      --output-dir "${RESULTS_DIR}/gpqa"
  fi

  # Record VRAM consumption
  echo "Current VRAM Usage on Radeon AI PRO R9700:"
  rocm-smi --showmeminfo vram || true
done

echo ""
echo "=========================================================================="
echo " Comparative Benchmark Complete!"
echo " Reports and predictions saved under: ${RESULTS_DIR}"
echo "=========================================================================="
