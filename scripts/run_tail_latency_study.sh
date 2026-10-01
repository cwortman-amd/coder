#!/usr/bin/env bash
#
# Master Orchestration Script for LLM Tail Latency & Agent Compounding Study.
# Replicates the empirical protocol from the DigitalOcean study on local/remote inference endpoints.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-Qwen3.8-27B-Quark-AWQ-MXFP4}"
CONCURRENCY="${CONCURRENCY:-1 5 20}"
REQUESTS_PER_CELL="${REQUESTS_PER_CELL:-75}"
NUM_CHAINS="${NUM_CHAINS:-30}"
CHAIN_LENGTH="${CHAIN_LENGTH:-10}"
TOKENS_PER_CALL="${TOKENS_PER_CALL:-64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/_results/tail_latency_study}"

echo "======================================================="
echo "  LLM Tail Latency & Agent Compounding Study Orchestrator"
echo "  Target URL:          ${BASE_URL}"
echo "  Model:               ${MODEL}"
echo "  Concurrency Sweep:   ${CONCURRENCY}"
echo "  Single-Call Reqs:    ${REQUESTS_PER_CELL} reqs / concurrency cell"
echo "  Chained Benchmark:   ${NUM_CHAINS} chains of ${CHAIN_LENGTH} calls"
echo "  Tokens Per Call:     ${TOKENS_PER_CALL}"
echo "======================================================="

# 1. Health Check
echo "[*] Checking endpoint health..."
health_endpoint="${BASE_URL%/}"
health_endpoint="${health_endpoint%/v1}"
if ! curl -sf -m 5 "${health_endpoint}/health" >/dev/null && ! curl -sf -m 5 "${BASE_URL}/models" >/dev/null; then
  echo "[-] ERROR: Endpoint at ${BASE_URL} is not responding." >&2
  exit 1
fi
echo "[+] Endpoint is healthy and responsive."

# 2. Prepare timestamped results directory
STAMP="$(date -u +"%Y%m%d_%H%M%S")"
TARGET_DIR="${OUTPUT_ROOT}/${STAMP}"
mkdir -p "${TARGET_DIR}"

# 3. Execute Agent Chain & Latency Benchmark
echo ""
echo "[*] Launching Agent Chain & Concurrency Benchmark..."
python3 "${ROOT}/scripts/bench_agent_chain.py" \
  --url "${BASE_URL}" \
  --model "${MODEL}" \
  --concurrency-list ${CONCURRENCY} \
  --requests-per-cell "${REQUESTS_PER_CELL}" \
  --num-chains "${NUM_CHAINS}" \
  --chain-length "${CHAIN_LENGTH}" \
  --tokens-per-call "${TOKENS_PER_CALL}" \
  --output-dir "${OUTPUT_ROOT}"

# Find the newly created manifest
MANIFEST="${TARGET_DIR}/agent_chain_manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  MANIFEST="$(find "${OUTPUT_ROOT}" -name "agent_chain_manifest.json" | sort | tail -n 1)"
  TARGET_DIR="$(dirname "${MANIFEST}")"
fi

# 4. Generate Diagnostic Analysis
echo ""
echo "[*] Running Diagnostic Tail Latency Analyzer..."
python3 "${ROOT}/scripts/analyze_tail_metrics.py" --input "${MANIFEST}"

# 5. Generate Publication Plots
echo ""
echo "[*] Generating 4-Panel Visualization Dashboard..."
FIGURE_PATH="${TARGET_DIR}/agent_chain_dashboard.png"
if [[ -f "${ROOT}/.venv/bin/python3" ]]; then
  "${ROOT}/.venv/bin/python3" "${ROOT}/scripts/plot_tail_distributions.py" \
    --input "${MANIFEST}" \
    --out "${FIGURE_PATH}"
else
  python3 "${ROOT}/scripts/plot_tail_distributions.py" \
    --input "${MANIFEST}" \
    --out "${FIGURE_PATH}"
fi

echo ""
echo "======================================================="
echo "  BENCHMARK CAMPAIGN COMPLETED SUCCESSFULLY"
echo "  • Manifest JSON:  ${MANIFEST}"
echo "  • Markdown Report: ${TARGET_DIR}/agent_chain_report.md"
echo "  • Figure Dashboard: ${FIGURE_PATH}"
echo "======================================================="
