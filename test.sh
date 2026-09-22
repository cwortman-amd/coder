#!/bin/bash
# ==============================================================================
# test.sh - Automated Dual Benchmark Test Runner & Results Summary
# ==============================================================================
# Executes:
#   1. SWE-bench evaluation:  docker compose run --rm benchmark
#   2. GPQA reasoning test:   python3 benchmark/run_gpqa.py --dataset sample
# Generates a unified benchmark summary table and markdown report.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ANSI Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${SCRIPT_DIR}/benchmark_results"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "${BLUE}${BOLD}     AMD ROCm AI Model Benchmarking Suite (SWE-bench & GPQA)          ${NC}"
echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "Hardware Target  : ${BOLD}AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB VRAM)${NC}"
echo -e "Execution Time   : $(date)"
echo ""

# ------------------------------------------------------------------------------
# Pre-flight Check: Ensure Inference Server is Healthy
# ------------------------------------------------------------------------------
echo -n "Checking local ROCm inference server health... "
HEALTH_URL="http://127.0.0.1:8000/health"
if ! curl -sf --connect-timeout 3 "$HEALTH_URL" >/dev/null 2>&1; then
    echo -e "${RED}${BOLD}[SERVER NOT HEALTHY]${NC}"
    echo -e "${YELLOW}The inference server at http://127.0.0.1:8000 is not responding.${NC}"
    echo -e "${YELLOW}Please start it first using: ${BOLD}./setup.sh${NC}"
    echo -e "${YELLOW}Or wait for it to become ready with: ${BOLD}./check.sh --wait 60${NC}"
    exit 1
fi
echo -e "${GREEN}${BOLD}[HEALTHY]${NC}"

# Detect active model
ACTIVE_MODEL=$(curl -s http://127.0.0.1:8000/v1/models 2>/dev/null | jq -r '.data[0].id // "Unknown"')
echo -e "Active Model     : ${GREEN}${BOLD}${ACTIVE_MODEL}${NC}"
echo ""

# Record start time
TESTS_START=$(date +%s)
SWE_STATUS="SKIPPED"
GPQA_STATUS="SKIPPED"

# ------------------------------------------------------------------------------
# Test Case 1: SWE-bench Benchmark (Docker Container)
# ------------------------------------------------------------------------------
echo -e "${CYAN}${BOLD}>>> [1/2] Running SWE-bench Evaluation (docker compose run --rm benchmark)...${NC}"
if docker compose run --rm benchmark; then
    SWE_STATUS="PASSED"
    echo -e "${GREEN}${BOLD}✓ SWE-bench test run completed successfully.${NC}"
else
    SWE_STATUS="FAILED"
    echo -e "${RED}${BOLD}✗ SWE-bench test run failed.${NC}"
fi
echo ""

# Ensure benchmark_results directory is accessible by host user
chmod -R ugo+rwX "$RESULTS_DIR" 2>/dev/null || true

# ------------------------------------------------------------------------------
# Test Case 2: GPQA Scientific Reasoning Benchmark (Host Python)
# ------------------------------------------------------------------------------
echo -e "${CYAN}${BOLD}>>> [2/2] Running GPQA Reasoning Benchmark (python3 benchmark/run_gpqa.py --dataset sample)...${NC}"
if python3 benchmark/run_gpqa.py --dataset sample; then
    GPQA_STATUS="PASSED"
    echo -e "${GREEN}${BOLD}✓ GPQA test run completed successfully.${NC}"
else
    GPQA_STATUS="FAILED"
    echo -e "${RED}${BOLD}✗ GPQA test run failed.${NC}"
fi
echo ""

TESTS_END=$(date +%s)
TOTAL_TIME=$(( TESTS_END - TESTS_START ))

# ------------------------------------------------------------------------------
# Locate Latest Metrics Files
# ------------------------------------------------------------------------------
LATEST_SWE_FILE=$(find "${RESULTS_DIR}" -maxdepth 2 -name "benchmark_metrics.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}')
LATEST_GPQA_FILE=$(find "${RESULTS_DIR}/gpqa" -maxdepth 2 -name "gpqa_summary.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}')

# ------------------------------------------------------------------------------
# Parse Metrics
# ------------------------------------------------------------------------------
SWE_TOTAL="N/A"
SWE_VALID="N/A"
SWE_RATE="N/A"
SWE_THROUGHPUT="N/A"
SWE_LATENCY="N/A"

if [ -n "$LATEST_SWE_FILE" ] && [ -f "$LATEST_SWE_FILE" ]; then
    SWE_TOTAL=$(jq -r '.total_instances // "?"' "$LATEST_SWE_FILE")
    SWE_VALID=$(jq -r '.valid_patches_generated // "?"' "$LATEST_SWE_FILE")
    SWE_RATE=$(jq -r '.patch_formatting_rate_pct // "?"' "$LATEST_SWE_FILE")
    SWE_THROUGHPUT=$(jq -r '.avg_tokens_per_sec // "?"' "$LATEST_SWE_FILE")
    SWE_LATENCY=$(jq -r '.avg_latency_per_sample_sec // "?"' "$LATEST_SWE_FILE")
fi

GPQA_TOTAL="N/A"
GPQA_CORRECT="N/A"
GPQA_ACCURACY="N/A"
GPQA_THROUGHPUT="N/A"
GPQA_LATENCY="N/A"
GPQA_PHYSICS="N/A"
GPQA_CHEMISTRY="N/A"
GPQA_BIOLOGY="N/A"

if [ -n "$LATEST_GPQA_FILE" ] && [ -f "$LATEST_GPQA_FILE" ]; then
    GPQA_TOTAL=$(jq -r '.total_questions // "?"' "$LATEST_GPQA_FILE")
    GPQA_CORRECT=$(jq -r '.correct_answers // "?"' "$LATEST_GPQA_FILE")
    GPQA_ACCURACY=$(jq -r '.accuracy_pct // "?"' "$LATEST_GPQA_FILE")
    GPQA_THROUGHPUT=$(jq -r '.average_throughput_tok_per_sec // "?"' "$LATEST_GPQA_FILE")
    GPQA_LATENCY=$(jq -r '.average_latency_sec // "?"' "$LATEST_GPQA_FILE")
    GPQA_PHYSICS=$(jq -r '.domain_breakdown.Physics.accuracy_pct // "N/A"' "$LATEST_GPQA_FILE")
    GPQA_CHEMISTRY=$(jq -r '.domain_breakdown.Chemistry.accuracy_pct // "N/A"' "$LATEST_GPQA_FILE")
    GPQA_BIOLOGY=$(jq -r '.domain_breakdown.Biology.accuracy_pct // "N/A"' "$LATEST_GPQA_FILE")
fi

# ------------------------------------------------------------------------------
# Terminal Summary Table
# ------------------------------------------------------------------------------
echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "${BLUE}${BOLD}                   UNIFIED BENCHMARK RESULTS SUMMARY                  ${NC}"
echo -e "${BLUE}${BOLD}======================================================================${NC}"
printf "%-14s | %-16s | %-15s | %-12s | %-12s\n" "Benchmark" "Task Type" "Score / Metric" "Throughput" "Latency / Item"
echo "----------------------------------------------------------------------"
printf "%-14s | %-16s | %-15s | %-12s | %-12s\n" \
  "SWE-bench" "Code Patching" "${SWE_RATE}% (${SWE_VALID}/${SWE_TOTAL})" "${SWE_THROUGHPUT} tok/s" "${SWE_LATENCY}s"
printf "%-14s | %-16s | %-15s | %-12s | %-12s\n" \
  "GPQA" "Sci. Reasoning" "${GPQA_ACCURACY}% (${GPQA_CORRECT}/${GPQA_TOTAL})" "${GPQA_THROUGHPUT} tok/s" "${GPQA_LATENCY}s"
echo "----------------------------------------------------------------------"
echo -e "${BOLD}GPQA Scientific Domain Breakdown:${NC}"
echo -e "  • Physics   : ${BOLD}${GPQA_PHYSICS}%${NC}"
echo -e "  • Chemistry : ${BOLD}${GPQA_CHEMISTRY}%${NC}"
echo -e "  • Biology   : ${BOLD}${GPQA_BIOLOGY}%${NC}"
echo "----------------------------------------------------------------------"
echo -e "Hardware Platform   : AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB VRAM)"
echo -e "Target Model Tested : ${BOLD}${ACTIVE_MODEL}${NC}"
echo -e "Total Test Duration : ${BOLD}${TOTAL_TIME} seconds${NC}"
echo -e "SWE-bench Status    : $([ "$SWE_STATUS" = "PASSED" ] && echo -e "${GREEN}${BOLD}PASSED${NC}" || echo -e "${RED}${BOLD}${SWE_STATUS}${NC}")"
echo -e "GPQA Status         : $([ "$GPQA_STATUS" = "PASSED" ] && echo -e "${GREEN}${BOLD}PASSED${NC}" || echo -e "${RED}${BOLD}${GPQA_STATUS}${NC}")"

# ------------------------------------------------------------------------------
# Generate Markdown Report File
# ------------------------------------------------------------------------------
SUMMARY_REPORT="${RESULTS_DIR}/test_run_summary_${TIMESTAMP}.md"
cat << EOF > "$SUMMARY_REPORT"
# Benchmark Evaluation Summary Report

- **Target Hardware**: AMD Radeon™ AI PRO R9700 (\`gfx1201\`, 32 GB GDDR6 VRAM)
- **Model Evaluated**: \`${ACTIVE_MODEL}\`
- **Execution Date**: $(date)
- **Total Runtime**: ${TOTAL_TIME} seconds

## Performance & Accuracy Summary Table

| Benchmark | Target Capability | Dataset | Score / Primary Metric | Throughput | Latency | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SWE-bench** | Software Engineering / Git Diff Patches | \`sample\` (3 instances) | **${SWE_RATE}%** (${SWE_VALID}/${SWE_TOTAL} valid patches) | **${SWE_THROUGHPUT} tok/s** | **${SWE_LATENCY} s/sample** | \`${SWE_STATUS}\` |
| **GPQA** | Scientific & Technical Multiple-Choice Reasoning | \`sample\` (3 questions) | **${GPQA_ACCURACY}%** (${GPQA_CORRECT}/${GPQA_TOTAL} correct) | **${GPQA_THROUGHPUT} tok/s** | **${GPQA_LATENCY} s/item** | \`${GPQA_STATUS}\` |

## GPQA Domain Accuracy Breakdown

- **Physics**: **${GPQA_PHYSICS}%**
- **Chemistry**: **${GPQA_CHEMISTRY}%**
- **Biology**: **${GPQA_BIOLOGY}%**

## Output Artifacts & Log Paths

- **SWE-bench Metrics**: \`${LATEST_SWE_FILE}\`
- **GPQA Metrics**: \`${LATEST_GPQA_FILE}\`
EOF

echo ""
echo -e "${GREEN}${BOLD}✓ Markdown Report Generated:${NC} ${SUMMARY_REPORT}"
echo -e "${BLUE}======================================================================${NC}"

if [ "$SWE_STATUS" = "PASSED" ] && [ "$GPQA_STATUS" = "PASSED" ]; then
    exit 0
else
    exit 1
fi
