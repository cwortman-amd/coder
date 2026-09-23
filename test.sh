#!/bin/bash
# ==============================================================================
# test.sh - Automated Dual Benchmark Test Runner & Results Summary
# ==============================================================================
# Executes:
#   1. SWE-bench evaluation:  docker compose run --rm --no-deps benchmark ...
#   2. GPQA reasoning test:   python3 benchmark/run_gpqa.py ...
# Generates a unified benchmark summary table and markdown report.
#
# Supported Tiers:
#   -s, --sample (Default) : Offline sample smoke test (3 SWE-bench, 3 GPQA)
#   -d, --diamond          : Diamond / Lite tier (SWE-bench Lite & GPQA Diamond)
#   -m, --main             : Main / Verified tier (SWE-bench Verified & GPQA Main)
#   -a, --all              : All tests (Full SWE-bench & Full GPQA Extended)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Load user environment (~/.env) if present to pull in HF_TOKEN
if [ -f "$HOME/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$HOME/.env"
    set +a
fi

# 2. Load local environment configuration (.env)
if [ -f "${SCRIPT_DIR}/.env" ]; then
    PREV_HF_TOKEN="${HF_TOKEN:-}"
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^[[:space:]]*#' "${SCRIPT_DIR}/.env" | grep -v '^[[:space:]]*$')
    set +a
    if [ -z "${HF_TOKEN:-}" ] && [ -n "${PREV_HF_TOKEN}" ]; then
        export HF_TOKEN="${PREV_HF_TOKEN}"
    fi
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="${HF_HOME:-${HF_CACHE_DIR:-$HOME/.cache/huggingface}}"

# ANSI Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Default Benchmark Configuration
PROFILE="sample"
SWE_DATASET="sample"
GPQA_SUBSET="sample"
NUM_SAMPLES=""
RUN_SWE=true
RUN_GPQA=true
RUN_EVAL=false
BENCHMARK_ENGINE="${INFERENCE_ENGINE:-vllm}"
COMPARE_ENGINES=false

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

AMD ROCm AI Model Benchmarking Suite (SWE-bench & GPQA)

Benchmark Tiers:
  -s, --sample             Run offline smoke test with sample subsets (Default)
                           • SWE-bench: 3 offline sample problems
                           • GPQA: 3 offline sample questions
  -d, --diamond            Run Diamond / Lite evaluation tier
                           • SWE-bench: princeton-nlp/SWE-bench_Lite (300 problems)
                           • GPQA: diamond (198 questions)
                           (Alias: -l, --lite)
  -m, --main               Run Main / Verified evaluation tier
                           • SWE-bench: princeton-nlp/SWE-bench_Verified (500 problems)
                           • GPQA: main (448 questions)
  -a, --all                Run all available benchmark tests
                           • SWE-bench: princeton-nlp/SWE-bench (all 2,294 problems)
                           • GPQA: extended (all 546 questions)

Inference Engine & Comparative Options:
  -e, --engine <engine>    Inference engine: 'vllm' (Default), 'llama.cpp', or 'sglang'
  --compare-engines        Run comparative benchmark between vLLM, llama.cpp, and SGLang

Execution Options:
  -n, --limit <N>          Limit execution to N instances per benchmark (e.g. -n 5)
  --num-samples <N>        Alias for -n / --limit
  --swe-only               Run only SWE-bench evaluation
  --gpqa-only              Run only GPQA scientific reasoning benchmark
  --eval, --run-eval       Execute SWE-bench Docker evaluation harness after patch generation
  -h, --help               Show this help message and exit

Examples:
  ./test.sh                 # Quick 3-question smoke test (default)
  ./test.sh -s              # Explicit sample smoke test
  ./test.sh -e sglang       # Benchmark active SGLang server
  ./test.sh --compare-engines # Compare vLLM vs llama.cpp vs SGLang
  ./test.sh -d              # SWE-bench Lite + GPQA Diamond
  ./test.sh -d -n 5         # SWE-bench Lite + GPQA Diamond, limited to 5 questions each
  ./test.sh -m              # SWE-bench Verified + GPQA Main
  ./test.sh -a              # All tests (Full SWE-bench + GPQA Extended)
  ./test.sh --gpqa-only -d  # GPQA Diamond only
EOF
    exit 0
}

# Parse Command-Line Arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--sample)
            PROFILE="sample"
            SWE_DATASET="sample"
            GPQA_SUBSET="sample"
            shift
            ;;
        -d|--diamond|-l|--lite)
            PROFILE="diamond"
            SWE_DATASET="princeton-nlp/SWE-bench_Lite"
            GPQA_SUBSET="diamond"
            shift
            ;;
        -m|--main)
            PROFILE="main"
            SWE_DATASET="princeton-nlp/SWE-bench_Verified"
            GPQA_SUBSET="main"
            shift
            ;;
        -a|--all)
            PROFILE="all"
            SWE_DATASET="princeton-nlp/SWE-bench"
            GPQA_SUBSET="extended"
            shift
            ;;
        -n|--limit|--num-samples)
            if [[ -z "${2:-}" ]] || [[ ! "$2" =~ ^[0-9]+$ ]]; then
                echo -e "${RED}Error: $1 requires an integer argument.${NC}" >&2
                exit 1
            fi
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --swe-only)
            RUN_SWE=true
            RUN_GPQA=false
            shift
            ;;
        --gpqa-only)
            RUN_SWE=false
            RUN_GPQA=true
            shift
            ;;
        -e|--engine)
            BENCHMARK_ENGINE="$2"
            shift 2
            ;;
        --compare-engines)
            COMPARE_ENGINES=true
            shift
            ;;
        --eval|--run-eval|--run-evaluation)
            RUN_EVAL=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            echo "Use --help for available options." >&2
            exit 1
            ;;
    esac
done

if [ "$COMPARE_ENGINES" = true ]; then
    echo -e "${CYAN}${BOLD}Invoking comparative engine benchmark (vLLM vs llama.cpp vs SGLang)...${NC}"
    COMP_ARGS=("--dataset" "$SWE_DATASET")
    if [ -n "$NUM_SAMPLES" ]; then
        COMP_ARGS+=("--num-samples" "$NUM_SAMPLES")
    fi
    exec "${SCRIPT_DIR}/benchmark/compare_engines.sh" "${COMP_ARGS[@]}"
fi

case "$PROFILE" in
    sample)
        PROFILE_LABEL="Sample Smoke Test (-s, --sample)"
        ;;
    diamond|lite)
        PROFILE_LABEL="Diamond / Lite Tier (-d, --diamond)"
        ;;
    main)
        PROFILE_LABEL="Main / Verified Tier (-m, --main)"
        ;;
    all)
        PROFILE_LABEL="All Tests (-a, --all)"
        ;;
esac

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${SCRIPT_DIR}/_results"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "${BLUE}${BOLD}     AMD ROCm AI Model Benchmarking Suite (SWE-bench & GPQA)          ${NC}"
echo -e "${BLUE}${BOLD}======================================================================${NC}"
echo -e "Hardware Target  : ${BOLD}AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB VRAM)${NC}"
echo -e "Execution Time   : $(date)"
echo -e "Benchmark Tier   : ${CYAN}${BOLD}${PROFILE_LABEL}${NC}"
echo -e "SWE-bench Target : ${BOLD}${SWE_DATASET}${NC}$([ "$RUN_SWE" = false ] && echo " (Disabled)" || echo "")"
echo -e "GPQA Target      : ${BOLD}${GPQA_SUBSET}${NC}$([ "$RUN_GPQA" = false ] && echo " (Disabled)" || echo "")"
if [ -n "$NUM_SAMPLES" ]; then
    echo -e "Instance Limit   : ${YELLOW}${BOLD}${NUM_SAMPLES} samples${NC}"
fi
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
# Test Case 1: SWE-bench Benchmark
# ------------------------------------------------------------------------------
if [ "$RUN_SWE" = true ]; then
    echo -e "${CYAN}${BOLD}>>> [1/2] Running SWE-bench Evaluation (${SWE_DATASET})...${NC}"
    SWE_CMD=("run_benchmark.py" "--base-url" "http://127.0.0.1:8000/v1" "--dataset" "$SWE_DATASET")
    if [ -n "$NUM_SAMPLES" ]; then
        SWE_CMD+=("--num-samples" "$NUM_SAMPLES")
    fi
    if [ "$RUN_EVAL" = true ]; then
        SWE_CMD+=("--run-evaluation")
    fi

    if docker compose run --rm --no-deps benchmark "${SWE_CMD[@]}"; then
        SWE_STATUS="PASSED"
        echo -e "${GREEN}${BOLD}✓ SWE-bench test run completed successfully.${NC}"
    else
        echo -e "${YELLOW}Docker benchmark container failed or was interrupted. Attempting direct host execution...${NC}"
        if python3 "${SCRIPT_DIR}/benchmark/run_benchmark.py" "${SWE_CMD[@]:1}"; then
            SWE_STATUS="PASSED"
            echo -e "${GREEN}${BOLD}✓ SWE-bench test run completed successfully (host fallback).${NC}"
        else
            SWE_STATUS="FAILED"
            echo -e "${RED}${BOLD}✗ SWE-bench test run failed.${NC}"
        fi
    fi
else
    echo -e "${YELLOW}>>> [1/2] SWE-bench Evaluation: SKIPPED (--gpqa-only specified)${NC}"
fi
echo ""

# Ensure benchmark_results directory is accessible by host user
chmod -R ugo+rwX "$RESULTS_DIR" 2>/dev/null || true

# ------------------------------------------------------------------------------
# Test Case 2: GPQA Scientific Reasoning Benchmark
# ------------------------------------------------------------------------------
if [ "$RUN_GPQA" = true ]; then
    echo -e "${CYAN}${BOLD}>>> [2/2] Running GPQA Reasoning Benchmark (${GPQA_SUBSET})...${NC}"
    GPQA_CMD=("python3" "${SCRIPT_DIR}/benchmark/run_gpqa.py" "--base-url" "http://127.0.0.1:8000/v1" "--subset" "$GPQA_SUBSET")
    if [ -n "$NUM_SAMPLES" ]; then
        GPQA_CMD+=("--num-samples" "$NUM_SAMPLES")
    fi

    if "${GPQA_CMD[@]}"; then
        GPQA_STATUS="PASSED"
        echo -e "${GREEN}${BOLD}✓ GPQA test run completed successfully.${NC}"
    else
        GPQA_STATUS="FAILED"
        echo -e "${RED}${BOLD}✗ GPQA test run failed.${NC}"
    fi
else
    echo -e "${YELLOW}>>> [2/2] GPQA Reasoning Benchmark: SKIPPED (--swe-only specified)${NC}"
fi
echo ""

TESTS_END=$(date +%s)
TOTAL_TIME=$(( TESTS_END - TESTS_START ))

# ------------------------------------------------------------------------------
# Locate Metrics Files for Current or Latest Run
# ------------------------------------------------------------------------------
LATEST_SWE_FILE=$(find "${RESULTS_DIR}" -maxdepth 2 -name "benchmark_metrics.json" -type f -newermt "@${TESTS_START}" 2>/dev/null | sort -nr | head -n1 || true)
if [ -z "$LATEST_SWE_FILE" ]; then
    LATEST_SWE_FILE=$(find "${RESULTS_DIR}" -maxdepth 2 -name "benchmark_metrics.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}' || true)
fi

LATEST_GPQA_FILE=$(find "${RESULTS_DIR}/gpqa" -maxdepth 2 -name "gpqa_summary.json" -type f -newermt "@${TESTS_START}" 2>/dev/null | sort -nr | head -n1 || true)
if [ -z "$LATEST_GPQA_FILE" ]; then
    LATEST_GPQA_FILE=$(find "${RESULTS_DIR}/gpqa" -maxdepth 2 -name "gpqa_summary.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}' || true)
fi

# ------------------------------------------------------------------------------
# Parse Metrics
# ------------------------------------------------------------------------------
SWE_TOTAL="N/A"
SWE_VALID="N/A"
SWE_RATE="N/A"
SWE_THROUGHPUT="N/A"
SWE_LATENCY="N/A"

if [ "$RUN_SWE" = true ] && [ -n "$LATEST_SWE_FILE" ] && [ -f "$LATEST_SWE_FILE" ]; then
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

if [ "$RUN_GPQA" = true ] && [ -n "$LATEST_GPQA_FILE" ] && [ -f "$LATEST_GPQA_FILE" ]; then
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
if [ "$RUN_SWE" = true ]; then
    printf "%-14s | %-16s | %-15s | %-12s | %-12s\n" \
      "SWE-bench" "Code Patching" "${SWE_RATE}% (${SWE_VALID}/${SWE_TOTAL})" "${SWE_THROUGHPUT} tok/s" "${SWE_LATENCY}s"
fi
if [ "$RUN_GPQA" = true ]; then
    printf "%-14s | %-16s | %-15s | %-12s | %-12s\n" \
      "GPQA" "Sci. Reasoning" "${GPQA_ACCURACY}% (${GPQA_CORRECT}/${GPQA_TOTAL})" "${GPQA_THROUGHPUT} tok/s" "${GPQA_LATENCY}s"
fi
echo "----------------------------------------------------------------------"
if [ "$RUN_GPQA" = true ]; then
    echo -e "${BOLD}GPQA Scientific Domain Breakdown:${NC}"
    echo -e "  • Physics   : ${BOLD}${GPQA_PHYSICS}%${NC}"
    echo -e "  • Chemistry : ${BOLD}${GPQA_CHEMISTRY}%${NC}"
    echo -e "  • Biology   : ${BOLD}${GPQA_BIOLOGY}%${NC}"
    echo "----------------------------------------------------------------------"
fi
echo -e "Hardware Platform   : AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB VRAM)"
echo -e "Target Model Tested : ${BOLD}${ACTIVE_MODEL}${NC}"
echo -e "Benchmark Tier      : ${BOLD}${PROFILE_LABEL}${NC}"
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
- **Benchmark Tier**: \`${PROFILE_LABEL}\`
- **Execution Date**: $(date)
- **Total Runtime**: ${TOTAL_TIME} seconds

## Performance & Accuracy Summary Table

| Benchmark | Target Capability | Dataset / Subset | Score / Primary Metric | Throughput | Latency | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SWE-bench** | Software Engineering / Git Diff Patches | \`${SWE_DATASET}\` (${SWE_TOTAL} instances) | **${SWE_RATE}%** (${SWE_VALID}/${SWE_TOTAL} valid patches) | **${SWE_THROUGHPUT} tok/s** | **${SWE_LATENCY} s/sample** | \`${SWE_STATUS}\` |
| **GPQA** | Scientific & Technical Multiple-Choice Reasoning | \`${GPQA_SUBSET}\` (${GPQA_TOTAL} questions) | **${GPQA_ACCURACY}%** (${GPQA_CORRECT}/${GPQA_TOTAL} correct) | **${GPQA_THROUGHPUT} tok/s** | **${GPQA_LATENCY} s/item** | \`${GPQA_STATUS}\` |

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
