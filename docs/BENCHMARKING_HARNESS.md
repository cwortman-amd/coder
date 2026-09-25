# SWE-bench & GPQA Benchmarking Harness Guide

This document describes the automated evaluation framework for testing model coding capability (SWE-bench), scientific reasoning (GPQA), and live serving throughput on AMD Radeon™ AI PRO R9700 and Instinct GPUs.

---

## Table of Contents
1. [Recommended Coding Models for 32 GB VRAM](#1-recommended-coding-models-for-32-gb-vram)
2. [SWE-bench Evaluation Architecture](#2-swe-bench-evaluation-architecture)
3. [SWE-bench Dataset Splits](#3-swe-bench-dataset-splits)
4. [Unified Test Runner (`test.sh`) & Accuracy Suite (`accuracy.sh`)](#4-unified-test-runner-testsh--accuracy-suite-accuracysh)
5. [Standalone SWE-bench Smoke Test](#5-standalone-swe-bench-smoke-test)
6. [Benchmarking SWE-bench Lite & Verified](#6-benchmarking-swe-bench-lite--verified)
7. [Automating Multi-Model Comparative Benchmarks (`compare_models.sh`)](#7-automating-multi-model-comparative-benchmarks-compare_modelssh)
8. [Automating Multi-Engine Comparisons (`compare_engines.sh`)](#8-automating-multi-engine-comparisons-compare_enginessh)
9. [Executing Functional Patch Resolution (`swebench.harness`)](#9-executing-functional-patch-resolution-swebenchharness)
10. [Empirical Benchmark Scores on Radeon AI PRO R9700](#10-empirical-benchmark-scores-on-radeon-ai-pro-r9700)

---

## 1. Recommended Coding Models for 32 GB VRAM

The table below outlines optimal coding models validated for the 32 GB VRAM capacity of the AMD Radeon AI PRO R9700:

| Model ID | Precision / Quant | Weights Size | Working VRAM (at max context) | Engine / Parser | Single vs. Dual R9700 Guidance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Qwen/Qwen3.8-27B-FP8`** | FP8 | ~27.5 GB | ~28.5 GB (8,192 ctx) | `vLLM` (Default) / `hermes` | **DEFAULT FOR vLLM**. Full FP8 precision on single 32GB R9700 with `MAX_MODEL_LEN=8192`. Uncompromised accuracy for SWE-bench & code generation. For 32k+ context, Dual R9700 TP=2 is recommended. |
| **`Qwen3.8-27B`** *(or `.gguf`)* | GGUF (Q4_K_M) | ~16.8 GB | ~22 GB (32k ctx) | `vLLM` (via plugin) or `llama.cpp` | **Extended Context (32k–64k)**. Compact weights leave 15+ GB free VRAM for deep repository context. Downloadable via `scripts/download_model.sh`. |
| **`Qwen/Qwen2.5-Coder-7B-Instruct`** | BF16 / FP16 | ~15 GB | ~18 GB (32k ctx) | `vLLM` / `hermes` | Blazing fast (>45 tok/s), strong tool calling, fits comfortably with 32k context on single R9700. |
| **`Qwen/Qwen2.5-Coder-14B-Instruct`** | BF16 | ~28 GB | ~30 GB (16k ctx) | `vLLM` / `hermes` | High coding intelligence on single R9700. Set `--max-model-len 16384` to prevent VRAM overflow. |
| **`Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`** | AWQ (4-bit) | ~19 GB | ~24 GB (32k ctx) | `vLLM` / `hermes` | **Best reasoning-to-VRAM ratio** on single R9700. Delivers 32B capability within 32 GB VRAM budget. |
| **`Qwen/Qwen2.5-Coder-32B-Instruct`** | BF16 / FP16 | ~65 GB | **Requires Dual R9700** | `vLLM` / `hermes` | Full-precision 32B dense coder on Dual R9700 (64 GB) with TP=2. |
| **`Qwen/Qwen3-0.6B`** | BF16 | ~1.4 GB | ~4 GB (32k ctx) | `vLLM` / `hermes` | Ultra-fast validation model for testing container pipelines. |

To switch models, run `./setup.sh` with flags or edit `.env`:
```bash
# Default: Launch vLLM with Qwen3.8-27B-FP8
./setup.sh --engine vllm

# Alternative: Launch llama.cpp with Qwen3.8-27B GGUF
./setup.sh --engine llama.cpp

# Or launch any custom Hugging Face model
./setup.sh -m Qwen/Qwen2.5-Coder-7B-Instruct
```

---

## 2. SWE-bench Evaluation Architecture

[SWE-bench](https://www.swebench.com/) (Princeton NLP) is the standard benchmark for evaluating LLMs on real-world software engineering tasks. It presents the model with real GitHub issues from popular open-source repositories and tests whether the model can generate a unified git diff (`model_patch`) that resolves the issue and passes the repository's test suite.

```text
+-----------------------------------------------------------------------------------------+
|                              SWE-bench Evaluation Pipeline                              |
|                                                                                         |
|   +--------------------------+                                                          |
|   |  SWE-bench Dataset       |                                                          |
|   |  - SWE-bench_Lite (300)  |                                                          |
|   |  - SWE-bench_Verified    |                                                          |
|   |  - Offline Sample Set    |                                                          |
|   +-------------+------------+                                                          |
|                 |                                                                       |
|                 |  1. Problem statement & repo context                                  |
|                 v                                                                       |
|   +-------------+------------------------------------+                                  |
|   |           Benchmark Runner (swebench-runner)     |                                  |
|   |                                                  |                                  |
|   |   - Queries OpenAI API (http://127.0.0.1:8000/v1)|                                  |
|   |   - Measures generation latency & throughput     |                                  |
|   |   - Strips reasoning tags and parses git diff    |                                  |
|   +-------------+------------------------------------+                                  |
|                 |                                                                       |
|                 |  2. Emits predictions.jsonl & benchmark_metrics.json                  |
|                 v                                                                       |
|   +-------------+------------------------------------+                                  |
|   |           SWE-bench Evaluation Harness           |                                  |
|   |                                                  |                                  |
|   |   - Spawns test environments via Docker socket   |                                  |
|   |   - Applies model_patch via git apply            |                                  |
|   |   - Executes repo unit tests (FAIL_TO_PASS)      |                                  |
|   |   - Outputs: % Resolved, Pass Rate, Error Rate   |                                  |
|   +--------------------------------------------------+                                  |
+-----------------------------------------------------------------------------------------+
```

---

## 3. SWE-bench Dataset Splits

| Dataset Identifier | Task Count | Description | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **`sample`** | 3 | Built-in offline sample problems from Astropy, SymPy, and Django. | Instant pipeline smoke-test; zero internet required. |
| **`princeton-nlp/SWE-bench_Lite`** | 300 | Curated subset of clean, self-contained issues from 12 popular repos. | Standard evaluation split for local open-source models. |
| **`princeton-nlp/SWE-bench_Verified`** | 500 | Human-validated issues filtered by SWE-bench researchers for clarity. | Gold-standard benchmark for state-of-the-art coding agents. |

---

## 4. Unified Test Runner (`test.sh`) & Accuracy Suite (`accuracy.sh`)

Use `accuracy.sh` for SWE-bench and GPQA. The unified `test.sh` runs both accuracy and throughput by default. GPU selection defaults to `auto`.

```bash
# Default: Offline smoke test (3 SWE-bench problems + 3 GPQA questions)
./accuracy.sh

# Diamond / Lite Tier: SWE-bench Lite (300 problems) + GPQA Diamond (198 questions)
./accuracy.sh -d

# Main / Verified Tier: SWE-bench Verified (500 problems) + GPQA Main (448 questions)
./accuracy.sh -m

# All Tests: Full SWE-bench (2,294 problems) + GPQA Extended (546 questions)
./accuracy.sh -a

# Quick sample limits (e.g., test first 5 questions of Diamond/Lite tier)
./accuracy.sh -d -n 5

# Benchmark a specific engine (defaults to vLLM)
./accuracy.sh -e vllm
./accuracy.sh -e llama.cpp

# Run both suites using automatic GPU detection
./test.sh -q

# Run only one suite through the dispatcher
./test.sh --accuracy -d --accuracy-limit 5
./test.sh --throughput -e vllm -c 8
```

---

## 5. Standalone SWE-bench Smoke Test

Run a rapid 3-problem benchmark against your active model without downloading external datasets:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps benchmark
```

*Example Output:*
```text
===========================================================================
 SWE-bench Model Evaluation & Performance Benchmark
===========================================================================
Server Base URL: http://127.0.0.1:8000/v1
Target Model   : Qwen/Qwen2.5-Coder-7B-Instruct
Dataset        : sample
Total Instances: 3
---------------------------------------------------------------------------
[1/3] Evaluating instance: astropy__astropy-12907 ... OK (3.21s, 48.2 tok/s, Patch valid)
[2/3] Evaluating instance: sympy__sympy-18057 ... OK (2.89s, 51.0 tok/s, Patch valid)
[3/3] Evaluating instance: django__django-11099 ... OK (2.15s, 52.4 tok/s, Patch valid)

===========================================================================
 BENCHMARK PERFORMANCE SUMMARY
===========================================================================
 Model Tested              : Qwen/Qwen2.5-Coder-7B-Instruct
 Total Instances Evaluated : 3
 Valid Git Patches Created : 3 (100.0%)
 Average Throughput        : 50.40 tokens/second
 Average Latency per Sample: 2.75 seconds
 Output Predictions File   : _results/Qwen_Qwen2.5-Coder-7B-Instruct_20260922/predictions.jsonl
 Metrics Report File       : _results/Qwen_Qwen2.5-Coder-7B-Instruct_20260922/benchmark_metrics.json
===========================================================================
```

---

## 6. Benchmarking SWE-bench Lite & Verified

Evaluate 10 instances of SWE-bench Lite:
```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps benchmark run_benchmark.py \
  --dataset princeton-nlp/SWE-bench_Lite \
  --num-samples 10 \
  --output-dir _results
```

Evaluate SWE-bench Verified:
```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps benchmark run_benchmark.py \
  --dataset princeton-nlp/SWE-bench_Verified \
  --num-samples 25 \
  --output-dir _results
```

---

## 7. Automating Multi-Model Comparative Benchmarks (`compare_models.sh`)

Use [`benchmark/compare_models.sh`](../benchmark/compare_models.sh) to automatically cycle through multiple models on the AMD Radeon AI PRO R9700, test each model, and record the results:

```bash
./benchmark/compare_models.sh sample 3
```

This automated script:
1. Recreates the `inference` container with model 1 (e.g. `Qwen/Qwen2.5-Coder-7B-Instruct`).
2. Waits for the vLLM server to report healthy on port 8000.
3. Executes the SWE-bench benchmark suite and logs throughput and patch validity.
4. Records GPU VRAM usage via `rocm-smi`.
5. Repeats for model 2 (`Qwen/Qwen2.5-Coder-14B-Instruct`) and model 3 (`Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`).
6. Saves aggregated reports in `_results/`.

---

## 8. Automating Multi-Engine Comparisons (`compare_engines.sh`)

Use [`benchmark/compare_engines.sh`](../benchmark/compare_engines.sh) to compare vLLM, llama.cpp, and SGLang side-by-side on TTFT, decode speed, VRAM consumption, and SLO pass rates:

```bash
./benchmark/compare_engines.sh --all
```

---

## 9. Executing Functional Patch Resolution (`swebench.harness`)

To run the official SWE-bench evaluation harness and compute functional resolution pass rates:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps benchmark \
  python3 -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path /app/_results/Qwen_Qwen2.5-Coder-7B-Instruct_20260922/predictions.jsonl \
    --run_id qwen25_7b_eval \
    --max_workers 4
```

---

## 10. Empirical Benchmark Scores on Radeon AI PRO R9700

Empirical benchmark performance measured on the **AMD Radeon™ AI PRO R9700** (32 GB GDDR6, RDNA 4 `gfx1201`):

| Model | Quantization | Working VRAM | Inference Speed (tok/s) | Avg Latency / Task | Valid Patch Format (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Qwen/Qwen3-0.6B`** *(Smoke Test)* | BF16 | ~4.0 GB | **269.9 tok/s** | 1.50 s | 100% |
| **`Qwen/Qwen2.5-Coder-7B-Instruct`** | BF16 | ~18.5 GB | **50.4 tok/s** | 2.75 s | 100% |
| **`Qwen/Qwen2.5-Coder-14B-Instruct`** | BF16 | ~29.5 GB | **28.6 tok/s** | 4.80 s | 100% |
| **`Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`**| AWQ (4-bit) | ~24.0 GB | **34.2 tok/s** | 3.90 s | 100% |
| **`Qwen/Qwen3.8-27B-FP8`** | FP8 | ~28.5 GB | **33.2 tok/s** | 3.05 s | 100% |
