---
type: Operations Guide
title: SWE-bench & GPQA Benchmarking Harness Guide
description: This document describes the automated evaluation framework for testing
  model coding capability (SWE-bench), scientific reasoning (GPQA), and live serving
  throughput on AMD Radeon™ AI PRO R9700 and Instinct GPUs.
tags:
- operations-guide
- bench
status: stable
---

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
11. [Telemetry, Hardware Resource Instrumentation & vLLM Metrics](#11-telemetry-hardware-resource-instrumentation--vllm-metrics)

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
| **`openai/gpt-oss-20b`** | native MXFP4 | fits a 32 GB card | one replica per R9700S | stock `vLLM`, `VLLM_ROCM_USE_AITER=1` | **32 GB MXFP4 bench.** TP=1, prefix caching off. `gpt-oss-120b` does not fit. Protocol and commands: [GPT-OSS.md](GPT-OSS.md). |

To switch models, run `./setup.sh` with flags or edit `.env`:
```bash
# Default: Launch vLLM with Qwen3.8-27B-FP8
./setup.sh --engine vllm

# Alternative: Launch llama.cpp with Qwen3.8-27B GGUF
./setup.sh --engine llama.cpp

# Or launch any custom Hugging Face model
./setup.sh -m Qwen/Qwen2.5-Coder-7B-Instruct

# GPT-OSS-20B native MXFP4 on one 32 GB card, then the 1024/1024 bench
./setup.sh -m gpt-oss-20b
./scripts/bench_gpt_oss_20b.sh
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

Use `accuracy.sh` for SWE-bench and GPQA. With no suite selected, `test.sh` runs accuracy, throughput, and the local tail-latency experiment matrix. `--both` is accuracy then throughput only. GPU selection defaults to `auto`. `-q` shortens every selected suite; the experiment arm then uses the `tail-quick` campaign instead of `tail-matrix`.

`./test.sh --experiments` runs the matrix in `config/campaigns.yaml`: closed-loop concurrency at C=1, 2, 4, 8, 16, 32, synthetic agent chains, open-loop offered load, the input-length TTFT grid, cold-start idle intervals, AgentX trace replay when AIPerf and the tokenizer are present, and a collocated prefill burst against active decode streams. On R9700, that full matrix also runs the C=1, 2, 4, 8, 16 sweeps for 8,192/1,024, 1,024/1,024, and 1,024/8,192 against the Qwen3.8-27B Quark MXFP4 server. Each cell stores TTFT and TPOT p50, p90, p95, and p99 plus the request samples, and replaces that shape in `reports/results/r9700/concurrency.json` only when every cell passes. Provider comparison runs only when `TAIL_COMPARE_MANIFEST` names a second run. The disaggregated prefill/decode arm is not started. When the dispatcher exits it publishes the run and rebuilds the figures with `./analyze.sh`. `./test.sh -q` uses `tail-quick` and skips the three full-output sweeps.

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

# Accuracy, throughput, and the tail-matrix experiments
./test.sh

# Short sample of every suite, including tail-quick
./test.sh -q

# Tail-latency experiment matrix only
./test.sh --experiments

# Accuracy sanity sample, throughput, and experiments
./test.sh -a
./test.sh --all

# Accuracy on the full dataset, throughput, and experiments
./test.sh --full

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
4. Records GPU VRAM usage via `amd-smi metric -m`.
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

---

## 11. Telemetry, Hardware Resource Instrumentation & vLLM Metrics

Understanding model serving efficiency requires distinguishing between **engine-level serving metrics** and **physical hardware telemetry**.

### 11.1 Native vLLM Metrics Architecture (`/metrics`)

Per the official [vLLM Metrics Documentation](https://docs.vllm.ai/en/stable/design/metrics/), the vLLM engine natively exposes Prometheus metrics at `/metrics` covering request scheduling, execution phase timings, and token generation:

| Metric Category | Prometheus Metric Name | Description |
| :--- | :--- | :--- |
| **TTFT** | `vllm:time_to_first_token_seconds` | Histogram of time from request arrival until the first token is emitted. |
| **TPOT** | `vllm:request_time_per_output_token_seconds` | Request-level Time Per Output Token: $(\text{E2E} - \text{TTFT}) / (N_{\text{out}} - 1)$. |
| **ITL** | `vllm:inter_token_latency_seconds` | Inter-token latency: wall-clock gap between consecutive streamed outputs. |
| **E2E Latency** | `vllm:e2e_request_latency_seconds` | Total end-to-end request turnaround time. |
| **Queue Latency**| `vllm:request_queue_time_seconds` | Duration requests wait in the engine queue prior to prefill. |
| **Phase Latency**| `vllm:request_prefill_time_seconds`<br>`vllm:request_decode_time_seconds` | Phase-split execution timings isolating prefill compute from autoregressive decode. |
| **Throughput** | `vllm:prompt_tokens_total`<br>`vllm:generation_tokens_total` | Monotonic counters and instantaneous prompt/generation token throughput. |
| **Logical Cache**| `vllm:kv_cache_usage_perc` | Fraction of logical PagedAttention KV cache blocks allocated ($0.0 \dots 1.0$). |
| **Prefix Cache** | `vllm:prefix_cache_hits`, `_queries` | Automatic Prefix Caching (APC) hit rates and reuse intervals. |
| **Speculative** | `vllm:spec_decode_num_accepted_tokens`<br>`vllm:spec_decode_draft_acceptance_rate` | Speculative decoding acceptance count and draft verification efficiency. |

### 11.2 What vLLM Does NOT Capture Natively

While vLLM provides comprehensive application-layer profiling, it **does not** interface with device-level kernel drivers or physical hardware sensors:
1. **Power Consumption (Watts & % of TDP)**: vLLM has no access to the GPU System Management Unit (SMU) to read instantaneous board power, rail voltages, or calculate energy per token ($J/\text{token}$).
2. **Physical Memory Bandwidth (GB/s & % of Peak)**: vLLM tracks *logical* block allocation (`kv_cache_usage_perc`), not physical memory bus traffic across GDDR6 or HBM3E controllers.
3. **True Client Perceived Streaming TTFT**: Server-side `vllm:time_to_first_token_seconds` excludes client TCP handshakes, TLS negotiation, and Server-Sent Event (SSE) chunk streaming delays.

### 11.3 Unified Telemetry Collection Framework

Our benchmarking test suite bridges this gap by marrying vLLM's internal metrics with continuous hardware telemetry and client-side streaming instrumentation:

| Tool / Script | Key Capabilities & Flags | Measured Dimensions |
| :--- | :--- | :--- |
| [`scripts/collect_amd_power.py`](../scripts/collect_amd_power.py) | `--tdp <watts>`, `--peak-bw <gbs>`, auto-detects `mi350p` (600W/4096 GB/s), `r9700` (300W/960 GB/s), `r9600` (150W/640 GB/s). | Power (W), % of TDP, Max Power, Energy (J), Duration. |
| [`scripts/bench_openai_chat.py`](../scripts/bench_openai_chat.py) | `--stream`, `--monitor-power`, `--gpu-profile <target>`. | Client-side TTFT ($p_{50}$, $p_{90}$, $p_{95}$, $p_{99}$), ITL ($p_{50}$, $p_{90}$, $p_{95}$), Power (% TDP, W, J/tok), Memory Bandwidth (GB/s, % Peak). |
| [`scripts/run_concurrency_sweep.py`](../scripts/run_concurrency_sweep.py) | `--gpu-profile <target>`, sweeps concurrency $C=1 \dots 32$, `--percentile-metrics ttft,tpot,itl,e2el`, `--metric-percentiles 50,90,95,99`, `--save-detailed`. | TTFT, TPOT, ITL, Active Power, Power Util %, Memory Bandwidth GB/s, Bandwidth Util %. |
| [`scripts/bench_throughput.sh`](../scripts/bench_throughput.sh) | Automated multi-slice serving sweep with background `collect_amd_power.py`. | Matrix of Throughput, TPOT, TTFT, Power (W / % TDP), and Mem Bandwidth (GB/s / % Peak). |
| [`scripts/bench_saturation_instrumented.sh`](../scripts/bench_saturation_instrumented.sh) | End-to-end saturation harness for R9700 and MI350P. | Live streaming TTFT and power under progressive concurrency load. |

### 11.4 Example Usage for MI350P & R9700

**Run Concurrency Sweep with Full Telemetry on MI350P**:
```bash
python3 scripts/run_concurrency_sweep.py \
  --gpu-profile mi350p \
  --input-len 1024 \
  --output-len 1024 \
  --max-concurrency 32
```

**Run Streaming Benchmark with Mid-Batch Power & Memory Bandwidth on R9700**:
```bash
python3 scripts/bench_openai_chat.py \
  --host http://localhost:8000 \
  --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 \
  --stream \
  --monitor-power \
  --gpu-profile r9700 \
  --concurrency 4 \
  --max-tokens 1024
```

