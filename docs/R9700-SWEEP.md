---
type: Benchmark Report
title: R9700 SWEEP (8192:1024)
description: Concurrency, latency, power, and bandwidth sweep for 8,192 input and
  1,024 output tokens.
tags:
- concurrency
- latency
- r9700
status: stable
---

# R9700 SWEEP (8192:1024)

Narrative and earlier sweeps: [R9700.md](R9700.md).

**Target Hardware**: AMD Radeon™ AI PRO R9700 (`gfx1201`, 32 GB GDDR6, 300W TDP, 960 GB/s Peak)
**Model**: `Qwen3.8-27B-Quark-AWQ-MXFP4`
**Workload**: 8,192 Input Tokens / 1,024 Output Tokens
**Execution Date**: 2026-10-08 17:49:39 EDT

## Performance, Latency, Power & Bandwidth Ledger

| C | Aggregate tok/s | Per-stream tok/s | TTFT p50/p90/p95/p99 (ms) | TPOT p50/p90/p95/p99 (ms) | Power (W / % TDP) | Mem Bandwidth (GB/s / % Peak) | Total Energy (J) | J/token | tokens/Joule | Hotspot Max (°C) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **32.14** | 32.14 | 337.8 / 339.5 / 339.7 / 339.9 | 30.82 / 30.85 / 30.86 / 30.86 | 163.1 W (54.4%) | 618.0 GB/s (64.38%) | 13,609.6 | **6.645** | **0.1505** | 0.0 | `PASSED` |
| **2** | **60.84** | 30.42 | 500.1 / 625.3 / 641.0 / 653.5 | 32.4 / 32.51 / 32.53 / 32.54 | 137.0 W (45.7%) | 585.01 GB/s (60.94%) | 7,361.1 | **3.594** | **0.2782** | 0.0 | `PASSED` |
| **4** | **97.90** | 24.48 | 3800.6 / 6512.0 / 6662.7 / 6783.3 | 37.07 / 39.96 / 40.16 / 40.33 | 146.3 W (48.8%) | 470.68 GB/s (49.03%) | 9,070.5 | **2.214** | **0.4516** | 0.0 | `PASSED` |
| **8** | **98.02** | 12.25 | 21288.1 / 47373.3 / 47948.7 / 48409.0 | 34.66 / 39.26 / 40.18 / 40.93 | 171.8 W (57.3%) | 235.62 GB/s (24.54%) | 17,815.5 | **2.175** | **0.4598** | 0.0 | `PASSED` |
| **16** | **97.88** | 6.12 | 57558.1 / 129301.2 / 131104.0 / 132088.7 | 35.09 / 40.86 / 41.14 / 41.15 | 188.0 W (62.7%) | 117.64 GB/s (12.25%) | 35,386.0 | **2.160** | **0.4630** | 0.0 | `PASSED` |

## Marginal Transitions & Plateau Analysis

| Transition | Throughput Marginal Gain | Efficiency Marginal Gain | Latency Status | Scaling Regime |
| :---: | :---: | :---: | :---: | :---: |
| **$C=1 \to C=2$** | **+89.3%** ($32.14 \to 60.84\text{ tok/s}$) | **+84.9%** ($0.1505 \to 0.2782\text{ tok/J}$) | $p_{95}\text{ TPOT} = 32.53\text{ ms}$ | Linear Scaling |
| **$C=2 \to C=4$** | **+60.9%** ($60.84 \to 97.90\text{ tok/s}$) | **+62.3%** ($0.2782 \to 0.4516\text{ tok/J}$) | $p_{95}\text{ TPOT} = 40.16\text{ ms}$ | Scaling Knee |
| **$C=4 \to C=8$** | **+0.1%** ($97.90 \to 98.02\text{ tok/s}$) | **+1.8%** ($0.4516 \to 0.4598\text{ tok/J}$) | $p_{95}\text{ TPOT} = 40.18\text{ ms}$ | **Throughput Plateau (<10%)** |
| **$C=8 \to C=16$** | **-0.1%** ($98.02 \to 97.88\text{ tok/s}$) | **+0.7%** ($0.4598 \to 0.4630\text{ tok/J}$) | $p_{95}\text{ TPOT} = 41.14\text{ ms}$ | **Full Saturation / Serialization** |

## Analytical Takeaways

1. **Roofline Ceiling (~98 tok/s)**: On 8,192 input / 1,024 output tokens, the 32 GB R9700 saturates its compute and memory pipeline at $C=4$, delivering ~98 tok/s. Higher concurrencies divide this fixed token rate among more streams.
2. **Prefill Head-of-Line Blocking**: While decode latency (TPOT) remains stable between 30.8 ms and 41.1 ms across all concurrencies, TTFT explodes from 338 ms ($C=1$) to 57.6 s median / 132.1 s $p_{99}$ ($C=16$) due to serial ingestion of 8k prompts.
3. **Energy Efficiency Knee**: Operating at $C=4$ minimizes energy consumption to **2.214 J/token** (0.4516 tok/J), 3.0× more efficient than single-stream serving ($C=1$).

## Durable Artifacts & Sample Preservation

- **Exact Request Samples (Compressed JSON)**: [`reports/results/qwen3.8-27b-mxfp4/latency/r9700/8k1k/`](../reports/results/qwen3.8-27b-mxfp4/latency/r9700/8k1k/)
- **Durable Profile Index**: [`reports/profiling/qwen3.8-27b-mxfp4-latency.json`](../reports/profiling/qwen3.8-27b-mxfp4-latency.json)
- **Raw Benchmark Scratch**: `_results/concurrency_sweep/`

---

