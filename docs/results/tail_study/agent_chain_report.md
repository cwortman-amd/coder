---
type: Benchmark Report
title: Complete-Task Agent Latency & Compounding Report
description: Measured single-call and complete-task latency distributions for sequential
  agent chains.
tags:
- agent-chain
- tail-latency
- benchmark
status: stable
---

# Complete-Task Agent Latency & Compounding Report

**Model:** `Qwen3.8-27B-Quark-AWQ-MXFP4`  
**Target URL:** `http://127.0.0.1:8000/v1`  
**Workload Mode:** `closed_loop_concurrency`  
**Length Buckets:** Prompt `controlled_~100_tokens` | Output `target_64_tokens`  
**Interactive Deadlines:** TTFT $\le 1000.0$ ms | Task $\le 30.0$ s  
**Generated:** 2026-10-01T11:09:46.446245+00:00  

---

## 1. Single-Call Latency Distributions (TTFT, TTFAT, and Decode Rates)

| Concurrency | Requests | TTFT p50/p95/p99 | TTFAT (1st Answer) p50/p95 | Decode Rate (M>1) | Total Latency p50/p99 | TTFT Miss (% >1000.0ms) |
|---:|---:|---:|---:|---:|---:|---:|
| C=1 | 75/75 | 36/39/66 ms | 36/39 ms | 100.8 tok/s | 661/1056 ms | **0.0%** |
| C=5 | 75/75 | 75/82/85 ms | 75/82 ms | 76.3 tok/s | 900/1101 ms | **0.0%** |
| C=20 | 75/75 | 183/200/201 ms | 183/200 ms | 64.4 tok/s | 1171/1187 ms | **0.0%** |

## 2. Complete-Task Agent Trajectory Benchmark

*Workload: 30 independent sequential chains of 10 calls per chain.*

| Metric | Empirical Measured Tasks | Resampling Baseline 1 (Independent) | Resampling Baseline 2 (Correlated Block) |
|:---|---:|---:|---:|
| **Sample Count ($N$)** | 30 | 5000 | 5000 |
| **Task Duration p50** | **7.05 s** | 6.62 s | 6.62 s |
| **Task Duration p95** | **7.55 s** | 7.23 s | 7.20 s |
| **Task Duration p99** | **8.01 s** | 7.59 s | 7.21 s |
| **Worst-Case Task** | **8.19 s** | 8.44 s | 7.21 s |
| **p99:p50 Spread Ratio** | **1.14x** | 1.15x | 1.09x |
| **Coefficient of Variation (CV)** | 18.8% | 3.7% | 3.3% |
| **Task Deadline Miss (% >30.0s)** | **0.0%** | 0.0% | 0.0% |
| **Tasks with >=1 Slow Call** | **100.0%** | N/A | N/A |

## 3. Mathematical Systems Foundations & Corrected Trade-Offs

### 3.1 Corrected TTFT-vs-Decode Crossover Formula
For a constant-gap model:
$$T_{\mathrm{complete}} \approx T_{\mathrm{TTFT}} + (M - 1) \cdot \overline{\mathrm{ITL}}$$

For Model A ($T_{\mathrm{TTFT}}=200\text{ ms}$, rate $= 20\text{ tok/s} \implies \overline{\mathrm{ITL}} = 0.05\text{ s}$) versus Model B ($T_{\mathrm{TTFT}}=1000\text{ ms}$, rate $= 100\text{ tok/s} \implies \overline{\mathrm{ITL}} = 0.01\text{ s}$):
$$\Delta T_{\mathrm{TTFT}} = 1.0 - 0.2 = 0.8\text{ s}$$
$$\Delta \overline{\mathrm{ITL}} = 0.05 - 0.01 = 0.04\text{ s/interval}$$
$$\text{Crossover at } M - 1 = \frac{0.8}{0.04} = 20 \implies M \approx 21\text{ output tokens}$$

*For any task generating more than approximately 21 tokens, Model B completes earlier despite having a $5\times$ slower initial token.*

### 3.2 Compounding Exceedance Reference vs. Complete Task Duration
Under the independent, identically distributed assumption:
$$P(\text{at least one call exceeds threshold } T) = 1 - (1 - q)^N$$

| Chain Length ($N$) | Independent Exceedance Reference ($q=1%$) | Independent Exceedance Reference ($q=5%$) |
|---:|---:|---:|
| 1 calls | 1.0% | **5.0%** |
| 2 calls | 1.99% | **9.75%** |
| 5 calls | 4.9% | **22.62%** |
| 10 calls | 9.56% | **40.13%** |
| 15 calls | 13.99% | **53.67%** |
| 20 calls | 18.21% | **64.15%** |

*Important distinction:* This predicts the probability of encountering an outlier call; it is **not** the workflow's p95 or p99 completion time. Sequential call durations add ($T_{\text{task}} = \sum T_i$), and shared queue congestion correlates outliers, causing the empirical task tail to diverge from independent convolutions.

