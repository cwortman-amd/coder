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
**Generated:** 2026-10-01T20:29:14.971474+00:00  

---

## 1. Single-Call Latency Distributions (TTFT, TTFAT, and Decode Rates)

| Concurrency | Requests | TTFT p50/p95/p99 | TTFAT (1st Answer) p50/p95 | Decode Rate (M>1) | Total Latency p50/p99 | TTFT Miss (% >1000.0ms) |
|---:|---:|---:|---:|---:|---:|---:|
| C=1 | 200/200 | 34/36/41 ms | 34/36 ms | 97.0 tok/s | 684/767 ms | **0.0%** |
| C=2 | 200/200 | 78/80/86 ms | 78/80 ms | 90.6 tok/s | 769/842 ms | **0.0%** |
| C=4 | 200/200 | 73/83/86 ms | 73/83 ms | 87.1 tok/s | 798/811 ms | **0.0%** |
| C=8 | 200/200 | 86/89/93 ms | 86/89 ms | 82.9 tok/s | 846/855 ms | **0.0%** |
| C=16 | 200/200 | 155/164/166 ms | 155/164 ms | 68.7 tok/s | 1073/1084 ms | **0.0%** |
| C=32 | 200/200 | 284/296/298 ms | 284/296 ms | 56.3 tok/s | 1405/1424 ms | **0.0%** |

## 2. Complete-Task Agent Trajectory Benchmark

*Workload: 30 independent sequential chains of 10 calls per chain.*

| Metric | Empirical Measured Tasks | Resampling Baseline 1 (Independent) | Resampling Baseline 2 (Correlated Block) |
|:---|---:|---:|---:|
| **Sample Count ($N$)** | 30 | 5000 | 5000 |
| **Task Duration p50** | **7.15 s** | 6.82 s | 6.87 s |
| **Task Duration p95** | **7.82 s** | 6.96 s | 6.96 s |
| **Task Duration p99** | **7.83 s** | 7.04 s | 7.10 s |
| **Worst-Case Task** | **7.84 s** | 7.27 s | 7.10 s |
| **p99:p50 Spread Ratio** | **1.10x** | 1.03x | 1.03x |
| **Coefficient of Variation (CV)** | 18.1% | 1.0% | 1.8% |
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

