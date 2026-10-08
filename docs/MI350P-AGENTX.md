---
type: Technical Report
title: MI350P AgentX concurrency tail
description: 'Date: 30 September 2026 Hardware: 1× AMD Instinct MI350P PCIe, GPU 0
  (rocm-inference-server) Model: Qwen3.8-27B-Quark-AWQ-MXFP4, context cap 65,536.'
tags:
- technical-report
- mi350p
- agentx
status: stable
---

# MI350P AgentX concurrency tail

**Date:** 30 September 2026  
**Hardware:** 1× AMD Instinct MI350P PCIe, GPU 0 (`rocm-inference-server`)  
**Model:** `Qwen3.8-27B-Quark-AWQ-MXFP4`, context cap 65,536  
**Workload:** AIPerf 0.13 `inferencex-agentx-mvp`, corpus `semianalysis_cc_traces_weka_062126`, seed `20260707`  
**Windows:** 900 s profiling at each concurrency, plus warmup and a 30 s grace period

This is the collocated AgentX load curve. It is separate from the fixed-length 1K/1K throughput campaign in [MI350P.md](MI350P.md) and from the controlled cold-prefill interference tests in [QWEN-TAIL-PDD.md](QWEN-TAIL-PDD.md).

## Headline

Output throughput peaks at configured concurrency **32 (46.80 tok/s)** and collapses at **64 (4.01 tok/s)**. The interactive tail moves earlier: token gaps above 1 s are absent at C=1, 4.59 per 1,000 intervals at C=16, and 23.39 per 1,000 at C=32. C=64 and C=128 are overload diagnostics. Both have a persistent waiting queue, KV usage near the cache ceiling, and **zero completed AgentX sessions**.

The engine’s full-context KV limit on this launch is about **22.3× at 65,536 tokens**. Configured AgentX concurrency is a client lane limit. Realized running requests stayed well below that limit until the cache filled.

## Method

The loader scanned 393 traces and kept **7** whose peak context fit in 65,536 tokens. AIPerf replayed their recorded parent/child dependencies and joins. The locked scenario cache-busts the first-turn prefix, caps system idle at 10 s, and forbids an open request-rate schedule. A lane starts another session only after its previous session releases it.

Each cell used the same endpoint, tokenizer, seed, context cap, and 900 s profiling duration:

```bash
python3 scripts/run_agentx_tail_sweep.py --concurrency 1 8 16
python3 scripts/run_agentx_tail_sweep.py \
  --concurrency 32 64 128 \
  --campaign-seconds 7200
python3 scripts/analyze_agentx_tail_sweep.py
```

All six cells exited 0 with `submission_valid: true` and zero request errors. Raw exports are in gitignored `_results/agentx_tail_sweep/20260930_180731/` and `_results/agentx_tail_sweep/20260930_224523/`. The durable summary is [results/mi350p/agentx_concurrency.json](results/mi350p/agentx_concurrency.json).

## Load curve

| C | Requests | Sessions sent / done | Output tok/s | TTFT p50 / p95 | Request p50 / p95 | Request-average ITL p50 / p95 | >1 s gaps | Prefix hit | Running avg | Waiting avg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23 | 2 / 1 | 11.59 | 0.68 / 2.07 s | 39.8 / 85.3 s | 109 / 124 ms | 0 / 10,740 | 92.3% | 0.94 | 0 |
| 8 | 57 | 7 / 1 | 19.31 | 0.89 / 2.79 s | 18.7 / 92.5 s | 101 / 115 ms | 33 / 17,793 | 93.1% | 1.92 | 0 |
| 16 | 132 | 12 / 3 | 33.14 | 1.05 / 3.97 s | 18.9 / 101.0 s | 113 / 162 ms | 146 / 31,800 | 92.7% | 4.17 | 0 |
| 32 | 201 | 26 / 6 | **46.80** | 1.25 / 14.47 s | 27.6 / 141.5 s | 155 / 320 ms | 1,279 / 54,682 | 88.5% | 11.11 | 0.10 |
| 64 | 31 | 36 / 0 | 4.01 | 42.1 / 261.0 s | 557 / 767 s | 3.71 / 4.47 s | 6,691 / 7,049 | 25.9% | 28.74 | 8.75 |
| 128 | 37 | 66 / 0 | 5.72 | 103.5 / 380.6 s | 466 / 734 s | 2.52 / 3.16 s | 7,236 / 8,950 | 30.8% | 27.81 | 48.10 |

At C=64 the KV cache averaged **83.2%** full, with p99 **98.5%**. At C=128 the average waiting queue was **48.1 requests** and its p95 was **59**. The server admitted about 28–33 running requests in both overload cells, close to the KV ceiling, so raising the client lane count from 64 to 128 added queue depth rather than decode capacity.

## Visualizations

Empirical distributions and the master operational dashboard for MI350P AgentX are published under:
* [`figures/agentx/mi350p/01_agentx_ttft_histogram.png`](figures/agentx/mi350p/01_agentx_ttft_histogram.png)
* [`figures/agentx/mi350p/02_agentx_master_dashboard.png`](figures/agentx/mi350p/02_agentx_master_dashboard.png)

## What the tail supports

C=1 through C=16 show a smooth load trend: TTFT p95 rises from 2.07 s to 3.97 s, and freezes above 1 s rise from 0 to 4.59 per 1,000 intervals. No server-side waiting queue forms. Prefix-cache hit rate stays above 92%.

C=32 is the raw-throughput maximum and already a poor interactive point. TTFT p95 is 14.5 s, request-average ITL p95 is 320 ms, and 23.4 of every 1,000 token intervals exceed 1 s. Its waiting queue is still small, so this degradation is concurrent prefill/decode and cache pressure, not a deep admission queue.

C=64 and C=128 complete no conversations. Their lower completed-request counts are a consequence of requests taking many minutes, not evidence of a lighter workload. C=128 has a lower freeze rate than C=64 (808 versus 949 per 1,000) while having a much worse TTFT and a much deeper queue. That rate is survivor-biased and does not make C=128 healthier.

End-to-end medians are not a pure load comparison. Median output length was 372 tokens at C=1 and 170 at C=8, which is why C=1 has the larger median request latency. Request latency split by output band is in each campaign’s `analysis.json`.

## Sample limits

These are bounded pilots, not publication-grade tail estimates.

| C | Empirical request-tail resolution | Completed sessions |
|---:|---:|---:|
| 1 | 4.35% | 1 |
| 8 | 1.75% | 1 |
| 16 | 0.76% | 3 |
| 32 | 0.50% | 6 |
| 64 | 3.23% | 0 |
| 128 | 2.70% | 0 |

Report p50 and p95 as observed cell summaries. A p99 from 23–201 requests is the extreme order statistic of that cell. Five total completed sessions across C=1–C=16, and six at C=32, are too few to bootstrap an AgentX task-completion tail. The vLLM inter-token histogram cannot assign a gap above 1 s to a request or conversation because the export stores one average ITL per request.

The seven eligible traces also bound the claim. Longer runs recycle those traces. They do not sample the 386 traces rejected by the 65,536-token cap, including the wide fan-out traces whose peaks approach 1M tokens.

## Next measurement

Deep-sample **C=16, C=24, and C=32** on this same server and corpus. C=16 is the last moderate-tail point, C=24 fills the gap, and C=32 is the measured throughput maximum. Keep C=64 and C=128 out of any sustainable-capacity comparison. A stable request p99 needs about 1,000 completed requests per point, and an affected-session rate needs per-request token timestamps plus enough finished conversations to resample.
