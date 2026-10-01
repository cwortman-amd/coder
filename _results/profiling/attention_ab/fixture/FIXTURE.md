---
type: Profiling Report
title: Paged-decode fixture (27 Sep 2026)
description: One decode query, 24 query heads, 4 KV heads, head size 256, bf16 Q/K/V,
  scale 1/sqrt(256), causal, no sliding window, no sinks. The same logical.
tags:
- profiling-report
- attention-ab
- fixture
status: stable
---

# Paged-decode fixture (27 Sep 2026)

One decode query, 24 query heads, 4 KV heads, head size 256, bf16 Q/K/V,
scale `1/sqrt(256)`, causal, no sliding window, no sinks. The same logical
tokens are packed into `[blocks, kv_heads, page, 2 * head]` and viewed as
`[blocks, page, kv_heads, head]` for K and V. Triton uses page 784. AITER
unified uses page 832. The reference is FP32 softmax over that logical
sequence. Times below are CUDA-graph replays, not the rocprof 11.64 µs gate.

Script: `scripts/attn_page_fixture.py`. Rows: `page_fixture.json`.

## Indexing

A single key position is spiked so the output must equal that position's V.
This covers the last token and both sides of pages 784 and 832, contexts 32
through 16,384, batches 1, 4, and 8, and mixed lengths.

Both kernels match the reference. The largest spike error is 3.5e-4 at 16,384
tokens. On every spike the two kernels match each other exactly (max abs 0).
A wrong page gather would have missed the spike. That did not happen at 784,
832, or the short context of 44 tokens where `reasoning_00` diverges.

## Random bf16

| Case | Triton max abs vs FP32 | AITER max abs vs FP32 |
|---|---:|---:|
| C1, 1,024 | 5.5e-4 | 5.8e-4 |
| C1, 8,192 | 2.2e-4 | 2.2e-4 |
| C8, 1,024 | 6.1e-4 | 6.8e-4 |
| C4 mixed 512–8,192 | 1.1e-3 | 9.8e-4 |

The largest Triton-versus-AITER gap on these outputs is 2.0e-3. Neither
backend is a page-index outlier relative to the reference.

## Graph-replay latency, one query token

| Shape | Triton | AITER unified |
|---|---:|---:|
| C1, 44 | 12.9 µs | 12.7 µs |
| C1, 1,024 | 13.3 µs | 13.2 µs |
| C1, 8,192 | 36.2 µs | 30.7 µs |
| C1, 16,384 | 60.7 µs | 40.4 µs |
| C8, 1,024 | 23.2 µs | 24.6 µs |
| C4, 8,192 | 54.7 µs | 76.9 µs |
| C8, 8,192 | 133.5 µs | 111.3 µs |

At C1 1K both sit next to the traced 11.64 µs kernel. AITER is faster at C1
16K and slower at C4 8K. No one kernel wins every measured shape.

## Decision

A new HipKittens paged-decode kernel is not the next step. Both existing
unified kernels already gather the hybrid page, agree with an independent
reference, and are within a few microseconds at low-batch 1K. The attention/KV
category is 2.11% of profiled duration, so replacing it does not open another
20% on the current C1 1K workload.

The greedy gaps are not a bad page lookup. They remain a multi-layer numerical
difference. The shape split in the latency table is the concrete item to
profile if a kernel change is revisited later.
