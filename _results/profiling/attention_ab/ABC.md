---
type: Profiling Report
title: Three-way attention isolation (27 Sep 2026)
description: Same Quark MXFP4 checkpoint, same vLLM image, VLLMROCMUSEAITER unset,
  FULLANDPIECEWISE graphs, C1/C8 harness, temperature 0. Only.
tags:
- profiling-report
- attention-ab
- abc
status: stable
---

# Three-way attention isolation (27 Sep 2026)

Same Quark MXFP4 checkpoint, same vLLM image, `VLLM_ROCM_USE_AITER` unset,
`FULL_AND_PIECEWISE` graphs, C1/C8 harness, temperature 0. Only
`--attention-backend` changes. Startup on every lane logs
`Using AiterMxfp4LinearKernel for MXFP4 GEMM`.

`ROCM_AITER_UNIFIED_ATTN` does not need the AITER master switch. Its causal
forward calls `aiter.ops.triton.unified_attention`, which launches
`kernel_unified_attention_3d` / `_2d`. `TRITON_ATTN` calls vLLM's own
`unified_attention`, traced as `kernel_unified_attention`. Both are Triton
unified attention. Stock `ROCM_ATTN` decode is `kernel_paged_attention_2d`.

Hybrid page size after Mamba alignment:

| Backend | Log | Effective attention page |
|---|---|---:|
| `ROCM_ATTN` | attention block raised for the Mamba page | 784 |
| `TRITON_ATTN` | same hybrid rule, no 64-token preference | 784 |
| `ROCM_AITER_UNIFIED_ATTN` | prefers 64, then raises the page | **832** |

The "block size 64" line is not the page this model runs.

## Unprofiled serving

AITER figures are three new repeats. Triton and stock figures are the
existing matched campaign.

| Backend | C1 tok/s | C8 tok/s | Warmed 1K→2K TTFT p50/p95 | ITL p95 | ITL p50 at 1.0–1.25K / 1.75–2.0K / 2.75–3.0K |
|---|---:|---:|---:|---:|---|
| `ROCM_ATTN` | 79.35 ± 0.57 | 553.58 ± 2.97 | 46.24 / 60.09 ms | 15.05 ms | 11.87 / 13.22 / 15.01 ms |
| `ROCM_AITER_UNIFIED_ATTN` | **99.58 ± 0.05** | **653.96 ± 0.44** | 47.97 / 65.69 ms | 10.14 ms | 9.98 / 10.01 / 10.07 ms |
| `TRITON_ATTN` | **100.36 ± 0.04** | **660.53 ± 8.69** | 45.65 / 58.84 ms | 10.04 ms | 9.90 / 9.96 / 10.03 ms |

Triton's C1 lead over AITER unified is **0.78 tok/s (0.8%)**. Both C1 series
are tight, so that gap is stable. At C8, Triton's mean is higher, and one of
its three runs (650.49) sits below every AITER run, so the C8 lead is inside
Triton's own spread. Both unified backends stay flat from 1K to 3K context.
Stock `ROCM_ATTN` slows by about 3 ms over that range.

## Named decode kernel, 1K/1K window

Profiler-on throughput is not a serving point (stock 57.35, Triton 59.00,
AITER unified 62.03 tok/s).

| Backend | Decode kernel | p50 | p95 | Cache-update p50 | Attention/KV robust share |
|---|---|---:|---:|---:|---:|
| `ROCM_ATTN` | `kernel_paged_attention_2d` | 173.45 µs | 222.09 µs | (in the 20.02% bucket) | 20.02% |
| `TRITON_ATTN` | `kernel_unified_attention` | 11.64 µs | 13.56 µs | 3.88 µs | 2.11% |
| `ROCM_AITER_UNIFIED_ATTN` | `kernel_unified_attention_3d` `BLOCK_SIZE_832` | **9.00 µs** | **9.68 µs** | 4.92 µs | 1.96% |

AITER's decode kernel is faster than Triton's in this window. Adding the
cache write, AITER is 9.00+4.92 µs and Triton is 11.64+3.88 µs. The 14.9×
figure remains Triton versus the stock paged kernel. It does not describe
Triton versus AITER unified. The prefill-shaped AITER kernel
(`ALL_DECODE_0`, 16 calls, one per full-attention layer) was 276 µs p50 and
is not the decode path.

## First-divergence logits

At the 18 prefixes where Triton first leaves the stored control token, the
live AITER server picks the stored control token on 13. It picks Triton's
token on `rag_01` (0.13 nat), `reasoning_04` (0.25 nat), `reasoning_00`
(3.63 nat), and `long_8k_02`. `long_8k_02` now emits `needle` on the current
control as well. `math_10` leaves the shared prefix before Triton's split, so
that position is not a teacher-forced comparison.

Of the five Triton gaps of at least 2 nats, AITER matches the control argmax
on `code_08`, `json_tool_10`, `math_16`, and `math_18`. Both unified backends
emit `Let` instead of `5` on `reasoning_00`, with a margin above 3.6 nats.

## Decision

Keep `ROCM_ATTN` as the default. The large-margin `reasoning_00` flip is
shared by both fast backends.

Between the two fast backends, AITER unified is the closer match to the
stock argmax on the prompts that blocked Triton, and it gives up about 0.8%
of Triton's C1 throughput. The improvement over stock belongs to leaving
`kernel_paged_attention_2d` on this hybrid page, not to a Triton-versus-AITER
kernel rewrite.
