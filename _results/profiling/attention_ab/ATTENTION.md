# MI350P attention-backend A/B (27 Sep 2026)

The 14.9× decode-kernel ratio is `TRITON_ATTN` versus stock `ROCM_ATTN`
(`kernel_paged_attention_2d` 173.45 µs p50 versus `kernel_unified_attention`
11.64 µs). It is not a comparison with `ROCM_AITER_UNIFIED_ATTN`. Attention
backend and MXFP4 linear kernel are separate switches. Every lane below kept
`VLLM_ROCM_USE_AITER` unset and logged `AiterMxfp4LinearKernel`.

The three-way isolation is in `ABC.md`.

## Serving result

Standard promotion workload is 1K input / 1K output. Control values are the
existing three-run campaign. Triton values are three new warmed runs.

| Backend | C1 tok/s | C8 aggregate tok/s | C1 vs control | C8 vs control |
|---|---:|---:|---:|---:|
| Auto `ROCM_ATTN` control | 79.35 ± 0.57 | 553.58 ± 2.97 | — | — |
| `ROCM_AITER_UNIFIED_ATTN` | 99.31 | 654.83 | +25.1% | +18.3% |
| `TRITON_ATTN` | **100.36 ± 0.04** | **660.53 ± 8.69** | **+26.5%** | **+19.3%** |

The 1K-prefill to 2K-decode streaming discriminator was measured after one
full warm run:

| Backend | Output tok/s | TTFT p50 / p95 | Token-arrival p95 |
|---|---:|---:|---:|
| `ROCM_ATTN` | 73.90 | 46.24 / 60.09 ms | 15.05 ms |
| `ROCM_AITER_UNIFIED_ATTN` | 99.31 | 48.21 / 61.24 ms | 10.10 ms |
| `TRITON_ATTN` | **100.18** | **45.65 / 58.84 ms** | **10.04 ms** |

The first request after startup was excluded from the TTFT comparison because
it triggered one-time warmup/JIT work. One Triton stream run was also excluded:
the greedy-quality suite ran concurrently and contaminated its throughput.

Token-arrival time also depends on KV depth. On the warmed 1K-input/2K-output
streams, p50/p95 ITL in milliseconds is:

| Backend | Context 1.0–1.25K | Context 1.75–2.0K | Context 2.75–3.0K |
|---|---:|---:|---:|
| `ROCM_ATTN` | 11.87 / 12.07 | 13.22 / 13.42 | 15.01 / 15.21 |
| `ROCM_AITER_UNIFIED_ATTN` | 10.00 / 10.02 | 10.03 / 10.06 | 10.08 / 10.12 |
| `TRITON_ATTN` | 9.90 / 9.94 | 9.96 / 9.99 | 10.03 / 10.07 |

Default attention slows by 3.1 ms from the start of decode to a 3K context.
Both alternatives stay flat, so the gain is larger late in the request than
the 20.02% trace share predicted. A 25%-faster attention component at that
share would remove only about 4% of traced GPU time; the measured serving
gain is 26.5% at C1 because `ROCM_ATTN` is on the wall-time critical path.

A backend-specific EngineCore trace confirms the kernel mechanism. Stock
`kernel_paged_attention_2d` was 173.45/222.09 µs p50/p95 and 19.58% of robust
dispatch duration. Triton `kernel_unified_attention` is
**11.64/13.56 µs** and 1.58%. Attention/KV as a category falls from 20.02% to
**2.11%**, an 11.2× reduction in robust duration including cache update.
Detail: `../enginecore_exec/20260927T151210Z_triton_attn/PROFILE.md`.

These are unprofiled graph-enabled serving results. Unlike the failed
shape-specific split-K candidates, the trace-ranked attention change transfers
to end-to-end serving. The gain exceeds attention's 20% profiled share because
the 1K→2K workload gives attention more weight than the profiler-perturbed
average and because the trace share is summed dispatch duration, not wall time.

## Correctness and quality

The current `ROCM_ATTN` control reproduces the stored control **116/116**.
Each alternative backend is internally deterministic across repeated runs,
but changes reductions enough to alter greedy continuations:

| Backend | Exact token sequences vs current control |
|---|---:|
| `ROCM_AITER_UNIFIED_ATTN` | 98/116 (84.5%) |
| `TRITON_ATTN` | 98/116 (84.5%) |

The mismatch sets are not identical. Many diverge at token zero, so this is
not a stale control or random repeatability failure. It is an attention
numerics change.

Production-sampling diagnostic (`T=1`, top-p 0.95, top-k 20, seed 1234):

| Backend | Score | Failures |
|---|---:|---|
| `ROCM_ATTN` control | 30/32 | `math_gsm_01`, `math_gsm_06` |
| `TRITON_ATTN` | 29/32 | `math_gsm_03`, `math_gsm_06`, `reasoning_00` |

All 32 Triton responses were structurally valid. This 32-item diagnostic is
too small to establish a statistically meaningful quality regression, but it
does not clear a strict “same output” gate either.

## Decision

`TRITON_ATTN` stays opt-in. Its first-divergence screen is in
`divergence/GATE1.md`. The matched AITER comparison is in `ABC.md`: both
unified backends leave the slow paged decode kernel, Triton leads AITER by
0.8% at C1, and AITER's decode kernel (`kernel_unified_attention_3d`, 9.00 µs
p50) is the faster of the two named kernels. AITER matches the stock argmax
on four of the five large Triton gaps. Both still flip `reasoning_00` by more
than 3.6 nats. Keep `ROCM_ATTN` as the default.

The expanded predeclared qualification is complete:
`../../quality/attention_qualification/RESULT.md`. AITER passes paired greedy
and three-seed sampling quality but fails latency. Repeated 1K/1K runs isolate
the miss to the steady C4/C8 token step (**11.25 / 11.97 ms** versus the
frozen 11 ms ceiling); TTFT passes and server waiting remains zero. The prior
hot 8K TTFT difference follows the uncached page remainder under Mamba
`align`, not the 8,192 chunk limit, while cold AITER long prefill is still
15–17% slower. Triton fails the reasoning-quality gate and narrowly exceeds
the strict C8 1K ITL ceiling. Neither is promoted. See
`../../quality/attention_qualification/latency_campaign/1K1K.md` and
`../../quality/attention_qualification/aiter_latency/DIAGNOSIS.md`.

Do not combine this result with global AITER MXFP4 ASM or split-K overrides.
Those remain rejected. Do not use either fast backend as the baseline for
duplicate-quant reuse until a backend meets the product gates.
