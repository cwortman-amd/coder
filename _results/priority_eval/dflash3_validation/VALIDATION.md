# DFLASH-3 promotion-gate validation (2026-09-25)

## Decision

DFLASH-3 is **not** promoted as the default interactive profile. It is the
preferred optional DFlash setting only for low-concurrency, long-output
throughput workloads. The production control remains the stock HF Quark MXFP4
configuration with `VLLM_ROCM_USE_AITER` unset and stock `M_LEQ_8`.

## Greedy equality

The 116-prompt text-only corpus used sequential C1 decoding with
`temperature=0`, `top_p=1`, `top_k=-1`, and `seed=1234`.

| Comparison | Exact token IDs and text |
|---|---:|
| DFLASH-3 vs MXFP4 control | **101/116 (87.1%)** |
| DFLASH-3 vs DFLASH-7 | **108/116 (93.1%)** |
| DFLASH-3 vs control, JSON/tool subset | **15/15** |

DFLASH-3 and DFLASH-7 have the same 15 mismatch prompt IDs versus the
sequential control. This fails the strict equality gate; it does not by itself
show that unverified tokens escaped.

Artifacts:

- `_results/quality/dflash3_greedy.json`
- `_results/quality/compare_baseline_vs_dflash3.json`
- `_results/quality/compare_dflash7_vs_dflash3.json`

## Matched C1 streaming latency

Protocol: 24 sequential requests, 1K input / 256 output, token arrival recorded
with `stream_interval=1`.

| Metric | Control | DFLASH-3 | Change |
|---|---:|---:|---:|
| Output throughput | 82.56 tok/s | 97.41 tok/s | **+18.0%** |
| TTFT p95 | 58.55 ms | 127.45 ms | **2.18× worse** |
| Mean token ITL | 11.95 ms | 9.80 ms | Better from burst delivery |
| Token ITL p95 | 12.07 ms | 18.84 ms | **+56.1% worse** |
| Chunk ITL p95 | 12.07 ms | 18.89 ms | **+56.6% worse** |

Artifacts:

- `_results/priority_eval/dflash3_validation/control_stream_c1_n24.json`
- `_results/priority_eval/dflash3_validation/stream_c1_n24.json`

## Throughput context

The existing 1K/1K depth sweep measured DFLASH-3 at **113.75 tok/s C1**
(**+44.3%** versus its matched control). The throughput crossover remains
between C5 and C6. These longer-output results do not override the failed
equality and interactive-latency gates.

## Next quality work

Measure production-sampling task scores and compare MXFP4 with FP8/BF16.
