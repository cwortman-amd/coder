# Exact-shape MXFP4 dispatch (27 Sep 2026)

Two candidates. Each one replaced only `M_LEQ_8` for one `(N, K)` via AITER’s
`GEMM-AFP4WFP4-N=…-K=….json` lookup. `DEFAULT.json` stayed stock.
`VLLM_ROCM_USE_AITER` stayed unset. Graphs were the normal O2 piecewise
capture. Neither candidate is promoted.

Eager M=1 times include host launch overhead. Split-K 1 also skips the reduce
launch, so the eager ratio is larger than the GPU-only gap inside a captured
graph.

## Fused gate-up (`N=34816`, `K=5120`)

Isolated M=1, bit-exact versus library default (`max|d|=0`):

| Config | Eager ms | vs library default |
|---|---:|---:|
| library default (runtime split-K 2) | 0.0651 | 1.00× |
| explicit stock `M_LEQ_8` | 0.0559 | 1.16× |
| split-K 1, `num_warps=2`, other tiles stock | **0.0344** | **1.90×** |

Serving file: `configs/aiter/gfx950/GEMM-AFP4WFP4-N=34816-K=5120.json`.
`_get_config(1, 34816, 2560)` returned `tuned=True`, `NUM_KSPLIT=1`,
`num_warps=2`. Down, out, and qkvz stayed on stock `M_LEQ_8` (split-K 4,
4 warps). Prefill `M=1024` still resolved to the stock `any` tile.

Unprofiled C1 1K/1K, 4 prompts: **79.42 tok/s**. Campaign control is
**79.35 ± 0.57**. That is inside the noise band. Stop. Do not reuse these
warps on another projection.

## Down-projection (`N=5120`, `K=17408`)

The gate-up winner did not transfer. Best isolated config kept 4 warps and
changed only `NUM_KSPLIT` to 1: **0.0325 ms** versus library default 0.0660 ms
(2.03×) and explicit stock 0.0556 ms. Bit-exact.

Serving file: `configs/aiter/gfx950/GEMM-AFP4WFP4-N=5120-K=17408.json`.
Dispatch check: down `tuned=True`, split-K 1, 4 warps; gate-up and out stayed
stock split-K 4.

Unprofiled C1 1K/1K: **75.35 tok/s** (−5.0% vs 79.35). Stop. This matches the
earlier global `NUM_KSPLIT=1` serving regression (75.93) even though only
down-projection changed.

## Not run

C8, TTFT, and p95 token-arrival were not collected. Throughput already failed
the promotion bar. ASM was not compared: `use_asm_gemm` is process-wide, and
`VLLM_ROCM_USE_AITER=1` remains rejected.
