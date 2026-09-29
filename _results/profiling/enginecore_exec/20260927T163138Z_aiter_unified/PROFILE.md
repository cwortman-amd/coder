# AITER unified attention kernel trace (27 Sep 2026)

Attribution only. Profiled one-request 1K/1K throughput was **62.03 tok/s**.
The unprofiled repeats are **99.58 ± 0.05 C1** and **653.96 ± 0.44 C8**.

`VLLM_ROCM_USE_AITER` was unset. Startup still logged
`Using AiterMxfp4LinearKernel for MXFP4 GEMM`, the same Quark linear kernel
the stock path uses. The only requested change was
`--attention-backend ROCM_AITER_UNIFIED_ATTN`. Graphs were
`FULL_AND_PIECEWISE`.

The backend first selects a 64-token KV block, then the hybrid Mamba
alignment raises the attention page to **832** tokens
(`kv lcm block sizes 832`). The nominal 64 does not survive on this model.

Clipped benchmark window, decode kernel:

| Kernel | Calls | p50 | p95 |
|---|---:|---:|---:|
| `kernel_unified_attention_3d` `BLOCK_SIZE_832` `ALL_DECODE_1` | 15,281 | **9.00 µs** | **9.68 µs** |
| `reshape_and_cache_flash_kernel` | 15,313 | 4.92 µs | 5.00 µs |
| `kernel_unified_attention_3d` `ALL_DECODE_0` (prefill) | 16 | 276.30 µs | 282.22 µs |

Attention/KV robust share of summed dispatch duration: **1.96%**.
Raw CSV was deleted after clipping. Row trace:
`benchmark_kernel_trace.csv.gz`.
