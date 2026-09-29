# Best-C1 kernel and memory profile: Triton attention

Configuration: one MI350P, Qwen3.8-27B Quark AWQ MXFP4, stock MXFP4
dispatch, `VLLM_ROCM_USE_AITER` unset, O2 `FULL_AND_PIECEWISE` graphs, and
explicit `--attention-backend TRITON_ATTN`.

The competitive unprofiled result is **100.36 ± 0.04 C1 tok/s**. EngineCore
profiling reduced this one-request 1K/1K run to **59.00 tok/s**. The profiler
number is attribution-only.

## Kernel result

The clipped benchmark window contains 1,597,961 dispatches and 81 kernel
names. Shares use each kernel's p99-winsorized summed dispatch duration.
Queues can overlap, so shares are attribution weights rather than wall time.

| Category | Stock `ROCM_ATTN` trace | Triton trace |
|---|---:|---:|
| MXFP4 GEMM/reduction | 49.18% | 59.85% |
| Elementwise/norm | 12.15% | 14.83% |
| Dynamic MXFP4 quant | 8.38% | 10.20% |
| Separate large split-K | 5.13% | 6.24% |
| GDN/state | 3.64% | 4.43% |
| **Attention/KV** | **20.02%** | **2.11%** |
| Other/copy/sampling | 1.50% | 2.35% |

The non-attention shares rise because attention was removed from the
denominator. Their kernels did not become slower.

The decode kernel comparison is direct:

| Backend kernel | Calls | p50 | p95 | Robust total | Full-trace share |
|---|---:|---:|---:|---:|---:|
| Stock `kernel_paged_attention_2d` | 15,815 | 173.45 µs | 222.09 µs | 2,745.6 ms | 19.58% |
| Triton `kernel_unified_attention` | 16,339 | **11.64 µs** | **13.56 µs** | **187.6 ms** | **1.58%** |

Triton is **14.9× faster at p50** and **16.4× faster at p95** for the named
decode kernel. Attention/KV robust duration including cache update drops from
2,807.3 to 250.8 ms, an **11.2× reduction**, even though the Triton window has
3.3% more decode calls. This backend-specific trace explains the 26.5%
unprofiled C1 gain and the flat late-decode ITL. It is not merely a graph-gap
effect.

The installed source explains the dispatch difference. `ROCM_ATTN` routes
through `chunked_prefill_paged_decode`; its native kernel supports only page
sizes 16 and 32, while this hybrid model requires a padded 784-token
attention/Mamba page. `TRITON_ATTN` is graph-aware and selects its 3D decode
kernel for low batch sizes. The trace confirms `kernel_unified_attention` is
actually selected.

## Memory-bandwidth interpretation

The practical streaming roof remains the independent three-run RVS Babel
Read median **3,793.4 GB/s**. Model loading reports **17.91 GiB**. If each C1
decode step reads that complete loaded weight set exactly once:

| Configuration | Conditional weight-stream rate | Fraction of Babel |
|---|---:|---:|
| Control, 79.35 tok/s | 1,526 GB/s | 40.2% |
| Triton, 100.36 tok/s | **1,930 GB/s** | **50.9%** |

This is a conditional lower-complexity model, not a measured DRAM byte count.
Weights can be reread, cached, or skipped differently; KV, scales, partials,
and state add traffic. Do not report 1,930 GB/s as an HBM counter.

`amd-smi metric` sampled the profiled request every 0.5 seconds. During the
19 samples with GFX activity at least 90%:

| Signal | Median | Range |
|---|---:|---:|
| GFX activity | 100% | 93–100% |
| UMC activity | 33% | 31–34% |
| Socket power | 406 W | 383–417 W |
| GFX clock | 2,197 MHz | 2,064–2,199 MHz |
| HBM clock | 2,000 MHz | fixed |

UMC activity is a device busy percentage, not calibrated GB/s. Together with
Babel and the conditional stream estimate, it rejects both “HBM is saturated”
and “raise MCLK/PPT” as explanations. Triton increases useful throughput while
HBM remains at the same clock and package power remains well below 600 W.

## Remaining bottleneck

After attention is reduced to 2.11%, the profile is dominated by:

- MXFP4 GEMM/reduction: **59.85%**.
- Elementwise/norm plus quantization: **25.03%**.
- Large split-K projection: **6.24%**.
- GDN/state: **4.43%**.

The failed exact-shape gate-up and down overrides still must not be mounted.
The only proven duplicate quantization is the 48-layer
`in_proj_qkvz`/`in_proj_ba` pair, worth about 1.3% of the old trace. The next
large engineering target is therefore a graph-native MXFP4 linear path that
reduces quant/reduction/launch overhead across multiple exact shapes, not
another global tile edit.

## Artifacts

- `benchmark_kernel_summary.json`: category and kernel statistics.
- `benchmark_kernel_trace.csv.gz`: clipped row-level dispatches.
- `benchmark_kernel_trace.perfetto.json.gz`: Perfetto import.
- `telemetry.jsonl` / `telemetry_summary.json`: 0.5-second host samples.
- `enginecore_kernel_stats.csv` / `enginecore_hip_api_stats.csv`: profiler
  aggregates.

The 459 MB raw kernel CSV and 4.9 MB raw HIP trace were deleted after clipping.
