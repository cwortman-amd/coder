# EngineCore kernel trace (27 Sep 2026)

## Result

**Kernel-level tracing now works.** The API parent is not profiled. Only the
Python `--multiprocessing-fork` child that becomes `VLLM::EngineCore` is
launched under `rocprofv3`.

Decode-dominant run:

- Frozen HF control: `VLLM_ROCM_USE_AITER` unset, stock `M_LEQ_8`, O2 graphs.
- Workload: C1, 1K input / 1K output, one request, `ignore_eos`.
- Profiled throughput: **57.35 tok/s** versus unprofiled campaign
  **79.35 ± 0.57 tok/s**. The trace is attribution evidence, not a performance
  point.
- Window: `20260927T102334Z/`, 1,530,943 dispatches, 81 kernel names.

Artifacts:

- `20260927T102334Z/benchmark_kernel_trace.perfetto.json.gz` — open in
  [Perfetto](https://ui.perfetto.dev) after decompressing.
- `20260927T102334Z/benchmark_kernel_trace.csv.gz` — row-level rocprof trace.
- `20260927T102334Z/benchmark_kernel_summary.json` — p50/p95/p99, totals, and
  robust category/kernel rankings.
- `20260927T102334Z/bench_c1_1k_1k.json` — profiled serving result.

The uncompressed full-startup CSVs were removed after clipping; they were
approximately 810 MB and are not needed to inspect the benchmark window.

## Why dynamic attach failed

1. `vllm serve` (API parent) starts with `ROCP_TOOL_ATTACH=1` and initializes
   ROCprofiler registration before EngineCore spawn.
2. vLLM starts EngineCore with Python `spawn`; CPython copies the parent's live
   environment without a vLLM filter.
3. EngineCore inherits `ROCPROFILER_REGISTER_LIBRARY` and maps
   `librocprofiler-register.so`/`librocprofiler-sdk.so`.
4. rocprofiler-register treats non-empty `ROCPROFILER_REGISTER_LIBRARY` as an
   active tool. At HSA registration it therefore skips loading
   `librocprofiler-sdk-attach.so`, even when child `sitecustomize` restores
   `ROCP_TOOL_ATTACH=1` before HIP initialization. No `rocp-bg-attach` thread
   is created, so `rocprofv3 --pid` correctly refuses before tracing.

`CAP_SYS_PTRACE` does not solve the missing attach thread. Wrapping
`vllm serve` traces the wrong process (parent HIP init only).

## Working launch route

`scripts/rocprof_attach_sitecustomize.py` changes Python's multiprocessing
executable in PID 1 when `ENGINECORE_ROCP_EXEC=1`.
`scripts/enginecore_rocprof_exec.sh` passes resource-tracker/helper processes
directly to Python and wraps only an invocation containing
`--multiprocessing-fork`:

```bash
stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "_results/profiling/enginecore_exec/${stamp}"
ENGINECORE_ROCP_EXEC=1 \
ENGINECORE_ROCPROF_DIR="/results/profiling/enginecore_exec/${stamp}" \
  ./scripts/launch_vllm_mxfp4.sh
```

After the benchmark, stop the server once so rocprof flushes CSV, then restore
the standard launch with all profiler variables unset.

This selector is validated for TP=1. At TP>1, additional MultiprocExecutor
workers can also use `--multiprocessing-fork` and would need explicit role
selection or separate output names.

## Decode-window ranking

The table uses each kernel's p99-winsorized total to prevent a few
profiler/stop outliers (hundreds of milliseconds) from dominating. Shares are
of summed GPU dispatch duration; queues can overlap, so shares are attribution
weights rather than wall-clock percentages.

| Category | Robust share |
|---|---:|
| MXFP4 GEMM + split-K reduction | **49.18%** |
| Attention + KV kernels | **20.02%** |
| Compiled elementwise / norm | **12.15%** |
| MXFP4 activation quantization | **8.38%** |
| Large split-K, likely LM head | **5.13%** |
| GDN / recurrent state | **3.64%** |
| Other + copy + sampling | **1.50%** |

Top kernels:

| Kernel | p50 | p95 | Robust share |
|---|---:|---:|---:|
| `_gemm_afp4wfp4`, M8/N64/K512, split-K 2 | 19.36 µs | 37.04 µs | **35.38%** |
| `kernel_paged_attention_2d` | 173.45 µs | 222.09 µs | **19.58%** |
| `dynamic_mxfp4_quant`, M1 | 3.88 µs | 4.08 µs | **8.34%** |
| `_gemm_afp4wfp4_reduce`, split-K 2 | 3.88 µs | 4.04 µs | **6.57%** |
| `wvSplitK_hf_sml` | 727.40 µs | 735.44 µs | **5.13%** |
| `_gemm_afp4wfp4`, split-K 4 | 10.44 µs | 10.84 µs | **4.73%** |
| GDN packed recurrent decode | 6.44 µs | 6.60 µs | **2.18%** |

## Interpretation

The missing time is not one gate GEMM. The captured full-model path is
dominated by the combination of MXFP4 linears (including quant/reduction),
paged attention whose cost grows with KV depth, and many compiled
elementwise/norm launches. This explains why globally replacing the isolated
M=1 GEMM did not improve serving.

Promotion still requires unprofiled graph-enabled serving against
**79.35 ± 0.57 C1** and **553.58 ± 2.97 C8**. The next optimization candidates
are shape-specific MXFP4 dispatch and attention/KV work; both must be tested
end-to-end.

## Normalized accounting

Shares are of **summed GPU dispatch duration**, not end-to-end wall time and
not HBM traffic. Queues can overlap. Categories in
`benchmark_kernel_summary.json` are disjoint and sum to 1. `wvSplitK_hf_sml`
is `large_splitk_likely_lm_head` at **5.13%**. It is not inside the **49.18%**
MXFP4 GEMM/reduction bucket, so those two shares may be added once. The kernel
name is consistent with unpacked `lm_head.weight` `[248320, 5120]`, and that
identity is still not a reason to tune it yet.

“25% faster” means a 1.25× rate, which removes 20% of that component’s time
(`1 - 1/1.25`), not 25%. Applied to these disjoint shares:

| Target | Traced share | Proposed improvement | Time removed |
|---|---:|---:|---:|
| MXFP4 GEMM/reduction | 49.18% | 25% faster | 9.84% |
| Attention/KV | 20.02% | 25% faster | 4.00% |
| Elementwise/norm + quantization | 20.53% | 30% faster | 4.74% |
| Large split-K / likely LM head | 5.13% | 20% faster | 0.86% |
| Combined | | | **19.4%** |

`1 / (1 - 0.194) ≈ 1.24` applied to the unprofiled 79.35 tok/s control is
about **98.5 tok/s**. That is an Amdahl illustration. The profiled run was
57.35 tok/s and may have changed relative costs, so it is not a forecast.

A 2× speedup on the fused gate-up’s 16.57% share removes at most 8.285% of
traced GPU duration (`1 / (1 - 0.08285) ≈ 1.09`), or about **86.5 tok/s** from
the control if that share transfers and nothing else changes.

### M=8 GEMM identity

`Grid_Size_X` is a thread count. Workgroup size is 256. Triton launches
`NUM_KSPLIT * cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)` programs, and decode uses
`BLOCK_M=8`, `BLOCK_N=64`, so `N = Grid_Size_X / 256 / NUM_KSPLIT * 64`.
`get_splitk` then separates equal-N projections by packed K. Call counts match
the checkpoint layer types (64 MLP layers, 48 `linear_attention`, 16
`full_attention`): 47349/48 = 15783/16 = 986.4375, and the 64-layer count
63131 is one call short of 48+16. Identities use the image
`packed_modules_mapping` plus those shapes, not grid size alone.

| Logical N | Runtime split-K | Calls | Layers | Projection |
|---|---:|---:|---:|---|
| 34816 | 2 | 63131 | 64 | fused `gate_up_proj` (`gate`+`up`, K=5120). Requested split-K 4 is reduced to 2 |
| 5120 | 2 | 63131 | 64 | `down_proj` (K=17408). Same N as `out_proj`, different K |
| 16384 | 2 | 47349 | 48 | fused `in_proj_qkvz` (10240+6144) |
| 5120 | 4 | 63131 | 64 | attention `out_proj` (K=6144 stays at split-K 4) |
| 96 (two N-tiles) | 2 | 47349 | 48 | fused `in_proj_ba` (48+48) |
| 14336 | 2 | 15783 | 16 | fused full-attention `qkv_proj` (12288+1024+1024) |

Fused gate-up is 16.57% of winsorized dispatch duration (p50 36.68 µs).
`down_proj` is the separate 8.84% bucket and must be tuned on its own.
