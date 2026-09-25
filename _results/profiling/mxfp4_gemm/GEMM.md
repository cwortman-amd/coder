# Isolated AITER MXFP4 GEMM vs Babel Read (25 Sep 2026)

Canonical: [`docs/MI350P_MXFP4_VLLM_EVAL.md`](../../docs/MI350P_MXFP4_VLLM_EVAL.md). Babel roof: [`C1_BABEL_ROOFLINE.md`](C1_BABEL_ROOFLINE.md).

Same-process `torch.ops.vllm.gemm_with_dynamic_quant` on **real** Qwen3.8-27B layer-0 shards. No EngineCore. Reproduce:

```bash
bash scripts/bench_mxfp4_gemm.sh   # stops :8000, restores HF control
```

## What ran

| Item | Result |
|---|---|
| Image default `VLLM_ROCM_USE_AITER` | **false** — `AiterMxfp4LinearKernel` still selected, **Triton** `gemm_afp4wfp4` |
| `VLLM_ROCM_USE_AITER=1` | `use_asm_gemm=true`, gfx950 `f4gemm_*_BpreShuffle` hsaco |
| Babel Read | 3793 GB/s |
| rocprof | **worked** (python is not EngineCore): `_results/profiling/mxfp4_gemm/rocprof/` |

Layer-0 packed sizes (after `process_weights_after_loading`):

| Layer | N × K | W+scale |
|---|---|---:|
| mlp.gate / up | 17408 × 5120 | 0.0473 GB |
| mlp.down | 5120 × 17408 | 0.0473 GB |
| linear_attn.out_proj | 5120 × 6144 | 0.0167 GB |
| linear_attn.in_proj_qkv | 10240 × 5120 | 0.0279 GB |

## Triton path (production default)

Wall-clock HIP events, 200 timed iters (includes dynamic MXFP4 quant of activations):

| Layer | M=1 GB/s | vs Babel | M=8 | M=64 |
|---|---:|---:|---:|---:|
| mlp.gate_proj | **469** | **12.4%** | 472 | 636 |
| mlp.up_proj | 460 | 12.1% | 465 | 631 |
| mlp.down_proj | 474 | 12.5% | 469 | 633 |
| out_proj | 170 | 4.5% | 170 | 224 |
| in_proj_qkv | 277 | 7.3% | 277 | 373 |

M=1–8 sit on a **~100 µs launch floor**. Larger M only helps a little (still ≤17% Babel on MLP).

rocprofv3 on M=1 gate_proj (51 GEMM dispatches). **GPU time only:**

| Kernel | Calls | Avg ns | Share of GPU |
|---|---:|---:|---:|
| `_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_8_…_NUM_KSPLIT_2` | 51 | **22.0 µs** | **81.6%** |
| `_dynamic_mxfp4_quant_kernel_BLOCK_SIZE_M_1_…` | 51 | 2.3 µs | 8.7% |
| `_gemm_afp4wfp4_reduce_kernel_…_KSPLIT_2` | 51 | 1.9 µs | 7.2% |

Weight-stream using **GEMM GPU time only**: \(0.0473\,\mathrm{GB} / 22.0\,\mu\mathrm{s} \approx \mathbf{2150}\,\mathrm{GB/s}\) (**57% of Babel Read**). Wall 469 GB/s is mostly **host/launch**, not HBM. Tile is `BLOCK_M=8` even at M=1 (padding). `NUM_KSPLIT=2` forces an extra reduce.

## ASM path (`VLLM_ROCM_USE_AITER=1`)

AITER logged **no tuned config** for `(M=1, N=17408, K=5120)` / `(M=1, N=5120, K=17408)` and used default `f4gemm_bf16_per1x32Fp4_BpreShuffle_*`.

| Layer | M=1 wall GB/s | vs Babel | vs Triton M=1 |
|---|---:|---:|---:|
| gate_proj | **1035** (0.046 ms) | 27.3% | **2.2×** |
| down_proj | 999 (0.047 ms) | 26.3% | **2.1×** |

Still far from 3793 GB/s on the wall clock. Untuned default tiles.

## End-to-end serving (do not promote)

HF recipe + `VLLM_ROCM_USE_AITER=1`, 1K/1K `ignore_eos`:

| | Control (AITER unset) | AITER=1 | Delta |
|---|---:|---:|---:|
| C1 | **79.53** | **71.40** | **−10.2%** |
| C8 (warm) | **553.58** mean | **518.82** | **−6.3%** |

Logs still say `AiterMxfp4LinearKernel`. Isolated ASM GEMM is faster; **full decode is slower** (untuned hsaco + graph/GDN interaction). Keep production at image default (`VLLM_ROCM_USE_AITER` unset/false).

## Bottleneck (aligned with serving ablations)

Eager M=1 GEMM and graph-enabled serving **do not move together**. Isolated Triton GPU ~22 µs vs wall ~101 µs; ASM and `NUM_KSPLIT=1` both sped the isolated GEMM and **slowed C1**. Babel Read **3793 GB/s** is available streaming bandwidth. C1 **79.53 tok/s ≈ 1.53 TB/s** only under a full 17.91 GiB stream/step model (**40%** of that roof). Do **not** assign the unused 60% to pack/unpack or gate_proj without an EngineCore timeline.

## Next

1. Freeze HF control. Promotion gate = unprofiled graph-enabled C1/C8, not eager GEMM.
2. Captured quant+GEMM+reduce replay (all representative shapes), still judged by serving.
3. EngineCore-native trace only if it hits the GPU-owning process.
4. DFLASH-3 greedy equality + TTFT/p95.

JSON: `summary.json`, `summary_aiter1.json`, `aiter1_c1.json`, `aiter1_c8.json`, `tile_sweep.json`, `ksplit1_c1.json`, `ksplit1_c8.json`, `kernel_top.json`.
