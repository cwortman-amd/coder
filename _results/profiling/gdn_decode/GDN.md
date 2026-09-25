# Isolated GDN decode vs C1 (25 Sep 2026)

Canonical: [`docs/MI350P_MXFP4_VLLM_EVAL.md`](../../docs/MI350P_MXFP4_VLLM_EVAL.md).

Qwen3.8-27B: **48 / 64** layers are `linear_attention` (interval 4). Production decode on this image is **not** fused CUDA and **not** AITER KDA.

| Path | Status on this stack |
|---|---|
| `VLLM_GDN_DECODE_KERNEL=cuda` → `fused_gdn_decode_post_conv_mtp` | **Not built** (`torch.ops._C` missing). Falls back to Triton/FLA. |
| `forward_hip` / AITER `fused_kda_decode` | Requires `VLLM_ROCM_USE_AITER=1` **and** `gqa_interleaved_layout` (Qwen3-**Next**). This checkpoint is Qwen3.5 **non-interleaved**, so AITER still unpacks and calls the generic FLA path. |
| Packed FLA (`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=True`) | **What C1 actually runs:** `causal_conv1d_update` + `fused_recurrent_gated_delta_rule_packed_decode` |

Reproduce: `python3 scripts/bench_gdn_decode.py` in the vLLM ROCm image (stops :8000 if you wrap it).

## Results (eager, Qwen GDN shapes)

`H_k=16`, `H_v=48`, `d=128`, `qkv=10240`, conv width 4.

| M | conv wall | recurrent wall | pair wall | ×48 layers |
|---:|---:|---:|---:|---:|
| 1 | 0.024 ms | 0.029 ms | **0.053 ms** | **2.56 ms/tok** |
| 8 | 0.024 ms | 0.029 ms | 0.054 ms | 2.57 ms/tok |

M=1–8 sit on the same ~50 µs launch floor (same pattern as MXFP4 GEMM).

rocprofv3 (M=1, same process):

| Kernel | Avg GPU | Share of GPU in this microbench |
|---|---:|---|
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | **1.59 µs** | 49% |
| `_causal_conv1d_update_kernel` | **1.16 µs** | 35% |

GPU-only GDN: **~2.75 µs/layer × 48 = 0.13 ms/token** (~1% of C1’s 12.6 ms). Wall 2.56 ms is launch; graphs already hide most of that (eager C1 is 0.18×).

## Implication

Missing `fused_gdn_decode_post_conv_mtp` is **not** the 79 vs 197 tok/s gap. Fusing conv+recurrent would save ~1–2 µs GPU and one launch per layer; even perfect GDN fusion cannot create ~100 tok/s.

Do **not** turn on `VLLM_ROCM_USE_AITER=1` to “get GDN AITER” on this model: interleaved fast-path does not apply, and AITER=1 already **regressed** C1 via ASM MXFP4.

C1 time is still dominated by **MXFP4 linear GEMM GPU work** (order 8 ms if ~378 weight tiles × ~22 µs) plus 16× `ROCM_ATTN` full-attention layers, norms, and graph bubbles — not GDN.

## Next

GDN fusion is not a C1 promotion candidate on GPU-time grounds (~0.13 ms/tok). Serving remains the gate. Next campaign items: DFLASH-3 greedy/TTFT; captured full-decode reproducer — not more eager GDN microbenches.

JSON: `summary.json`, `kernel_top.json`, `rocprof/gdn_m1_kernel_stats.csv`.
