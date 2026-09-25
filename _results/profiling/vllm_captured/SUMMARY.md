# vLLM captured mini-decode

- UTC: `2026-09-25T17:18:27Z`
- Status: **completed**
- Live `:8000` detected: `False`
- GPU free/total GiB: `143.67` / `143.98`
- Compile path: vLLM `VLLM_COMPILE` (mode 3/O2), `FULL_AND_PIECEWISE`, capture sizes 1 and 8
- Control: `VLLM_ROCM_USE_AITER` unset; stock `M_LEQ_8`; no overlay

## Result

vLLM captured M=1 and M=8 decode windows completed.

## Decode windows

| M | generated | steady window ms | tok/s |
|---:|---:|---:|---:|
| 1 | 16 | 246.227 | 60.92 |
| 8 | 128 | 914.061 | 131.28 |

## Projection ranking

| Rank | Projection | attributed GPU ms | evidence |
|---:|---|---:|---|
| 1 | lm_head | 38.0385 | capture shape + profiler external-id kernel correlation |
| 2 | mlp.gate_proj | 9.0406 | shared fused gate_up_proj; values are not additive |
| 3 | mlp.up_proj | 9.0406 | shared fused gate_up_proj; values are not additive |
| 4 | mlp.down_proj | unavailable | unavailable: no projection-to-replay-kernel correlation |
| 5 | attention.projections | unavailable | unavailable: no projection-to-replay-kernel correlation |

Gate and up are represented separately in the report but share the same fused `gate_up_proj` measurement when vLLM merges them.

## Limitations

- This is an intermediate captured-decode screen, not the unprofiled serving promotion gate.
- PyTorch module hooks do not execute during full-graph replay. Projection ranking therefore requires capture-shape/external-id correlation; the script reports unavailable if ROCm omits it.

## CUDA kernel ranking (torch profiler, perturbed)

Source: `torch_profiles/profiler_out_0.txt`. Self CUDA total **433 ms**. Captured M=1 was **60.92 tok/s** (profiler-on, same band as rocprof wrap 60–65), **not** the unprofiled 79.35 campaign mean.

| Kernel | Self CUDA | Share | avg | calls |
|---|---:|---:|---:|---:|
| `_gemm_afp4wfp4_kernel` BLOCK_M=8 | 171.9 ms | **39.7%** | 22.4 µs | 7680 |
| `_dynamic_mxfp4_quant_kernel` BLOCK_M=8 | 38.4 ms | 8.9% | 4.0 µs | 9728 |
| `_gemm_afp4wfp4_reduce_kernel` | 30.1 ms | 6.9% | 3.9 µs | 7680 |
| `_rocm_C::wvSplitK` (likely LM-head) | 25.4 ms | 5.9% | 746 µs | 34 |
| extra `_gemm_afp4wfp4_kernel` BLOCK_M=8 | 21.6 ms | 5.0% | 10.5 µs | 2048 |
| `fused_recurrent_gated_delta_rule_packed_decode` | 17.6 ms | 4.1% | 12.2 µs | 1440 |

MXFP4 GEMM + quant + split-K reduce are about **half of profiled CUDA time**. GDN packed decode is ~4%. The projection-table `lm_head` 38 ms line is **not** the preferred ranking; it likely collided with the quant kernel's 38.4 ms. Down-proj and attention projections did not correlate. This remains an intermediate, profiler-on timeline—not the serving promotion gate.

