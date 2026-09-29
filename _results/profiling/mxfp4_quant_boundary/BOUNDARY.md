# MXFP4 quantization-boundary check (27 Sep 2026)

Elementwise/norm is 12.15% of traced GPU dispatch duration and dynamic MXFP4
quantization is 8.38%. Together they are 20.53%. The installed fusion passes
do not cover this checkpoint.

`fuse_act_quant` matches SiLU-and-mul plus FP8 or NVFP4 quantization.
`fuse_norm_quant` matches RMSNorm plus the same FP8/NVFP4 ops. Neither pattern
matches AITER `dynamic_mxfp4_quant`. Turning those flags on does not fuse the
Quark MXFP4 launches. The GDN output comment that mentions `fuse_norm_quant`
is describing that FP8/NVFP4 pass, not an MXFP4 kernel available here.

## What can be reused

`dynamic_mxfp4_quant` is a pure function of the activation. Two calls on the
same `[1, 5120]` BF16 tensor produced identical packed values and scales.

ROCm `QwenGatedDeltaNetAttention.forward_hip` calls both projections on that
same tensor:

```text
in_proj_qkvz(hidden_states)
in_proj_ba(hidden_states)
```

Each linear then calls `gemm_with_dynamic_quant`, which quantizes again when
`x_scales` is absent. The second quant does not change scales or outputs.
This is the only same-source pair in the decode step:

| Projection pair | Same source? | Reuse |
|---|---|---|
| `in_proj_qkvz`, `in_proj_ba` | Yes, post-norm hidden state, 48 linear layers | Safe |
| Fused `qkv_proj` | One projection | Already one quant |
| Fused `gate_up_proj` | One projection | Already one quant |
| `down_proj` | SiLU(gate) × up, not the norm output | Must quantize separately |
| `out_proj` | Attention output, not the norm output | Must quantize separately |

Per token that is 48 duplicate quants out of about 304
(`48 × 5 + 16 × 4`). At equal quant cost that is 1.3% of traced GPU duration,
or about 80.4 tok/s from 79.35 if nothing else changes. The op already accepts
precomputed `x_scales`, but the linear caller does not pass them.

A cache keyed by `data_ptr` is not safe. CUDA-graph replay reuses the same
storage with new values, so a pointer cache would replay a stale quantization.

## Decision

No fusion flag was enabled and no serving candidate was launched. The proven
reuse is real and small. It does not stack with the rejected gate-up or
down-projection split-K overrides. The serving result that already clears
100 tok/s is `TRITON_ATTN`, documented in
`_results/profiling/attention_ab/ATTENTION.md`.
