---
type: Profiling Report
title: 'Real-tensor attention isolation: reasoning00 (27 Sep 2026)'
description: Prompt length 44. Stock emits token 20 (5). Triton and AITER unified
  emit token 9764 (Let).
tags:
- profiling-report
- attention-ab
- real-capture
- reasoning
- '00'
status: stable
---

# Real-tensor attention isolation: `reasoning_00` (27 Sep 2026)

Prompt length 44. Stock emits token 20 (`5`). Triton and AITER unified emit
token 9764 (`Let`).

The diagnostic launcher captured Q, K, V and the attention output for all 16
full-attention layers under each backend. The first full-attention layer is
model layer 3. Before that operation, Q, K, and V are bitwise identical across
all three servers.

At layer 3:

- Triton and AITER produce the same last-token attention output.
- Their output differs from stock by max 0.0078125 and RMSE 0.0003237.
- The stock output RMS is 0.36445, so the difference is 0.089% RMS.
- 321 of 6,144 bf16 output elements differ.
- Against CPU FP64 causal GQA on the captured tensors, last-token RMSE is
  0.0006558 for stock and 0.0006415 for each unified backend.

The unified output is slightly closer to the independent reference. There is
no large operator-level error and no page lookup involved at 44 tokens.

The small layer-3 rounding difference is amplified by the model. At layer 7,
after three intervening linear-attention layers, the last-token Q differs from
stock by 0.634 max. By layer 27 it differs by 2.06 and the attention output
differs by 7.25. At layer 63 the stock-versus-Triton output gap reaches 17.72.

Triton and AITER remain effectively the same path: their captured Q is
bitwise equal through layer 59. Their attention outputs are usually bitwise
equal; isolated differences are at bf16-scale. At layer 63 their Q differs by
0.717 and output by 1.36, but both still choose `Let`.

## Conclusion

The large first-token logit flip is downstream amplification of a small
prefill-attention numerical difference. It is not evidence of a mask, page,
or indexing defect in either unified kernel. Stock `ROCM_ATTN` is not closer
to FP64 on the first operation.

This clears the attention-correctness defect hypothesis for this reproducer.
It does not itself clear task quality: `5` follows the user's “Integer only”
instruction while `Let` does not. Promotion remains gated on the larger paired
task evaluation.

Artifacts:

- `reasoning_00_stock/`, `reasoning_00_triton/`,
  `reasoning_00_aiter/`: 16 layer captures each.
- `reasoning_00_replay.json`: all kernels replayed on stock Q/K/V.
- `reasoning_00_layers.json`: actual-backend layer-by-layer comparison and
  each output versus its own FP64 reference.
