# P1–P4 service-layer comparison (1K/1K)

Roofline, energy, and report wording: [`docs/MI350P_MXFP4_VLLM_EVAL.md`](../../docs/MI350P_MXFP4_VLLM_EVAL.md) §5.

Captured with `scripts/profile_capture.sh --skip-attach`. GPU kernel traces are not available yet (EngineCore attach blocked). Queue time is ~0 in all four windows.

| ID | tok/s | mean TTFT s | mean ITL s | draft accept | accepted/draft-step | notes |
|---|---:|---:|---:|---:|---:|---|
| P1 control C1 | 79.73 | 0.046 | 0.013 | — | — | GFX ~100%, ~390 W, 2.2 GHz |
| P2 DFlash-7 C1 | 102.47 | 0.140 | 0.022 | 17.9% | 1.25 | C1 gain; TTFT worse; ITL events 1820 vs 4092 (burst delivery) |
| P3 control C8 | 558.61 | 0.313 | 0.014 | — | — | 16 requests / conc 8 |
| P4 DFlash-7 C8 | 416.71 | 0.510 | 0.040 | 17.4% | 1.22 | C8 loss **not** from acceptance collapse |

DFlash C8 still accepts ~17% of draft tokens and ~2.22 output tokens per draft step, nearly identical to C1. The C8 regression is therefore extra draft/verify work and worse decode ITL (~40 ms vs ~14 ms), not a collapse of speculative quality.

vLLM 0.30 ROCm does not export `spec_decode_efficiency` or `emitted_tokens`. `num_drafts_total` tracks verify steps (matches ITL event count on DFlash).
