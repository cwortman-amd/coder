# Predeclared fast-attention promotion gates (27 Sep 2026)

Written before generating or running the expanded paired suite.

Profiles: stock `ROCM_ATTN`, `ROCM_AITER_UNIFIED_ATTN`, and `TRITON_ATTN`.
Same Qwen3.8-27B Quark MXFP4 checkpoint, vLLM image, graph mode, prompts,
chat template, and decoding parameters. Stock remains production control.

## Deterministic task-quality gate

Primary run: greedy (`temperature=0`, `top_p=1`, `top_k=-1`, seed 1234,
thinking disabled). Score useful outcomes, not prose or token equality.

The suite must contain at least 20 independently scored items in each of
math, reasoning, executable Python, and JSON/tool calls, plus retrieval probes
near 2K, 8K, and the service's 16K limit. It must include `reasoning_00`,
`code_08`, and `math_18`.

A fast backend passes only if all conditions hold:

1. No request failures.
2. Overall paired accuracy delta versus stock is at least -2 percentage
   points.
3. Each of math, reasoning, and executable code has a paired accuracy delta
   of at least -5 percentage points and no more than one net paired loss
   (`stock pass / candidate fail` minus the reverse).
4. JSON/tool calls have 100% parse/schema validity and zero newly introduced
   semantic failures on cases stock passes.
5. Long-context retrieval has zero newly introduced failures at each tested
   context band or needle position.
6. Report paired wins/losses/ties and a paired bootstrap 95% confidence
   interval for every category. The point thresholds above still apply when
   the suite is too small for a narrow confidence interval.
7. The known `reasoning_00` instruction failure is counted normally; it is not
   waived because the underlying attention operator is numerically valid.

The two fast backends are also compared directly. If both pass, prefer the
one with fewer critical JSON/tool or long-context failures, then fewer net
reasoning losses. Performance breaks a remaining quality tie.

## Production-sampling confirmation

After a backend passes greedy quality, rerun the critical math, reasoning,
code, and JSON/tool subset at `temperature=1`, `top_p=0.95`, `top_k=20`
with seeds 1234, 1235, and 1236. It passes if:

- JSON/tool validity remains 100%.
- No category's mean pass rate is more than 5 percentage points below stock.
- There are no newly introduced failures on designated JSON/tool safety cases.

This confirmation is not run for a backend that fails the primary gate.

## Serving latency and capacity gate

Unprofiled `FULL_AND_PIECEWISE` serving after warmup:

- C1 1K/1K output throughput must be at least 98 tok/s.
- C8 1K/1K aggregate throughput must be at least 640 tok/s.
- At C1/C4/C8, TTFT p95 may not regress more than 10% or 5 ms (whichever is
  larger) versus stock for the same context.
- Token-arrival p95 may not regress versus stock and may not exceed 11 ms at
  the 1K workload.
- 8K and near-16K C1/C4/C8 runs must complete without OOM, graph fallback,
  or request failure. Compare TTFT and p95 token-arrival latency directly;
  among quality-passing fast backends, lower long-context tails are preferred.

Profiler-on tok/s is excluded. Startup logs must identify the requested
backend, `AiterMxfp4LinearKernel`, and `FULL_AND_PIECEWISE`.

## Promotion

Promotion requires both quality and serving gates. Otherwise keep
`ROCM_ATTN` as default. DFLASH-3 is outside this decision.
