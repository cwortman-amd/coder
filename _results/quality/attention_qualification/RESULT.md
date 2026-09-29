# Fast-attention qualification result (27 Sep 2026)

Gates were frozen in `GATES.md` before generating or running the 100-item
suite. Same Quark checkpoint, vLLM image, graph mode, prompts, chat template,
and greedy parameters were used for every backend.

## Decision

Neither fast backend is promoted. Keep `ROCM_ATTN` as production default.

- AITER unified **passes task quality** but **fails the predeclared latency
  gate**, primarily because 8K TTFT regresses.
- Triton **fails task quality** on reasoning and narrowly fails the strict
  C8 1K token-arrival threshold.
- Both remain opt-in diagnostics. Do not start duplicate-quant reuse against
  either backend as a promotion baseline yet.

## Greedy paired quality

| Category | Stock | AITER unified | AITER wins/losses | Triton | Triton wins/losses |
|---|---:|---:|---:|---:|---:|
| Overall | 86/100 | **87/100** | 3 / 2 | 85/100 | 2 / 3 |
| Math | 18/24 | 18/24 | 1 / 1 | 18/24 | 1 / 1 |
| Reasoning | 13/20 | **14/20** | 2 / 1 | **11/20** | 0 / 2 |
| Executable code | 20/21 | 20/21 | 0 / 0 | **21/21** | 1 / 0 |
| JSON/tool | 20/20 | 20/20 | 0 / 0 | 20/20 | 0 / 0 |
| Long retrieval | 15/15 | 15/15 | 0 / 0 | 15/15 | 0 / 0 |

AITER overall paired delta is +1 percentage point, bootstrap 95% CI
[-3, +5]. Reasoning is +5 points, CI [-10, +20]. It loses
`reasoning_00`, but wins two other stock failures. It has no JSON/tool, code,
or long-context regression.

Triton overall delta is -1 point, CI [-5, +3]. Reasoning is -10 points,
CI [-25, 0], with two net losses and no wins. This violates the frozen
-5-point / one-net-loss reasoning gate. Triton does gain the original
`code_08` regex case.

All three retrieve all five needles at each actual prompt band:
approximately 1,944, 8,024, and 15,625 tokens, with start/middle/end
positions.

## AITER production-sampling confirmation

Three seeds (1234–1236), `temperature=1`, `top_p=0.95`, `top_k=20`:

| Category | Stock | AITER unified | Delta |
|---|---:|---:|---:|
| Math | 54/72 | 54/72 | 0 |
| Reasoning | 37/60 | 36/60 | -1.7 points |
| Executable code | 61/63 | 61/63 | 0 |
| JSON/tool | 60/60 | 60/60 | 0 |

JSON/tool validity is 100%. AITER passes the sampling confirmation. Triton
sampling was not run because the frozen policy excludes candidates that fail
the primary greedy gate.

## Warmed serving latency

Numbers are p95 milliseconds. Each shape was warmed immediately before the
measured run. Long-context output length was 128; 1K output length was 512.

| Shape | Stock TTFT / ITL | AITER TTFT / ITL | Triton TTFT / ITL |
|---|---:|---:|---:|
| C1 1K | 60.53 / 12.48 | 62.01 / **10.11** | **58.32 / 9.99** |
| C4 1K | 257.41 / 13.34 | 259.20 / 11.20 | **255.94 / 10.87** |
| C8 1K | 470.52 / 13.87 | **421.93** / 11.80 | 461.70 / **11.51** |
| C1 8K | 89.51 / 24.51 | **155.01** / 10.57 | 92.32 / **10.47** |
| C4 8K | 388.32 / 25.25 | **567.86** / 12.66 | 363.43 / **11.38** |
| C8 8K | 751.92 / 25.80 | **996.34** / 14.72 | 702.36 / **12.63** |
| C1 15K | 128.11 / 37.05 | **114.00 / 10.83** | 137.35 / 10.97 |
| C4 15K | 568.78 / 38.11 | **398.78** / 13.89 | 508.94 / **11.94** |
| C8 15K | 1014.69 / 38.19 | **777.02** / 17.23 | 997.56 / **13.81** |

Bold TTFT in the AITER 8K rows indicates a regression, not a win. The frozen
TTFT allowance is stock plus the larger of 10% or 5 ms; AITER exceeds it at
C1/C4/C8 8K. AITER also exceeds the strict 11 ms 1K ITL ceiling at C4 and C8.

Triton meets the TTFT rule at every shape and materially improves all ITL
values, but its C8 1K p95 is 11.51 ms, above the frozen 11 ms ceiling.

All requests completed. No OOM or graph fallback was observed. Existing
unprofiled 1K/1K references remain:

- AITER unified: 99.58 ± 0.05 C1, 653.96 ± 0.44 C8.
- Triton: 100.36 ± 0.04 C1, 660.53 ± 8.69 C8.

## Next action

Do not proceed to duplicate-quant reuse as a production optimization yet.
The 8K TTFT investigation is in `aiter_latency/DIAGNOSIS.md`. The measured
gap follows the uncached page remainder (`prompt_tokens % page_size`), not a
9-token overrun of the 8192 chunk limit. The strict 11 ms 1K ITL ceiling at
C4/C8 remains a separate, steady decode miss. Any revised gate must be
justified operationally and evaluated in a new campaign; it must not be
retroactively changed to pass these results.

Machine-readable paired analysis: `analysis.json`.
