# Docs

Grouped by machine. The current production record is [MI350P.md](MI350P.md).

## MI350P

| File | Contents |
|---|---|
| [MI350P.md](MI350P.md) | Quark MXFP4 vLLM results, gates, and next work |
| [TCO.md](TCO.md) | 16× R9600D, 8× R9700S, 8× MI350P, 8× RTX PRO 6000 cost and tokens per dollar |
| [MI350P-BRINGUP.md](MI350P-BRINGUP.md) | 24 Sep FP8 bring-up and engine matrix |
| [MI350P-PD.md](MI350P-PD.md) | Two-GPU prefill/decode on this host |
| [MI350P-EP.md](MI350P-EP.md) | Qwen3.5-35B-A3B TP vs EP, same burst as the RTX PRO 6000 / B200 post. Eager result on this host: **0.93×** output tok/s. Scorecard: [`_results/ep_mi350p/COMPARE.md`](../_results/ep_mi350p/COMPARE.md) |
| [FLASH-NEXT.md](FLASH-NEXT.md) | Qwen3.8-Flash-Next, separate from the dense control |

## R9700

| File | Contents |
|---|---|
| [R9700.md](R9700.md) | Engines, accuracy, phase profiles, concurrency, and SLO tracks |
| [GPT-OSS.md](GPT-OSS.md) | GPT-OSS-20B native MXFP4 serve and 1,024/1,024 bench on a 32 GB R9700S |
| [R9700-PD.md](R9700-PD.md) | Dual-GPU plan and capacity model |
| [TESTPLAN.md](TESTPLAN.md) | Dual-card method and acceptance criteria |

## How to run

| File | Contents |
|---|---|
| [OPS.md](OPS.md) | Compose files and troubleshooting |
| [BENCH.md](BENCH.md) | SWE-bench and GPQA harness |
