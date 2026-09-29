# Docs

Grouped by machine. The current production record is [MI350P.md](MI350P.md).

## MI350P

| File | Contents |
|---|---|
| [MI350P.md](MI350P.md) | Quark MXFP4 vLLM results, gates, and next work |
| [TCO.md](TCO.md) | 16× R9600D, 8× R9700S, 8× MI350P, 8× RTX PRO 6000 cost, tokens per dollar, and presentation plots |
| [MI350P-BRINGUP.md](MI350P-BRINGUP.md) | 24 Sep FP8 bring-up and engine matrix |
| [MI350P-PD.md](MI350P-PD.md) | Two-GPU prefill/decode on this host |
| [MI350P-EP.md](MI350P-EP.md) | Qwen3.5-35B-A3B TP vs EP, same burst as the RTX PRO 6000 / B200 post. Eager result on this host: **0.93×** output tok/s. Scorecard: [`_results/ep_mi350p/COMPARE.md`](../_results/ep_mi350p/COMPARE.md) |
| [MI350P-AIMS.md](MI350P-AIMS.md) | GPT-OSS-120B AIM RC (vLLM 0.19.1, ROCm 7.13), one MI350P, random-dataset concurrency sweep |
| [MI350P-MLPERF.md](MI350P-MLPERF.md) | GPT-OSS-120B MLPerf Inference v6.1. One-GPU Offline 5,736 tok/s VALID. Server at 5 QPS INVALID |
| [FLASH-NEXT.md](FLASH-NEXT.md) | Qwen3.8-Flash-Next, separate from the dense control |

## R9700

| File | Contents |
|---|---|
| [R9700.md](R9700.md) | Engines, accuracy, phase profiles, concurrency, and SLO tracks |
| [R9700-SWEEP.md](R9700-SWEEP.md) | Empirical concurrency sweep (8,192:1,024) with latency & power telemetry |
| [R9700-SWEEP-1024-1024.md](R9700-SWEEP-1024-1024.md) | Concurrency sweep (1,024:1,024) with latency & power telemetry |
| [R9700-SWEEP-1024-8192.md](R9700-SWEEP-1024-8192.md) | Concurrency sweep (1,024:8,192) with latency & power telemetry |
| [GPT-OSS.md](GPT-OSS.md) | GPT-OSS-20B native MXFP4 serve and 1,024/1,024 bench on a 32 GB R9700S |
| [R9700-PD.md](R9700-PD.md) | Dual-GPU plan and capacity model |
| [TESTPLAN.md](TESTPLAN.md) | Dual-card method and acceptance criteria |

## Disaggregated Serving & Architecture Evaluation (PDD vs. DP)

| File | Contents |
|---|---|
| [PDD-FRAMEWORK.md](PDD-FRAMEWORK.md) | Theoretical & presales evaluation framework, three-part value narrative, 5-level evidence ladder, break-even crossover analysis ($\eta < 0.50$), and enterprise tokenomics model |
| [PDD-EVAL.md](PDD-EVAL.md) | Comprehensive evaluation process, testbed architectures (2-card 1P1D, 8-card 1P:7D, 16-card 2P:14D), interactive SLO metrics, queue stability criteria, 6-exhibit visual suite, fleet router/simulator tooling, live connector diagnostics, and 4-track execution roadmap |
| [KV_CONNECTOR.md](KV_CONNECTOR.md) | MI350P 1P1D KV connector: HIP-IPC patch apply script, GPU 1→GPU 0 serving (72.49 tok/s, token-ID gate still open) |

## Published measurements

Benches write under `_results`. That directory, plus the local NIXL checkout `_src` and install prefix `_opt`, is gitignored. The plot scripts copy the records they need into `docs/` and then plot those copies.

| Command | Published input | Figure |
|---|---|---|
| `python3 scripts/generate_kv_plots.py` | `docs/results/` | `docs/figures/kv/` |
| `python3 scripts/generate_tco_plots.py` | `docs/profiling/power_bandwidth.json` and `docs/results/r9700/concurrency.json` | `docs/figures/tco/` |
| `python3 scripts/generate_pd_plots.py` | `docs/results/pd/pd_emulator_summary.json` | `docs/figures/pd/` |

`docs/profiling/power_bandwidth.json` holds the R9700 and MI350P power and bandwidth summaries. The PDD summary uses the 29 Sep contract: TTFT ≤ 3 s, TPOT ≤ 20 ms, p95 ITL ≤ 20 ms, p99 ITL ≤ 50 ms, and peak ITL ≤ 100 ms. Sample traces are not kept. When `_results` is absent, the scripts use the copies already in `docs/`.

## How to run

| File | Contents |
|---|---|
| [OPS.md](OPS.md) | Compose files (single GPU, TP=2, DP=2, P/D, 8-card 1P:7D, DP=8, 16-card 2P:14D), fleet router, and troubleshooting (MoRIIO, KFD, OOM) |
| [BENCH.md](BENCH.md) | SWE-bench and GPQA harness |

