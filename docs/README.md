---
type: Documentation Catalog
title: Docs
description: Grouped by machine. The current production record is MI350P.md.
tags:
- documentation
- catalog
status: stable
---

# Docs

Grouped by machine. The current production record is [MI350P.md](MI350P.md).

## MI350P

| File | Contents |
|---|---|
| [MI350P.md](MI350P.md) | Quark MXFP4 vLLM results, gates, and next work |
| [MI350P-AGENTX.md](MI350P-AGENTX.md) | Faithful AgentX concurrency curve on one MI350P. Throughput peaks at C=32 and collapses at C=64 |
| [MI350P-TP.md](MI350P-TP.md) | Two-GPU PCIe TP. The decode gap follows a ~51 ms launch stall on `0001:c7:00.0`. The same card also misses the RVS bf16 GST target (`scripts/debug.sh`). `scripts/pcie_path_bench.cu` times the HIP peer path |
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
| [QWEN-TAIL-PDD.md](QWEN-TAIL-PDD.md) | Qwen3.8-27B MXFP4 blog: the token freeze throughput charts miss, prefix-cache and chunk baselines, and the MI350P PCIe handoff still awaiting a DP=2 comparison |
| [AGENTX-TAIL.md](AGENTX-TAIL.md) | Multi-turn Claude Code traces sweep on R9700 (65k context): prefix cache collapse from 93% to 0%, the C=8 knee, and C=16/32 KV eviction cliff |
| [TAIL-LATENCY-STUDY.md](TAIL-LATENCY-STUDY.md) | Empirical tail latency & agent compounding study: single-call sweeps, 10-call chained benchmarks, TTFT-vs-task inversion detection, and publication dashboard |
| [TAIL-EVALUATION-PLAN.md](TAIL-EVALUATION-PLAN.md) | Enterprise 2-track evaluation blueprint: 7-experiment matrix, unified request schema, open-loop controls, P/D interference tests, and defensible evidence criteria |

## Published measurements

Benches write under `_results`. That directory, plus the local NIXL checkout `_src` and install prefix `_opt`, is gitignored. `./test.sh` copies the results from that run into `docs/results` when it finishes. Plot scripts only read `docs/`.

| Command | Published input | Figure |
|---|---|---|
| `./analyze.sh` | `docs/results/` and `docs/profiling/` | Every report and figure below |
| `python3 scripts/generate_kv_plots.py` | `docs/results/` | `docs/figures/kv/` |
| `python3 scripts/plot_gpu_utilization.py` | `docs/profiling/gpu_metrics.json` | `docs/figures/utilization/` |
| `python3 scripts/generate_tco_plots.py` | `docs/profiling/power_bandwidth.json` and `docs/results/r9700/concurrency.json` | `docs/figures/tco/` |
| `python3 scripts/generate_pd_plots.py` | `docs/results/pd/pd_emulator_summary.json` | `docs/figures/pd/` |
| `python3 scripts/generate_tp_compare_plots.py` | `docs/results/tp_compare/mi350p/` | `docs/figures/tp/` |
| `python3 scripts/generate_latency_histogram_plots.py` | `docs/results/qwen3.8-27b-mxfp4/latency/r9700/` | `docs/figures/latency/` |
| `python3 scripts/plot_agentx_tail_sweep.py` | `docs/results/agentx/` | `docs/figures/agentx/` |
| `.venv/bin/python3 scripts/plot_tail_distributions.py --input docs/results/tail_study/agent_chain_manifest.json --out docs/figures/tail_study_dashboard.png` | `docs/results/tail_study/` | `docs/figures/tail_study_dashboard.png` |
| `python3 scripts/publish_latency_results.py --source-dir <qwen-run-dir> --gpu-profile <profile>` | `docs/results/qwen3.8-27b-mxfp4/latency/` | `docs/profiling/qwen3.8-27b-mxfp4-latency.json` |
| `python3 scripts/run_agentx_tail_sweep.py` then `python3 scripts/analyze_agentx_tail_sweep.py` | `docs/results/mi350p/agentx_concurrency.json` | [MI350P-AGENTX.md](MI350P-AGENTX.md) |

`docs/profiling/power_bandwidth.json` holds the R9700 and MI350P concurrency-sweep power and bandwidth series used by the TCO figures. `docs/profiling/gpu_metrics.json` holds the suite throughput profile: socket power, the UMC memory-bandwidth estimate, and amd-smi PCIe traffic. The PDD summary uses the 29 Sep contract: TTFT ≤ 3 s, TPOT ≤ 20 ms, p95 ITL ≤ 20 ms, p99 ITL ≤ 50 ms, and peak ITL ≤ 100 ms. Qwen latency runs are the exception to the older “sample traces are not kept” policy: test scripts now retain per-request TTFT, TPOT, E2E, and per-token ITL under `docs/results/qwen3.8-27b-mxfp4/latency/`, with report-ready percentiles and histograms in `docs/profiling/qwen3.8-27b-mxfp4-latency.json`. AgentX multi-turn traces are archived under `docs/results/agentx/`, and empirical complete-task agent chain and open-loop sweep results are archived under `docs/results/tail_study/`. A `./test.sh` run also publishes its accuracy report, throughput directory, and experiment tree under `docs/results/`. Generated text is omitted.

For a report-grade tail run (1,000 requests per cell, 0.1% empirical resolution):

```bash
python3 scripts/bench_qwen_tail_latency.py \
  --gpu-profile mi350p \
  --input-lens 1024 8192 \
  --concurrency-list 1 16 64
```

## Presentation & Decks

| File | Contents |
|---|---|
| [presentation/tail-latency.md](presentation/tail-latency.md) | 19-slide comprehensive technical deck: tail latency, AgentX multi-turn dynamics, P/D architecture, PCIe transport, and tokenomics |
| [presentation/tail-latency.html](presentation/tail-latency.html) | Standalone interactive browser-based presentation (HTML) |
| [presentation/tail-latency.pdf](presentation/tail-latency.pdf) | High-resolution publication presentation deck (PDF) |
| [presentation/tail-latency.pptx](presentation/tail-latency.pptx) | Editable PowerPoint slide deck (PPTX) |
| [presentation/coder-tco.md](presentation/coder-tco.md) | On-premises coding-agent TCO: subscription list prices against measured R9700S and MI350P token cost |
| [presentation/coder-tco.html](presentation/coder-tco.html) | Standalone interactive browser-based presentation (HTML) |
| [presentation/coder-tco.pdf](presentation/coder-tco.pdf) | High-resolution publication presentation deck (PDF) |
| [presentation/coder-tco.pptx](presentation/coder-tco.pptx) | Editable PowerPoint slide deck (PPTX) |

Compile or refresh decks via `./scripts/build_presentation.sh`.

SVG sources and the PNG rasters the decks embed are in [assets/](assets/). Each name has both files.

| File | Use |
|---|---|
| [llm-router](assets/llm-router.svg) | Coding request to local, frontier, or specialized |
| [llm-router-tools](assets/llm-router-tools.svg) | Tools, response processing, and the return path |
| [llm-router-detail](assets/llm-router-detail.svg) | Appendix: the earlier multi-vendor router |
| [route-flow](assets/route-flow.svg) | Coder TCO slide 3: what moves onto the GPU |
| [router-diagram](assets/router-diagram.svg) | Coder TCO slide 8: route by requirements |
| [routing-classify](assets/routing-classify.svg) | Classify the prompt, then choose a model |
| [routing-architecture](assets/routing-architecture.svg) | Requirements, filter, quality check, outcomes |
| [routing-factors](assets/routing-factors.svg) | Hard constraints before soft preferences |
| [routing-cascade](assets/routing-cascade.svg) | Cascade versus fallback |
| [routing-eval](assets/routing-eval.svg) | Evaluation loop |

## How to run

| File | Contents |
|---|---|
| [OPS.md](OPS.md) | Compose files (single GPU, TP=2, DP=2, P/D, 8-card 1P:7D, DP=8, 16-card 2P:14D), fleet router, and troubleshooting (MoRIIO, KFD, OOM) |
| [BENCH.md](BENCH.md) | SWE-bench and GPQA harness |

## OKF v0.2 Knowledge Bundle

Documentation is organized as an [Open Knowledge Format v0.2](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) bundle. Use [index.md](index.md) for progressive disclosure. Non-reserved Markdown concepts require typed frontmatter; `index.md` files are reserved catalogs.

After adding or changing documentation, run:

```bash
python3 scripts/okf_docs.py migrate
python3 scripts/okf_docs.py index
python3 scripts/okf_docs.py validate
```

Migration derives deterministic metadata from repository content. Do not add verification or provenance claims unless they are independently established.

