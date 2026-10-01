# Guides and Technical Reports

* [Multi-Turn Agentic Serving Dynamics on AMD Radeon™ AI PRO R9700](AGENTX-TAIL.md) - Hardware Under Test: Single AMD Radeon™ AI PRO R9700 (32 GB GDDR6, 256-bit, PCIe Gen 5.0 x16, gfx1201).
* [SWE-bench & GPQA Benchmarking Harness Guide](BENCH.md) - This document describes the automated evaluation framework for testing model coding capability (SWE-bench), scientific reasoning (GPQA), and live serving throughput on AMD Radeon™ AI PRO R9700 and Instinct GPUs.
* [Qwen3.8-Flash-Next on two MI350P GPUs](FLASH-NEXT.md) - Separate from the frozen dense-27B MXFP4 control (MI350P.md). The published AMD recipe is 4× MI355X, TP=4. Nothing below is a validated.
* [GPT-OSS-20B native MXFP4](GPT-OSS.md) - openai/gpt-oss-20b ships native MXFP4 and fits one 32 GB card. That is the Radeon AI PRO R9700S (and the R9700, the same 64 CU / 32 GB / 300 W GPU). openai/gpt-oss-120b does not fit that card. The 120B AIM sweep on...
* [KV connector — MI350P 1P1D](KV_CONNECTOR.md) - This host has 2 × AMD Instinct MI350P PCIe (0x75a8, gfx950). GPU 0 is 0000:8b:00.0. GPU 1 is 0001:c7:00.0. rocm-smi reports the link as PCIE, 3 hops, weight 72. The two cards are on different PCI domains.
* [MI350P AgentX concurrency tail](MI350P-AGENTX.md) - Date: 30 September 2026 Hardware: 1× AMD Instinct MI350P PCIe, GPU 0 (rocm-inference-server) Model: Qwen3.8-27B-Quark-AWQ-MXFP4, context cap 65,536.
* [MI350P GPT-OSS-120B AIM sweep](MI350P-AIMS.md) - Date: 29 September 2026 SKU: 1× AMD Instinct MI350P (gfx950, PCI 0x75a8, 144 GB HBM3E), GPU 0. GPU 1 stayed idle.
* [MI350P bring-up](MI350P-BRINGUP.md) - Earlier single-GPU notes from 24 Sep. The current Quark MXFP4 serving record is MI350P.md. Dual-GPU prefill/decode is MI350P-PD.md.
* [Expert parallelism on two MI350P cards](MI350P-EP.md) - Same burst as the NVIDIA forum comparison of 2× RTX PRO 6000 Blackwell on PCIe against 2× B200 on NVLink, run here on 2× Instinct MI350P.
* [MI350P GPT-OSS-120B MLPerf Inference v6.1](MI350P-MLPERF.md) - Date: 29 September 2026 SKU: 1× AMD Instinct MI350P (gfx950, 128 CUs), GPU 0, TP=1 Image: rocm/amd-mlperf:mi355xgptoss120binference6.1 (sha256:4f17b38a81f735274caa8792c57ed884c7164fd79c41238e02f249f005d881ae).
* [Dual Instinct MI350P PCIe — Phase Isolation and P/D Report](MI350P-PD.md) - The cards on this host are AMD Instinct MI350P PCIe.
* [Tensor parallelism over PCIe on two MI350P cards](MI350P-TP.md) - Measured 30 September 2026 on the dual-socket MI350P PCIe host. This is a functional TP=2 result, but it is not a performance-qualified deployment.
* [MI350P vLLM Quark MXFP4 evaluation (reproducible)](MI350P.md) - Campaign: 25–28 September 2026 SKU: 1× AMD Instinct MI350P PCIe (gfx950, PCI 0x75a8, 144 GB HBM3E, 600 W cap), GPU 0 only.
* [Operations](OPS.md) - Container layout and fixes for this repo. Benchmark commands are in BENCH.md.
* [Prefill/Decode Disaggregation (P/D) Comprehensive Evaluation & Architecture Specification](PDD-EVAL.md) - Document Reference: docs/PDD-EVAL.md Target Model: Qwen3.8-27B MXFP4 (local/vllm-mxfp4:gfx1201) Target Hardware: Dual AMD Radeon™ AI PRO R9700 (64 CUs, 32 GB GDDR6, PCIe Gen 5.0 x16, $300\text{W}$ TDP).
* [Prefill/Decode Disaggregation (P/D) vs. Data Parallelism (DP=2): Presales & Engineering Evaluation Framework](PDD-FRAMEWORK.md) - Target Model: Qwen3.8-27B MXFP4 (local/vllm-mxfp4:gfx1201) Hardware Baseline: Dual AMD Radeon™ AI PRO R9700 (64 CUs, 32 GB GDDR6, PCIe Gen 5.0 x16, $300\text{W}$ TDP).
* [The Token Freeze That Throughput Charts Miss](QWEN-TAIL-PDD.md) - A coding agent can deliver an impressive number of tokens per second and still feel broken. The failure often happens between tokens: a response is streaming smoothly, a new long prompt arrives, and the existing...
* [R9700 dual-GPU plan](R9700-PD.md) - Tensor parallel, data parallel, and prefill/decode disaggregation for two R9700 cards, plus the single-card counterfactual model. Method: TESTPLAN.md. Measured single-card phases: R9700.md.
* [R9700 concurrency sweep (1024:1024)](R9700-SWEEP-1024-1024.md) - Narrative and earlier sweeps: R9700.md.
* [R9700 concurrency sweep (1024:8192)](R9700-SWEEP-1024-8192.md) - Narrative and earlier sweeps: R9700.md.
* [R9700 concurrency sweep](R9700-SWEEP.md) - Narrative and earlier sweeps: R9700.md.
* [R9700 results](R9700.md) - Radeon AI PRO R9700 (gfx1201) measurements. This is not the MI350P record. Current Instinct MXFP4 serving is MI350P.md. Concurrency sweeps with full power/thermal telemetry are recorded in R9700-SWEEP.md...
* [Docs](README.md) - Grouped by machine. The current production record is MI350P.md.
* [Enterprise LLM Tail Latency & Agent Serving Evaluation Plan](TAIL-EVALUATION-PLAN.md) - Document ID: EVAL-TAIL-AGENT-2026.1 Target Hardware: Single & Dual AMD Radeon™ AI PRO R9700 (32 GB GDDR6) / AMD Instinct™ MI350P (288 GB HBM3e).
* [LLM Inference Tail Latency & Agent Compounding Framework](TAIL-LATENCY-STUDY.md) - This framework reproduces and expands upon the empirical findings from the DigitalOcean tail latency study, grounded in the systems principles of Jeffrey Dean and Luiz André Barroso's foundational work The Tail at...
* [TCO: 16× R9600D, 8× R9700S, 8× MI350P, 8× RTX PRO 6000](TCO.md) - Planning comparison where host DRAM capacity matches total GPU VRAM capacity for each server fill: R9700S (256 GB host DRAM at $37,000), R9600D (512 GB host DRAM at $49,000), RTX PRO 6000 (768 GB at $62,000), and...
* [Comprehensive Dual-Card Radeon™ AI PRO R9700 Performance & Disaggregation Test Plan](TESTPLAN.md) - Document ID: TESTPLAN-R9700-PD-2026.1 Target Hardware: Dual AMD Radeon™ AI PRO R9700 (32 GB GDDR6, 256-bit, PCIe Gen 5.0 x16, gfx1201).

# Published Results

* [Published Results Catalog](results/index.md) - Progressive-disclosure index for published benchmark results.

# Presentation

* [Serving Agentic LLMs at the Edge](presentation/slides.md) - Empirical Evaluation of Qwen3.8-27B MXFP4 on AMD Radeon™ AI PRO R9700 & Instinct™ MI350P.
