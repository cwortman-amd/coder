# Checks

* [Inference Health & Verification Check Report](checks/check_summary_20260923_071840.md) - Date: Wed Sep 23 07:18:43 AM EDT 2026 Target Server: http://127.0.0.1:8000 Active Model: Qwen/Qwen3.8-27B-FP8.
* [Inference Health & Verification Check Report](checks/check_summary_20260923_073047.md) - Date: Wed Sep 23 07:31:58 AM EDT 2026 Target Server: http://127.0.0.1:8000 Active Model: Qwen/Qwen3.8-27B-FP8.
* [Inference Health & Verification Check Report](checks/check_summary_20260923_074614.md) - Date: Wed Sep 23 07:46:24 AM EDT 2026 Target Server: http://127.0.0.1:8000 Active Model: Qwen/Qwen3.8-27B-FP8.
* [Inference Health & Verification Check Report](checks/check_summary_20260923_081511.md) - Date: Wed Sep 23 08:15:59 AM EDT 2026 Target Server: http://127.0.0.1:8000 Active Model: Qwen/Qwen3.8-27B-FP8.
* [Inference Health & Verification Check Report](checks/check_summary_20260923_081606.md) - Date: Wed Sep 23 08:16:08 AM EDT 2026 Target Server: http://127.0.0.1:8000 Active Model: Qwen/Qwen3.8-27B-FP8.
* [Inference Health & Verification Check Report](checks/check_summary_20260924_230933.md) - Date: Thu Sep 24 11:09:34 PM UTC 2026 Target Server: http://127.0.0.1:8000 Active Model: Qwen/Qwen3.8-27B-FP8.
* [Inference Health & Verification Check Report](checks/check_summary_20260928_115536.md) - Date: Mon Sep 28 11:55:38 AM EDT 2026 Target Server: http://127.0.0.1:8000 Active Model: Qwen3.8-27B-Quark-AWQ-MXFP4.
* [Inference Health & Verification Check Report](checks/check_summary_20260928_134009.md) - Date: Mon Sep 28 01:40:10 PM EDT 2026 Target Server: http://127.0.0.1:8000 Active Model: Qwen3.8-27B-Quark-AWQ-MXFP4.
* [Inference Health & Verification Check Report](checks/check_summary_20260928_143528.md) - Date: Mon Sep 28 02:35:30 PM EDT 2026 Target Server: http://127.0.0.1:8000 Active Model: Qwen3.8-27B-Quark-AWQ-MXFP4.

# Ep Mi350P

* [Qwen3.5-35B-A3B expert parallel, same burst](ep_mi350p/COMPARE.md) - Protocol match for the NVIDIA developer forum post: Qwen/Qwen3.5-35B-A3B, TP=2, --gpu-memory-utilization 0.9, --max-model-len 32768, then the same serve command with --enable-expert-parallel. Client: vllm bench serve...

# Pd Emulator

* [Counterfactual Capacity & Interference Report: 1P1D vs. DP=2 on Radeon™ AI PRO R9700](pd_emulator/pd_emulator_report_20260923_202929.md) - Evaluation Date: September 23, 2026 Hardware Platform: AMD Radeon™ AI PRO R9700 (gfx1201, 64 CUs, 32 GB GDDR6).
* [Counterfactual Capacity & Interference Report: 1P1D vs. DP=2 on Radeon™ AI PRO R9700](pd_emulator/pd_emulator_report_20260923_212012.md) - Evaluation Date: September 23, 2026 Hardware Platform: AMD Radeon™ AI PRO R9700 (gfx1201, 64 CUs, 32 GB GDDR6).
* [Counterfactual Capacity & Interference Report: 1P1D vs. DP=2 on Radeon™ AI PRO R9700](pd_emulator/pd_emulator_report_20260923_213222.md) - Evaluation Date: September 23, 2026 Hardware Platform: AMD Radeon™ AI PRO R9700 (gfx1201, 64 CUs, 32 GB GDDR6).
* [Counterfactual Capacity & Interference Report: 1P1D vs. DP=2 on Radeon™ AI PRO R9700](pd_emulator/pd_emulator_report_20260929_042526.md) - Evaluation Date: September 29, 2026 Hardware Platform: AMD Radeon™ AI PRO R9700 (gfx1201, 64 CUs, 32 GB GDDR6).

# Priority Eval

* [MI350P MXFP4 control + graph/batch ablations (2026-09-25)](priority_eval/EVAL.md) - Canonical write-up for reports: docs/MI350P.md. This file is the lab ledger (JSON paths, per-run notes).
* [DFLASH-3 promotion-gate validation (2026-09-25)](priority_eval/dflash3_validation/VALIDATION.md) - DFLASH-3 is not promoted as the default interactive profile. It is the preferred optional DFlash setting only for low-concurrency, long-output.
* [DFlash depth 1–5 (matched protocol, 25 Sep 2026)](priority_eval/dflash_depth/DEPTH.md) - Harness: scripts/dflashdepthsweep.sh. Control rows are the prior crossover process (crossover/baselinematchedc.json, 78.80 / 318.04 / 426.95 / 557.14). DFLASH-7 rows are the prior crossover, not re-run in this process.
* [Instrumented C32–C128 (HF MXFP4 control)](priority_eval/saturation_c32_c128/SATURATION.md) - Harness: scripts/benchsaturationinstrumented.sh Date: 2026-09-25 Server: frozen HF control on 127.0.0.1:8000, VLLMROCMUSEAITER unset, stock MLEQ8, graphs O2.

# Profiling

* [C1 control vs RVS BABEL HBM roof (25 Sep 2026)](profiling/C1_BABEL_ROOFLINE.md) - Canonical report: docs/MI350P.md.
* [P1–P4 service-layer comparison (1K/1K)](profiling/P1_P4_METRICS.md) - Roofline, energy, and report wording: docs/MI350P.md §5.
* [Profiling Qwen3.8-27B MXFP4 on MI350P](profiling/PROFILE.md) - Campaign summary: docs/MI350P.md.
* [Three-way attention isolation (27 Sep 2026)](profiling/attention_ab/ABC.md) - Same Quark MXFP4 checkpoint, same vLLM image, VLLMROCMUSEAITER unset, FULLANDPIECEWISE graphs, C1/C8 harness, temperature 0. Only.
* [MI350P attention-backend A/B (27 Sep 2026)](profiling/attention_ab/ATTENTION.md) - The 14.9× decode-kernel ratio is TRITONATTN versus stock ROCMATTN (kernelpagedattention2d 173.45 µs p50 versus kernelunifiedattention.
* [TRITONATTN gate 1: first-divergence logits (27 Sep 2026)](profiling/attention_ab/divergence/GATE1.md) - C1 chat, temperature 0, topp 1, topk -1, seed 1234, enablethinking false, FULLANDPIECEWISE graphs. Same Quark MXFP4 checkpoint and vLLM image.
* [Paged-decode fixture (27 Sep 2026)](profiling/attention_ab/fixture/FIXTURE.md) - One decode query, 24 query heads, 4 KV heads, head size 256, bf16 Q/K/V, scale 1/sqrt(256), causal, no sliding window, no sinks. The same logical.
* [Real-tensor attention isolation: reasoning00 (27 Sep 2026)](profiling/attention_ab/real_capture/REASONING_00.md) - Prompt length 44. Stock emits token 20 (5). Triton and AITER unified emit token 9764 (Let).
* [EngineCore attach experiment (25 Sep 2026)](profiling/enginecore_attach/ATTACH.md) - Goal: GPU-owner kernel timeline without wrapping vllm serve.
* [Best-C1 kernel and memory profile: Triton attention](profiling/enginecore_exec/20260927T151210Z_triton_attn/PROFILE.md) - Configuration: one MI350P, Qwen3.8-27B Quark AWQ MXFP4, stock MXFP4 dispatch, VLLMROCMUSEAITER unset, O2 FULLANDPIECEWISE graphs, and.
* [AITER unified attention kernel trace (27 Sep 2026)](profiling/enginecore_exec/20260927T163138Z_aiter_unified/PROFILE.md) - Attribution only. Profiled one-request 1K/1K throughput was 62.03 tok/s. The unprofiled repeats are 99.58 ± 0.05 C1 and 653.96 ± 0.44 C8.
* [EngineCore kernel trace (27 Sep 2026)](profiling/enginecore_exec/TRACE.md) - Kernel-level tracing now works. The API parent is not profiled. Only the Python --multiprocessing-fork child that becomes VLLM::EngineCore is.
* [Isolated GDN decode vs C1 (25 Sep 2026)](profiling/gdn_decode/GDN.md) - Canonical: docs/MI350P.md.
* [Captured MXFP4 mini-decode attempt (25 Sep 2026)](profiling/mxfp4_captured/CAPTURE.md) - Goal: graph-replay real Qwen3.8-27B MXFP4 shapes under stock Triton and shape-local split-K variants, as an intermediate screen before serving.
* [Exact-shape MXFP4 dispatch (27 Sep 2026)](profiling/mxfp4_gateup/SERVING.md) - Two candidates. Each one replaced only MLEQ8 for one (N, K) via AITER’s GEMM-AFP4WFP4-N=…-K=….json lookup. DEFAULT.json stayed stock.
* [Isolated AITER MXFP4 GEMM vs Babel Read (25 Sep 2026)](profiling/mxfp4_gemm/GEMM.md) - Canonical: docs/MI350P.md. Babel roof: C1BABELROOFLINE.md.
* [MXFP4 quantization-boundary check (27 Sep 2026)](profiling/mxfp4_quant_boundary/BOUNDARY.md) - Elementwise/norm is 12.15% of traced GPU dispatch duration and dynamic MXFP4 quantization is 8.38%. Together they are 20.53%. The installed fusion passes.
* [vLLM captured mini-decode](profiling/vllm_captured/SUMMARY.md) - UTC: 2026-09-25T17:18:27Z Status: completed Live :8000 detected: False GPU free/total GiB: 143.67 / 143.98.

# Quality

* [MXFP4 vs FP8/BF16 production-sampling comparison](quality/MXFP4_VS_FP8.md) - Blocked until an exclusive-GPU window. The frozen MXFP4 control is live on 127.0.0.1:8000 and was not restarted or displaced for this work. Running a.
* [MXFP4 vs DFlash greedy quality (text-only)](quality/QUALITY.md) - Campaign summary: docs/MI350P.md.
* [Predeclared fast-attention promotion gates (27 Sep 2026)](quality/attention_qualification/GATES.md) - Written before generating or running the expanded paired suite.
* [Fast-attention qualification result (27 Sep 2026)](quality/attention_qualification/RESULT.md) - Gates were frozen in GATES.md before generating or running the 100-item suite. Same Quark checkpoint, vLLM image, graph mode, prompts, chat template,.
* [AITER unified latency diagnosis (27 Sep 2026)](quality/attention_qualification/aiter_latency/DIAGNOSIS.md) - Stock ROCMATTN is the running service again. VLLMROCMUSEAITER was not set. No scheduler or chunk-size change was applied. The 8192-token.
* [1K/1K latency campaign, first cell (27 Sep 2026)](quality/attention_qualification/latency_campaign/1K1K.md) - Gates were copied into GATESSNAPSHOT.md before these runs. Both profiles used the pinned vLLM 0.30.0 image, the local Quark MXFP4 checkpoint, O2.
* [Frozen latency gates copied before the 1K/1K campaign](quality/attention_qualification/latency_campaign/GATES_SNAPSHOT.md) - Source: results/quality/attentionqualification/GATES.md, written before the paired quality suite. These limits are not being revised for this run.

# Throughput

* [Throughput Benchmark Performance Report](throughput/20260922_163433/throughput_benchmark_report.md) - Model Evaluated: Qwen/Qwen3-0.6B Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM).
* [Throughput Benchmark Performance Report](throughput/20260922_182344/throughput_benchmark_report.md) - Model Evaluated: Qwen/Qwen3.8-27B-FP8 Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM).
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260923_064358/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Engines Tested: vllm Benchmark Suite: vLLM Throughput Matrix (vllm bench serve).
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260923_065256/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Engines Tested: vllm Benchmark Suite: vLLM Throughput Matrix (vllm bench serve).
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260923_065722/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Engines Tested: llama.cpp Benchmark Suite: vLLM Throughput Matrix (vllm bench serve).
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260923_070213/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Engines Tested: vllm llama.cpp sglang.
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260923_070815/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Engines Tested: vllm llama.cpp sglang.
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260923_071852/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Engines Tested: vllm Benchmark Suite: vLLM Throughput Matrix (vllm bench serve).
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260923_072606/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Engines Tested: vllm Benchmark Suite: vLLM Throughput Matrix (vllm bench serve).
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260923_073204/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Engines Tested: vllm Benchmark Suite: vLLM Throughput Matrix (vllm bench serve).
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260923_073413/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Engines Tested: llama.cpp Benchmark Suite: vLLM Throughput Matrix (vllm bench serve).
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260923_092248/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Engines Tested: llama.cpp Benchmark Suite: vLLM Throughput Matrix (vllm bench serve).
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260923_092944/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Engines Tested: vllm Benchmark Suite: vLLM Throughput Matrix (vllm bench serve).
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260928_143005/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6) GPU Profile: r9700 ROCm ISA: gfx1201.
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260928_143607/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6) GPU Profile: r9700 ROCm ISA: gfx1201.
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260928_143749/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6) GPU Profile: r9700 ROCm ISA: gfx1201.
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260928_143758/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6) GPU Profile: r9700 ROCm ISA: gfx1201.
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260928_143852/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6) GPU Profile: r9700 ROCm ISA: gfx1201.
* [Multi-Engine Throughput Benchmark Performance Report](throughput/20260928_144716/throughput_benchmark_report.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6) GPU Profile: r9700 ROCm ISA: gfx1201.

# Top Level

* [Accuracy Benchmark Summary](accuracy_summary_20260924_230947.md) - Device: AMD Instinct™ MI350P (gfx950, 144 GB HBM3E) GPU profile: mi350p ROCm ISA: gfx950 Engine: vllm.
* [Benchmark Evaluation Summary Report](test_run_summary_20260922_161016.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Model Evaluated: Qwen/Qwen3-0.6B.
* [Benchmark Evaluation Summary Report](test_run_summary_20260922_191036.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Model Evaluated: Qwen/Qwen3.8-27B-FP8.
* [Benchmark Evaluation Summary Report](test_run_summary_20260923_062250.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Model Evaluated: Qwen/Qwen3.8-27B-FP8.
* [Benchmark Evaluation Summary Report](test_run_summary_20260924_052545.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Model Evaluated: Qwen3.8-27B-Quark-AWQ-MXFP4.
* [Benchmark Evaluation Summary Report](test_run_summary_20260924_055126.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Model Evaluated: Qwen/Qwen3.8-27B-FP8.
* [Benchmark Evaluation Summary Report](test_run_summary_20260924_062825.md) - Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6 VRAM) Model Evaluated: Qwen3.8-27B-Q4KM.gguf.

# Upstream

* [Upstream issues to file (measured on vLLM 0.30 ROCm + MI350P)](upstream/ISSUES.md) - These are lab-ready reports. File against the listed projects; do not treat this folder as a substitute for the tracker.
