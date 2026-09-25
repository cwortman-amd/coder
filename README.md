# On-Premise AI Coding Agent with AMD Radeon™ AI PRO R9700 and OpenCode

An enterprise-grade, privacy-first, on-premises AI coding assistant stack powered by **AMD Radeon™ AI PRO R9700 GPUs**, **AMD ROCm**, and **OpenCode**.

Designed for high-throughput local code generation, repository-level multi-file refactoring, and agentic task execution without sending code off-premise.

---

## Objective: Why Run a Coding Agent On-Premises?

Modern AI software development agents require deep repository context, terminal execution privileges, and continuous workspace indexing. Relying on commercial cloud AI providers introduces critical security and economic challenges that on-premises local deployment eliminates:

- **IP Protection & Zero Data Leakage**: Proprietary codebases, trade secrets, and compliance-restricted repositories (SOC2, HIPAA, ISO 27001, defense) remain 100% air-gapped within your workstation or private cluster. Prompts and code never leave your network, eliminating third-party logging and model training risks.
- **Zero Cloud API Fees & Unlimited Usage**: Eliminates recurring per-token cloud API billing and strict rate limits during multi-turn agentic coding sessions. Run unconstrained repository-wide refactoring and automated test loops at zero incremental cost.
- **Deterministic Low Latency**: Delivers predictable, low-latency prompt prefill and generation on local compute without internet latency, cloud queue congestion, or subscription throttles.
- **Safe Autonomous Tool Execution**: Executes multi-file editing, git diff patching, bash command execution, and unit test suites locally inside isolated containerized environments.
- **Hardware-Optimized for AMD Compute**: Fully leverages the 32 GB VRAM capacity, native FP8 GEMM kernels, and MxFP4 quantization of the **AMD Radeon™ AI PRO R9700** (`gfx1201`) for enterprise workstation performance.

---

## Key Features

- **100% Private & Air-Gapped**: Runs entirely on local hardware—no code snippet or prompt leaves your workstation network.
- **Native AMD ROCm Acceleration**: Tuned for RDNA 4 (`gfx1201`) and Instinct GPUs with kernel optimizations, FP8 GEMM, and low-latency KV prefix caching.
- **OpenCode Client Support**: Integrated Web UI and Terminal TUI featuring workspace tree search, git diff generation, and bash tool execution.
- **Multi-Engine Runtime Support**: Switch seamlessly between **vLLM** (production FP8 baseline), **llama.cpp** (deep context GGUF), **SGLang** (RadixAttention), and **vLLM Radiance** (MxFP4 W4A8).
- **Built-in Benchmarking Harness**: Standardized evaluation pipelines for **SWE-bench**, **GPQA Diamond**, and multi-engine throughput sweeps.

---

## Key Components

The architecture consists of four primary software and hardware layers:

1. **Inference Server Engines (`docker/`)**:
   - **vLLM (FP8 Baseline)**: Production serving engine with PagedAttention, KV prefix caching, and Hermes function tool parsing.
   - **llama.cpp (HIP Build)**: Native ROCm 7.x HIP build (`local/llama.cpp:rocm10-gfx1201`) optimized for ultra-deep 32k–64k context GGUF models.
   - **vLLM Radiance (MxFP4)**: 4-bit microscopic floating-point W4A8 quantization with AITER unified attention.
   - **SGLang**: Alternative serving runtime featuring RadixAttention KV cache sharing.
2. **OpenCode Agent Client (`opencode`)**:
   - Interactive Web UI (`http://localhost:4096`) and Terminal TUI (`./setup.sh -i`) featuring automated multi-file editing, git diff synthesis, workspace search, and terminal tool execution.
3. **Hardware Telemetry & Profiling Stack (`scripts/`)**:
   - High-frequency 250ms sysfs hwmon power collector (`collect_amd_power.py`), `rocm-smi` monitoring, and GPU architecture profile auto-detection (`gpu_profile.py`).
4. **Evaluation & Benchmarking Harness (`benchmark/`)**:
   - Automated SWE-bench Lite/Verified GitHub issue resolution pipeline, GPQA Diamond scientific reasoning harness, and multi-engine throughput sweeper.

---

## Prerequisites

### 1. Hardware Requirements
- **GPU**: AMD Radeon™ AI PRO R9700 (32 GB GDDR6, RDNA 4 `gfx1201`) or AMD Instinct MI350P (`gfx950`).
- **System Memory**: 32 GB system RAM minimum (64 GB+ recommended).
- **Storage**: 50 GB free NVMe disk space for container images, benchmarks, and model weights.

### 2. Software Requirements
- **Operating System**: Linux (Ubuntu 22.04 LTS or 24.04 LTS recommended).
- **AMD ROCm Driver**: Host ROCm driver installed (v6.3, v7.1, or v10.0 runtime).
- **Container Runtime**: Docker Engine 24.0+ and Docker Compose v2.20+.

### 3. Device Node Permissions
Grant your user account access to AMD KFD and DRI GPU device nodes (`/dev/kfd` and `/dev/dri`):

```bash
sudo usermod -aG video,render $USER
```
*(Log out and log back in for group membership updates to take effect).*

---

## Quickstart Guide

### 1. Launch the Stack
Initialize environment defaults and launch the vLLM inference container and OpenCode agent:

```bash
chmod +x setup.sh check.sh test.sh demo.sh accuracy.sh throughput.sh
./setup.sh
```

### 2. Verify Server Health
Check that the ROCm inference engine is healthy and responding to queries on port 8000:

```bash
./check.sh
```

### 3. Interact with OpenCode
Open your browser to [`http://localhost:4096`](http://localhost:4096) to use the OpenCode Web UI, or run interactive TUI mode:

```bash
./setup.sh -i
```

---

## Repository Directory Structure Overview

```text
/home/amd/workspace/coder/
├── setup.sh                  # Quickstart launcher (auto-routes vLLM, llama.cpp, or SGLang)
├── check.sh                  # Live health check & prompt verification script
├── test.sh                   # Unified accuracy + throughput dispatcher
├── accuracy.sh               # SWE-bench & GPQA accuracy runner
├── throughput.sh             # Primary throughput benchmark entrypoint
├── demo.sh                   # HTML5 physics simulation benchmark challenge demo
├── opencode.json             # Provider configuration for OpenCode client
├── scripts/                  # Executable python & shell benchmark scripts
│   ├── bench_throughput.sh   # Throughput benchmark suite
│   ├── bench_dual_gpu.sh     # Dual-GPU multi-mode evaluation suite (TP=2, P/D, DP=2)
│   ├── download_model.sh     # Model downloader for Qwen3.8-27B-Q4_K_M.gguf (~16.8 GB)
│   ├── inspect_dual_gpu.py   # Hardware topology & KV connector probe
│   ├── collect_amd_power.py  # Sysfs hwmon GPU power telemetry sampler
│   ├── measure_power.py      # AMD-SMI power telemetry daemon
│   ├── run_concurrency_sweep.py # Automated concurrency sweeper
│   └── qwen3_5.py            # Architecture patch for Qwen 3.5/3.8 models
├── docker/                   # Container Dockerfiles and Compose orchestration files
│   ├── Dockerfile.llamacpp-rocm-gfx1201 # ROCm 7.x gfx1201 HIP image build for llama.cpp
│   ├── Dockerfile.vllm-mxfp4  # vLLM container for Radiance MxFP4 W4A8
│   ├── docker-compose.yml    # Primary vLLM FP8 + OpenCode orchestration
│   ├── docker-compose.gguf.yml # llama.cpp GGUF orchestration
│   ├── docker-compose.mxfp4.yml # Radiance MXFP4 W4A8 orchestration
│   └── docker-compose.tp2.yml   # Tensor Parallelism (TP=2) 64 GB orchestration
├── docs/                     # Architectural reports, specifications, and deep-dive documentation
├── models/                   # Local GGUF and model weight storage
└── benchmark/                # SWE-bench & GPQA benchmarking harness scripts
```

### Directory Breakdown
- **Root (`/`)**: Contains top-level entrypoint scripts (`setup.sh`, `check.sh`, `demo.sh`, `test.sh`, `throughput.sh`, `accuracy.sh`) for stack initialization, verification, and evaluation.
- **`scripts/`**: Internal automation utilities, hardware probes (`inspect_dual_gpu.py`), power monitors (`collect_amd_power.py`), model downloaders (`download_model.sh`), and runtime architecture patches (`qwen3_5.py`).
- **`docker/`**: Container definition files (`Dockerfile.*`) and Docker Compose configurations (`docker-compose.*.yml`) for vLLM FP8, llama.cpp HIP GGUF, Radiance MxFP4, and multi-GPU topologies (`tp2`, `dp2`, `pd`).
- **`docs/`**: Technical documentation, architectural specifications, test plans, troubleshooting manuals, and empirical benchmark reports.
- **`benchmark/`**: SWE-bench and GPQA evaluation harness scripts (`run_benchmark.py`, `run_gpqa.py`, `bench_phases.py`, `pd_router.py`), requirements, and sample datasets.
- **`models/`**: Staging directory for local model weights, tokenizers, Jinja chat templates, and GGUF quantization files.

---

## Entrypoint Scripts Guide

The root directory contains 6 primary automation and benchmarking scripts. Below is a detailed breakdown of each script's purpose, usage, CLI flags, and expected outputs.

---

### 1. `setup.sh` — Stack Launcher & Environment Bootstrapper

- **Objective**: Bootstraps environment variables (`.env`), configures GPU profiles, downloads required container images, and starts the containerized ROCm inference engine (`vllm`, `llama.cpp`, `sglang`, or `mxfp4`) alongside the OpenCode agent.
- **Usage**:
  ```bash
  ./setup.sh                         # Default: Launch vLLM FP8 engine with Qwen3.8-27B-FP8
  ./setup.sh -e llama.cpp            # Launch llama.cpp GGUF engine
  ./setup.sh -e mxfp4                # Launch vLLM Radiance MxFP4 engine
  ./setup.sh -m Qwen/Qwen2.5-Coder-7B-Instruct # Override model selection
  ```
- **Key Options & Flags**:
  - `-e, --engine <vllm|llama.cpp|sglang|mxfp4>`: Inference engine runtime target (default: `vllm`).
  - `-m, --model <name>`: Model Hugging Face repository ID or local GGUF filename.
  - `-p, --port <port>`: Inference API port (default: `8000`).
  - `--opencode-port <port>`: OpenCode Web UI port (default: `4096`).
  - `-g, --gpu-profile <r9700|mi350p|auto>`: GPU profile preset (default: `auto`).
- **Expected Results & Outputs**:
  - Background Docker containers launched (`rocm-inference-server` and `opencode-client`).
  - Local environment configuration updated in `.env`.
  - OpenAI-compatible API ready at `http://localhost:8000/v1` and OpenCode Web UI ready at `http://localhost:4096`.

---

### 2. `check.sh` — Server Health & API Verification Check

- **Objective**: Performs automated health checks and functional verification against the active inference server, testing API endpoints, model registry, tool calling (Hermes parser), and code generation.
- **Usage**:
  ```bash
  ./check.sh                         # Run immediate health check against http://127.0.0.1:8000
  ./check.sh --wait 60               # Wait up to 60s for server weights to load on cold boot
  ./check.sh -p 8080                 # Check custom inference port 8080
  ```
- **Key Options & Flags**:
  - `-p, --port <port>`: Target inference server port (default: `8000` or from `.env`).
  - `-w, --wait <seconds>`: Wait/retry timeout for server `/health` to report HTTP 200 (default: `0`).
- **Expected Results & Outputs**:
  - Colorized terminal report detailing `/health` status, model registry response, structured function tool-calling validation, and code generation output.
  - Reviewable execution logs and markdown report saved to `_results/checks/check_summary_<timestamp>.md` and `check_<timestamp>.log`.

---

### 3. `demo.sh` — HTML5 Physics Coding Benchmark & Structural Audit

- **Objective**: Evaluates the active model on an industry-standard coding challenge (generating a complete, single-file HTML5 Canvas 2D water simulation) and executes an automated 6-point structural physics audit on the generated artifact.
- **Usage**:
  ```bash
  ./demo.sh                          # Generate water simulation and run automated 6-point audit
  ./demo.sh -s                       # Generate simulation and launch live preview server on http://localhost:3000
  ./demo.sh -i                       # Launch interactive OpenCode terminal TUI in demo workspace
  ./demo.sh -d                       # Display multi-model evaluation dashboard instructions
  ```
- **Key Options & Flags**:
  - `-i, --interactive`: Run OpenCode in interactive terminal TUI mode inside `opencode-water-sim/`.
  - `-s, --serve`: Automatically start a local HTTP preview server at completion.
  - `-p, --port <port>`: Preview web server port (default: `3000`).
  - `-m, --model <model>`: Override target model name (default: auto-detected from active server).
  - `-d, --dashboard`: Show OpenCode visual evaluation dashboard instructions.
- **Expected Results & Outputs**:
  - Generated application file `opencode-water-sim/index.html` containing Canvas element, particle math, collision bounds, and CSS styling.
  - Terminal scorecard reporting 6 structural audit checks (Canvas element, `requestAnimationFrame` loop, gravity/velocity physics math, collision bounds, mouse ripples, and CSS styling).
  - Live HTTP web preview server at `http://localhost:3000` (when using `-s`).

---

### 4. `test.sh` — Master Unified Benchmark Dispatcher

- **Objective**: Master dispatcher script that coordinates running both accuracy (SWE-bench & GPQA) and throughput benchmarking suites sequentially or selectively.
- **Usage**:
  ```bash
  ./test.sh -q                       # Run quick smoke test (3 accuracy samples + 128:64 throughput test)
  ./test.sh --accuracy -d --accuracy-limit 5  # Run accuracy suite on Diamond split (limit 5 samples)
  ./test.sh --throughput -e vllm -c 8 --test-cases 8192:1024 # Run 8192:1024 throughput at C=8
  ./test.sh --both -g auto -e mxfp4 -d -c 8   # Full accuracy + throughput evaluation
  ```
- **Key Options & Flags**:
  - **Suites**: `--both` (Default, runs accuracy then throughput), `--accuracy` (Run accuracy only), `--throughput` (Run throughput only).
  - **Shared Options**: `-g, --gpu-profile <auto|r9700|mi350p>`, `-e, --engine <vllm|mxfp4|llama.cpp|sglang>`, `-q, --quick`.
  - **Accuracy Flags**: `-s` (Sample), `-d` (Diamond/Lite), `-m` (Main/Verified), `-a` (All), `--accuracy-limit <N>`, `--swe-only`, `--gpqa-only`, `--eval`.
  - **Throughput Flags**: `-c, --concurrency <N>`, `--num-prompts <N>`, `--test-cases <I:O,...>`, `--compare-engines`.
- **Expected Results & Outputs**:
  - Aggregated performance summary output to terminal covering task accuracy, generation throughput, and TTFT latency.
  - Comprehensive archival report saved to `_results/test_run_summary_<timestamp>.md`.

---

### 5. `throughput.sh` — Token Throughput & Latency Sweep Suite

- **Objective**: Benchmarks live serving performance across prompt/output token length matrices ($8192:1024$, $1024:8192$, $1024:1024$), concurrency levels ($C=1, 2, 4, 8, 16$), and inference engines (vLLM, llama.cpp, SGLang, Radiance MxFP4).
- **Usage**:
  ```bash
  ./throughput.sh -q                 # Quick 128:64 smoke test
  ./throughput.sh -e vllm -c 8 --test-cases 8192:1024 # Benchmark 8192 input / 1024 output at C=8
  ./throughput.sh --compare-engines -q # Side-by-side benchmark comparing all engines
  ```
- **Key Options & Flags**:
  - `-e, --engine <vllm|mxfp4|llama.cpp|sglang|all>`: Target inference engine.
  - `--engines <list>`: Comma-separated list of engines to benchmark.
  - `-c, --concurrency <N>`: Concurrent request stream count (default: `1`).
  - `-n, --num-prompts <N>`: Total prompts per workload test case.
  - `--test-cases <I:O,...>`: Input:output token length pairs (e.g. `8192:1024,1024:8192`).
  - `-q, --quick`: Single 128:64 smoke test.
  - `--compare-engines`: Automated side-by-side comparison matrix across engines.
- **Expected Results & Outputs**:
  - Metrics detailing prompt ingestion rate (prefill tok/s), generation rate (decode tok/s), Time-To-First-Token (TTFT ms), Inter-Token Latency (ITL ms), and peak VRAM allocation.
  - JSON result files and markdown reports written to `_results/throughput/<timestamp>/`.

---

### 6. `accuracy.sh` — Model Accuracy Evaluator (SWE-bench & GPQA)

- **Objective**: Evaluates model software engineering task accuracy on **SWE-bench** (generating git patches for GitHub issues) and scientific reasoning accuracy on **GPQA Diamond**.
- **Usage**:
  ```bash
  ./accuracy.sh                      # Offline 3-sample smoke test
  ./accuracy.sh -d -n 5              # Evaluate first 5 problems of SWE-bench Lite & GPQA Diamond
  ./accuracy.sh --gpqa-only -d       # Evaluate GPQA Diamond scientific reasoning only
  ./accuracy.sh -m --eval            # Run SWE-bench Verified and execute Docker unit test harness
  ```
- **Key Options & Flags**:
  - **Dataset Tiers**: `-s, --sample` (Default 3-sample offline split), `-d, --diamond, -l, --lite` (SWE-bench Lite & GPQA Diamond), `-m, --main` (SWE-bench Verified & GPQA Main), `-a, --all` (Full dataset split).
  - `-n, --limit <N>`: Limit evaluation to N samples per benchmark.
  - `--swe-only`: Run SWE-bench software engineering tasks only.
  - `--gpqa-only`: Run GPQA scientific reasoning tasks only.
  - `--eval`: Execute automated Docker unit test verification using `swebench.harness`.
  - `-e, --engine <vllm|mxfp4|llama.cpp|sglang>`: Target inference engine.
  - `-g, --gpu-profile <auto|r9700|mi350p>`: Target GPU profile.
- **Expected Results & Outputs**:
  - Output predictions file (`predictions.jsonl`) containing generated git diff patches.
  - Benchmark metrics JSON (`benchmark_metrics.json`) recording patch formatting validity rate, task pass rate, and execution latency.
  - Archival report directory saved to `_results/<model_tag>_<timestamp>/`.

---

## Supported Engines & Model Precision

| Engine | Model Precision | Weights Footprint | Supported Context | Best Used For |
| :--- | :--- | :--- | :--- | :--- |
| **vLLM (Default)** | `Qwen/Qwen3.8-27B-FP8` | ~27.5 GB | 8,192 – 9,600 | Production baseline; uncompromised accuracy & tool calling. |
| **llama.cpp** | `Qwen3.8-27B-Q4_K_M.gguf` | ~16.8 GB | 32,768 – 65,536 | Deep context windows on single 32 GB GPU. |
| **vLLM Radiance** | `Qwen3.8-27B-Quark-AWQ-MXFP4` | ~14.2 GB | 16,384 – 32,768 | Microscopic FP4 W4A8 quantization with AITER attention. |
| **SGLang** | `Qwen3.8-27B-Q4_K_M.gguf` | ~16.8 GB | 16,384 – 32,768 | High-concurrency RadixAttention caching. |

Switch inference engines at startup:
```bash
./setup.sh --engine vllm       # Default FP8 vLLM engine
./setup.sh --engine llama.cpp  # GGUF llama.cpp engine
./setup.sh --engine mxfp4      # Radiance MxFP4 engine
```

---

## Documentation Sitemap (`docs/`)

For in-depth architectural analysis, benchmark datasets, container build specifications, and operational manuals, see the documentation in [`docs/`](docs/):

- **[MI350P vLLM Quark MXFP4 evaluation](docs/MI350P_MXFP4_VLLM_EVAL.md)**  
  Reproducible 25 Sep Instinct campaign: freeze HF MXFP4 control **79.35 C1 / 553.58 C8**; Babel Read **3793 GB/s**; C1 **40%** of that *conditional* roof. Isolated ASM/ksplit=1 GEMM wins **regressed serving**. DFLASH-3 is opt-in long-output only (101/116 greedy; TTFT/ITL p95 regress). Lab: `_results/priority_eval/EVAL.md`.

- **[Docker Compose & Container Specification Guide](docs/DOCKER_COMPOSE_GUIDE.md)**  
  Full Docker Compose file specifications, environment variables, `Dockerfile` build scripts, and plugin instructions.
- **[SWE-bench & GPQA Benchmarking Harness Guide](docs/BENCHMARKING_HARNESS.md)**  
  Complete benchmark harness setup, dataset splits, offline testing, and comparative evaluation scripts (`compare_engines.sh`, `compare_models.sh`).
- **[Troubleshooting & Operational Guide](docs/TROUBLESHOOTING.md)**  
  Resolutions for permission issues, iGPU collisions, OOM error tuning, kernel panics, and legacy ROCm image traps.
- **[Comprehensive Benchmark & Ablation Report](docs/COMPREHENSIVE_BENCHMARK_REPORT.md)**  
  Single-variable controlled ablations comparing ROCm 10 FP8 vs. vLLM-MXFP4 Radiance and llama.cpp HIP.
- **[Optimization Tracks Empirical Report](docs/OPTIMIZATION_TRACKS_EMPIRICAL_REPORT.md)**  
  Empirical results for prefix caching speedup (9.24× gain), chunk size sweep (2048 sweet spot), and interactive SLO pass rates.
- **[Dual R9700 Evaluation & P/D Architecture Report](docs/DUAL_R9700_EVALUATION_ARCHITECTURE.md)**  
  Dual-card hardware topology, Tensor Parallelism (TP=2) vs. Data Parallelism (DP=2), and Prefill/Decode Disaggregation analysis.
- **[Single R9700 Phase Profiling & Interference Report](docs/SINGLE_R9700_PHASE_PROFILING.md)**  
  Isolated prefill vs. steady decode, thermodynamic power profiles, and contention jitter analysis.
- **[Quantization Accuracy & Task Fidelity Report](docs/QUANTIZATION_ACCURACY_COMPARISON_REPORT.md)**  
  Accuracy breakdown across FP8, MxFP4, and Q4_K_M on SWE-bench and GPQA Diamond.
- **[Master Testplan Specification](docs/TESTPLAN.md)**  
  Standardized test methodology, formulas, efficiency criteria, and 6-stage connector readiness model.
