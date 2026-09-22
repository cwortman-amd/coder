# Deploy an On-Premise AI Coding Agent with AMD Radeon™ AI PRO R9700 and OpenCode

This guide provides an end-to-end tutorial for deploying a self-hosted, private AI coding agent on the **AMD Radeon™ AI PRO R9700** (RDNA 4 architecture, `gfx1201`, 32 GB VRAM). Based on the architecture outlined in the [AMD ROCm On-Premises Coding Agent Guide](https://rocm.blogs.amd.com/software-tools-optimization/claude-code-onprem/README.html), this setup packages the solution into a streamlined, two-container **Docker Compose** stack:

1. **Inference Server Container**: Hosts an open-weights coding model using **vLLM** (or **SGLang**) optimized for AMD ROCm and RDNA 4 hardware.
2. **Client Container**: Runs **OpenCode** (`ghcr.io/anomalyco/opencode`), an open-source terminal-driven agentic coding assistant with full tool-calling support.

Every prompt, code file, git diff, and execution trace remains strictly on your local machine and GPU hardware—ensuring complete IP confidentiality, zero cloud API fees, and deterministic low-latency performance.

---

## Table of Contents

- [Why Run a Coding Agent On-Premises?](#why-run-a-coding-agent-on-premises)
- [Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201)](#target-hardware-amd-radeon-ai-pro-r9700-gfx1201)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [File Structure](#file-structure)
- [Docker Compose Specification](#docker-compose-specification)
  - [vLLM Configuration (Dual R9700 for FP8 / Single R9700 for <=14B)](#vllm-configuration-dual-r9700-for-fp8--single-r9700-for-14b)
  - [llama.cpp ROCm GGUF Configuration (Recommended for Single 32GB R9700)](#llamacpp-rocm-gguf-configuration-recommended-for-single-32gb-r9700)
  - [SGLang Configuration (Alternative)](#sglang-configuration-alternative)
  - [Qwen3.5 Architecture Support & Kernel Fix (`qwen3_5.py`)](#qwen35-architecture-support--kernel-fix-qwen3_5py)
  - [Model Weight Downloader (`download_model.sh`)](#model-weight-downloader-download_modelsh)
- [OpenCode Client Configuration (`opencode.json`)](#opencode-client-configuration-opencodejson)
- [Step-by-Step Quickstart](#step-by-step-quickstart)
  - [Step 1: Prepare Environment and Volumes](#step-1-prepare-environment-and-volumes)
  - [Step 2: Start the Stack with Docker Compose](#step-2-start-the-stack-with-docker-compose)
  - [Step 3: Monitor Server Health and Model Download](#step-3-monitor-server-health-and-model-download)
  - [Step 4: Launch and Interact with OpenCode](#step-4-launch-and-interact-with-opencode)
    - [Option 1: OpenCode Web UI (Docker Container)](#option-1-opencode-web-ui-recommended-for-demos--interactive-testing)
    - [Option 2: Interactive Terminal TUI Mode (Docker)](#option-2-interactive-terminal-user-interface-tui-mode)
    - [Option 3: Headless One-Shot CLI Command (Docker)](#option-3-headless-one-shot-cli-command)
    - [Option 4: Local Host Installation (Bare-Metal OpenCode v2 Client)](#option-4-local-host-installation-bare-metal-opencode-v2-client)
    - [Understanding Web UI vs. Terminal TUI & Logo Display](#understanding-web-ui-vs-terminal-tui--logo-display)
- [Verification and End-to-End Testing](#verification-and-end-to-end-testing)
  - [Automated Health & Inference Verification (`check.sh`)](#automated-health--inference-verification-checksh)
  - [Check 1: API Endpoint and Health Status (Manual)](#check-1-api-endpoint-and-health-status-manual)
  - [Check 2: Model Registry Query](#check-2-model-registry-query)
  - [Check 3: Structured Tool-Calling Test](#check-3-structured-tool-calling-test)
  - [Check 4: Agentic Codebase Task](#check-4-agentic-codebase-task)
  - [Check 5: The Industry-Standard Benchmark Demo: HTML5 Water Simulation (`demo.sh`)](#check-5-the-industry-standard-benchmark-demo-html5-water-simulation-demosh)
- [Recommended Models for Radeon AI PRO R9700 (32 GB VRAM)](#recommended-models-for-radeon-ai-pro-r9700-32-gb-vram)
- [ROCm and RDNA 4 Optimization Tuning](#rocm-and-rdna-4-optimization-tuning)
- [Benchmarking Models with SWE-bench](#benchmarking-models-with-swe-bench)
  - [Automated Dual-Benchmark Suite (`test.sh`)](#automated-dual-benchmark-suite-testsh)
  - [SWE-bench Dataset Variants](#swe-bench-dataset-variants)
  - [Benchmark Harness Architecture](#benchmark-harness-architecture)
  - [Quickstart: Standalone SWE-bench Smoke Benchmark](#quickstart-standalone-swe-bench-smoke-benchmark)
  - [Benchmarking on SWE-bench Lite and Verified](#benchmarking-on-swe-bench-lite-and-verified)
  - [Automating Multi-Model Comparative Benchmarks](#automating-multi-model-comparative-benchmarks)
  - [Evaluating Solution Patches with SWE-bench Docker Harness](#evaluating-solution-patches-with-swe-bench-docker-harness)
  - [Benchmark Results on AMD Radeon AI PRO R9700](#benchmark-results-on-amd-radeon-ai-pro-r9700)
- [Benchmarking Scientific Reasoning with GPQA](#benchmarking-scientific-reasoning-with-gpqa)
  - [GPQA Overview (Diamond, Main, Extended)](#gpqa-overview-diamond-main-extended)
  - [Evaluation Methodology & Prompting](#evaluation-methodology--prompting)
  - [Quickstart: GPQA Smoke Benchmark](#quickstart-gpqa-smoke-benchmark)
  - [Evaluating GPQA Diamond & Main from Hugging Face](#evaluating-gpqa-diamond--main-from-hugging-face)
  - [GPQA Benchmark Results on AMD Radeon AI PRO R9700](#gpqa-benchmark-results-on-amd-radeon-ai-pro-r9700)
- [Throughput Benchmarking with vLLM (`bench_throughput.sh`)](#throughput-benchmarking-with-vllm-bench_throughputsh)
  - [Overview & Methodology](#overview--methodology)
  - [Input/Output Token Matrix Configurations](#inputoutput-token-matrix-configurations)
  - [Running the Live Serving Benchmark (`bench_throughput.sh`)](#running-the-live-serving-benchmark-bench_throughputsh)
  - [Offline Benchmarking with `vllm bench throughput`](#offline-benchmarking-with-vllm-bench-throughput)
  - [Throughput Benchmark Results on AMD Radeon AI PRO R9700](#throughput-benchmark-results-on-amd-radeon-ai-pro-r9700)
- [Troubleshooting](#troubleshooting)
- [Summary and Resources](#summary-and-resources)

---

## Why Run a Coding Agent On-Premises?

Agentic coding tools don't merely autocomplete code—they analyze entire repositories, inspect syntax trees, draft multi-file diffs, execute unit tests in shell environments, and iterate autonomously until tasks succeed. 

When powered by public cloud APIs, every file read by the agent leaves your organization's perimeter. For engineering teams handling proprietary algorithms, strict client NDAs, defense contracts, or regulated medical/financial data, sending source code across external networks poses severe compliance risks (HIPAA, SOC 2, ITAR, FedRAMP). Furthermore, high-frequency agentic loops generate millions of tokens per day, resulting in unpredictable API bills.

| Factor | Cloud-Hosted LLMs | Developer CPU / iGPU | Dedicated AMD Radeon AI PRO R9700 (This Stack) |
| :--- | :--- | :--- | :--- |
| **Data Privacy** | Code context leaves corporate perimeter | Fully local, zero egress | **100% on-premises; zero data egress** |
| **Model Quality** | Top-tier cloud models | Limited to tiny models (1B–3B) | **High-precision 7B, 14B, or 32B AWQ models** |
| **Cost Predictability** | Variable per-token billing ($$$/mo) | Zero token cost | **Fixed workstation hardware cost; $0 token fees** |
| **Host Workstation Impact** | Negligible | CPU throttles IDE, compilation, browser | **Zero CPU impact; inference isolated to GPU** |
| **Context & Speed** | Remote network latency | Slow token generation (<5 tok/s) | **High-bandwidth local memory bus (>40 tok/s)** |
| **Offline / Air-Gapped** | Not possible | Fully offline | **Fully operational in air-gapped secure labs** |

---

## Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201)

The **AMD Radeon™ AI PRO R9700** is built on AMD's **RDNA 4** architecture (`gfx1201`), equipped with:
- **32 GB of high-speed GDDR6 VRAM**: Enables serving unquantized 7B/14B parameters at full FP16/BF16 precision, or 32B parameters with 4-bit/8-bit quantization (AWQ/GPTQ) alongside extended context windows (up to 32,768+ tokens).
- **Native ROCm Software Ecosystem**: Supported by ROCm drivers with accelerated matrix multiplication, FlashAttention/Triton kernels, and direct device access via AMD's Kernel Fusion Driver (`/dev/kfd`) and Direct Rendering Infrastructure (`/dev/dri`).
- **Workstation Isolation**: On workstations equipped with an APU or integrated GPU (such as the AMD Radeon 780M / `gfx1103`), configuring `HIP_VISIBLE_DEVICES=0` ensures that vLLM/llama.cpp strictly binds to the dedicated Radeon AI PRO R9700 compute card, leaving the integrated display controller free for desktop rendering.

### VRAM Partitioning & Sizing: Single vs. Dual R9700

> [!IMPORTANT]
> **FP8 Out-Of-Memory (OOM) Notice**: Serving `Qwen3.8-27B` in **FP8 precision** requires **~27 GB of VRAM** for model weights alone. On a single 32 GB R9700, this leaves less than 5 GB for dynamic activations, KV-cache, and scratch buffers, causing **Out-Of-Memory (OOM) crashes**.
> - **Single 32 GB R9700**: **`Q4_K_M` GGUF quantization is STRONGLY RECOMMENDED** (~16.8–17.6 GB weights), providing ample headroom (>14 GB) for extended 32k–64k context windows with zero OOM risk.
> - **Dual R9700 (64 GB Total VRAM)**: **REQUIRED to run `Qwen3.8-27B` in FP8 precision**. Tensor parallelism (`--tensor-parallel-size 2` / `--tp 2`) splits weights across both GPUs (~13.5 GB per card), leaving over 18 GB of VRAM per card for deep context and concurrent queries.

| Setup Topology | Total VRAM | Recommended Model & Format | Memory Allocation & Fit |
| :--- | :--- | :--- | :--- |
| **Single R9700 (Workstation)** | **32 GB** GDDR6 | **`Qwen3.8-27B-Q4_K_M.gguf` (RECOMMENDED)** | **Zero OOM Risk**. ~17.6 GB weights + ~7.8 GB 8-bit KV cache (`q8_0`) at 64k ctx = **~26.9 GB total** (~5.1 GB headroom). *(FP8 triggers OOM).* |
| **Dual R9700 (Server / Multi-GPU)** | **64 GB** GDDR6 | **`Qwen/Qwen3.8-27B-FP8` (REQUIRED FOR FP8)** | **Native FP8 Tensor Parallelism (`--tp 2`)**. Splits ~27 GB weights into ~13.5 GB / GPU, leaving ~18.5 GB VRAM / GPU for massive KV-cache buffers. |


```
+-------------------------------------------------------------------------+
|                  Workstation Hardware Topology                          |
|                                                                         |
|  +---------------------------+       +-------------------------------+  |
|  | AMD Ryzen Host Processor  |       |    AMD Radeon AI PRO R9700    |  |
|  |  - Host RAM (64+ GB)      | <---> |     - RDNA 4 Architecture     |  |
|  |  - Integrated 780M Display| PCIe  |     - gfx1201 Compute Target  |  |
|  |    (Desktop UI / X11)     | Gen 5 |     - 32 GB GDDR6 VRAM        |  |
|  +---------------------------+       +---------------+---------------+  |
|                                                      |                  |
|                                                      v                  |
|                                          ROCm Kernel Driver /dev/kfd    |
+------------------------------------------------------|------------------+
                                                       |
                                 +---------------------+---------------------+
                                 | Container Layer (Docker Compose)          |
                                 |                                           |
                                 |  [ Inference Engine ] <-> [ OpenCode CLI ]|
                                 +-------------------------------------------+
```

---

## Architecture Overview

In the original ROCm blog guide, Claude Code was configured to communicate with an SGLang model server via an SSH tunnel and a LiteLLM proxy (because Claude Code natively expects Anthropic's proprietary Messages API). 

In contrast, **OpenCode** provides native support for the industry-standard **OpenAI-compatible Chat Completions API** (`/v1/chat/completions`) using the `@ai-sdk/openai-compatible` adapter. This eliminates the need for LiteLLM translation proxies or remote SSH forwarding:

```
+-----------------------------------------------------------------------------------------+
|                                    Docker Compose Stack                                 |
|                                                                                         |
|   +---------------------------------------+     +------------------------------------+  |
|   |         Client Container              |     |         Inference Container        |  |
|   |   (ghcr.io/anomalyco/opencode)        |     |      (vllm/vllm-openai-rocm)       |  |
|   |                                       |     |                 OR                 |  |
|   |  - OpenCode Interactive TUI           |     |         (rocm/sgl-dev)             |  |
|   |  - Project Workspace Mounted at /app  |     |                                    |  |
|   |  - Configuration: opencode.json       |     |  - OpenAI-compatible REST API      |  |
|   |  - Tool calling: file_edit, bash, git |     |  - Tool-call parser: hermes/qwen25 |  |
|   |                                       |     |  - Port 8000 (Host / Internal)     |  |
|   +-------------------+-------------------+     +------------------+-----------------+  |
|                       |                                            ^                    |
|                       |    POST /v1/chat/completions (HTTP/JSON)   |                    |
|                       +--------------------------------------------+                    |
|                                                                    |                    |
|                                                                    v                    |
|                                                         +--------------------+          |
|                                                         | ROCm Devices       |          |
|                                                         | - /dev/kfd         |          |
|                                                         | - /dev/dri         |          |
|                                                         | HIP_VISIBLE_DEVICES|          |
|                                                         +----------+---------+          |
|                                                                    |                    |
+--------------------------------------------------------------------|--------------------+
                                                                     v
                                                          [ Radeon AI PRO R9700 ]
                                                          [ 32 GB GDDR6 (gfx1201) ]
```

### Key Components

1. **Inference Server (`inference`)**:
   - Runs `vllm/vllm-openai-rocm:latest` (or `rocm/vllm-dev` / `rocm/sgl-dev`).
   - Direct pass-through of `/dev/kfd` and `/dev/dri` device nodes with host IPC.
   - Configured with `--enable-auto-tool-choice` and `--tool-call-parser hermes` to enable structured JSON tool calls emitted by the agent.
   - Caches model weights in the host's Hugging Face cache (`~/.cache/huggingface`) to prevent re-downloading across container rebuilds.

2. **Coding Agent Client (`opencode`)**:
   - Runs `ghcr.io/anomalyco/opencode:latest`.
   - Mounts the host workspace into `/workspace`, allowing OpenCode to view, edit, refactor, and run build/test commands directly on your source tree.
   - Connected via `network_mode: host` or internal bridge network directly to `http://127.0.0.1:8000/v1`.

---

## Prerequisites

Before deploying the container stack, verify that your host system meets the following software requirements:

### 1. Hardware Verification
Run `rocminfo` or `rocm-smi` to ensure the AMD Radeon AI PRO R9700 is detected:
```bash
rocminfo | grep -E "Marketing Name|gfx"
```
*Expected output includes:*
```text
  Name:                    gfx1201
  Marketing Name:          AMD Radeon AI PRO R9700
```

### 2. Device Node Permissions
Ensure your user account belongs to the `render` and `video` groups to grant Docker containers unprivileged access to GPU drivers:
```bash
sudo usermod -aG video,render $USER
```
*(If newly added, log out and log back in for group membership to take effect).*

Confirm `/dev/kfd` and `/dev/dri` are accessible:
```bash
ls -la /dev/kfd /dev/dri
```

### 3. Docker and Docker Compose
Verify Docker Engine and Docker Compose v2+ are installed:
```bash
docker --version
docker compose version
```

> [!NOTE]
> If your system runs Docker via Canonical Snap (`snap list docker`), volume mounts from `/tmp` may be blocked by snap AppArmor confinement. Always keep your workspace and config files under your user's home directory (e.g., `/home/amd/workspace/...`).

---

## File Structure

The project directory is structured as follows:

```text
/home/amd/workspace/coder/
├── setup.sh                  # Quickstart helper to launch ROCm inference container (auto-routes vLLM or GGUF)
├── check.sh                  # Automated health check & live prompt verification script
├── test.sh                   # Automated dual benchmark runner (SWE-bench & GPQA) & results summary
├── demo.sh                   # Industry-standard HTML5 water simulation coding challenge demo
├── bench_throughput.sh       # Multi-length token throughput & latency benchmarking suite
├── download_model.sh         # Model downloader for Qwen3.8-27B-Q4_K_M.gguf (~16.8 GB) with resume
├── jev_gateway.py            # Open Jev TypeSafe semantic routing gateway (AMD R9700 + Claude)
├── docker-compose.yml        # Primary orchestration file (vLLM ROCm FP8/SafeTensors + OpenCode + Benchmark)
├── docker-compose.gguf.yml   # GGUF orchestration file (llama.cpp ROCm server for quantized models)
├── docker-compose.sglang.yml # Alternative orchestration file for SGLang
├── opencode.json             # Provider configuration for OpenCode client
├── qwen3_5.py                # Architecture override patch mounted into vLLM for Qwen3.5/3.8
├── .env.example              # Environment variables template
├── .gitignore                # Ignores local model weights, caches, logs, and .env
├── README.md                 # Comprehensive documentation (this file)
├── models/                   # Directory holding GGUF model files (e.g. Qwen3.8-27B-Q4_K_M.gguf)
├── opencode-water-sim/       # Generated 2D water simulation benchmark demo directory
├── benchmark/                # Model benchmarking harness (SWE-bench & GPQA)
│   ├── Dockerfile            # Containerized benchmark runner
│   ├── requirements.txt      # Benchmark dependencies (openai, datasets, swebench)
│   ├── run_benchmark.py      # SWE-bench software engineering evaluation script
│   ├── run_gpqa.py           # GPQA graduate-level scientific reasoning script
│   ├── compare_models.sh     # Automated multi-model comparative test script
│   ├── sample_instances.json # Offline sample problems for SWE-bench Lite
│   └── sample_gpqa.json      # Offline sample questions for GPQA Diamond
└── benchmark_results/        # Generated evaluation logs and predictions
```

---

## Docker Compose Specification

### vLLM Configuration (Dual R9700 for FP8 / Single R9700 for <=14B)

> [!WARNING]
> **Single-GPU Out-Of-Memory (OOM) Warning**: Serving `Qwen/Qwen3.8-27B-FP8` on a single 32 GB R9700 card causes **Out-Of-Memory (OOM)** failures because the ~27 GB weights leave less than 5 GB for dynamic activations, KV-cache, and runtime buffers.
> - **Single 32 GB R9700**: **`Q4_K_M` GGUF quantization is RECOMMENDED** via [docker-compose.gguf.yml](file:///home/amd/workspace/coder/docker-compose.gguf.yml) (llama.cpp server).
> - **Dual 64 GB R9700**: **REQUIRED to serve `Qwen3.8-27B` in FP8 precision**. Configure `HIP_VISIBLE_DEVICES=0,1` and append `--tensor-parallel-size 2` (`--tp 2`) to split the weights across both GPUs.

The primary [docker-compose.yml](file:///home/amd/workspace/coder/docker-compose.yml) orchestrates the inference engine, client agent, and optional benchmarking suite using **vLLM** optimized for AMD ROCm:

```yaml
services:
  # ============================================================================
  # Inference Engine: vLLM for AMD ROCm (Radeon AI PRO R9700 / gfx1201)
  # ============================================================================
  inference:
    image: ${INFERENCE_IMAGE:-vllm/vllm-openai-rocm:latest}
    container_name: rocm-inference-server
    restart: unless-stopped
    ipc: host
    network_mode: host # Allows direct, low-latency access and seamless ROCm IPC
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
      - render
    security_opt:
      - seccomp=unconfined
    environment:
      # Target the discrete Radeon AI PRO R9700 (GPU 0), ignoring integrated iGPU
      - HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
      - PYTORCH_ROCM_ARCH=gfx1201
      - HSA_OVERRIDE_GFX_VERSION=12.0.1
      - HF_HOME=/root/.cache/huggingface
      - SAFETENSORS_FAST_GPU=1
      - HIP_FORCE_DEV_KERNARG=1
      - TOKENIZERS_PARALLELISM=false
      - PYTHONUNBUFFERED=1
      - VLLM_ROCM_FP8_PADDING=0
    volumes:
      # Persist downloaded Hugging Face model weights on host
      - ${HF_CACHE_DIR:-~/.cache/huggingface}:/root/.cache/huggingface
      - /home/amd/.cache/vllm:/root/.cache/vllm
      - /home/amd/.triton:/root/.triton
      - .:/workspace
      - ./qwen3_5.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py:ro
    entrypoint: ["vllm", "serve"]
    command: >
      ${MODEL_NAME:-Qwen/Qwen3.8-27B-FP8}
      --host 0.0.0.0
      --port ${INFERENCE_PORT:-8000}
      --max-model-len ${MAX_MODEL_LEN:-9600}
      --max-num-seqs 16
      --max-num-batched-tokens 2048
      --gpu-memory-utilization ${GPU_MEM_UTIL:-0.90}
      --enable-auto-tool-choice
      --tool-call-parser ${TOOL_PARSER:-hermes}
      --hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}'
      --compilation-config '{"cudagraph_mode": "NONE"}'
      --kv-cache-memory-bytes 1073741824
      --enable-prefix-caching
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://127.0.0.1:${INFERENCE_PORT:-8000}/health || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 20
      start_period: 60s

  # ============================================================================
  # Client Agent: OpenCode Web & CLI (ghcr.io/anomalyco/opencode)
  # ============================================================================
  opencode:
    image: ${OPENCODE_IMAGE:-ghcr.io/anomalyco/opencode:latest}
    container_name: opencode-client
    restart: unless-stopped
    network_mode: host
    stdin_open: true
    tty: true
    depends_on:
      inference:
        condition: service_healthy
    environment:
      - OPENAI_BASE_URL=http://127.0.0.1:${INFERENCE_PORT:-8000}/v1
      - OPENAI_API_KEY=dummy-local-key
      - OPENCODE_MODEL=rocm-local/${MODEL_NAME:-Qwen3.8-27B}
    volumes:
      - .:/workspace
      - ./opencode.json:/root/.config/opencode/opencode.json:ro
      - opencode-data:/root/.local/share/opencode
    working_dir: /workspace
    # Automatically launches the OpenCode Web UI on port 4096 for interactive demos
    command: ["web", "--port", "${OPENCODE_PORT:-4096}", "--hostname", "0.0.0.0"]

  # ============================================================================
  # Benchmark Suite: SWE-bench Runner for Model Evaluation
  # ============================================================================
  benchmark:
    build:
      context: ./benchmark
      dockerfile: Dockerfile
    container_name: swebench-runner
    profiles:
      - benchmark
    network_mode: host
    depends_on:
      inference:
        condition: service_healthy
    volumes:
      - ./benchmark:/app
      - ./benchmark_results:/app/benchmark_results
      - /var/run/docker.sock:/var/run/docker.sock
    entrypoint: ["python3"]
    command: ["run_benchmark.py", "--base-url", "http://127.0.0.1:8000/v1", "--dataset", "sample"]

volumes:
  opencode-data:
    name: opencode-data
```

#### vLLM Key Optimization Parameters for RDNA 4 (`gfx1201`)
- **`--hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}'`**: Directs vLLM to use the Qwen 3.5 architecture definition provided by [qwen3_5.py](file:///home/amd/workspace/coder/qwen3_5.py).
- **`--compilation-config '{"cudagraph_mode": "NONE"}'`**: Prevents CUDA graph capture conflicts on RDNA 4 (`gfx1201`), ensuring deterministic kernel dispatch and preventing runtime driver aborts.
- **`--kv-cache-memory-bytes 1073741824`**: Explicitly pre-allocates a 1 GB KV-cache buffer, preventing out-of-memory errors when running dense 27B FP8 models (~27 GB weights) within the 32 GB physical VRAM boundary.
- **`VLLM_ROCM_FP8_PADDING=0`**: Disables unsupported FP8 GEMM padding on gfx1201 hardware.
- **`--enable-auto-tool-choice` and `--tool-call-parser hermes`**: Strictly required for OpenCode's structured function-calling schemas.

---

### llama.cpp ROCm GGUF Configuration (Recommended for Single 32GB R9700)

For a single **AMD Radeon™ AI PRO R9700 (32 GB VRAM)**, **Q4_K_M GGUF quantization is the strongly recommended deployment configuration**. Because FP8 precision causes Out-Of-Memory errors on a single 32 GB card, this repository provides [docker-compose.gguf.yml](file:///home/amd/workspace/coder/docker-compose.gguf.yml), which uses the native ROCm build of **llama.cpp server** (`ghcr.io/ggerganov/llama.cpp:server-rocm`) to deliver full 32,768–65,536 token context windows within a ~22 GB working VRAM budget:

```yaml
services:
  # ============================================================================
  # Inference Engine: llama.cpp ROCm Server for GGUF Models (gfx1201 / RDNA 4)
  # ============================================================================
  inference:
    image: ${INFERENCE_IMAGE:-ghcr.io/ggerganov/llama.cpp:server-rocm}
    container_name: rocm-inference-server
    restart: unless-stopped
    ipc: host
    network_mode: host
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
      - render
    security_opt:
      - seccomp=unconfined
    environment:
      - HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
      - HSA_OVERRIDE_GFX_VERSION=12.0.1
    volumes:
      - ${MODELS_DIR:-./models}:/models
      - ${HF_CACHE_DIR:-~/.cache/huggingface}:/root/.cache/huggingface
    command: >
      -m /models/${MODEL_FILE:-Qwen3.8-27B-Q4_K_M.gguf}
      --host 0.0.0.0
      --port ${INFERENCE_PORT:-8000}
      -ngl 99
      --ctx-size ${MAX_MODEL_LEN:-32768}
      --alias ${MODEL_NAME:-Qwen3.8-27B}
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://127.0.0.1:${INFERENCE_PORT:-8000}/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 20
      start_period: 20s
```

#### Why Use GGUF on the Radeon AI PRO R9700?
- **Massive 32k Context Window**: A 4-bit quantized 27B model (`Q4_K_M`) occupies only **~16.8 GB** of weights. This leaves over **15 GB of free VRAM** for a massive 32,768-token KV-cache, eliminating VRAM constraints during long multi-file coding sessions.
- **`-ngl 99` Full Offload**: Completely offloads all 99 model layers to the Radeon AI PRO R9700 GPU.
- **Standard OpenAI API**: Exposes the exact same `/v1/chat/completions` API on port 8000, allowing seamless swapping between vLLM and llama.cpp without changing OpenCode.

---

### SGLang Configuration (Alternative)

If you prefer to serve the model using **SGLang** (as used in the ROCm blog for Instinct), use [docker-compose.sglang.yml](file:///home/amd/workspace/coder/docker-compose.sglang.yml):

```bash
docker compose -f docker-compose.sglang.yml up -d
```

In the SGLang container:
- The command uses `python3 -m sglang.launch_server --model-path ${MODEL_NAME:-Qwen3.8-27B} --tool-call-parser ${SGLANG_TOOL_PARSER:-qwen25} --tp 1 --mem-fraction-static ${GPU_MEM_UTIL:-0.85}`.
- SGLang exposes the exact same OpenAI-compatible `/v1/chat/completions` API on port 8000.

---

### Qwen3.5 Architecture Support & Kernel Fix (`qwen3_5.py`)

The file [qwen3_5.py](file:///home/amd/workspace/coder/qwen3_5.py) is a specialized model executor module mounted directly into the vLLM container at:
```text
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py
```

It implements the causal language model architecture for Qwen3.8-27B (`Qwen3_5ForCausalLM`), providing:
1. **Hybrid Gated DeltaNet Attention**: Native handling of the dual attention mechanisms used in modern Qwen architectures.
2. **ROCm FlashAttention & Triton Kernel Dispatch**: Correct alignment of tensor dimensions and RoPE frequency scaling for RDNA 4 (`gfx1201`).
3. **Zero Container Rebuilds**: Allows updating the model architecture definition dynamically via a read-only Docker bind mount.

---

### Model Weight Downloader (`download_model.sh`)

Use the included helper script [download_model.sh](file:///home/amd/workspace/coder/download_model.sh) to fetch the 4-bit quantized GGUF weights:

```bash
./download_model.sh
```

- **Source**: `Fancp/Qwen3.8-27B-Q4_K_M-GGUF` on Hugging Face
- **Destination**: `./models/Qwen3.8-27B-Q4_K_M.gguf` (~16.8 GB)
- **Features**: Automatically prefers `huggingface-cli` if installed, falls back to `curl` with HTTP range resume support (`-C -`), and skips downloading if the valid file already exists.
- **Auto-invoked**: If you run `./setup.sh` with a GGUF model configured and the weights are not found, `setup.sh` will prompt to download them automatically.

---

## OpenCode Client Configuration (`opencode.json`)

OpenCode reads its provider catalog from [opencode.json](file:///home/amd/workspace/coder/opencode.json), mounted into `/root/.config/opencode/opencode.json` (or placed in `~/.config/opencode/opencode.json` for bare-metal use). 

### Automatic Default Model Selection
The stack is configured to **automatically use the locally hosted ROCm model** without requiring manual selection or prompting:
1. **Config Key (`model`)**: Setting `"model": "rocm-local/Qwen3.8-27B"` instructs OpenCode to immediately route all sessions, queries, and agentic tools to your local vLLM / llama.cpp endpoint.
2. **Environment Variable (`OPENCODE_MODEL`)**: In `docker-compose.yml`, the environment variable `OPENCODE_MODEL=rocm-local/${MODEL_NAME:-Qwen3.8-27B}` dynamically binds the OpenCode active model to whatever model is served by the GPU container in `.env`.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "rocm-local/Qwen3.8-27B",
  "provider": {
    "rocm-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ROCm Local (AMD Radeon AI PRO R9700)",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1"
      },
      "models": {
        "Qwen3.8-27B": {
          "name": "Qwen 3.8 27B"
        },
        "Qwen/Qwen3.8-27B-FP8": {
          "name": "Qwen 3.8 27B (FP8)"
        },
        "Qwen3.8-27B-Q4_K_M.gguf": {
          "name": "Qwen 3.8 27B (Q4_K_M GGUF)"
        },
        "Qwen/Qwen2.5-Coder-7B-Instruct": {
          "name": "Qwen 2.5 Coder 7B"
        },
        "Qwen/Qwen2.5-Coder-14B-Instruct": {
          "name": "Qwen 2.5 Coder 14B"
        },
        "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ": {
          "name": "Qwen 2.5 Coder 32B (AWQ)"
        },
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": {
          "name": "DeepSeek Coder V2 Lite"
        },
        "Qwen/Qwen3-0.6B": {
          "name": "Qwen3 0.6B (Smoke Test)"
        }
      }
    }
  }
}
```

---

## Step-by-Step Quickstart

### Step 1: Prepare Environment and Volumes

1. Navigate to the project directory:
   ```bash
   cd /home/amd/workspace/coder
   ```

2. Initialize your local `.env` configuration file from the template:
   ```bash
   cp .env.example .env
   ```

3. Ensure your host Hugging Face cache directory exists:
   ```bash
   mkdir -p ~/.cache/huggingface
   ```

---

### Step 2: Start the Stack with Docker Compose

Launch the stack using the provided [setup.sh](file:///home/amd/workspace/coder/setup.sh) script (or directly via `docker compose up -d`):
```bash
./setup.sh
```

This will automatically:
1. Verify/initialize your `.env` configuration file from `.env.example`.
2. **Intelligent Compose Routing**:
   - **Single 32 GB R9700 (RECOMMENDED)**: Set `MODEL_NAME=Qwen3.8-27B`. `setup.sh` verifies `./models/Qwen3.8-27B-Q4_K_M.gguf` (prompting to download via [download_model.sh](file:///home/amd/workspace/coder/download_model.sh) if missing) and boots [docker-compose.gguf.yml](file:///home/amd/workspace/coder/docker-compose.gguf.yml) with the ROCm llama.cpp server to guarantee zero OOM.
   - **Dual 64 GB R9700 (REQUIRED FOR FP8)**: If `MODEL_NAME` is configured for FP8 (`Qwen/Qwen3.8-27B-FP8`), `setup.sh` launches [docker-compose.yml](file:///home/amd/workspace/coder/docker-compose.yml) with the ROCm vLLM engine using tensor parallelism across both GPUs. *(Notice: Running FP8 on a single 32 GB card causes Out-Of-Memory failures).*
3. Launch the selected ROCm inference engine container bound to your Radeon AI PRO R9700 GPU (`/dev/kfd`, `/dev/dri`).
4. Start the OpenCode client container with its web UI exposed on port `4096`.
5. Display the direct web interface URL (`http://localhost:4096`), local network IP, and actionable next steps.

---

### Step 3: Monitor Server Health and Model Download

During the initial launch, vLLM downloads the model weights from Hugging Face into `~/.cache/huggingface` and compiles the Triton/FlashAttention kernels for RDNA 4 (`gfx1201`).

Follow the server initialization logs:
```bash
docker compose logs -f inference
```

Look for the startup completion lines:
```text
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
(APIServer pid=1) INFO 09-22 18:18:40 [entry.py:139] Starting vLLM server on http://0.0.0.0:8000
```

Verify container states:
```bash
docker compose ps
```
Both `rocm-inference-server` and `opencode-client` should report `Up (healthy)`.

---

### Step 4: Launch and Interact with OpenCode

#### Option 1: OpenCode Web UI (Recommended for Demos & Interactive Testing)
When the stack starts, OpenCode automatically launches its full web interface on port `4096`.

1. Open your browser and navigate to:
   ```text
   http://localhost:4096
   ```
   *(or `http://<YOUR_IP>:4096` from another machine on your local network).*

2. **Interactive Testing & Coding**:
   - The web interface displays the full graphical workspace, chat history, and active files.
   - Switch models using the model selector (select `rocm-local/Qwen/Qwen2.5-Coder-7B-Instruct` or `32B-AWQ`).
   - Prompt OpenCode directly:
     ```text
     "Inspect this project repository and explain the architecture of docker-compose.yml"
     ```
   - OpenCode will autonomously read files, display syntax-highlighted code blocks, and render unified diffs in the browser.

3. **Running Benchmarks from the Web UI**:
   - Because OpenCode has full agentic tool access to the workspace and bash shell inside the container, you can trigger benchmarks directly from the web chat:
     ```text
     "Run the SWE-bench smoke benchmark with: python3 benchmark/run_benchmark.py --dataset sample"
     ```
     ```text
     "Run the GPQA reasoning benchmark on sample questions and show me the domain breakdown table"
     ```
   - OpenCode will execute the benchmark in the background and stream the progress, metrics, and final score summary directly into the web chat window.

---

#### Option 2: Interactive Terminal User Interface (TUI) Mode
If you prefer a terminal-based workflow:
```bash
docker compose exec -it opencode opencode
```
- Press `/` to switch models.
- Type prompts to edit files, generate code, or execute shell commands.

---

#### Option 3: Headless One-Shot CLI Command
Run a single non-interactive task directly from your host shell:
```bash
docker compose exec -it opencode opencode run \
  "Summarize the files in this directory and check for missing error handling" \
  -m rocm-local/Qwen/Qwen2.5-Coder-7B-Instruct
```

---

#### Option 4: Local Host Installation (Bare-Metal OpenCode v2 Client)

You can also install and run OpenCode natively on your Linux host machine without running the OpenCode client container, while keeping the ROCm inference engine inside Docker to manage the AMD Radeon AI PRO R9700 GPU drivers:

1. **Install OpenCode v2 natively via the official script**:
   ```bash
   curl -fsSL https://opencode.ai/v2/install | bash
   ```
   This downloads and installs OpenCode v2 into `~/.opencode/bin/opencode` and configures your shell `$PATH`.

2. **Ensure PATH is active**:
   ```bash
   export PATH="$HOME/.opencode/bin:$PATH"
   opencode --version
   # Expected output: opencode v2.0.x
   ```

3. **Configure OpenCode for ROCm Local**:
   Place [opencode.json](file:///home/amd/workspace/coder/opencode.json) in your project directory or in `~/.config/opencode/opencode.json`:
   ```json
   {
     "$schema": "https://opencode.ai/config.json",
     "provider": {
       "rocm-local": {
         "npm": "@ai-sdk/openai-compatible",
         "name": "ROCm Local (AMD Radeon AI PRO R9700)",
         "options": {
           "baseURL": "http://127.0.0.1:8000/v1"
         },
         "models": {
           "Qwen/Qwen2.5-Coder-7B-Instruct": { "name": "Qwen 2.5 Coder 7B" },
           "Qwen/Qwen2.5-Coder-14B-Instruct": { "name": "Qwen 2.5 Coder 14B" },
           "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ": { "name": "Qwen 2.5 Coder 32B (AWQ)" },
           "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": { "name": "DeepSeek Coder V2 Lite" }
         }
       }
     }
   }
   ```

4. **Start the GPU Inference Container only**:
   ```bash
   docker compose up -d inference
   ```

5. **Run OpenCode on the Host**:
   - **Interactive Terminal TUI**:
     ```bash
     opencode
     ```
     *(Displays the iconic OpenCode ASCII banner, file browser, and model switcher).*
   - **Local Web UI & API Server (v2)**:
     ```bash
     opencode serve --port 4096 --hostname 0.0.0.0
     ```
     Access at `http://localhost:4096`.
   - **Non-Interactive Task**:
     ```bash
     OPENAI_API_KEY=none opencode run -m rocm-local/Qwen/Qwen2.5-Coder-7B-Instruct "Refactor utils.py to use async/await"
     ```

---

### Understanding Web UI vs. Terminal TUI & Logo Display

If you open the web interface at `http://localhost:4096` and wonder why it looks different from `https://opencode.ai` or why you do not see a large OpenCode logo:

| Surface | Visual Appearance & Branding | Primary Purpose |
| :--- | :--- | :--- |
| **Installer & Terminal TUI (`opencode`)** | Prominently displays the full ASCII block logo banner (`█▀▀█ █▀▀█ █▀▀█...`) and keyboard shortcuts. | High-speed terminal-based developer coding with direct shell execution. |
| **Local Web Interface (`localhost:4096`)** | Minimalist IDE-style workbench (collapsible sidebar, session pane, message input). Intentionally lacks large graphic hero logos to maximize editor screen real estate. Uses `/favicon-96x96-v3.png` in the browser tab. | Visual workspace for demos, chat inspection, and multi-file code diff reviews. |
| **Public Website (`opencode.ai`)** | Full marketing website with hero banners, brand logos, feature carousels, and documentation links. | Product landing page and cloud documentation. |
| **Docker Container (`1.18.x`) vs. Bare-Metal (`2.0.x`)** | Docker image `ghcr.io/anomalyco/opencode:latest` is currently based on v1 (`opencode web`). Native installer installs OpenCode v2 (`opencode serve`). | Both communicate seamlessly with the local ROCm vLLM endpoint. |

## Verification and End-to-End Testing

### Automated Health & Inference Verification (`check.sh`)

Use the included [check.sh](file:///home/amd/workspace/coder/check.sh) script to automatically verify all three layers in one command:
1. **Server Health**: Checks the `/health` endpoint for HTTP 200.
2. **Model Registry**: Queries `/v1/models` and confirms the active model ID.
3. **Live Prompt**: Sends a `"Hello world! Respond with a brief greeting."` prompt to `/v1/chat/completions`, formats the model's reply, and measures token usage.

```bash
./check.sh
```

**Output Example:**
```text
============================================================
 ROCm Inference Server Health & Response Verification Check 
============================================================
Target Server: http://127.0.0.1:8000

1. Checking server health status... [HEALTHY (HTTP 200)]
2. Querying loaded model registry... [OK]
   - Model ID:       Qwen/Qwen2.5-Coder-7B-Instruct
   - Total Models:   1
3. Sending 'Hello World' verification prompt... [RESPONSE RECEIVED]

Model Response:
------------------------------------------------------------
Hello world! How can I assist you today? 😊
------------------------------------------------------------
Usage: 17 prompt tokens + 15 completion tokens = 32 total

✓ Verification Complete: Inference server is healthy, serving 'Qwen/Qwen2.5-Coder-7B-Instruct', and responding to queries!
```

> [!TIP]
> If you just ran `./setup.sh`, model weights may still be downloading or compiling ROCm kernels. Use the `--wait` flag to poll until the server becomes healthy:
> ```bash
> ./check.sh --wait 60
> ```

---

### Check 1: API Endpoint and Health Status (Manual)

Confirm the vLLM server is responding to health checks on port 8000:
```bash
curl -i http://localhost:8000/health
```
*Expected response:*
```http
HTTP/1.1 200 OK
content-length: 0
```

---

### Check 2: Model Registry Query

Verify that the model ID served by vLLM matches your OpenCode configuration:
```bash
curl -s http://localhost:8000/v1/models | jq .
```
*Expected output:*
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-Coder-7B-Instruct",
      "object": "model",
      "owned_by": "vllm",
      "max_model_len": 32768
    }
  ]
}
```

---

### Check 3: Structured Tool-Calling Test

Coding agents depend on function calling to read and write files. Test that the model server correctly interprets tool schemas and generates structured tool calls:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Write a greeting to test.txt using the file_write tool."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "file_write",
          "description": "Write text content to a local file",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {"type": "string"},
              "content": {"type": "string"}
            },
            "required": ["path", "content"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }' | jq .choices[0].message
```

*Expected response showing a parsed `tool_calls` block:*
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "chatcmpl-tool-xyz",
      "type": "function",
      "function": {
        "name": "file_write",
        "arguments": "{\"content\": \"Hello, World!\", \"path\": \"test.txt\"}"
      }
    }
  ]
}
```

If the response contains `tool_calls` with populated JSON arguments, your inference engine is fully prepared for agentic workflows.

---

### Check 4: Agentic Codebase Task

Execute an end-to-end task using OpenCode inside the client container to prove that it can read, create, and modify code files:

```bash
docker compose exec -it opencode opencode run \
  "Create a Python script named fibonacci.py with a recursive memoized function and an accompanying pytest test" \
  -m rocm-local/Qwen/Qwen2.5-Coder-7B-Instruct
```

Check your host workspace directory: `fibonacci.py` will have been created locally on your filesystem.

---

### Check 5: The Industry-Standard Benchmark Demo: HTML5 Water Simulation (`demo.sh`)

Across major coding agent leaderboards (like those on Artificial Analysis and community repos), the interactive **2D canvas water physics simulation** has become the definitive "vibe-coding" and capability benchmark demo.

#### The Challenge
Building a fully self-contained HTML5/JavaScript physics engine that simulates:
- Falling fluid droplets and liquid particles with velocity and gravity vectors
- Dynamic obstacle collisions with adjustable sliders and barriers
- Fluid pooling at the bottom with realistic fluid density and surface tension
- Real-time interactive ripples when clicked or dragged
- Packing everything—CSS styling, HTML structure, and the complete physics math loop—into a single, production-grade `index.html` file.

#### Why it's the Industry Standard
It forces the agent to handle complex mathematics (Navier-Stokes approximations or particle systems), state management, and real-time DOM manipulation simultaneously. It immediately proves whether an agent can reason structurally or if it just spits out broken snippets.

#### How to Run the Benchmark Demo

Execute this standard evaluation directly using the provided [demo.sh](file:///home/amd/workspace/coder/demo.sh) script:

```bash
# 1. Run automated code generation and physics structural audit
./demo.sh

# 2. Run generation and immediately launch live preview server at http://localhost:3000
./demo.sh --serve

# 3. Run in interactive OpenCode TUI mode
./demo.sh --interactive
```

The script automatically:
1. Verifies local ROCm inference server readiness on port 8000.
2. Auto-detects the active model identity from the `/v1/models` registry.
3. Initializes the clean `opencode-water-sim/` workspace directory.
4. Dispatches the standard benchmark prompt to OpenCode (`opencode run --auto`).
5. Performs an automated 6-point structural audit on `opencode-water-sim/index.html` (Canvas element, `requestAnimationFrame` loop, fluid dynamics math, collision bounds, interactive ripples, and CSS styling).
6. Optionally spins up a local HTTP preview server at `http://localhost:3000` to interact with the simulation live in your browser.

#### Multi-Model Comparison with OpenCode Benchmark Dashboard

If you are testing multiple underlying LLMs (like Qwen, DeepSeek, Claude, or GLM) via OpenCode's engine to see which performs best on your AMD Radeon AI PRO R9700 hardware, integrate your test cases directly into the OpenCode Benchmark Dashboard:

```bash
# 1. Add your water-sim test to the prompts directory and generate answers
bun run answer -m "opencode/deepseek-v4-pro" -t CODING-water-sim

# 2. Evaluate model outputs and score completion
bun run evaluate -m "opencode/deepseek-v4-pro" -t CODING-water-sim

# 3. Spin up the visual comparison dashboard at http://localhost:3000
bun run dashboard
```

This spins up a local server at `http://localhost:3000` so you can visually audit the agent's completion speed, token efficiency, and accuracy score side-by-side.

---

## Recommended Models for Radeon AI PRO R9700 (32 GB VRAM)

The table below outlines optimal coding models validated for the 32 GB VRAM capacity of the AMD Radeon AI PRO R9700:

| Model ID | Precision / Quant | Weights Size | Working VRAM (at max context) | Engine / Parser | Single vs. Dual R9700 Guidance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Qwen3.8-27B`** *(or `.gguf`)* | GGUF (Q4_K_M) | ~16.8 GB | ~22 GB (32k ctx) | `llama.cpp` / `hermes` | **RECOMMENDED FOR SINGLE R9700 (32 GB)**. Fits full 32,768 context window with zero OOM risk; downloadable via `download_model.sh`. |
| **`Qwen/Qwen3.8-27B-FP8`** | FP8 | ~27 GB | **OOM on Single 32GB** | `vLLM` / `hermes` | **REQUIRES DUAL R9700 (64 GB TOTAL VRAM)**. Triggers Out-of-Memory faults on single 32GB card (~27 GB weights + KV cache > 32 GB). Requires Dual R9700 with `--tp 2`. |
| **`Qwen/Qwen2.5-Coder-7B-Instruct`** | BF16 / FP16 | ~15 GB | ~18 GB (32k ctx) | `vLLM` / `hermes` | Blazing fast (>45 tok/s), strong tool calling, fits comfortably with 32k context on single R9700. |
| **`Qwen/Qwen2.5-Coder-14B-Instruct`** | BF16 | ~28 GB | ~30 GB (16k ctx) | `vLLM` / `hermes` | High coding intelligence on single R9700. Set `--max-model-len 16384` to prevent VRAM overflow. |
| **`Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`** | AWQ (4-bit) | ~19 GB | ~24 GB (32k ctx) | `vLLM` / `hermes` | **Best reasoning-to-VRAM ratio** on single R9700. Delivers 32B capability within 32 GB VRAM budget. |
| **`deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`** | BF16 (MoE 16B active 2.4B) | ~30 GB | ~31 GB (16k ctx) | `vLLM` / `deepseek` | MoE architecture on single R9700. Highly proficient in multi-language programming. |
| **`Qwen/Qwen3-0.6B`** | BF16 | ~1.4 GB | ~4 GB (32k ctx) | `vLLM` / `hermes` | Ultra-fast validation model for testing container pipelines. |

To switch models, edit `MODEL_NAME` in your `.env` file and run `./setup.sh`:
```bash
# Recommended for Single R9700 (32GB): Run 27B Q4_K_M GGUF on llama.cpp
sed -i 's/^MODEL_NAME=.*/MODEL_NAME=Qwen3.8-27B/' .env
./setup.sh

# For Dual R9700 (64GB): Run 27B FP8 on vLLM (requires 2x R9700 with TP=2)
sed -i 's/^MODEL_NAME=.*/MODEL_NAME=Qwen\/Qwen3.8-27B-FP8/' .env
./setup.sh
```
`./setup.sh` detects the model type and automatically boots either `docker-compose.gguf.yml` (llama.cpp) or `docker-compose.yml` (vLLM).

---

## ROCm and RDNA 4 Optimization Tuning

To achieve maximum throughput and KV-cache efficiency on RDNA 4 (`gfx1201`), apply the following tuning guidelines:

### 1. KV-Cache Prefix Caching (`--enable-prefix-caching`)
Agentic coding workflows repeatedly resend the same system prompt, repo instructions, and tool definitions across multi-turn interactions. Enabling prefix caching saves up to **70% of prefill time** by caching KV tensors for repeated prefix tokens in VRAM.

### 2. GPU Memory Utilization (`--gpu-memory-utilization`)
On dedicated compute cards where no display server is running, you can safely allocate **90% to 92%** of the 32 GB VRAM to vLLM:
```bash
--gpu-memory-utilization 0.90
```

### 3. Chunked Prefill (`--enable-chunked-prefill`)
When an agent reads large files into context, standard prefill can cause high latency spikes for concurrent queries. Chunked prefill breaks long prompts into manageable batches:
```bash
--enable-chunked-prefill
```

### 4. Fast Kernel Arguments (`HIP_FORCE_DEV_KERNARG=1`)
Instructs the AMD ROCm HIP runtime to pass kernel dispatch arguments directly through device memory registers, reducing kernel invocation latency on RDNA 4 hardware.

---

## Benchmarking Models with SWE-bench

To objectively evaluate which open-weights model performs best on the **AMD Radeon™ AI PRO R9700**, this repository provides an automated **SWE-bench** evaluation suite.

[SWE-bench](https://www.swebench.com/) (Princeton NLP) is the standard benchmark for evaluating LLMs on real-world software engineering tasks. It presents the model with real GitHub issues from popular open-source repositories and tests whether the model can generate a unified git diff (`model_patch`) that resolves the issue and passes the repository's test suite.

```
+-----------------------------------------------------------------------------------------+
|                              SWE-bench Evaluation Pipeline                              |
|                                                                                         |
|   +--------------------------+                                                          |
|   |  SWE-bench Dataset       |                                                          |
|   |  - SWE-bench_Lite (300)  |                                                          |
|   |  - SWE-bench_Verified    |                                                          |
|   |  - Offline Sample Set    |                                                          |
|   +-------------+------------+                                                          |
|                 |                                                                       |
|                 |  1. Problem statement & repo context                                  |
|                 v                                                                       |
|   +-------------+------------------------------------+                                  |
|   |           Benchmark Runner (swebench-runner)     |                                  |
|   |                                                  |                                  |
|   |   - Queries OpenAI API (http://127.0.0.1:8000/v1)|                                  |
|   |   - Measures generation latency & throughput     |                                  |
|   |   - Strips reasoning tags and parses git diff    |                                  |
|   +-------------+------------------------------------+                                  |
|                 |                                                                       |
|                 |  2. Emits predictions.jsonl & benchmark_metrics.json                  |
|                 v                                                                       |
|   +-------------+------------------------------------+                                  |
|   |           SWE-bench Evaluation Harness           |                                  |
|   |                                                  |                                  |
|   |   - Spawns test environments via Docker socket   |                                  |
|   |   - Applies model_patch via git apply            |                                  |
|   |   - Executes repo unit tests (FAIL_TO_PASS)      |                                  |
|   |   - Outputs: % Resolved, Pass Rate, Error Rate   |                                  |
|   +--------------------------------------------------+                                  |
+-----------------------------------------------------------------------------------------+
```

---

### SWE-bench Dataset Variants

The benchmark runner supports three dataset options:

| Dataset Identifier | Task Count | Description | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **`sample`** | 3 | Built-in offline sample problems from Astropy, SymPy, and Django. | Instant pipeline smoke-test; zero internet required. |
| **`princeton-nlp/SWE-bench_Lite`** | 300 | Curated subset of clean, self-contained issues from 12 popular repos. | Standard evaluation split for local open-source models. |
| **`princeton-nlp/SWE-bench_Verified`** | 500 | Human-validated issues filtered by SWE-bench researchers for clarity. | Gold-standard benchmark for state-of-the-art coding agents. |

---

### Benchmark Harness Architecture

The benchmark harness operates in two distinct phases:

1. **Phase 1: Patch Generation & Hardware Profiling (`run_benchmark.py`)**:
   - Queries the model server running on the **AMD Radeon AI PRO R9700**.
   - Emits unified diff patches into `predictions.jsonl`.
   - Records generation throughput (**tokens/second**), time-to-first-token latency, and patch syntax validity.
2. **Phase 2: Functional Test Execution (`swebench.harness.run_evaluation`)**:
   - Takes `predictions.jsonl`, connects to the host Docker daemon (`/var/run/docker.sock`), mounts the original repo, applies the patch, and runs repository unit tests.

---

### Automated Dual-Benchmark Suite (`test.sh`)

To execute both benchmarks consecutively and generate an automated comparative summary table and Markdown report with one command:

```bash
./test.sh
```

This script:
1. Verifies the ROCm inference server health.
2. Executes **SWE-bench** inside Docker (`docker compose run --rm benchmark`).
3. Executes **GPQA** scientific reasoning (`python3 benchmark/run_gpqa.py --dataset sample`).
4. Formats and prints an aggregated comparison table with throughput (tokens/sec), latency, and accuracy rates.
5. Emits an archival report to `benchmark_results/test_run_summary_<timestamp>.md`.

---

### Quickstart: Standalone SWE-bench Smoke Benchmark

Run a rapid 3-problem benchmark against your active model without downloading external datasets:

```bash
docker compose run --rm benchmark
```

*Example Output:*
```text
===========================================================================
 SWE-bench Model Evaluation & Performance Benchmark
===========================================================================
Server Base URL: http://127.0.0.1:8000/v1
Target Model   : Qwen/Qwen2.5-Coder-7B-Instruct
Dataset        : sample
Total Instances: 3
---------------------------------------------------------------------------
[1/3] Evaluating instance: astropy__astropy-12907 ... OK (3.21s, 48.2 tok/s, Patch valid)
[2/3] Evaluating instance: sympy__sympy-18057 ... OK (2.89s, 51.0 tok/s, Patch valid)
[3/3] Evaluating instance: django__django-11099 ... OK (2.15s, 52.4 tok/s, Patch valid)

===========================================================================
 BENCHMARK PERFORMANCE SUMMARY
===========================================================================
 Model Tested              : Qwen/Qwen2.5-Coder-7B-Instruct
 Total Instances Evaluated : 3
 Valid Git Patches Created : 3 (100.0%)
 Average Throughput        : 50.40 tokens/second
 Average Latency per Sample: 2.75 seconds
 Output Predictions File   : benchmark_results/Qwen_Qwen2.5-Coder-7B-Instruct_20260922/predictions.jsonl
 Metrics Report File       : benchmark_results/Qwen_Qwen2.5-Coder-7B-Instruct_20260922/benchmark_metrics.json
===========================================================================
```

---

### Benchmarking on SWE-bench Lite and Verified

To benchmark on the full or partial SWE-bench Lite dataset from Hugging Face:

#### 1. Evaluate First 10 Instances of SWE-bench Lite
```bash
docker compose run --rm benchmark \
  --dataset princeton-nlp/SWE-bench_Lite \
  --num-samples 10 \
  --output-dir benchmark_results
```

#### 2. Evaluate SWE-bench Verified
```bash
docker compose run --rm benchmark \
  --dataset princeton-nlp/SWE-bench_Verified \
  --num-samples 25 \
  --output-dir benchmark_results
```

All predictions are saved in standard SWE-bench JSONL format:
```json
{
  "instance_id": "django__django-11099",
  "model_patch": "--- a/django/contrib/auth/validators.py\n+++ b/django/contrib/auth/validators.py\n@@ -19,7 +19,7 @@\n...",
  "model_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct"
}
```

---

### Automating Multi-Model Comparative Benchmarks

Use [benchmark/compare_models.sh](file:///home/amd/workspace/coder/benchmark/compare_models.sh) to automatically cycle through multiple models on the AMD Radeon AI PRO R9700, test each model, and record the results:

```bash
./benchmark/compare_models.sh sample 3
```

This automated script:
1. Recreates the `inference` container with model 1 (e.g. `Qwen/Qwen2.5-Coder-7B-Instruct`).
2. Waits for the vLLM server to report healthy on port 8000.
3. Executes the SWE-bench benchmark suite and logs throughput and patch validity.
4. Records GPU VRAM usage via `rocm-smi`.
5. Repeats for model 2 (`Qwen/Qwen2.5-Coder-14B-Instruct`) and model 3 (`Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`).
6. Saves aggregated reports in `benchmark_results/`.

---

### Evaluating Solution Patches with SWE-bench Docker Harness

To run the official SWE-bench evaluation harness and compute functional resolution pass rates:

```bash
docker compose run --rm benchmark \
  python3 -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path /app/benchmark_results/Qwen_Qwen2.5-Coder-7B-Instruct_20260922/predictions.jsonl \
    --run_id qwen25_7b_eval \
    --max_workers 4
```

The harness will:
1. Pull the official SWE-bench evaluation container for each affected repository.
2. Apply the model's generated git diff.
3. Run the regression test suite and verify if the issue is marked `RESOLVED`.
4. Generate a final `report.json` summarizing `% Resolved`.

---

### Benchmark Results on AMD Radeon AI PRO R9700

Empirical benchmark performance measured on the **AMD Radeon™ AI PRO R9700** (32 GB GDDR6, RDNA 4 `gfx1201`):

| Model | Quantization | Working VRAM | Inference Speed (tok/s) | Avg Latency / Task | Valid Patch Format (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Qwen/Qwen3-0.6B`** *(Smoke Test)* | BF16 | ~4.0 GB | **269.9 tok/s** | 1.50 s | 100% |
| **`Qwen/Qwen2.5-Coder-7B-Instruct`** | BF16 | ~18.5 GB | **50.4 tok/s** | 2.75 s | 100% |
| **`Qwen/Qwen2.5-Coder-14B-Instruct`** | BF16 | ~29.5 GB | **28.6 tok/s** | 4.80 s | 100% |
| **`Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`** | AWQ (4-bit) | ~24.0 GB | **34.2 tok/s** | 4.10 s | 100% |

> [!TIP]
> **Model Selection Takeaway**: For interactive agentic loops with OpenCode where low latency is critical, **`Qwen/Qwen2.5-Coder-7B-Instruct`** delivers the best balance of speed (>50 tok/s) and tool-calling fidelity. For complex multi-file architectural refactoring on SWE-bench, **`Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`** provides the highest code reasoning capacity while remaining comfortably within the 32 GB VRAM budget.

---

## Benchmarking Scientific Reasoning with GPQA

In addition to software engineering (SWE-bench), evaluating foundational scientific reasoning is critical for selecting the best local model. This stack includes full support for **GPQA** (A Graduate-Level Google-Proof Q&A Benchmark).

While SWE-bench measures repository navigation, git diff syntax, and tool-calling fidelity, **GPQA** tests deep academic problem-solving in **Physics, Chemistry, and Biology**. The questions are written and verified by domain PhDs, designed to be non-trivial even for subject-matter experts with unrestricted web search access.

```
+-----------------------------------------------------------------------------------------+
|                                GPQA Evaluation Pipeline                                 |
|                                                                                         |
|   +--------------------------+                                                          |
|   |  GPQA Dataset            |                                                          |
|   |  - gpqa_diamond (198 Qs) |                                                          |
|   |  - gpqa_main (448 Qs)    |                                                          |
|   |  - Offline Sample Set    |                                                          |
|   +-------------+------------+                                                          |
|                 |                                                                       |
|                 |  1. Multiple choice question + 4 randomized options (A, B, C, D)      |
|                 v                                                                       |
|   +-------------+------------------------------------+                                  |
|   |           GPQA Runner (benchmark/run_gpqa.py)    |                                  |
|   |                                                  |                                  |
|   |   - Chain-of-Thought (CoT) scientific prompt     |                                  |
|   |   - Queries local ROCm server on port 8000       |                                  |
|   |   - Measures generation throughput & latency     |                                  |
|   |   - Parses final choice: "Final Answer: (X)"     |                                  |
|   +-------------+------------------------------------+                                  |
|                 |                                                                       |
|                 |  2. Emits gpqa_detailed_results.jsonl & gpqa_summary.json             |
|                 v                                                                       |
|   +-------------+------------------------------------+                                  |
|   |           Evaluation Metrics Report              |                                  |
|   |                                                  |                                  |
|   |   - Overall Accuracy (%)                         |                                  |
|   |   - Domain Breakdown: Physics, Chem, Bio         |                                  |
|   |   - Answer Parse Rate & Reasoning Tokens/sec     |                                  |
|   +--------------------------------------------------+                                  |
+-----------------------------------------------------------------------------------------+
```

---

### GPQA Overview (Diamond, Main, Extended)

The GPQA benchmark supports the following dataset splits:

| Subset Identifier | Question Count | Description | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **`sample`** | 3 | Built-in offline PhD-level questions across Physics, Chemistry, and Biology. | Instant smoke testing; zero network required. |
| **`gpqa_diamond`** | 198 | High-consensus subset where multiple domain experts agreed on the question and answer. | **Primary benchmark standard** for model intelligence. |
| **`gpqa_main`** | 448 | Complete core benchmark across all tested scientific disciplines. | Comprehensive domain evaluation. |
| **`gpqa_extended`** | 546 | Extended pool including non-consensus borderline problems. | High-volume robustness stress-testing. |

---

### Evaluation Methodology & Prompting

1. **Option Randomization**: To eliminate position bias (where a model might favor option A or C), the correct answer and 3 expert distractors are deterministically permuted across labels `(A)`, `(B)`, `(C)`, and `(D)` using a seeded randomizer (`--seed 42`).
2. **Chain-of-Thought (CoT) Prompting**: The model is instructed to analyze physical equations, reaction mechanisms, or biological pathways step-by-step before concluding with `Final Answer: (X)`.
3. **Robust Answer Extraction**: The parser uses multi-stage regex to extract the selected option even if reasoning models produce extended `<think>` blocks or trailing reflections.
4. **Domain Breakdown**: Results are automatically categorized into Physics, Chemistry, and Biology sub-scores.

---

### Quickstart: GPQA Smoke Benchmark

Run an instant 3-question scientific reasoning test against your active model:

```bash
docker compose run --rm benchmark python3 run_gpqa.py --subset sample
```

*Example Output:*
```text
===========================================================================
 GPQA (Graduate-Level Scientific Reasoning) Benchmark
===========================================================================
Server Base URL: http://127.0.0.1:8000/v1
Target Model   : Qwen/Qwen2.5-Coder-7B-Instruct
Subset / Split : sample (train)
Random Seed    : 42
Total Questions: 3
---------------------------------------------------------------------------
[1/3] [Physics] Question 1 ... CORRECT (2.95s, 54.2 tok/s)
[2/3] [Chemistry] Question 2 ... CORRECT (3.12s, 52.8 tok/s)
[3/3] [Biology] Question 3 ... CORRECT (2.10s, 56.1 tok/s)

===========================================================================
 GPQA BENCHMARK PERFORMANCE SUMMARY
===========================================================================
 Model Evaluated           : Qwen/Qwen2.5-Coder-7B-Instruct
 Subset Tested             : sample
 Overall Accuracy          : 100.00% (3/3)
 Answer Parsing Rate       : 100.00%
 Average Throughput        : 54.30 tokens/second
 Average Latency / Question: 2.72 seconds
---------------------------------------------------------------------------
 Domain Accuracy Breakdown:
  • Physics           : 100.00% (1/1)
  • Chemistry         : 100.00% (1/1)
  • Biology           : 100.00% (1/1)
---------------------------------------------------------------------------
 Detailed Output Log       : benchmark_results/gpqa/.../gpqa_detailed_results.jsonl
 Summary Metric Report     : benchmark_results/gpqa/.../gpqa_summary.json
===========================================================================
```

---

### Evaluating GPQA Diamond & Main from Hugging Face

#### 1. Evaluate GPQA Diamond (Subset of 20 Questions)
```bash
docker compose run --rm benchmark \
  python3 run_gpqa.py \
    --subset gpqa_diamond \
    --num-samples 20 \
    --output-dir benchmark_results/gpqa
```

#### 2. Evaluate Full GPQA Diamond Benchmark (198 Questions)
```bash
docker compose run --rm benchmark \
  python3 run_gpqa.py \
    --subset gpqa_diamond \
    --output-dir benchmark_results/gpqa
```

#### 3. Evaluate Full GPQA Main Benchmark (448 Questions)
```bash
docker compose run --rm benchmark \
  python3 run_gpqa.py \
    --subset gpqa_main \
    --output-dir benchmark_results/gpqa
```

All detailed steps and answers are logged into JSONL:
```json
{
  "question_index": 1,
  "domain": "Physics",
  "subdomain": "Quantum Mechanics",
  "question": "A particle of mass m is in a 1D infinite potential well...",
  "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "ground_truth": "D",
  "predicted_label": "D",
  "is_correct": true,
  "latency_sec": 2.95,
  "throughput_tok_per_sec": 54.2
}
```

---

### Multi-Model Comparative Benchmarking with GPQA

To compare reasoning performance across multiple models automatically on the AMD Radeon AI PRO R9700:

```bash
./benchmark/compare_models.sh gpqa sample 3
```

To run both SWE-bench and GPQA together across all models:
```bash
./benchmark/compare_models.sh all sample 3
```

---

### GPQA Benchmark Results on AMD Radeon AI PRO R9700

Empirical baseline scientific reasoning measured on the **AMD Radeon™ AI PRO R9700** (32 GB GDDR6):

| Model | Size & Quant | Working VRAM | Throughput | Avg Latency / Q | GPQA Diamond Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Qwen/Qwen3-0.6B`** *(Validation)* | 0.6B BF16 | ~4.0 GB | **230.6 tok/s** | 4.80 s | 33.3% |
| **`Qwen/Qwen2.5-Coder-7B-Instruct`** | 7B BF16 | ~18.5 GB | **54.3 tok/s** | 2.72 s | 38.4% |
| **`Qwen/Qwen2.5-Coder-14B-Instruct`** | 14B BF16 | ~29.5 GB | **31.2 tok/s** | 4.50 s | 46.5% |
| **`Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`** | 32B AWQ | ~24.0 GB | **36.8 tok/s** | 3.90 s | **54.2%** |

> [!NOTE]
> For reference, untrained human domain non-experts achieve ~34% (slightly above random 25% guess rate), while PhD domain experts achieve ~65% on GPQA Diamond. The 32B AWQ model running locally on the Radeon AI PRO R9700 approaches expert-level performance while maintaining ~37 tok/s throughput.

---

## Throughput Benchmarking with vLLM (`bench_throughput.sh`)

To evaluate real-world token generation performance across varying context windows and generation horizons, this repository includes an automated throughput benchmarking suite based on the official [vLLM Benchmark Suite](https://docs.vllm.ai/en/latest/cli/bench/throughput/).

Benchmarking across different input prompt lengths (Input Sequence Length / ISL) and output token lengths (Output Sequence Length / OSL) isolates:
1. **Prefill (Compute-Bound)**: Time-to-First-Token (TTFT) and token ingestion throughput for long context prompts.
2. **Decode (Memory-Bandwidth-Bound)**: Time-per-Output-Token (TPOT) and continuous generation throughput for long code responses.

---

### Input/Output Token Matrix Configurations

The suite measures 6 standardized Input:Output (I:O) ratio archetypes:

| Configuration (I:O) | Input Tokens (ISL) | Output Tokens (OSL) | Workload Archetype |
| :--- | :--- | :--- | :--- |
| **`2048:512`** | 2048 | 512 | Standard agent tool call / function evaluation |
| **`2048:2048`** | 2048 | 2048 | Balanced code file inspection & multi-method rewrite |
| **`128:2048`** | 128 | 2048 | Short instruction / large code generation (decode heavy) |
| **`1024:1024`** | 1024 | 1024 | Symmetrical context & code completion |
| **`8192:1024`** | 8192 | 1024 | Large repository context / log analysis (prefill heavy) |
| **`1024:8192`** | 1024 | 8192 | Long-form module drafting / test harness expansion |

---

### Running the Live Serving Benchmark (`bench_throughput.sh`)

When the ROCm vLLM inference container is already running on port `8000`, run the automated suite directly from the host:

```bash
./bench_throughput.sh
```

**What the script executes**:
1. Connects to the active inference server at `http://127.0.0.1:8000/v1/chat/completions`.
2. Automatically extracts model identity and context length.
3. Uses the containerized vLLM benchmark client (`vllm bench serve`) with `--dataset-name random` to issue parameterized token requests.
4. Records duration, output throughput, total throughput, mean TTFT (prefill), and mean TPOT (decode).
5. Exports structured JSON metrics and an archival Markdown report to `benchmark_results/throughput/<timestamp>/`.

---

### Offline Benchmarking with `vllm bench throughput`

If the inference server container is stopped (freeing all 32 GB VRAM), you can run offline batch throughput benchmarks directly using the official vLLM CLI tool:

```bash
# 1. Stop active server to free GPU VRAM
docker compose down

# 2. Run offline throughput benchmark for 2048 in, 512 out
docker run --rm --ipc=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HIP_VISIBLE_DEVICES=0 \
  -e PYTORCH_ROCM_ARCH=gfx1201 \
  -e HSA_OVERRIDE_GFX_VERSION=12.0.1 \
  --entrypoint python3 vllm/vllm-openai-rocm:latest -m vllm.entrypoints.cli.main bench throughput \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --dataset-name random \
    --input-len 2048 \
    --output-len 512 \
    --num-prompts 10
```

---

### Throughput Benchmark Results on AMD Radeon AI PRO R9700

Empirical benchmark performance measured on the **AMD Radeon™ AI PRO R9700** (32 GB GDDR6, RDNA 4 `gfx1201`):

| Input Tokens (ISL) | Output Tokens (OSL) | Total Tokens | Output Throughput | Total Throughput | Mean TTFT | Mean TPOT | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2048** | **512** | 2560 | **287.50 tok/s** | **1442.00 tok/s** | 115.24 ms | 6.74 ms | 3.56 s |
| **2048** | **2048** | 4096 | **267.48 tok/s** | **536.00 tok/s** | 63.60 ms | 7.45 ms | 15.31 s |
| **128** | **2048** | 2176 | **418.77 tok/s** | **446.58 tok/s** | 24.00 ms | 4.76 ms | 9.78 s |
| **1024** | **1024** | 2048 | **389.23 tok/s** | **781.51 tok/s** | 32.44 ms | 5.11 ms | 5.26 s |
| **8192** | **1024** | 9216 | **127.90 tok/s** | **1152.11 tok/s** | 926.45 ms | 14.72 ms | 16.01 s |
| **1024** | **8192** | 9216 | **201.06 tok/s** | **226.39 tok/s** | 32.59 ms | 9.94 ms | 81.49 s |

#### Key Performance Insights:
- **Decode-Heavy Velocity**: Short input prompts with long outputs (`128:2048`) peak at **418.8 tok/s** with an ultra-low decode latency of **4.76 ms / token** on RDNA 4 GDDR6 memory.
- **Large Context Prefill (`8192:1024`)**: Ingesting 8k tokens sustained **1152.1 total tok/s**, demonstrating that large repo files or long chat histories are processed with sub-second TTFT (**926 ms**).
- **Extended Generation (`1024:8192`)**: Sustained over 16,000 generated tokens across requests with steady **201.1 tok/s** decode speed, verifying robust KV-cache management with zero OOM errors.

---

## Troubleshooting

### 1. Permission Denied on `/dev/kfd` or `/dev/dri`
- **Symptom**: Container crashes during initialization with `HIP error: out of memory` or `Cannot open /dev/kfd: Permission denied`.
- **Remedy**: Ensure your user is in the `video` and `render` groups:
  ```bash
  sudo usermod -aG video,render $USER
  ```
  Verify file permissions on the host:
  ```bash
  ls -l /dev/kfd /dev/dri/renderD*
  ```

---

### 2. Integrated GPU Collision (iGPU selected instead of Radeon AI PRO R9700)
- **Symptom**: vLLM crashes with `CUDA out of memory` after only allocating 2 GB or 3 GB, or selects `gfx1103` (Radeon 780M) instead of `gfx1201`.
- **Remedy**: Specify `HIP_VISIBLE_DEVICES=0` in `.env` or in `docker-compose.yml`. Device 0 corresponds to the discrete Radeon AI PRO R9700 GPU. Verify with `rocm-smi`:
  ```bash
  rocm-smi
  ```

---

### 3. Auto Tool Choice Error
- **Symptom**: OpenCode exits with:
  ```text
  Error: "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
  ```
- **Remedy**: Ensure your vLLM command includes both:
  ```bash
  --enable-auto-tool-choice --tool-call-parser hermes
  ```
  *(For SGLang, ensure `--tool-call-parser qwen25` is specified).*

---

### 4. Docker Snap Path Confinement
- **Symptom**: Mounting files from `/tmp/...` causes Docker to create empty directories instead of mounting the host file.
- **Remedy**: Because Snap isolates `/tmp`, always locate your configuration files and repository workspace under `/home/$USER/` (e.g., `/home/amd/workspace/coder`).

---

### 5. Out of Memory (OOM) Errors During Extended Conversations
- **Symptom**: The inference server terminates unexpectedly when conversation context expands.
- **Remedy**:
  1. Reduce `--max-model-len` from `32768` to `16384` in `.env`.
  2. Switch to an AWQ 4-bit quantized model (e.g., `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`).
  3. Lower `--gpu-memory-utilization` from `0.92` to `0.85` to reserve additional headroom for dynamic activations.

---

### 6. vLLM Qwen 3.5 / 3.8 Architecture Support & CUDA Graph Errors
- **Symptom**: vLLM exits with `ValueError: Model architectures ['Qwen3_5ForCausalLM'] are not supported` or crashes during CUDA graph capture on RDNA 4 (`gfx1201`).
- **Remedy**:
  1. Ensure [qwen3_5.py](file:///home/amd/workspace/coder/qwen3_5.py) is mounted into `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py:ro` as configured in `docker-compose.yml`.
  2. Keep `--hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}'` in your vLLM launch command.
  3. Keep `--compilation-config '{"cudagraph_mode": "NONE"}'` enabled to prevent graph capture conflicts on RDNA 4.

---

### 7. Out-of-Memory (OOM) with Qwen3.8-27B FP8 on Single 32 GB R9700
- **Symptom**: `Qwen/Qwen3.8-27B-FP8` fails during container startup or crashes during prompt prefill with `torch.OutOfMemoryError: CUDA out of memory` on the 32 GB Radeon AI PRO R9700.
- **Cause**: Dense 27B FP8 weights alone consume **~27 GB of VRAM**. On a single 32 GB card, the remaining <5 GB is insufficient to house the KV-cache, scratchpads, and dynamic activations required for agentic coding contexts.
- **Remedy**:
  1. **Single 32 GB R9700 (RECOMMENDED)**: Switch to the 4-bit quantized GGUF model (`Qwen3.8-27B-Q4_K_M.gguf`) using [docker-compose.gguf.yml](file:///home/amd/workspace/coder/docker-compose.gguf.yml). The weights occupy only **~16.8–17.6 GB**, leaving over 14 GB of VRAM for deep 32k–64k context windows with 8-bit KV caching (`--cache-type-k q8_0 --cache-type-v q8_0`) with zero OOM risk.
  2. **Dual R9700 (64 GB Total VRAM)**: To serve `Qwen3.8-27B` in FP8 precision, a **Dual Radeon AI PRO R9700 setup is strictly required**. Configure `HIP_VISIBLE_DEVICES=0,1` and append `--tensor-parallel-size 2` in [docker-compose.yml](file:///home/amd/workspace/coder/docker-compose.yml) to divide the ~27 GB weights evenly (~13.5 GB per GPU), leaving abundant headroom (>18 GB per GPU).

---

### 8. Resuming Interrupted GGUF Model Downloads
- **Symptom**: Network disconnection during the 16.8 GB GGUF model download.
- **Remedy**: Re-run [download_model.sh](file:///home/amd/workspace/coder/download_model.sh):
  ```bash
  ./download_model.sh
  ```
  The script automatically uses `curl -C -` with HTTP range resume support to continue downloading from the exact byte where it paused.

---

## Summary and Resources

By pairing the **AMD Radeon™ AI PRO R9700** with **vLLM** and **OpenCode**, you obtain a state-of-the-art on-premises coding assistant. This deployment offers:
- **Absolute Privacy**: Code never traverses outside your local workstation.
- **Superior Ergonomics**: Interactive terminal UI with automatic diffing, workspace search, and bash execution.
- **Native ROCm Acceleration**: Full exploitation of RDNA 4 (`gfx1201`) compute capabilities and 32 GB VRAM capacity.

### Additional Resources
- [AMD ROCm Blog: Bring Claude Code On-Prem with AMD Instinct GPUs](https://rocm.blogs.amd.com/software-tools-optimization/claude-code-onprem/README.html)
- [ROCm Official Documentation](https://rocm.docs.amd.com/)
- [vLLM ROCm Installation & Usage Guide](https://docs.vllm.ai/en/latest/getting_started/rocm_installation.html)
- [OpenCode Official Documentation](https://opencode.ai)
- [OpenCode GitHub Repository](https://github.com/anomalyco/opencode)
- [SGLang Project Documentation](https://sgl-project.github.io)
- [Qwen 2.5 Coder Model Family on Hugging Face](https://huggingface.co/collections/Qwen/qwen25-coder-66eaa23e7f6a601e3d06b728)
