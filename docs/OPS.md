# Operations

Container layout and fixes for this repo. Benchmark commands are in [BENCH.md](BENCH.md).

## Containers

## Docker Compose & Container Architecture Guide

This guide provides complete specifications for container orchestration, Dockerfiles, engine parameters, and optimization settings for serving AI coding models on AMD Radeon™ AI PRO R9700 and Instinct GPUs.

---

### Table of Contents
1. [vLLM Container Specification (`docker/docker-compose.yml`)](#1-vllm-container-specification-dockerdocker-composeyml)
2. [vLLM Key Optimization Parameters for RDNA 4 (`gfx1201`)](#2-vllm-key-optimization-parameters-for-rdna-4-gfx1201)
3. [llama.cpp ROCm GGUF Configuration (`docker/docker-compose.gguf.yml`)](#3-llamacpp-rocm-gguf-configuration-dockerdocker-composeggufyml)
4. [Building llama.cpp for gfx1201 (`docker/Dockerfile.llamacpp-rocm-gfx1201`)](#4-building-llamacpp-for-gfx1201-dockerdockerfilellamacpp-rocm-gfx1201)
5. [Serving GGUF Models on vLLM (via `vllm-gguf-plugin`)](#5-serving-gguf-models-on-vllm-via-vllm-gguf-plugin)
6. [SGLang ROCm Configuration (`docker/docker-compose.sglang.yml`)](#6-sglang-rocm-configuration-dockerdocker-composesglangyml)
7. [Radiance MxFP4 W4A8 Engine (`docker/docker-compose.mxfp4.yml`)](#7-radiance-mxfp4-w4a8-engine-dockerdocker-composemxfp4yml)
8. [Multi-GPU Orchestration (`tp2`, `dp2`, `pd`, `pd.8card`, `dp8`, `pd.16card`)](#8-multi-gpu-orchestration-tp2-dp2-pd-pd8card-dp8-pd16card)
9. [Qwen 3.5 Architecture Patch (`scripts/qwen3_5.py`)](#9-qwen-35-architecture-patch-scriptsqwen3_5py)

---

### 1. vLLM Container Specification (`docker/docker-compose.yml`)

> [!NOTE]
> **VRAM Allocation & Context Tuning on 32 GB R9700**: Serving `Qwen/Qwen3.8-27B-FP8` requires ~27.5 GB for weights alone.
> - **Single 32 GB R9700 (FP8 Production)**: Supported with bounded context (`MAX_MODEL_LEN=9600` or `8192`) and `--kv-cache-memory-bytes 1073741824` (1.0 GB pre-allocated KV cache). This allocates ~28.5 GB total VRAM, leaving ~3.5 GB safe operating headroom.
> - **Single 32 GB R9700 (Ultra-Deep Context)**: For 32,768–65,536 token context windows on a single card, use **`Q4_K_M` GGUF quantization** via [`docker/docker-compose.gguf.yml`](../docker/docker-compose.gguf.yml) (llama.cpp ROCm 7.x server), which consumes only ~16.8 GB for weights.
> - **Dual 64 GB R9700 (Scale-Up)**: Configure `HIP_VISIBLE_DEVICES=0,1` and append `--tensor-parallel-size 2` (`--tp 2`) to divide the weights (~13.7 GB per GPU), providing 18+ GB headroom per card for extended 64k+ context.

The primary [`docker/docker-compose.yml`](../docker/docker-compose.yml) orchestrates the inference engine, client agent, and optional benchmarking suite using **vLLM** optimized for AMD ROCm:

```yaml
services:
  # ============================================================================
  # Inference Engine: vLLM for AMD ROCm (Radeon AI PRO R9700 / gfx1201)
  # ============================================================================
  inference:
    image: ${VLLM_IMAGE:-rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0}
    container_name: rocm-inference-server
    restart: unless-stopped
    ipc: host
    network_mode: host # Allows direct, low-latency access and seamless ROCm IPC
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - "44"
      - "109"
    security_opt:
      - seccomp=unconfined
      - apparmor=unconfined
    environment:
      # Target the discrete Radeon AI PRO R9700 (GPU 0), ignoring integrated iGPU
      - HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
      - PYTORCH_ROCM_ARCH=gfx1201
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - HF_HOME=/root/.cache/huggingface
      - HF_TOKEN=${HF_TOKEN:-}
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
      - ..:/workspace
      - ../_results:/results
      - ../_results:/workspace/_results
      - ${MODELS_DIR:-../models}:/models
      - ../scripts/qwen3_5_rocm10.py:/opt/python/lib/python3.14/site-packages/vllm/model_executor/models/qwen3_5.py:ro
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
```

---

### 2. vLLM Key Optimization Parameters for RDNA 4 (`gfx1201`)

- **`--hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}'`**: Directs vLLM to use the Qwen 3.5 architecture definition provided by [`scripts/qwen3_5.py`](../scripts/qwen3_5.py).
- **`--compilation-config '{"cudagraph_mode": "NONE"}'`**: Prevents CUDA graph capture conflicts on RDNA 4 (`gfx1201`), ensuring deterministic kernel dispatch and preventing runtime driver aborts.
- **`--kv-cache-memory-bytes 1073741824`**: Explicitly pre-allocates a 1 GB KV-cache buffer, preventing out-of-memory errors when running dense 27B FP8 models (~27 GB weights) within the 32 GB physical VRAM boundary.
- **`VLLM_ROCM_FP8_PADDING=0`**: Disables unsupported FP8 GEMM padding on gfx1201 hardware.
- **`--enable-auto-tool-choice` and `--tool-call-parser hermes`**: Strictly required for OpenCode's structured function-calling schemas.

---

### 3. llama.cpp ROCm GGUF Configuration (`docker/docker-compose.gguf.yml`)

For GGUF quantization (e.g., `Qwen3.8-27B-Q4_K_M.gguf`), this repository provides a dedicated, high-performance **llama.cpp ROCm server** targeted directly at the RDNA 4 architecture (`gfx1201`).

```yaml
services:
  inference:
    image: ${GGUF_IMAGE:-local/llama.cpp:rocm10-gfx1201}
    container_name: rocm-llama-server
    restart: unless-stopped
    ipc: host
    network_mode: host
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - "44"
      - "109"
    security_opt:
      - seccomp=unconfined
      - apparmor=unconfined
    environment:
      - HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
      - HF_HOME=/root/.cache/huggingface
      - HF_TOKEN=${HF_TOKEN:-}
    volumes:
      - ${MODELS_DIR:-../models}:/models
      - ${HF_HOME:-${HF_CACHE_DIR:-/home/amd/.cache/huggingface}}:/root/.cache/huggingface
      - ../_results:/results
    entrypoint: ["llama-server"]
    command: >
      -m /models/${MODEL_FILE:-Qwen3.8-27B-Q4_K_M.gguf}
      --host 0.0.0.0
      --port ${INFERENCE_PORT:-8000}
      -ngl 999
      -fa on
      -c ${MAX_MODEL_LEN:-12288}
      -b 1024
      -ub 512
      -np 1
      --alias ${MODEL_ALIAS:-${MODEL_FILE:-Qwen3.8-27B-Q4_K_M.gguf}}
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://127.0.0.1:${INFERENCE_PORT:-8000}/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 20
      start_period: 20s
```

---

### 4. Building llama.cpp for gfx1201 (`docker/Dockerfile.llamacpp-rocm-gfx1201`)

> [!WARNING]
> **Legacy ROCm 5.6 Images Trigger Tensile Failure & CPU Fallback**:
> Prebuilt generic images compiled against legacy ROCm 5.6 trigger `rocBLAS error: Could not initialize Tensile host` on AMD RDNA 4 (`gfx1201`), causing CPU fallback.

To build a native ROCm 7.x HIP binary specifically compiled for `gfx1201`, use [`docker/Dockerfile.llamacpp-rocm-gfx1201`](../docker/Dockerfile.llamacpp-rocm-gfx1201):

```dockerfile
FROM rocm/dev-ubuntu-24.04:7.1-complete

ARG LLAMA_CPP_REF=master

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake git curl libcurl4-openssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} \
    https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp

WORKDIR /opt/llama.cpp

RUN HIPCXX="$(hipconfig -l)/clang" \
    HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP=ON \
      -DAMDGPU_TARGETS=gfx1201 \
      -DLLAMA_CURL=ON \
    && cmake --build build --config Release -j"$(nproc)"

ENV PATH="/opt/llama.cpp/build/bin:${PATH}"
ENTRYPOINT ["llama-server"]
```

Build the container image on the host:
```bash
docker build -f docker/Dockerfile.llamacpp-rocm-gfx1201 -t local/llama.cpp:rocm7-gfx1201 .
```

---

### 5. Serving GGUF Models on vLLM (via `vllm-gguf-plugin`)

The `vllm-gguf-plugin` package adds native `.gguf` quantization support directly into vLLM:

```bash
pip install vllm-gguf-plugin
```

Launch vLLM with GGUF weights:
```bash
vllm serve ./models/Qwen3.8-27B-Q4_K_M.gguf \
  --quantization gguf \
  --load-format gguf \
  --tokenizer Qwen/Qwen3.8-27B \
  --port 8000
```

---

### 6. SGLang ROCm Configuration (`docker/docker-compose.sglang.yml`)

SGLang provides efficient radix-attention caching. [`docker/docker-compose.sglang.yml`](../docker/docker-compose.sglang.yml) orchestrates SGLang on ROCm:

```yaml
services:
  inference:
    image: ${SGLANG_IMAGE:-lmsysorg/sglang:v0.4.3.post2-rocm630}
    container_name: rocm-sglang-server
    restart: unless-stopped
    ipc: host
    network_mode: host
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    environment:
      - HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
      - PYTORCH_ROCM_ARCH=gfx1201
    volumes:
      - ${MODELS_DIR:-../models}:/models
      - ${HF_HOME:-${HF_CACHE_DIR:-/home/amd/.cache/huggingface}}:/root/.cache/huggingface
      - ../_results:/results
    entrypoint: ["python3", "-m", "sglang.launch_server"]
    command: >
      --model-path ${SGLANG_MODEL_PATH:-/models/Qwen3.8-27B-Q4_K_M.gguf}
      --tokenizer-path ${SGLANG_TOKENIZER_PATH:-Qwen/Qwen3.8-27B}
      --host 0.0.0.0
      --port ${INFERENCE_PORT:-8000}
      --mem-fraction-static 0.85
```

---

### 7. Radiance MxFP4 W4A8 Engine (`docker/docker-compose.mxfp4.yml`)

Radiance MxFP4 leverages 4-bit microscopic floating-point quantization with AITER unified attention:

```bash
docker compose -f docker/docker-compose.mxfp4.yml up -d
```

Key features:
- **W4A8 Microscopic Floating Point**: 4-bit weights with 8-bit activations.
- **A-tiled GEMM Kernels**: Custom tuned kernels for RDNA 4 WMMA execution units.
- **Memory Footprint**: ~14.2 GB model weights, leaving ~17 GB VRAM for extended KV cache.

---

### 8. Multi-GPU Orchestration (`tp2`, `dp2`, `pd`, `pd.8card`, `dp8`, `pd.16card`)

#### Dual-GPU Tensor Parallelism (`docker/docker-compose.tp2.yml`)
Pools two matching cards into one tensor-parallel server. The default image tag is `gfx1201`; set `GPU_PROFILE=mi350p`, `PYTORCH_ROCM_ARCH=gfx950`, and `GPU_ISA=gfx950` for the MI350P pair.
```bash
docker compose -f docker/docker-compose.tp2.yml up -d
```
The compose file sets `HIP_VISIBLE_DEVICES` only, plus `HSA_FORCE_FINE_GRAIN_PCIE=1` and `HSA_NO_SCRATCH_RECLAIM=1`. It leaves `ROCR_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, and `NCCL_PROTO` unset. `NCCL_DEBUG_SUBSYS` covers `INIT,P2P,COLL,GRAPH,TUNING`. The MXFP4 image and `docker-compose.mxfp4.yml` also no longer force `NCCL_PROTO=Simple`. The measured two-MI350P result and the HIP path probe are in [MI350P-TP.md](MI350P-TP.md).

#### Dual-GPU Data Parallelism (`docker/docker-compose.dp2.yml`)
Spins up 2 independent 32 GB serving replicas with a round-robin proxy router on port 8000. Doubles prompt concurrency:
```bash
docker compose -f docker/docker-compose.dp2.yml up -d
```

#### Dual-GPU Prefill/Decode Disaggregation (`docker/docker-compose.pd.yml`)
Separates GPU 0 (Prefill engine, port 8001) and GPU 1 (Decode engine, port 8002) connected via high-speed P2P transport to eliminate prefill ITL stalls:
```bash
docker compose -f docker/docker-compose.pd.yml up -d
```

#### 8-Card Server Disaggregated Cluster (1P:7D) ([`docker/docker-compose.pd.8card.yml`](file:///home/amd/workspace/coder/docker/docker-compose.pd.8card.yml))
Orchestrates an 8-GPU R9700 server with 1 dedicated prefill engine on GPU 0 (port 8001) and 7 dedicated decoders on GPUs 1–7 (ports 8002–8008) communicating via host-staged `/dev/shm` IPC:
```bash
docker compose -f docker/docker-compose.pd.8card.yml up -d
```

#### 8-Card Data Parallel Baseline (DP=8) ([`docker/docker-compose.dp8.yml`](file:///home/amd/workspace/coder/docker/docker-compose.dp8.yml))
Deploys 8 independent collocated prefill+decode replicas across GPUs 0–7 (ports 8001–8008) for multi-replica goodput and interference benchmarking:
```bash
docker compose -f docker/docker-compose.dp8.yml up -d
```

#### 16-Card Dual-Node Cluster (2P:14D) ([`docker/docker-compose.pd.16card.yml`](file:///home/amd/workspace/coder/docker/docker-compose.pd.16card.yml))
Orchestrates a dual-chassis rack architecture pooling 512 GB VRAM across 16 GPUs (2 prefill engines on GPU 0 of each node, 14 distributed decoders) interconnected with RDMA networking:
```bash
docker compose -f docker/docker-compose.pd.16card.yml up -d
```

#### Asynchronous Fleet Router ([`scripts/pd_fleet_router.py`](file:///home/amd/workspace/coder/scripts/pd_fleet_router.py))
An enterprise-grade FastAPI reverse proxy providing unified OpenAI-compatible routing (`/v1/chat/completions`):
```bash
# Launch in Disaggregated P/D Mode
python3 scripts/pd_fleet_router.py --mode pd --prefill-urls http://localhost:8001 --decode-urls http://localhost:8002 http://localhost:8003 http://localhost:8004 http://localhost:8005 http://localhost:8006 http://localhost:8007 http://localhost:8008 --port 8000

# Launch in Cache-Affine Data Parallel Mode
python3 scripts/pd_fleet_router.py --mode dp --dp-urls http://localhost:8001 http://localhost:8002 http://localhost:8003 http://localhost:8004 http://localhost:8005 http://localhost:8006 http://localhost:8007 http://localhost:8008 --port 8000
```

#### Inter-GPU Topology & Connector Diagnostics ([`scripts/inspect_dual_gpu.py`](file:///home/amd/workspace/coder/scripts/inspect_dual_gpu.py))
Inspects ROCm KFD agent topology, validates homogeneous ISA matching, and diagnoses in-container vLLM KV transfer connector readiness:
```bash
python3 scripts/inspect_dual_gpu.py
```


---

### 9. Qwen 3.5 Architecture Patch (`scripts/qwen3_5.py`)

vLLM 0.27.0 lacks native Qwen 3.5/3.8 hybrid architecture definitions. The patch [`scripts/qwen3_5.py`](../scripts/qwen3_5.py) is bind-mounted into the container at runtime:
```yaml
volumes:
  - ../scripts/qwen3_5_rocm10.py:/opt/python/lib/python3.14/site-packages/vllm/model_executor/models/qwen3_5.py:ro
```
This enables native execution of `Qwen3_5ForCausalLM` without modifying the underlying container image.

## Troubleshooting

## Troubleshooting & Operational Guide

This document provides resolutions for common issues, error messages, device node permissions, and edge cases encountered when deploying inference engines on AMD Radeon™ AI PRO R9700 and Instinct GPUs.

---

### Table of Contents
1. [Permission Denied on `/dev/kfd` or `/dev/dri`](#1-permission-denied-on-devkfd-or-devdri)
2. [Integrated GPU Collision (iGPU selected instead of dGPU)](#2-integrated-gpu-collision-igpu-selected-instead-of-dgpu)
3. [Auto Tool Choice Error](#3-auto-tool-choice-error)
4. [Docker Snap Path Confinement](#4-docker-snap-path-confinement)
5. [Out of Memory (OOM) Errors During Extended Conversations](#5-out-of-memory-oom-errors-during-extended-conversations)
6. [vLLM Qwen 3.5 / 3.8 Architecture Support & CUDA Graph Errors](#6-vllm-qwen-35--38-architecture-support--cuda-graph-errors)
7. [Out-of-Memory (OOM) with Qwen3.8-27B FP8 on Single 32 GB R9700](#7-out-of-memory-oom-with-qwen38-27b-fp8-on-single-32-gb-r9700)
8. [Resuming Interrupted GGUF Model Downloads](#8-resuming-interrupted-gguf-model-downloads)
9. [llama.cpp Tensile Host Initialization Error](#9-llamacpp-tensile-host-initialization-error)
10. [Hugging Face 404 Repository Lookup Error with GGUF Models](#10-hugging-face-404-repository-lookup-error-with-gguf-models)
11. [vLLM Disaggregated Serving: MoRIIO is not available & Connector Diagnostics](#11-vllm-disaggregated-serving-moriio-is-not-available--connector-diagnostics)

---

#### 1. Permission Denied on `/dev/kfd` or `/dev/dri`
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

#### 2. Integrated GPU Collision (iGPU selected instead of dGPU)
- **Symptom**: vLLM crashes with `CUDA out of memory` after only allocating 2 GB or 3 GB, or selects `gfx1103` (Radeon 780M) instead of `gfx1201`.
- **Remedy**: Specify `HIP_VISIBLE_DEVICES=0` in `.env` or in `docker/docker-compose.yml`. Device 0 corresponds to the discrete Radeon AI PRO R9700 GPU. Verify with `amd-smi`:
  ```bash
  amd-smi list
  ```

---

#### 3. Auto Tool Choice Error
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

#### 4. Docker Snap Path Confinement
- **Symptom**: Mounting files from `/tmp/...` causes Docker to create empty directories instead of mounting the host file.
- **Remedy**: Because Snap isolates `/tmp`, always locate your configuration files and repository workspace under `/home/$USER/` (e.g., `/home/amd/workspace/coder`).

---

#### 5. Out of Memory (OOM) Errors During Extended Conversations
- **Symptom**: The inference server terminates unexpectedly when conversation context expands.
- **Remedy**:
  1. Reduce `--max-model-len` from `32768` to `16384` in `.env`.
  2. Switch to an AWQ 4-bit quantized model (e.g., `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`).
  3. Lower `--gpu-memory-utilization` from `0.92` to `0.85` to reserve additional headroom for dynamic activations.

---

#### 6. vLLM Qwen 3.5 / 3.8 Architecture Support & CUDA Graph Errors
- **Symptom**: vLLM exits with `ValueError: Model architectures ['Qwen3_5ForCausalLM'] are not supported` or crashes during CUDA graph capture on RDNA 4 (`gfx1201`).
- **Remedy**:
  1. Ensure [`scripts/qwen3_5.py`](../scripts/qwen3_5.py) is mounted into `/opt/python/lib/python3.14/site-packages/vllm/model_executor/models/qwen3_5.py:ro` as configured in [`docker/docker-compose.yml`](../docker/docker-compose.yml).
  2. Keep `--hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}'` in your vLLM launch command.
  3. Keep `--compilation-config '{"cudagraph_mode": "NONE"}'` enabled to prevent graph capture conflicts on RDNA 4.

---

#### 7. Out-of-Memory (OOM) with Qwen3.8-27B FP8 on Single 32 GB R9700
- **Symptom**: `Qwen/Qwen3.8-27B-FP8` fails during container startup or crashes during prompt prefill with `torch.OutOfMemoryError: CUDA out of memory` on the 32 GB Radeon AI PRO R9700.
- **Cause**: Dense 27B FP8 weights alone consume **~27.5 GB of VRAM**. If `--max-model-len` is set too high (e.g. 16k–32k) or KV cache memory is left unbounded, dynamic activations will exceed the 32 GB physical boundary.
- **Remedy**:
  1. **Single 32 GB R9700 (FP8 Serving)**: Bound the context length to `MAX_MODEL_LEN=9600` (or `8192`) and explicitly pre-allocate 1 GB KV cache: `--kv-cache-memory-bytes 1073741824`. This fits reliably in ~28.5 GB allocated VRAM with ~3.5 GB headroom.
  2. **Single 32 GB R9700 (Ultra-Deep Context)**: For long multi-file 32k–64k context windows, switch to the 4-bit quantized GGUF model (`Qwen3.8-27B-Q4_K_M.gguf`) using [`docker/docker-compose.gguf.yml`](../docker/docker-compose.gguf.yml). The weights occupy only **~16.8 GB**, leaving over 15 GB of VRAM for deep KV caching with zero OOM risk.
  3. **Dual R9700 (64 GB Total VRAM)**: To serve `Qwen3.8-27B` at unconstrained 32k–64k context in FP8 precision, use Dual Radeon AI PRO R9700 GPUs. Configure `HIP_VISIBLE_DEVICES=0,1` and append `--tensor-parallel-size 2` in [`docker/docker-compose.yml`](../docker/docker-compose.yml) to divide the ~27.5 GB weights evenly (~13.7 GB per GPU).

---

#### 8. Resuming Interrupted GGUF Model Downloads
- **Symptom**: Network disconnection during the 16.8 GB GGUF model download.
- **Remedy**: Re-run [`scripts/download_model.sh`](../scripts/download_model.sh):
  ```bash
  ./scripts/download_model.sh
  ```
  The script automatically uses `curl -C -` with HTTP range resume support to continue downloading from the exact byte where it paused.

---

#### 9. llama.cpp Tensile Host Initialization Error
- **Symptom**: `llama-server` container logs:
  ```text
  rocBLAS error: Could not initialize Tensile host: No devices found
  ```
  The server starts, but queries execute with 0% GPU utilization and catastrophic latency (**17.6 tok/s prefill, 2.67 tok/s decode** on CPU).
- **Cause**: The prebuilt image `ghcr.io/ggerganov/llama.cpp:server-rocm` was compiled against ROCm 5.6, which lacks the RDNA 4 (`gfx1201`) ISA code objects and Tensile host definitions.
- **Remedy**: Build the dedicated ROCm 7.x image using [`docker/Dockerfile.llamacpp-rocm-gfx1201`](../docker/Dockerfile.llamacpp-rocm-gfx1201):
  ```bash
  docker build -f docker/Dockerfile.llamacpp-rocm-gfx1201 -t local/llama.cpp:rocm7-gfx1201 .
  ```
  This compiles llama.cpp with `GGML_HIP=ON` and `-DAMDGPU_TARGETS=gfx1201`, delivering **1,069+ tok/s prefill** and **29+ tok/s decode** on the GPU.

---

#### 10. Hugging Face 404 Repository Lookup Error with GGUF Models
- **Symptom**: Running `vllm bench serve` against llama.cpp exits with:
  ```text
  huggingface_hub.utils._errors.RepositoryNotFoundError: 404 Client Error ... Repository Not Found for url: https://huggingface.co/api/models/Qwen3.8-27B-Q4_K_M.gguf
  ```
- **Cause**: GGUF endpoints expose the model alias as `Qwen3.8-27B-Q4_K_M.gguf`. Benchmark clients infer the tokenizer name from the server model name, attempting to query Hugging Face for a non-existent repo `Qwen3.8-27B-Q4_K_M.gguf`.
- **Remedy**: Decouple the tokenizer from the server model alias by explicitly passing `--tokenizer Qwen/Qwen3.8-27B-FP8`. The client will load tokenizer metadata from the local Hugging Face cache without network 404 lookups. The included [`throughput.sh`](../throughput.sh) script handles this decoupling automatically.

---

#### 11. vLLM Disaggregated Serving: MoRIIO is not available & Connector Diagnostics
- **Symptom**: During P/D startup or connector diagnostic probe, logs show:
  ```text
  ERROR [moriio_engine.py:59] MoRIIO is not available
  ERROR [moriio_connector.py:95] MoRIIO is not available
  ```
- **Cause**: vLLM's `MoRIIOConnector` requires both the Python package `msgpack` and AMD's native C++ shared library (`mori.io`). In standard ROCm container images, the native C++ driver is not pre-installed.
- **Remedy**:
  1. **Python Dependency**: Install `msgpack` inside the container:
     ```bash
     docker exec -it <container_name> pip install msgpack
     ```
  2. **Zero-Dependency Fallback Runtime**: Use [`SimpleCPUOffloadConnector`](file:///home/amd/workspace/coder/scripts/inspect_dual_gpu.py) or host-staged shared memory (`/dev/shm`). `SimpleCPUOffloadConnector` and `ExampleConnector` are verified **`[RUNTIME_READY]`** with zero external C++ dependencies, delivering fast inter-process transfer ($8\text{--}18\text{ ms}$) across host PCIe.
  3. **Verification**: Run [`scripts/inspect_dual_gpu.py`](file:///home/amd/workspace/coder/scripts/inspect_dual_gpu.py) to confirm connector lifecycle readiness across all 16 registered factory connectors.


