# Docker Compose & Container Architecture Guide

This guide provides complete specifications for container orchestration, Dockerfiles, engine parameters, and optimization settings for serving AI coding models on AMD Radeon™ AI PRO R9700 and Instinct GPUs.

---

## Table of Contents
1. [vLLM Container Specification (`docker/docker-compose.yml`)](#1-vllm-container-specification-dockerdocker-composeyml)
2. [vLLM Key Optimization Parameters for RDNA 4 (`gfx1201`)](#2-vllm-key-optimization-parameters-for-rdna-4-gfx1201)
3. [llama.cpp ROCm GGUF Configuration (`docker/docker-compose.gguf.yml`)](#3-llamacpp-rocm-gguf-configuration-dockerdocker-composeggufyml)
4. [Building llama.cpp for gfx1201 (`docker/Dockerfile.llamacpp-rocm-gfx1201`)](#4-building-llamacpp-for-gfx1201-dockerdockerfilellamacpp-rocm-gfx1201)
5. [Serving GGUF Models on vLLM (via `vllm-gguf-plugin`)](#5-serving-gguf-models-on-vllm-via-vllm-gguf-plugin)
6. [SGLang ROCm Configuration (`docker/docker-compose.sglang.yml`)](#6-sglang-rocm-configuration-dockerdocker-composesglangyml)
7. [Radiance MxFP4 W4A8 Engine (`docker/docker-compose.mxfp4.yml`)](#7-radiance-mxfp4-w4a8-engine-dockerdocker-composemxfp4yml)
8. [Multi-GPU Orchestration (`tp2`, `dp2`, `pd`)](#8-multi-gpu-orchestration-tp2-dp2-pd)
9. [Qwen 3.5 Architecture Patch (`scripts/qwen3_5.py`)](#9-qwen-35-architecture-patch-scriptsqwen3_5py)

---

## 1. vLLM Container Specification (`docker/docker-compose.yml`)

> [!NOTE]
> **VRAM Allocation & Context Tuning on 32 GB R9700**: Serving `Qwen/Qwen3.8-27B-FP8` requires ~27.5 GB for weights alone.
> - **Single 32 GB R9700 (FP8 Production)**: Supported with bounded context (`MAX_MODEL_LEN=9600` or `8192`) and `--kv-cache-memory-bytes 1073741824` (1.0 GB pre-allocated KV cache). This allocates ~28.5 GB total VRAM, leaving ~3.5 GB safe operating headroom.
> - **Single 32 GB R9700 (Ultra-Deep Context)**: For 32,768–65,536 token context windows on a single card, use **`Q4_K_M` GGUF quantization** via [`docker/docker-compose.gguf.yml`](docker-compose.gguf.yml) (llama.cpp ROCm 7.x server), which consumes only ~16.8 GB for weights.
> - **Dual 64 GB R9700 (Scale-Up)**: Configure `HIP_VISIBLE_DEVICES=0,1` and append `--tensor-parallel-size 2` (`--tp 2`) to divide the weights (~13.7 GB per GPU), providing 18+ GB headroom per card for extended 64k+ context.

The primary [`docker/docker-compose.yml`](docker-compose.yml) orchestrates the inference engine, client agent, and optional benchmarking suite using **vLLM** optimized for AMD ROCm:

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

## 2. vLLM Key Optimization Parameters for RDNA 4 (`gfx1201`)

- **`--hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}'`**: Directs vLLM to use the Qwen 3.5 architecture definition provided by [`scripts/qwen3_5.py`](../scripts/qwen3_5.py).
- **`--compilation-config '{"cudagraph_mode": "NONE"}'`**: Prevents CUDA graph capture conflicts on RDNA 4 (`gfx1201`), ensuring deterministic kernel dispatch and preventing runtime driver aborts.
- **`--kv-cache-memory-bytes 1073741824`**: Explicitly pre-allocates a 1 GB KV-cache buffer, preventing out-of-memory errors when running dense 27B FP8 models (~27 GB weights) within the 32 GB physical VRAM boundary.
- **`VLLM_ROCM_FP8_PADDING=0`**: Disables unsupported FP8 GEMM padding on gfx1201 hardware.
- **`--enable-auto-tool-choice` and `--tool-call-parser hermes`**: Strictly required for OpenCode's structured function-calling schemas.

---

## 3. llama.cpp ROCm GGUF Configuration (`docker/docker-compose.gguf.yml`)

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

## 4. Building llama.cpp for gfx1201 (`docker/Dockerfile.llamacpp-rocm-gfx1201`)

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

## 5. Serving GGUF Models on vLLM (via `vllm-gguf-plugin`)

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

## 6. SGLang ROCm Configuration (`docker/docker-compose.sglang.yml`)

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

## 7. Radiance MxFP4 W4A8 Engine (`docker/docker-compose.mxfp4.yml`)

Radiance MxFP4 leverages 4-bit microscopic floating-point quantization with AITER unified attention:

```bash
docker compose -f docker/docker-compose.mxfp4.yml up -d
```

Key features:
- **W4A8 Microscopic Floating Point**: 4-bit weights with 8-bit activations.
- **A-tiled GEMM Kernels**: Custom tuned kernels for RDNA 4 WMMA execution units.
- **Memory Footprint**: ~14.2 GB model weights, leaving ~17 GB VRAM for extended KV cache.

---

## 8. Multi-GPU Orchestration (`tp2`, `dp2`, `pd`)

### Tensor Parallelism (`docker/docker-compose.tp2.yml`)
Pools 2x R9700 cards into a single 64 GB logical VRAM pool. Splits layers across GPUs:
```bash
docker compose -f docker/docker-compose.tp2.yml up -d
```

### Data Parallelism (`docker/docker-compose.dp2.yml`)
Spins up 2 independent 32 GB serving replicas with a round-robin proxy router on port 8000. Doubles prompt concurrency:
```bash
docker compose -f docker/docker-compose.dp2.yml up -d
```

### Prefill/Decode Disaggregation (`docker/docker-compose.pd.yml`)
Separates GPU 0 (Prefill engine) and GPU 1 (Decode engine) connected via high-speed P2P transport to eliminate prefill ITL stalls:
```bash
docker compose -f docker/docker-compose.pd.yml up -d
```

---

## 9. Qwen 3.5 Architecture Patch (`scripts/qwen3_5.py`)

vLLM 0.27.0 lacks native Qwen 3.5/3.8 hybrid architecture definitions. The patch [`scripts/qwen3_5.py`](../scripts/qwen3_5.py) is bind-mounted into the container at runtime:
```yaml
volumes:
  - ../scripts/qwen3_5_rocm10.py:/opt/python/lib/python3.14/site-packages/vllm/model_executor/models/qwen3_5.py:ro
```
This enables native execution of `Qwen3_5ForCausalLM` without modifying the underlying container image.
