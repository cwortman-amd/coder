# Troubleshooting & Operational Guide

This document provides resolutions for common issues, error messages, device node permissions, and edge cases encountered when deploying inference engines on AMD Radeon™ AI PRO R9700 and Instinct GPUs.

---

## Table of Contents
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

---

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

### 2. Integrated GPU Collision (iGPU selected instead of dGPU)
- **Symptom**: vLLM crashes with `CUDA out of memory` after only allocating 2 GB or 3 GB, or selects `gfx1103` (Radeon 780M) instead of `gfx1201`.
- **Remedy**: Specify `HIP_VISIBLE_DEVICES=0` in `.env` or in `docker/docker-compose.yml`. Device 0 corresponds to the discrete Radeon AI PRO R9700 GPU. Verify with `rocm-smi`:
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
  1. Ensure [`scripts/qwen3_5.py`](../scripts/qwen3_5.py) is mounted into `/opt/python/lib/python3.14/site-packages/vllm/model_executor/models/qwen3_5.py:ro` as configured in [`docker/docker-compose.yml`](../docker/docker-compose.yml).
  2. Keep `--hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}'` in your vLLM launch command.
  3. Keep `--compilation-config '{"cudagraph_mode": "NONE"}'` enabled to prevent graph capture conflicts on RDNA 4.

---

### 7. Out-of-Memory (OOM) with Qwen3.8-27B FP8 on Single 32 GB R9700
- **Symptom**: `Qwen/Qwen3.8-27B-FP8` fails during container startup or crashes during prompt prefill with `torch.OutOfMemoryError: CUDA out of memory` on the 32 GB Radeon AI PRO R9700.
- **Cause**: Dense 27B FP8 weights alone consume **~27.5 GB of VRAM**. If `--max-model-len` is set too high (e.g. 16k–32k) or KV cache memory is left unbounded, dynamic activations will exceed the 32 GB physical boundary.
- **Remedy**:
  1. **Single 32 GB R9700 (FP8 Serving)**: Bound the context length to `MAX_MODEL_LEN=9600` (or `8192`) and explicitly pre-allocate 1 GB KV cache: `--kv-cache-memory-bytes 1073741824`. This fits reliably in ~28.5 GB allocated VRAM with ~3.5 GB headroom.
  2. **Single 32 GB R9700 (Ultra-Deep Context)**: For long multi-file 32k–64k context windows, switch to the 4-bit quantized GGUF model (`Qwen3.8-27B-Q4_K_M.gguf`) using [`docker/docker-compose.gguf.yml`](../docker/docker-compose.gguf.yml). The weights occupy only **~16.8 GB**, leaving over 15 GB of VRAM for deep KV caching with zero OOM risk.
  3. **Dual R9700 (64 GB Total VRAM)**: To serve `Qwen3.8-27B` at unconstrained 32k–64k context in FP8 precision, use Dual Radeon AI PRO R9700 GPUs. Configure `HIP_VISIBLE_DEVICES=0,1` and append `--tensor-parallel-size 2` in [`docker/docker-compose.yml`](../docker/docker-compose.yml) to divide the ~27.5 GB weights evenly (~13.7 GB per GPU).

---

### 8. Resuming Interrupted GGUF Model Downloads
- **Symptom**: Network disconnection during the 16.8 GB GGUF model download.
- **Remedy**: Re-run [`scripts/download_model.sh`](../scripts/download_model.sh):
  ```bash
  ./scripts/download_model.sh
  ```
  The script automatically uses `curl -C -` with HTTP range resume support to continue downloading from the exact byte where it paused.

---

### 9. llama.cpp Tensile Host Initialization Error
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

### 10. Hugging Face 404 Repository Lookup Error with GGUF Models
- **Symptom**: Running `vllm bench serve` against llama.cpp exits with:
  ```text
  huggingface_hub.utils._errors.RepositoryNotFoundError: 404 Client Error ... Repository Not Found for url: https://huggingface.co/api/models/Qwen3.8-27B-Q4_K_M.gguf
  ```
- **Cause**: GGUF endpoints expose the model alias as `Qwen3.8-27B-Q4_K_M.gguf`. Benchmark clients infer the tokenizer name from the server model name, attempting to query Hugging Face for a non-existent repo `Qwen3.8-27B-Q4_K_M.gguf`.
- **Remedy**: Decouple the tokenizer from the server model alias by explicitly passing `--tokenizer Qwen/Qwen3.8-27B-FP8`. The client will load tokenizer metadata from the local Hugging Face cache without network 404 lookups. The included [`throughput.sh`](../throughput.sh) script handles this decoupling automatically.
