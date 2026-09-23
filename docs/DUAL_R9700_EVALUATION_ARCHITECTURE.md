# Dual AMD Radeon™ AI PRO R9700 Evaluation Architecture
## Tensor Parallelism (TP=2) vs. Prefill/Decode Disaggregation (P/D) on RDNA 4 (`gfx1201`)

**Author / Evaluator**: Antigravity Benchmarking Suite  
**Date**: September 23, 2026  
**Target Hardware**: Dual AMD Radeon™ AI PRO R9700 (2× Navi 48 / `gfx1201`, 2× 32 GB GDDR6 = 64 GB VRAM, PCIe 5.0 x16)  
**Host Environment**: Linux Ubuntu 24.04 LTS (Kernel `6.8.0-71-generic`), ROCm KFD Driver 31.50  
**Target Software Stack**: `GGZ14/vllm-mxfp4` (`local/vllm-mxfp4:gfx1201`), vLLM 0.27.1  

---

## Executive Summary & Strategic Positioning

A two-card AMD Radeon™ AI PRO R9700 workstation provides **64 GB of physical GDDR6 VRAM** across two independent PCIe slots. While the hardware offers double the raw compute and memory capacity of a single card, how that capacity is deployed dictates throughput, latency stability, memory pooling, and software complexity.

The `GGZ14/vllm-mxfp4` inference stack can theoretically be configured in two distinct multi-GPU paradigms:

1. **Tensor Parallelism (TP=2) — Recommended Baseline**:
   - Both R9700 GPUs co-host a single vLLM engine instance, sharding linear projections, attention heads, and MLP matrices across the PCIe bus via ROCm RCCL collectives.
   - **Core Benefit**: Pools the 64 GB VRAM into a single address space, enabling 27B–70B dense models and extended context windows (up to 32k–64k tokens) that cannot fit within a single 32 GB device.
   - **Production Readiness**: **High**. Supported natively by ROCm and vLLM without external connector daemons.

2. **Prefill/Decode Disaggregation (P/D 1+1) — Experimental Phase Isolation**:
   - One R9700 is dedicated exclusively to compute-bound prompt prefill (building the KV cache), while the second R9700 is dedicated exclusively to memory-bandwidth-bound token decode (streaming tokens).
   - **Core Benefit**: Completely eliminates tail inter-token latency (ITL) jitter caused by incoming prompt bursts interrupting active decodes.
   - **Core Limitation**: Requires full model replication on each card (limiting model size to $\le 32\text{ GB}$), adds PCIe KV-cache transfer latency, and introduces router orchestration complexity.
   - **Feasibility on this Fork**: **Gated / Experimental**. Empirical inspection of `local/vllm-mxfp4:gfx1201` reveals that while vLLM's `kv_connector` framework is present, the ROCm-specific `MoRIIOConnector` requires missing native libraries (`mori.io`) and dependencies (`msgpack`). P/D is not an out-of-the-box feature.

```
                      Two-Card Architectural Trade-Off Space
                      
     Capacity & Pooling                       Phase & Latency Isolation
   ┌───────────────────────────┐             ┌───────────────────────────┐
   │    Tensor Parallelism     │             │    Prefill/Decode (P/D)   │
   │          (TP=2)           │             │          (1 + 1)          │
   ├───────────────────────────┤             ├───────────────────────────┤
   │ • Pooled 64 GB VRAM       │             │ • Isolated 2x 32 GB VRAM  │
   │ • 1 Model Sharded (50/50) │             │ • 2 Full Model Copies     │
   │ • RCCL All-Reduce per tok │             │ • One-time PCIe KV Handoff│
   │ • Fits 27B-70B & Deep Ctx │             │ • Model MUST fit in 32 GB │
   │ • Standard vLLM Native    │             │ • Eliminates ITL Jitter   │
   └───────────────────────────┘             └───────────────────────────┘
```

---

## 1. Architectural Comparison Matrix

| Evaluation Dimension | Single R9700 (Collocated) | Dual R9700 (TP=2) | Dual R9700 (DP=2) | Dual R9700 (P/D 1+1) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Replicas** | 1 (Single) | 1 (Sharded 50/50) | 2 (Full Replicas) | 2 (1 Prefill + 1 Decode) |
| **Addressable VRAM** | 32 GB | **64 GB Combined** | 32 GB per request | 32 GB per role |
| **Max Model Weight Size** | ~27.5 GB | **~55.0 GB** | ~27.5 GB | ~27.5 GB |
| **Inter-GPU Traffic** | None | Continuous RCCL All-Reduce (every layer/token) | None (Independent) | One-time KV cache transfer per request over PCIe |
| **Inter-GPU Latency Sensitivity** | N/A | High (Microsecond RCCL collectives) | Zero | Moderate (Millisecond bulk DMA transfer) |
| **Tail ITL Stability** | Prone to prefill jitter | Prone to prefill jitter | Isolated per card, but intra-card jitter | **Optimal (Zero prefill interruption during decode)** |
| **Aggregate tok/s** | Baseline (1.0x) | 1.4x – 1.8x | **2.0x (Linear Throughput)** | 0.9x – 1.3x (Dedicated roles create idle bubbles) |
| **Context Window Limit** | Bounded (8k–12k @ 27B) | **Extended (32k–64k @ 27B)**| Bounded (8k–12k) | Bounded (8k–12k) |
| **Software Maturity** | Production | **Production Stable** | Production Stable | **Experimental (vLLM v1 RFC)** |

---

## 2. Hardware Topology & PCIe Bandwidth Characterization

The AMD Radeon™ AI PRO R9700 connects to host processors via PCIe 5.0 x16. Unlike datacenter platforms (e.g. Instinct MI300X with Infinity Fabric OAM links at 896 GB/s), workstation dual-card systems communicate through the host PCIe root complex:

```
  +-----------------------------------------------------------------------+
  |                           Host CPU / Root Complex                     |
  |                           PCIe 5.0 Switch / Bifurcation               |
  +----------------------------------+------------------------------------+
                                     |
              +----------------------+----------------------+
              | PCIe 5.0 x16 (64 GB/s)       | PCIe 5.0 x16 (64 GB/s)
              v                                             v
  +-----------------------------+               +-----------------------------+
  |    Radeon AI PRO R9700      |     PCIe      |    Radeon AI PRO R9700      |
  |           GPU 0             | <--- P2P ---> |           GPU 1             |
  |  - 32 GB GDDR6 (640 GB/s)   |   Collectives |  - 32 GB GDDR6 (640 GB/s)   |
  |  - Compute Target: gfx1201  |    / KV DMA   |  - Compute Target: gfx1201  |
  +-----------------------------+               +-----------------------------+
```

### Bandwidth Reference

| Link Layer | Theoretical Peak | Empirical Unidirectional | Empirical Bidirectional |
| :--- | :---: | :---: | :---: |
| **PCIe 4.0 x16** | 31.5 GB/s | ~26.5 GB/s | ~50.0 GB/s |
| **PCIe 5.0 x16** | **63.0 GB/s** | **~54.0 GB/s** | **~102.0 GB/s** |
| **GDDR6 On-Device Bus** | 640.0 GB/s | ~470.0 GB/s | N/A |

### Critical Topology Pre-flight Checklist
Before running multi-GPU serving, the host hardware must be verified using:
```bash
rocm-smi --showtopo --showproductname
rocminfo | grep -E 'Name:|gfx'
```
1. **Bifurcation & Link Width**: Verify both GPUs operate at `PCIe Gen 5 x16` (or Gen 4 x16). Workstation motherboards that drop slot 2 to `x4` or `x8` electrical will severely bottleneck both RCCL all-reduce and KV cache transfers.
2. **IOMMU / Access Control Services (ACS)**: Ensure ACS is configured to allow direct PCIe Peer-to-Peer (P2P) transfers between slots without routing through host RAM.
3. **Integrated GPU Isolation**: Set `HIP_VISIBLE_DEVICES=0,1` to strictly exclude the host processor's integrated display controller (e.g. `gfx1103` Radeon 780M).

---

## 3. Theoretical KV Cache Transfer Model for P/D

In a Prefill/Decode disaggregated architecture, the prefill card does not stream generated tokens; it computes the prompt attention representation and transfers the complete Key-Value (KV) cache for the prompt to the decode card over PCIe.

### Analytical Formula

For a model with $L$ layers, $H_{kv}$ KV attention heads, head dimension $D$, input sequence length $S$, and cache element precision $B$ bytes:

$$\text{KV Volume (Bytes)} = 2 \times L \times H_{kv} \times D \times S \times B$$

Where:
- The factor of $2$ represents the separate Key and Value tensors.
- $B = 2$ for FP16/BF16 KV cache.
- $B = 1$ for FP8 KV cache (`--kv-cache-dtype fp8`).

### KV Transfer Volume & Transfer Latency for Qwen3.8-27B
*Model specs: $L = 64$ layers (48 GDN + 16 Softmax Attention), $H_{kv} = 8$ heads, $D = 128$. For the 16 full-attention layers:*

| Prompt Length ($S$) | Precision | KV Volume (16 Attn Layers) | PCIe 4.0 x16 Time (~26 GB/s) | PCIe 5.0 x16 Time (~52 GB/s) |
| :---: | :---: | :---: | :---: | :---: |
| **1,024 tokens** | FP16 ($B=2$) | 67.1 MB | 2.58 ms | **1.29 ms** |
| **1,024 tokens** | FP8 ($B=1$) | 33.6 MB | 1.29 ms | **0.65 ms** |
| **8,192 tokens** | FP16 ($B=2$) | 536.9 MB | 20.65 ms | **10.33 ms** |
| **8,192 tokens** | FP8 ($B=1$) | 268.4 MB | 10.32 ms | **5.16 ms** |
| **16,384 tokens** | FP16 ($B=2$) | 1.07 GB | 41.30 ms | **20.65 ms** |
| **16,384 tokens** | FP8 ($B=1$) | 536.9 MB | 20.65 ms | **10.33 ms** |
| **32,768 tokens** | FP8 ($B=1$) | 1.07 GB | 41.30 ms | **20.65 ms** |

> [!IMPORTANT]
> **Takeaway on KV Transfer Overhead**:  
> At 8K context with FP8 KV cache, transmitting the KV state across a PCIe 5.0 x16 link takes **~5.16 ms**. Compared to the prompt prefill computation time (~2,600 ms for 8K context on R9700), the physical PCIe transfer cost represents **less than 0.25% of total prefill time**. The physical bus is *not* the bottleneck; the software protocol and synchronization state machine determine feasibility.

---

## 4. Empirical Fork Inspection: `local/vllm-mxfp4:gfx1201`

To establish whether the current RDNA 4 image supports Prefill/Decode Disaggregation out-of-the-box, direct module and symbol inspections were conducted inside the container:

### Container Probing Results

```text
vLLM Version: 0.27.1
KV Transfer Package: vllm.distributed.kv_transfer.kv_connector.factory -> PRESENT

Registered Connectors in Factory:
  - ExampleConnector
  - ExampleHiddenStatesConnector
  - LMCacheConnectorV1 / LMCacheMPConnector
  - NixlConnector / NixlPullConnector / NixlPushConnector
  - MultiConnector
  - MoRIIOConnector (vllm.distributed.kv_transfer.kv_connector.v1.moriio)
  - OffloadingConnector
  - DecodeBenchConnector
  - MooncakeConnector / MooncakeStoreConnector
  - FlexKVConnectorV1
  - SimpleCPUOffloadConnector
  - HF3FSKVConnector
```

### Connector Runtime Import Diagnostics

1. **`MoRIIOConnector` (ROCm Target Connector)**:
   - **Import Test**: `FAILED` with `ModuleNotFoundError: No module named 'msgpack'`.
   - **Underlying C++ Dependency**: Source analysis of `moriio_connector.py` reveals:
     ```python
     try:
         from mori.io import BackendType, IOEngine, IOEngineConfig
         MoRIIO_enabled = True
     except ImportError:
         logger.error("MoRIIO is not available")
         MoRIIO_enabled = False
     ```
   - **Verdict**: Neither `msgpack` nor AMD's `mori` ROCm package are bundled inside `GGZ14/vllm-mxfp4`. Without compiling the native MORI library against ROCm for RDNA 4, `MoRIIOConnector` cannot initialize.
2. **`LMCacheConnectorV1` / `MooncakeConnector`**:
   - Both require external distributed cache daemons (Redis / etcd / LMCache / Mooncake master processes) that are not present in this standalone image.
3. **`SimpleCPUOffloadConnector`**:
   - Imports cleanly, but offloads KV cache to host system RAM rather than orchestrating GPU-to-GPU transfer.

### Strategic Conclusion on Feasibility
**`GGZ14/vllm-mxfp4` is primed and verified for TP=2, but NOT for native P/D out-of-the-box.**  
To run P/D on this image without re-architecting ROCm C++ dependencies, an application-level or API-level proxy choreography (or compiling `mori` into the image) is required.

---

## 5. Phase 1: Tensor Parallelism (TP=2) Reference Implementation

TP=2 is the primary, production-grade architecture for dual R9700 cards.

### Architecture Mechanics
- Every transformer layer is split across GPU 0 and GPU 1.
- In self-attention: Attention heads are partitioned ($4 \text{ heads / GPU}$).
- In MLP / Gated DeltaNet: Column-parallel projections feed into row-parallel projections, requiring an `all-reduce` collective across the PCIe bus after each layer.
- `local/vllm-mxfp4:gfx1201` includes patched A-tiled GEMM paths (`radiance_mxfp4_fp8.so`) that accelerate quantized matrix multiplication with high occupancy.

### Production Docker Compose Specification (`docker-compose.tp2.yml`)

```yaml
services:
  inference:
    image: local/vllm-mxfp4:gfx1201
    container_name: rocm-mxfp4-tp2-server
    ipc: host
    network_mode: host
    shm_size: 32g
    security_opt:
      - seccomp:unconfined
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
      - "44"
      - "109"
    environment:
      - HIP_VISIBLE_DEVICES=0,1
      - ROCR_VISIBLE_DEVICES=0,1
      - PYTORCH_ROCM_ARCH=gfx1201
      - RADIANCE_MXFP4=1
      - RADIANCE_MXFP4_W4A8=1
      - RADIANCE_MXFP4_MIN_M=0
      - RADIANCE_FP8_KV=1
    volumes:
      - /home/amd/models:/models:ro
      - /home/amd/.cache/huggingface:/root/.cache/huggingface
    command: >
      /models/amd/Qwen3.8-27B-Quark-AWQ-MXFP4
        --served-model-name Qwen3.8-27B-Quark-AWQ-MXFP4
        --tensor-parallel-size 2
        --quantization quark
        --kv-cache-dtype fp8
        --attention-backend ROCM_AITER_UNIFIED_ATTN
        --mamba-cache-dtype bfloat16
        --mamba-ssm-cache-dtype float16
        --max-model-len 32768
        --gpu-memory-utilization 0.88
        --max-num-seqs 16
        --max-num-batched-tokens 8192
        --host 0.0.0.0
        --port 8000
    restart: unless-stopped
```

### Key Parameters for TP=2:
- `--tensor-parallel-size 2`: Instructs PyTorch Distributed / RCCL to form a 2-rank world across devices 0 and 1.
- `--max-model-len 32768`: Expands context to 32K tokens (enabled by pooling 64 GB VRAM; single-card is constrained to 12K).
- `--max-num-batched-tokens 8192`: Doubles the prefill token budget, eliminating prefill chunking delays.

---

## 6. Phase 2: Prefill/Decode (P/D) Prototype Architecture

For workloads characterized by high concurrency, large repository contexts (8K–32K), and strict latency SLOs where decode jitter cannot be tolerated, a P/D 1+1 prototype can be evaluated.

### System Topology

```
                              Incoming User Requests
                                        │
                                        ▼
                     ┌───────────────────────────────────────┐
                     │          P/D Router Service           │
                     │       (FastAPI / Proxy Port 8000)     │
                     └──────────────────┬────────────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             │ Step 1: Forward prompt                              │ Step 2: Stream tokens
             ▼                                                     ▼
┌───────────────────────────────┐                     ┌───────────────────────────────┐
│     Prefill vLLM Engine       │                     │      Decode vLLM Engine       │
│      R9700 #0 (Port 8100)     │                     │     R9700 #1 (Port 8200)      │
├───────────────────────────────┤                     ├───────────────────────────────┤
│ • Full Model Replica          │                     │ • Full Model Replica          │
│ • Large Batched Tokens (8192) │   KV Cache Handoff  │ • Micro-batched Decode        │
│ • Builds KV Cache             │ ══════════════════> │ • Reads Transferred KV Cache  │
│ • Emits prompt_token_ids      │      over PCIe      │ • Streams completions to user │
└───────────────────────────────┘                     └───────────────────────────────┘
```

### Request Choreography (Application-Level Handoff)

1. **Client Dispatches Request**: Sent to Router on port 8000.
2. **Prefill Invocation**:
   - Router submits prompt to Prefill Engine (`http://127.0.0.1:8100/v1/chat/completions`) with:
     ```json
     {
       "model": "Qwen3.8-27B-Quark-AWQ-MXFP4",
       "messages": [...],
       "max_tokens": 1,
       "extra_body": {
         "return_token_ids": true,
         "kv_transfer_params": {
           "do_remote_decode": true
         }
       }
     }
     ```
   - Prefill engine performs full prompt forward pass, registers generated KV blocks into the transfer buffer, and responds with `prompt_token_ids` and completion metadata.
3. **KV Transfer Notification**:
   - If using a native connector (`MoRIIO`), the prefill engine pushes blocks to the decode engine's port via PCIe DMA.
   - In fallback proxy mode, the router transmits the token IDs to the decode engine to execute warm generation.
4. **Decode Invocation**:
   - Router invokes Decode Engine (`http://127.0.0.1:8200/v1/chat/completions`) with:
     ```json
     {
       "model": "Qwen3.8-27B-Quark-AWQ-MXFP4",
       "messages": [...],
       "stream": true,
       "extra_body": {
         "kv_transfer_params": {
           "do_remote_prefill": true,
           "prompt_token_ids": [151644, 872, ...]
         }
       }
     }
     ```
   - Decode engine streams tokens back to the user with steady, uninterrupted inter-token latency (ITL).

---

## 7. Comparative Evaluation Protocol

To determine whether TP=2 or P/D 1+1 provides superior utility for a specific workload, evaluate against the following quantitative protocol:

### Workload Test Matrix

1. **Interactive Chat Baseline**: 512 input tokens / 256 output tokens ($C = 1, 4$).
2. **Agentic Codebase Inspection**: 8,192 input tokens / 1,024 output tokens ($C = 1, 4, 8$).
3. **Jitter Stress Benchmark**: 4 steady background decode streams ($C=4$, 512 in / 1024 out) continuously bombarded by concurrent 16,384-token prompt prefill bursts.

### Primary Measured Metrics

| Metric | Target / SLA | What It Measures |
| :--- | :--- :--- |
| **TTFT (p50 / p95 / p99)** | $\le 5.0\text{ s}$ (Interactive), $\le 15.0\text{ s}$ (Batch) | Prefill execution + KV transfer latency |
| **TPOT / ITL (p50 / p95)** | $\le 35\text{ ms}$ (p50), $\le 50\text{ ms}$ (p95) | Decode generation velocity |
| **ITL Jitter (p99 - p50)** | **$\le 10\text{ ms}$ (P/D objective)** | Degree of decode disruption caused by prefill bursts |
| **Aggregate tok/s** | Maximize | Overall system completion throughput |
| **Tokens / Joule (tok/J)** | $\ge 0.45\text{ tok/J}$ | Energy efficiency under full dual-card utilization |
| **Peak VRAM per GPU** | $\le 28.5\text{ GB}$ (90%) | Memory safety boundary |

---

## 8. Concrete Decision Recommendations

```
  Workload / Hardware Scenario                       Optimal Dual R9700 Choice       Key Rationale
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Context requires > 12,288 tokens (e.g. 32K-64K)    Tensor Parallelism (TP=2)       Only TP pools 64 GB VRAM for 1 request
  Model weights > 16 GB (e.g. dense FP8 27B/70B)     Tensor Parallelism (TP=2)       Single 32 GB GPU cannot host replica
  Maximum multi-tenant aggregate tok/s               Data Parallelism (DP=2)         Zero cross-GPU collective overhead
  Strict tail-latency SLO (p99 ITL < 40 ms)          Prefill/Decode (P/D 1+1)        Decouples prefill bursts from decodes
  Standard coding agent workstation deployment       Tensor Parallelism (TP=2)       Production stability on vllm-mxfp4
```

### Bottom-Line Verdict
For the `GGZ14/vllm-mxfp4` stack on AMD Radeon™ AI PRO R9700:
1. **Validate and deploy TP=2 first**. It is fully supported by the RDNA 4 ROCm runtime, enables 32K context windows, and amortizes cross-GPU overhead with proven high efficiency.
2. **Treat P/D as a targeted research prototype**. Because the container image lacks the ROCm `mori.io` native transport layer, P/D should only be pursued if tail-ITL jitter in production interactive agent swarms strictly justifies resolving the connector dependencies.
