# Dual AMD Radeon™ AI PRO R9700 Evaluation Architecture
## Tensor Parallelism (TP=2), Data Parallelism (DP=2) & Prefill/Decode Disaggregation (P/D) on RDNA 4 (`gfx1201`)

**Author / Evaluator**: Antigravity Benchmarking Suite  
**Date**: September 23, 2026 (Updated with Host Boundary Refinements)  
**Target Architecture**: Dual AMD Radeon™ AI PRO R9700 (2× Navi 48 / `gfx1201`, 2× 32 GB GDDR6 = 64 GB VRAM, PCIe 5.0 x16)  
**Current Development Host Boundary**: **1× Radeon AI PRO R9700 (`gfx1201`, 64 CUs, 32 GB)** + **1× Radeon 780M Graphics (`gfx1103`, 12 CUs, integrated APU)**  
**Host Environment**: Linux Ubuntu 24.04 LTS (Kernel `6.8.0-71-generic`), ROCm KFD Driver 31.50  
**Target Software Stack**: `GGZ14/vllm-mxfp4` (`local/vllm-mxfp4:gfx1201`), vLLM 0.27.1  

---

> [!WARNING]
> **CRITICAL HARDWARE BASELINE & SYSTEM BOUNDARY NOTICE**:
> - **Current Development Host**: Contains **one dedicated Radeon AI PRO R9700 (`gfx1201`)** and **one integrated Radeon 780M APU (`gfx1103`)**.
> - **Heterogeneous Multi-GPU Execution Prohibited**: The integrated 780M APU has a different architecture (`gfx1103`), 12 CUs, and shared system UMA memory. Attempting `--tensor-parallel-size 2` or P/D across `gfx1201` and `gfx1103` is fundamentally invalid and will cause allocator faults or severe pipeline stalls.
> - **Fail-Fast Policy**: All multi-card deployment configurations (`tp2`, `pd`, `dp2`) are gated by a strict **two-homogeneous-`gfx1201`-R9700 preflight check**. On the current single-card workstation, dual-card runs fail-fast with an informative diagnostic notice, while single-GPU baselines, chunked-prefill benchmarks, router mock choreography, and connector inspections remain fully executable.
> - **Dual-R9700 Material Scope**: The dual-GPU architectures detailed herein represent the **Target Production Design**, pending physical installation of the second Radeon AI PRO R9700 card.

---

## Executive Summary & Strategic Positioning

A two-card AMD Radeon™ AI PRO R9700 workstation provides **64 GB of physical GDDR6 VRAM** across two PCIe 5.0 x16 slots. Deploying this hardware effectively requires distinguishing three distinct multi-GPU operating modes:

1. **Tensor Parallelism (TP=2) — Immediate Production Priority**:
   - Both R9700s shard every transformer layer 50/50 via ROCm RCCL collectives over PCIe.
   - **Core Benefit**: Pools the physical memory into an addressable **~64 GB VRAM space**, enabling dense 27B–70B models and deep 32K–64K context windows that exceed a single 32 GB card.
   - **Readiness**: **Production Stable**. Natively supported by ROCm and vLLM on homogeneous `gfx1201` devices.

2. **Data Parallelism (DP=2) — Throughput Scaling Baseline**:
   - Each R9700 runs a complete, independent model replica behind a round-robin proxy.
   - **Core Benefit**: Delivers linear 2.0× aggregate token throughput with **zero cross-GPU collective overhead** for models that fit within 32 GB.
   - **Readiness**: **Production Stable**.

3. **Prefill/Decode Disaggregation (P/D 1+1) — Gated Phase Isolation Prototype**:
   - R9700 #0 is dedicated to compute-bound prompt prefill; R9700 #1 is dedicated to memory-bandwidth-bound decode.
   - **Core Benefit**: Completely decouples prompt prefill bursts from continuous token streaming, eliminating tail inter-token latency (ITL) jitter (preventing the 1.35-second prefill-chunk stalls). It does **not** accelerate single-stream decode throughput (which is bounded at ~34 tok/s by serial dependency $x_{t+1}=f(x_{\leq t})$), but guarantees strict SLA predictability under mixed load.
   - **Core Limitation**: Requires two full model copies (limiting model size to $\le 32\text{ GB}$), adds PCIe KV transfer overhead, and requires an external request router.
   - **Feasibility on this Fork**: **Gated / Experimental**. In-container probing of `local/vllm-mxfp4:gfx1201` proves that while the `kv_connector` factory exists, `MoRIIOConnector` lacks `msgpack` and AMD's native C++ `mori.io` runtime.

```
                      Dual R9700 Multi-GPU Decision Space
                      
     Capacity & Pooling          Throughput Scaling          Phase & Latency Isolation
   ┌──────────────────────┐    ┌──────────────────────┐    ┌───────────────────────────┐
   │  Tensor Parallelism  │    │   Data Parallelism   │    │    Prefill/Decode (P/D)   │
   │       (TP=2)         │    │       (DP=2)         │    │          (1 + 1)          │
   ├──────────────────────┤    ├──────────────────────┤    ├───────────────────────────┤
   │ • Pooled 64 GB VRAM  │    │ • 2 Complete Replicas│    │ • Dedicated Roles (P + D) │
   │ • Shards every layer │    │ • 0 Collective cost  │    │ • Eliminates ITL Jitter   │
   │ • RCCL per token     │    │ • Linear tok/s gain  │    │ • 2x Model copies in VRAM │
   │ • Fits 27B-70B & 32K │    │ • Max independent req│    │ • Gated by MoRI-IO stack  │
   │ • Production Ready   │    │ • Production Ready   │    │ • Experimental Prototype  │
   └──────────────────────┘    └──────────────────────┘    └───────────────────────────┘
```

---

## 1. Five-Way Serving Comparison Matrix

| Evaluation Dimension | Single R9700 (Collocated) | Single R9700 (Chunked Prefill) | Dual R9700 (TP=2) | Dual R9700 (DP=2) | Dual R9700 (P/D 1+1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Replicas** | 1 | 1 | 1 (Sharded 50/50) | 2 (Full Replicas) | 2 (1 Prefill + 1 Decode) |
| **Addressable VRAM** | 32 GB | 32 GB | **64 GB Combined** | 32 GB per request | 32 GB per role |
| **Max Model Weight Size** | ~27.5 GB | ~27.5 GB | **~55.0 GB** | ~27.5 GB | ~27.5 GB |
| **Inter-GPU Traffic** | None | None | Continuous RCCL All-Reduce | None | One-time PCIe KV transfer per request |
| **Inter-GPU Latency Risk** | None | None | High (microsecond collectives) | Zero | Moderate (bulk DMA transfer) |
| **Tail ITL Under Burst** | High jitter | Moderate jitter reduction | High jitter | Isolated per replica | **Minimal (True phase isolation)** |
| **Aggregate tok/s** | 1.0× (Baseline) | 0.95× – 1.0× | 1.4× – 1.8× | **2.0× (Linear)** | 0.9× – 1.3× (Role idle bubbles) |
| **Context Window Limit** | Bounded (8K–12K @ 27B) | Bounded (8K–12K) | **Extended (32K–64K @ 27B)**| Bounded (8K–12K) | Bounded (8K–12K) |
| **Power Profile** | ~296 W under load | ~296 W under load | ~580 W combined continuous | ~590 W combined parallel | **Asymmetric: GPU0 bursty, GPU1 steady** |
| **Thermodynamic Duty** | Cyclic thermal soak | Cyclic thermal soak | Symmetric thermal soak | Symmetric thermal soak | **GPU0: 14W $\leftrightarrow$ 296W; GPU1: 297W / 88°C GDDR6** |
| **Energy Disparity** | 0.094 J/in, 8.88 J/out | 0.094 J/in, 8.88 J/out | ~0.08 J/in, ~7.5 J/out | 0.094 J/in, 8.88 J/out | **0.094 J/prefill tok vs 8.88 J/decode tok** |
| **Implementation Maturity**| Production | Production | **Production Priority** | Production | **Gated Prototype** |

---

## 2. Hardware Topology & PCIe Bandwidth Characterization

In a dual Radeon AI PRO R9700 workstation, both cards communicate across the host PCIe 5.0 root complex:

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
| **PCIe 5.0 x16** | **63.0 GB/s** | **~52.0 – 54.0 GB/s** | **~102.0 GB/s** |
| **GDDR6 On-Device Bus** | 640.0 GB/s | ~470.0 GB/s | N/A |

### Pre-flight Hardware Verification Checklist
Before running multi-card workloads on dual R9700 hardware:
1. **Topology Detection**: Run `rocm-smi --showtopo --showproductname` to verify both slots identify as `gfx1201` APM107573.
2. **PCIe Slot Bifurcation**: Ensure motherboard PCIe slot 2 has not silently negotiated down to `x4` or `x8` lanes.
3. **IOMMU / Access Control Services (ACS)**: Enable PCIe P2P direct memory access in the system BIOS.
4. **iGPU Masking**: Ensure `HIP_VISIBLE_DEVICES=0,1` strictly references the two dGPUs, isolating the integrated `gfx1103` display controller.

### Thermodynamic & Power Asymmetry: Prefill vs. Decode Duty Cycles

Empirical telemetry on the R9700 reveals a critical physical divergence between prefill and decode:
- **Instantaneous Power Equivalence**: Both roles draw **~296 W** sustained (peaking at 312 W – 328 W).
- **The 94× Energy-per-Token Disparity**: A prefill token costs **0.094 Joules**, while a generated decode token costs **8.87 Joules**.
- **Role Duty Cycles in P/D 1+1**:
  - **GPU 0 (Prefill)**: Experiences a **bursty power profile**. It idles at ~14 W between incoming requests, spikes to ~296 W for ~2.60 seconds during an 8,192-token prompt ingestion, and returns immediately to idle.
  - **GPU 1 (Decode)**: Experiences a **continuous thermal soak**. It draws ~297 W continuously while streaming tokens, causing its GDDR6 memory to run **~13 °C hotter (87.6 °C vs 72.4 °C)** due to 100% memory bus duty cycle.
- **Power Provisioning**: Unlike TP=2 or DP=2 where both cards share identical symmetric load, P/D requires thermal cooling and power provisioning optimized for continuous memory soak on the decode card.

---

## 3. KV Cache Transfer Model for P/D & PCIe Lower Bounds

In a Prefill/Decode disaggregated topology, prompt processing on GPU 0 generates a Key-Value cache that is transferred to GPU 1 before token decode begins.

### Hybrid KV-Cache Dimension Formula

For hybrid architectures (such as `Qwen3.8-27B`, which combines 48 Gated DeltaNet linear attention layers and 16 Softmax Attention layers), standard KV cache is **only allocated by the full-attention layers**. Linear attention state is tracked separately in recurrent Mamba state buffers.

$$\text{KV Volume (Bytes)} = \sum_{l \in \text{KV-bearing layers}} 2 \times H_{kv,l} \times D_l \times S_l \times B_l$$

Where:
- $l \in [1..16]$ Softmax attention layers.
- $H_{kv} = 8$ KV attention heads.
- $D = 128$ head dimension.
- $S$ is prompt sequence length (tokens).
- $B$ is cache element precision ($B=1$ for FP8, $B=2$ for FP16).

For $S = 8,192$ tokens with FP8 KV cache ($B=1$):
$$\text{KV Volume} = 2 \times 16 \times 8 \times 128 \times 8,192 \times 1 = 268,435,456\text{ bytes} \approx \mathbf{268.4\text{ MB}}$$

### Ideal Payload-Copy Floor vs. Practical End-to-End Handoff Latency

The theoretical data transfer time across a PCIe 5.0 x16 bus (~52 GB/s empirical sustained) is:
$$T_{\text{data-movement}} = \frac{268.4\text{ MB}}{52.0\text{ GB/s}} \approx \mathbf{5.16\text{ ms}}$$

> [!IMPORTANT]
> **Payload Floor vs. End-to-End Handoff**:  
> The 5.16 ms figure represents the **ideal physical payload-copy floor**. It assumes zero-overhead descriptor exchange, immediate buffer availability, and frictionless kernel synchronization.  
> The true end-to-end handoff latency experienced by a request is:
> $$T_{\text{handoff}} = T_{\text{serialize}} + T_{\text{control-plane}} + T_{\text{buffer-ready}} + T_{\text{data-movement}} + T_{\text{decode-admission}}$$
> Where:
> - $T_{\text{serialize}}$: Prefill engine packaging token IDs and KV block descriptors.
> - $T_{\text{control-plane}}$: Router REST/ZeroMQ request dispatch and acknowledgement.
> - $T_{\text{buffer-ready}}$: Decode engine allocator allocating matching physical KV blocks.
> - $T_{\text{data-movement}}$: Bulk PCIe DMA transfer (the 5.16 ms physical floor).
> - $T_{\text{decode-admission}}$: Scheduler admitting the sequence into the active decode queue.

### Analytical Transfer Floor Reference Table

| Prompt Tokens ($S$) | Precision | KV Size (16 Attn Layers) | PCIe 4.0 x16 Floor (~26 GB/s) | PCIe 5.0 x16 Floor (~52 GB/s) |
| :---: | :---: | :---: | :---: | :---: |
| **1,024** | FP8 ($B=1$) | 33.6 MB | 1.29 ms | **0.65 ms** |
| **1,024** | FP16 ($B=2$) | 67.1 MB | 2.58 ms | **1.29 ms** |
| **8,192** | FP8 ($B=1$) | 268.4 MB | 10.32 ms | **5.16 ms** |
| **8,192** | FP16 ($B=2$) | 536.9 MB | 20.65 ms | **10.33 ms** |
| **16,384** | FP8 ($B=1$) | 536.9 MB | 20.65 ms | **10.33 ms** |
| **16,384** | FP16 ($B=2$) | 1.07 GB | 41.30 ms | **20.65 ms** |
| **32,768** | FP8 ($B=1$) | 1.07 GB | 41.30 ms | **20.65 ms** |

---

## 4. Empirical Container Inspection: `local/vllm-mxfp4:gfx1201`

Direct symbol and module diagnostics executed inside `local/vllm-mxfp4:gfx1201` yielded clear findings:

```text
vLLM Version: 0.27.1
KV Connector Factory: PRESENT (16 registered connectors)
  • MoRIIOConnector, LMCacheConnectorV1, NixlConnector, MooncakeConnector, etc.

Detailed Import Diagnostics:
  [MISSING DEP] MoRIIOConnector: Missing Python module 'msgpack'
  [AVAILABLE]   LMCacheConnectorV1 (requires external lmcache daemon)
  [AVAILABLE]   NixlConnector (WARNING: NIXL C++ runtime unavailable)
  [AVAILABLE]   MooncakeConnector (WARNING: MooncakeTransferEngine unavailable)
  [AVAILABLE]   SimpleCPUOffloadConnector (CPU host RAM offload only)
  [AVAILABLE]   ExampleConnector
  [NATIVE MORI] mori.io C++ / Python extension is NOT installed.
```

### Analysis: Registry Visibility vs. Functional Readiness
The presence of `MoRIIOConnector` in `KVConnectorFactory` indicates architectural intent, not operational readiness. Source inspection of `moriio_connector.py` reveals:
```python
try:
    from mori.io import BackendType, IOEngine, IOEngineConfig
    MoRIIO_enabled = True
except ImportError:
    logger.error("MoRIIO is not available")
    MoRIIO_enabled = False
```
Because AMD's native `mori.io` C++ library and `msgpack` are absent, `MoRIIOConnector` cannot initialize in the current image. Attempting to deploy P/D on this image today requires resolving these upstream ROCm dependencies.

---

## 5. Seven Functional Validation Gates for P/D

Before benchmarking P/D performance on dual homogeneous R9700 hardware, the setup must pass these seven sequential validation gates:

| Gate | Validation Target | Pass Criterion |
| :---: | :--- | :--- |
| **1. Connector Import Gate** | Clean module resolution | `MoRIIOConnector` imports without `ModuleNotFoundError` or missing `.so` dependencies. |
| **2. Producer/Consumer Smoke Gate** | Functional KV handoff | Decode engine accepts transferred KV blocks and generates tokens **without recomputing the prompt**. |
| **3. Token Equivalence Gate** | Correctness vs baseline | Deterministic greedy decoding produces identical token sequences to collocated single-GPU serving. |
| **4. KV Transfer Metric Gate** | Engine telemetry verification | Decode engine logs verify `accepted_remote_blocks > 0` and `prompt_tokens_evaluated = 0`. |
| **5. Long-Context Stability Gate** | High-load stability | Sustains 8K, 16K, and 32K context transfers without allocator memory leaks or descriptor timeouts. |
| **6. Burst Isolation Gate** | Latency jitter mitigation | Under concurrent 16K prefill bombardment, decode $p95/p99$ ITL demonstrates $\ge 20\text{--}30\%$ jitter reduction vs collocated baseline. |
| **7. Failure Recovery Gate** | Fault tolerance | Router handles prefill/decode worker termination gracefully, returning clean HTTP 502/504 errors without leaving orphaned KV blocks. |

---

## 6. Acceptance Criteria for Dual Homogeneous R9700 Hardware

Upon arrival and physical installation of the second Radeon AI PRO R9700 card:

### TP=2 Acceptance Criteria
- Both cards identify as `gfx1201` APM107573 with full PCIe Gen 5 x16 link width.
- ROCm RCCL initializes across devices 0 and 1 without deadlocks.
- The 27B model loads with `--max-model-len 32768`, utilizing the pooled 64 GB VRAM headroom.
- A 30-minute concurrency soak test ($C=1, 2, 4, 8, 16$) completes with 0% dropped requests.
- Full telemetry captured: 250 ms power per GPU, junction/memory temperatures, and RCCL collective stability.

### P/D Acceptance Criteria
- Python runtime dependency resolved: `msgpack` (installed and verified in `local/vllm-mxfp4:gfx1201`).
- Transport qualification: AMD native `mori.io` runtime integrated for direct P2P, or host-staged shared-memory (`/dev/shm` / `SimpleCPUOffloadConnector`) lifecycle harness validated.
- Passes all 7 functional validation gates above.
- Demonstrates measurable tail-ITL smoothing under burst conditions justifying the loss of 64 GB pooled capacity.

---

## 7. Immediate Action Sequence

```
  Step 1: Enforce two-card fail-fast preflight check on all dual-GPU scripts and compose files.
  Step 2: Use current host for Single-R9700 baselines, Chunked Prefill tuning, and P/D Router API validation.
  Step 3: Deploy runnable DP=2 orchestration configuration ready for hardware arrival.
  Step 4: On arrival of second R9700: Verify PCIe Gen 5 link negotiation, P2P ACS, and validate TP=2 baseline.
  Step 5: Execute comprehensive dual-card benchmark matrix according to TESTPLAN.md.
```

---

## 8. Counterfactual Capacity & Interference Modeling (Single-Card Emulation)

To rigorously evaluate the capacity case for P/D on a single Radeon AI PRO R9700 before card 2 arrives, a discrete-event pipeline emulator was developed in [`benchmark/pd_capacity_emulator.py`](../benchmark/pd_capacity_emulator.py) (callable via `python3 benchmark/bench_phases.py --mode pd-capacity-emulator`).

For complete mathematical derivations, trace datasets, and full evaluation results, see the dedicated [Counterfactual Capacity & Interference Report](PD_CAPACITY_COUNTERFACTUAL_MODEL.md).

### Key Architectural Findings
1. **The Raw Capacity Proof ($\eta < 0.5$)**:
   - P/D exceeds DP=2 in raw output tok/s **if and only if** collocated decode degrades below $17.04\text{ tok/s}$ ($\eta < 0.5$).
   - In the $J3$ prompt-saturation regime (measured at **11.42 tok/s** per replica), $\eta = 0.335 < 0.5$.
   - **Verdict**: P/D beats DP=2 in raw output tok/s by **1.29× to 1.71×** (**29.55–39.21 tok/s** vs. **22.85–22.98 tok/s**).
2. **The SLO-Goodput Proof (Streaming Quality of Service)**:
   - In collocated DP=2, incoming cold 8K prompt chunks schedule alongside active decode steps, inflicting **609.7 ms forward execution stalls** that cause **0.0% streaming SLO compliance** on collided decode streams.
   - P/D achieves **100.0% streaming SLO compliance**, delivering **24.11 to 39.01 SLO-qualified tok/s** compared to **<0.30 qualified tok/s** for DP=2.
3. **KV Handoff Robustness**:
   - Sweeping handoff latency $H \in [5.16, 25.0, 50.0, 100.0, 250.0]\text{ ms}$ proves P/D retains over **99.5% of its qualified goodput** across the entire 5.16 to 100 ms range. P/D is **not fragile to connector latency**.

---

## 9. Master Test Plan & Evaluation Methodology (`TESTPLAN.md`)

To standardize the empirical benchmarking of the dual-R9700 platform upon hardware arrival, the complete testing methodology has been codified in [`TESTPLAN.md`](../TESTPLAN.md) (and symlinked at [`docs/TESTPLAN.md`](TESTPLAN.md)).

### Core Elements of the Test Plan
1. **P/D as Goodput & Efficiency**: Evaluates whether separating phases recovers enough capacity otherwise lost to collocated prefill/decode interference to outweigh:
   - The sacrificed second decode replica (halved decode concurrency).
   - Duplicate model weights in VRAM (each GPU hosting a 15.7 GB model copy).
   - Cross-GPU KV transfer overhead across PCIe Gen 5.0 x16.
2. **Three-Stage Progressive Evaluation**:
   - *Stage 1 — Single-R9700 Calibration*: Isolated prefill $P(S, C, h)$, isolated decode $D(K, C)$, and collocated mixed-load degradation ($\eta_{\text{collocated}}$).
   - *Stage 2 — Two-R9700 A/B Testing*: Replaying deterministic JSONL traces across 5 workload families (Decode-Dominant Chat, Balanced Agent, Cold Long-Context, Warm Coding, and Ingest-Heavy RAG) comparing Single-Card, DP=2 Round-Robin, DP=2 Cache-Affine, TP=2, and P/D 1P1D.
   - *Stage 3 — Efficiency & Goodput Analysis*: Measuring decode isolation efficiency ($E_{\text{decode isolation}} \ge 0.90$), recoverable interference efficiency ($E_{\text{recovery}}$), phase-pool balance ($B$), SLO-qualified goodput gain, and board-level energy efficiency ($J/\text{token}_{\text{qual}}$).
3. **Explicit Pass/Fail Targets**:
   - Decode Isolation Efficiency: $\ge 90\%$ of isolated decode rate.
   - KV Handoff Latency ($p95$): $< 50\text{ ms}$ (stretch $< 100\text{ ms}$).
   - $p95$ ITL under cold 8K bursts: $\ge 2\times$ better than DP=2 Cache-Affine.
   - $p99$ ITL under cold 8K bursts: $\le 250\text{ ms}$ (Standard Interactive Tier ceiling).
   - SLO-Qualified Goodput Gain: $\ge 1.20\times$ DP=2 Goodput.
   - Energy per Qualified Output Token: $\le 1.15\times$ DP=2 Energy Rate.
4. **Container Readiness & 6-Stage Lifecycle Audit**:
   - Container `local/vllm-mxfp4:gfx1201` verified: `msgpack-1.2.2` installed and baked into [`Dockerfile.vllm-mxfp4`](../Dockerfile.vllm-mxfp4).
   - Connector classification: `MoRIIOConnector` is `[NATIVE_RUNTIME_MISSING]` / `[NOT_READY]` due to absent `mori.io` C++ extension; `SimpleCPUOffloadConnector` and `ExampleConnector` are `[RUNTIME_READY]`.
   - Host-staged `/dev/shm` fallback: Classified as a lifecycle and correctness harness ($T_{\text{D2H}} + T_{\text{metadata}} + T_{\text{sync}} + T_{\text{H2D}} + T_{\text{admission}}$ across two PCIe hops), not a zero-copy transport.

