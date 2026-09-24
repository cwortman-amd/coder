# Comprehensive Dual-Card Radeon™ AI PRO R9700 Performance & Disaggregation Test Plan

**Document ID:** `TESTPLAN-R9700-PD-2026.1`  
**Target Hardware:** Dual AMD Radeon™ AI PRO R9700 (32 GB GDDR6, 256-bit, PCIe Gen 5.0 x16, `gfx1201`)  
**Host Architecture:** AMD Ryzen™ 9 7900X (or EPYC™ workstation), PCIe 5.0 root complex, bifurcation/switch support  
**Software Baseline:** ROCm 7.14, PyTorch 2.11.0+rocm7.14, vLLM `0.27.1` (`local/vllm-mxfp4:gfx1201`)  
**Served Model:** Qwen3.8-27B-Quark-AWQ-MXFP4 (W4A8 FP8-WMMA GEMM, FP8 KV-Cache)  
**Execution Companion Tools:** [`benchmark/bench_phases.py`](file:///home/amd/workspace/coder/benchmark/bench_phases.py), [`benchmark/pd_capacity_emulator.py`](file:///home/amd/workspace/coder/benchmark/pd_capacity_emulator.py), [`benchmark/pd_router.py`](file:///home/amd/workspace/coder/benchmark/pd_router.py)

---

## 1. Executive Summary & Architectural Premise

Prefill/Decode Disaggregation (P/D 1+1) is **not** an architecture designed to accelerate isolated single-sequence autoregressive decode rate beyond the hardware memory-bandwidth roofline (~34 tok/s for Qwen3.8-27B on a single 32 GB R9700). Rather, P/D must be evaluated as a **goodput-and-efficiency architecture**.

In a collocated deployment (such as Data Parallelism with two independent instances, DP=2), incoming long-prompt prefill bursts contend directly for compute units, memory buses, and scheduler slots against active token generation streams. This causes severe inter-token latency (ITL) jitter and forward-execution stalls (e.g., 609.7 ms stalls for 2K chunk prefills), destroying interactive Service Level Objectives (SLOs).

```
                      COLOCATED SERVING INTERFERENCE (DP=2)
 ┌────────────────────────────────────────────────────────────────────────┐
 │ GPU 0 (Replica A): [ Active Decode Stream: 29ms TPOT ]                 │
 │                    ───► [ Incoming Cold 8K Prefill Burst Arrives ] ────►│
 │                    Execution Stall: Active token decode halted for     │
 │                    609.7ms (Chunk 1) + 609.7ms (Chunk 2)...             │
 │                    Result: Stream ITL spikes to 610ms -> SLO BREACH!   │
 └────────────────────────────────────────────────────────────────────────┘

                 PREFILL / DECODE DISAGGREGATION (P/D 1+1)
 ┌─────────────────────────┐   PCIe 5.0 x16 KV Handoff   ┌────────────────────────┐
 │ GPU 0: Prefill Worker   │ ──────────────────────────► │ GPU 1: Decode Worker   │
 │ Dedicated to Prompt     │      H ≈ 5.16ms - 25ms      │ Dedicated to Streaming │
 │ Compute & KV Ingestion  │                             │ Zero-Jitter Cadence    │
 └─────────────────────────┘                             └────────────────────────┘
```

### The Architectural Trade-Off Thesis
Separating prefill and decode across two dedicated GPUs recovers execution capacity otherwise lost to collocated phase interference. However, this recovery must be quantitatively proven to outweigh three substantial architectural costs:
1. **Sacrificing a Second Decode Replica:** DP=2 provides two concurrent decode engines; 1P1D dedicates only one GPU to token generation.
2. **Duplicate Model Weights:** Each 32 GB GPU must host a full 27B model replica (~15.7 GB VRAM footprint).
3. **KV-Transfer Overhead:** Serializing, transferring over PCIe Gen 5.0 x16, and importing KV tensors introduces latency ($H$) and consumes PCIe bandwidth.

### Three-Stage Progressive Validation Strategy
To evaluate this trade-off rigorously, the test plan defines a three-stage progression:
* **Stage 1 — Single-R9700 Empirical Calibration:** Quantify isolated prefill capacity $P(S, C, h)$, isolated decode capacity $D(K, C)$, and collocated mixed-load degradation ($\eta_{\text{collocated}}$).
* **Stage 2 — Two-R9700 A/B Test:** Execute identical deterministic request traces across Single-Card Baseline, DP=2 Round-Robin, DP=2 Cache-Affine, TP=2, and P/D 1P1D.
* **Stage 3 — Efficiency & Goodput Analysis:** Measure SLO-qualified goodput, GPU phase utilization balance, and energy per qualified output token ($J/\text{token}_{\text{qual}}$).

---

## 2. Performance Metrics & Service Level Objectives (SLOs)

Performance cannot be collapsed into a single "tokens/sec" metric. All evaluations must measure and report seven distinct performance dimensions.

### 2.1 Core Metrics Definitions

| Metric | Formal Mathematical Definition | Primary Significance |
|---|---|---|
| **Raw Output Throughput** | $\displaystyle \frac{\sum_{i=1}^N O_i}{T_{\text{wall}}}$ | Total token generation capacity across the cluster, irrespective of latency quality. |
| **Completed-Request Throughput** | $\displaystyle \frac{N_{\text{completed}}}{T_{\text{wall}}}$ | High-level application transaction rate under steady-state load. |
| **SLO-Qualified Goodput** | $\displaystyle \frac{\sum_{i \in \text{Passed}} O_i}{T_{\text{wall}}}$ | **Primary interactive evaluation metric**: Throughput of tokens meeting both TTFT and ITL SLOs. |
| **Prefill Throughput** | $\displaystyle \frac{\sum_{i=1}^N S_i}{\sum_{i=1}^N T_{\text{prefill\_service}, i}}$ | Efficiency and saturation density of the prefill compute pool. |
| **Decode Throughput** | $\displaystyle \frac{\sum_{i=1}^N (O_i - 1)}{\sum_{i=1}^N T_{\text{decode\_interval}, i}}$ | Effective autoregressive generation rate of the decode engine. |
| **GPU Utilization Efficiency** | $\displaystyle \frac{\sum \text{Useful Phase Tokens}}{\int_0^{T_{\text{wall}}} P_{\text{GPU}}(t)\,dt \text{ or } T_{\text{busy}}}$ | Ratio of productive tensor computation vs. idle time or pipeline stall waste. |
| **Energy-Qualified Goodput** | $\displaystyle \frac{\sum_{i \in \text{Passed}} O_i}{\int_0^{T_{\text{wall}}} [P_0(t) + P_1(t)]\,dt}$ | Output tokens meeting strict SLOs per Joule of total board-level energy consumed. |

### 2.2 Explicit Service Objectives (SLO Tiers)

```text
================================================================================
SERVICE OBJECTIVE SPECIFICATIONS
================================================================================

1. Standard Interactive Tier (Default Agentic & Chat Workloads):
   • Time to First Token (TTFT) p95 : ≤ 3,500 ms (includes queue + prefill + handoff)
   • Inter-Token Latency (ITL) p95  : ≤ 100 ms
   • Inter-Token Latency (ITL) p99  : ≤ 250 ms
   • Peak ITL Ceiling (Max Stall)   : < 500 ms (hard disconnect / timeout threshold)

2. Premium Streaming Tier (Low-Latency Code Completion & Interactive IDE):
   • Time to First Token (TTFT) p95 : ≤ 3,500 ms
   • Inter-Token Latency (ITL) p95  : ≤ 50 ms
   • Inter-Token Latency (ITL) p99  : ≤ 100 ms
   • Peak ITL Ceiling (Max Stall)   : < 250 ms
================================================================================
```

> [!IMPORTANT]
> **Boundary Calibration Note:** Single-GPU mixed-load profiling on `gfx1201` demonstrates that at concurrency $C=2$, the median ITL is already $p50 \approx 51.0\text{ ms}$ due to memory-bus sharing across concurrent sequences. Therefore, **50 ms must NOT be used as a general standard-tier ITL goal** for $C \ge 2$. It is reserved strictly for $C=1$ dedicated streaming tiers or scaled-out decoder pools.

---

## 3. Stage 1: Single-R9700 Empirical Baseline Calibration

Before testing dual-card configurations, the single R9700 server must produce parameterized calibration curves across isolated and collocated execution modes.

### 3.1 Isolated Prefill Curve: $P(S, C, h)$
Measures prefill-only execution times ($O=1$) across sequence length $S$, batch concurrency $C$, and prefix cache hit fraction $h$.

$$\displaystyle P(S, C, h) = \frac{S \cdot C}{T_{\text{prefill\_service}}(S, C, h)}$$

* **Sequence Lengths ($S$):** 1,024, 4,096, 8,192, 12,288 tokens.
* **Concurrency ($C$):** 1, 2, 4, 8 requests.
* **Prefix Cache Hit ($h$):** 0% (Cold), 25%, 50%, 75%, 100% (Warm).
* **Chunk Budgets:** `--max-num-batched-tokens` = 2048 vs. 4096.

#### Measured Empirical Anchors (Qwen3.8-27B MXFP4 on R9700):
* **Cold 8K Prefill (Chunk 2048):** $T_{\text{prefill}} = 2,776.9\text{ ms}$ (Prompt Rate: $2,688.7\text{ tok/s}$, Mean Power: $296.8\text{ W}$, Energy: $824.2\text{ J}$).
* **Cold 8K Prefill (Chunk 4096):** $T_{\text{prefill}} = 3,047.4\text{ ms}$ (Prompt Rate: $2,950.5\text{ tok/s}$, Mean Power: $299.8\text{ W}$).
* **25% Prefix Hit (6K Suffix):** $T_{\text{prefill}} = 2,221.7\text{ ms}$ ($1.25\times$ speedup, $555.2\text{ ms}$ saved).
* **50% Prefix Hit (4K Suffix):** $T_{\text{prefill}} = 1,758.0\text{ ms}$ ($1.58\times$ speedup, $1,018.9\text{ ms}$ saved).
* **75% Prefix Hit (2K Suffix):** $T_{\text{prefill}} = 913.9\text{ ms}$ ($3.04\times$ speedup, $1,863.0\text{ ms}$ saved).
* **100% Prefix Hit (0K Suffix):** $T_{\text{prefill}} = 300.5\text{ ms}$ ($9.24\times$ speedup, metadata lookup floor).

### 3.2 Isolated Decode Curve: $D(K, C)$
Measures steady-state autoregressive token generation ($O = 1,024$ tokens) beginning after first-token emission.

$$\displaystyle D(K, C) = \frac{(O - 1) \cdot C}{T_{\text{decode\_interval}}}$$

* **Context Footprints ($K$):** 128, 1,024, 4,096, 8,192 prompt tokens.
* **Output Lengths ($O$):** 1,024, 4,096 tokens.
* **Concurrency Levels ($C$):** 1, 2, 4, 8.

#### Measured Empirical Anchors:
* **Single-Stream ($C=1$):** $D = 34.07\text{ tok/s}$, $\text{TPOT} = 29.35\text{ ms}$, $\text{ITL } p95 = 29.9\text{ ms}$, $\text{Peak ITL} = 31.2\text{ ms}$, Mean Power: $274.6\text{ W}$, Energy: $8.78\text{ J/token}$.
* **Two-Stream ($C=2$):** Aggregate $D = 41.53\text{ tok/s}$ ($20.76\text{ tok/s/seq}$), $\text{TPOT} = 48.16\text{ ms}$, $p50\text{ ITL} = 51.0\text{ ms}$.
* **Four-Stream ($C=4$):** Aggregate $D = 91.46\text{ tok/s}$ ($22.86\text{ tok/s/seq}$), $\text{TPOT} = 43.73\text{ ms}$.
* **Eight-Stream ($C=8$):** Aggregate $D = 138.67\text{ tok/s}$ ($17.33\text{ tok/s/seq}$), $\text{TPOT} = 57.69\text{ ms}$.

### 3.3 Collocated Mixed-Load Interference Curve
Measures decode throughput degradation when active generation streams are bombarded by incoming cold prefill bursts.

$$\eta_{\text{collocated}}(C, \lambda, h) = \frac{D_{\text{mixed}}(C, \lambda, h)}{D_{\text{isolated}}(C)}$$

$$L_{\text{interference}} = 1 - \eta_{\text{collocated}}$$

* **Active Streams ($C$):** 1, 2, 4 continuous decodes ($O=1,024$).
* **Arrival Rates ($\lambda$):** 0.05, 0.1, 0.2, 0.5, 1.0 cold 8K prefill req/s.
* **Chunk Stall Quanta:**
  * Cold 2K Chunk Forward Execution: **609.7 ms stall**
  * Cold 4K Chunk Forward Execution: **1,108.5 ms stall**
  * Warm Prefix Forward Execution: **253.9 ms stall**

#### Measured Empirical Interference Factors:
* **$J0$ (Zero Interference):** $\eta = 1.000$, $D = 34.07\text{ tok/s}$, $L = 0.0\%$, Peak ITL = $31.2\text{ ms}$.
* **$J1$ (Light Burst, 1 prefill / 30s):** $\eta = 0.916$, $D = 31.22\text{ tok/s}$, $L = 8.4\%$, Peak ITL = $609.7\text{ ms}$.
* **$J2$ (Moderate Burst, 1 prefill / 10s):** $\eta = 0.670$, $D = 22.84\text{ tok/s}$, $L = 33.0\%$, Peak ITL = $609.7\text{ ms}$.
* **$J3$ (Sustained Saturation, 1 prefill / 3s):** $\eta = 0.335$, $D = 11.42\text{ tok/s}$, $L = 66.5\%$, Peak ITL = $609.7\text{ ms}$.

```
           MEASURED DECODE RETENTION RATIO (η) VS PREFILL LOAD
   1.0 ┼─── J0: η=1.000 (34.07 tok/s)
       │
   0.8 ┼─────────── J1: η=0.916 (31.22 tok/s)
       │
   0.6 ┼─────────────────────── J2: η=0.670 (22.84 tok/s)
   0.5 ┼ - - - - - - - - - - - - - - - - - - - - - - - - - - [1P1D Raw Win Threshold: η < 0.5]
   0.4 ┼
       │
   0.2 ┼─────────────────────────────────── J3: η=0.335 (11.42 tok/s)
   0.0 ┴───────┬───────────────────┬───────────────────┬───────────────────►
             Zero                Light               Moderate            Sustained
                            Incoming Prefill Arrival Rate (λ)
```

---

## 4. Stage 2: Dual-R9700 A/B Architectural Evaluation

When two physical Radeon™ AI PRO R9700 cards are active, all architectures are benchmarked under identical deterministic traces.

### 4.1 Comparative Architecture Matrix

| Architecture ID | Card 0 Role | Card 1 Role | Interconnect / Comm | Primary Evaluated Capability |
|---|---|---|---|---|
| **Single-Card Baseline** | Collocated Prefill+Decode | *Idle / Power-Off* | None | Single-GPU density & energy ceiling. |
| **DP=2 Round-Robin** | Full Model Replica | Full Model Replica | Client Round-Robin HTTP | Naive horizontal scaling baseline. |
| **DP=2 Cache-Affine** | Full Replica + Session Shard A | Full Replica + Session Shard B | Hash / Prefix-Affine Router | Production-grade collocated baseline. |
| **TP=2** | Sharded Layers (Rank 0) | Sharded Layers (Rank 1) | PCIe Gen 5.0 x16 / RCCL | Memory scaling for contexts/models > 32 GB. |
| **P/D 1P1D (Cold)** | Prefill-Only Engine | Decode-Only Engine | PCIe Gen 5.0 x16 KV Handoff | Phase isolation, zero-jitter decode cadence. |
| **P/D 1P1D (Warm)** | Prefill Engine + Warm Cache | Decode-Only Engine | PCIe Gen 5.0 x16 KV Handoff | Evaluates P/D efficiency under high prefix reuse. |

### 4.2 Deterministic Workload Trace Schema
Workloads are specified as replayable `.jsonl` trace files to guarantee identical request arrivals, token counts, and prefix distributions:

```json
{"id": "r0001", "arrival_ms": 0, "session": "sess_A", "input_tokens": 8192, "output_tokens": 1024, "prefix_hit": 0.00, "slo_tier": "interactive_standard"}
{"id": "r0002", "arrival_ms": 1000, "session": "sess_B", "input_tokens": 8192, "output_tokens": 1024, "prefix_hit": 0.00, "slo_tier": "interactive_standard"}
{"id": "r0003", "arrival_ms": 2500, "session": "sess_A", "input_tokens": 8192, "output_tokens": 1024, "prefix_hit": 0.75, "slo_tier": "interactive_standard"}
{"id": "r0004", "arrival_ms": 3200, "session": "sess_C", "input_tokens": 4096, "output_tokens": 1024, "prefix_hit": 0.25, "slo_tier": "interactive_standard"}
```

### 4.3 Five Workload Evaluation Families

| Workload Family | Input : Output Shape | Prefix Hit ($h$) | Architectural Hypothesis & Expected Outcome |
|---|---|---|---|
| **1. Decode-Dominant Chat** | 1,024 : 4,096 | Low (< 25%) | **DP=2 Strong Win**: Two active decode replicas provide double the output token concurrency ($2\times D$). Prefill interference is negligible. |
| **2. Balanced Agent** | 4,096 : 1,024 | Moderate (25–50%) | **Competitive Boundary**: Prefill chunks induce moderate stalls. DP=2 leads in raw volume; P/D leads in tail ITL stability. |
| **3. Cold Long-Context Agent** | 8,192 : 1,024 | Cold (0%) | **P/D Strong Win**: Collocated DP degrades to $\eta < 0.5$ (J3 regime). P/D wins in both raw throughput (1.29×–1.71×) and SLO goodput (>86×). |
| **4. Warm Coding Agent** | 8,192 : 1,024 | High (75–100%) | **Cache-Affine DP=2 Win**: Prefix hits reduce prefill service times to 300–913 ms (stall 253 ms), preserving decode cadence while retaining dual decode replicas. |
| **5. Ingest-Heavy RAG** | 16,384 : 128 | Low (0–25%) | **P/D Pool Saturation Test**: Massive prefill compute; evaluates whether prefill worker saturates while decode worker starves. |

---

## 5. Fair Architectural Serving Configurations

To prevent biased results, both DP=2 and P/D must be tuned specifically for their architectural operating points.

```
+--------------------------------------------------------------------------------+
|                        FAIR SERVING PARAMETER MATRIX                           |
+------------------------------------+-------------------------------------------+
| DP=2 Collocated Configuration      | P/D 1P1D Disaggregated Configuration      |
+------------------------------------+-------------------------------------------+
| • Chunked Prefill: 2048 tokens     | Prefill Worker (Port 8100, GPU 0):        |
| • Prefix Caching: ENABLED          | • Max Batched Tokens: 8192 (or 16384)     |
| • Routing: Cache-Affine            | • Max Sequences: 8                        |
| • Max Sequences: 16                | • Prefill Queue: Deep FIFO                |
| • Max Batched Tokens: 2048         | • Objective: Maximize input tok/s & TTFT  |
| • Admission: Shared Global Queue   |                                           |
|                                    | Decode Worker (Port 8200, GPU 1):         |
|                                    | • Max Batched Tokens: 2048                |
|                                    | • Max Sequences: 16                       |
|                                    | • KV Capacity: 100% reserved for decode   |
|                                    | • Objective: Protect 29-30ms TPOT cadence |
+------------------------------------+-------------------------------------------+
```

---

## 6. Stage 3: Full P/D Instrumentation & Efficiency Analysis

### 6.1 Comprehensive Timestamp Instrumentation

The router and engine wrappers must record all eleven lifecycle timestamps for each request:

```
 t_arrival
     │
     ▼
 [ Prefill Admission Queue ] ──► t_prefill_queue_start
     │
     ▼
 [ Prefill Engine Compute ]  ──► t_prefill_start  ──► t_prefill_done
     │
     ▼
 [ KV Export & Serialization] ──► t_kv_export_start ──► t_kv_export_done
     │
     ▼
 [ PCIe Gen 5.0 Transport ]
     │
     ▼
 [ KV Import & Deserialization] ──► t_kv_import_start ──► t_kv_import_done
     │
     ▼
 [ Decode Admission Queue ]  ──► t_decode_queue_start
     │
     ▼
 [ Continuous Batch Decode ] ──► t_decode_start ──► t_first_token ──► t_last_token
```

#### Derived Latency Intervals:
* $\displaystyle T_{\text{prefill\_queue}} = t_{\text{prefill\_start}} - t_{\text{arrival}}$
* $\displaystyle T_{\text{prefill\_service}} = t_{\text{prefill\_done}} - t_{\text{prefill\_start}}$
* $\displaystyle T_{\text{kv\_handoff}} = t_{\text{kv\_import\_done}} - t_{\text{kv\_export\_start}}$
* $\displaystyle T_{\text{decode\_queue}} = t_{\text{decode\_start}} - t_{\text{kv\_import\_done}}$
* $\displaystyle \text{TTFT} = t_{\text{first\_token}} - t_{\text{arrival}}$
* $\displaystyle \text{TPOT} = \frac{t_{\text{last\_token}} - t_{\text{first\_token}}}{N_{\text{output}} - 1}$

> [!CAUTION]
> **Measurement Integrity Warning:** Do NOT report the theoretical 5.16 ms PCIe copy time as the P/D handoff duration. The reported $T_{\text{kv\_handoff}}$ must capture the complete end-to-end path: tensor export, RPC/ZMQ handshake, DMA transfer, page registration, and decode scheduler admission.

### 6.2 Efficiency Analysis Formulations

#### 1. Decode Isolation Efficiency ($E_{\text{decode isolation}}$)
Evaluates whether the dedicated decode worker achieves its isolated hardware capability:

$$E_{\text{decode isolation}} = \frac{D_{\text{P/D decode}}}{D_{\text{isolated decode}}} \quad (\text{Target} \ge 0.90)$$

*If $E_{\text{decode isolation}} < 0.90$, investigate KV cache memory layout fragmentation, allocator lock contention, or unintentional prefill execution on the decode worker.*

#### 2. Recoverable Interference Efficiency ($E_{\text{recovery}}$)
Quantifies what fraction of the interference loss caused by collocated serving is successfully recovered:

$$E_{\text{recovery}} = \frac{D_{\text{P/D decode}} - D_{\text{collocated mixed}}}{D_{\text{isolated decode}} - D_{\text{collocated mixed}}}$$

* $E_{\text{recovery}} = 1.0$: Complete recovery; decode rate matches isolated baseline.
* $E_{\text{recovery}} = 0.0$: Zero recovery; P/D performs no better than collocated serving.
* $E_{\text{recovery}} < 0.0$: P/D pipeline/queueing overhead worsens performance.

#### 3. Phase-Pool Balance Index ($B$)
Measures capacity utilization balance across the two specialized GPUs over observation window $W$:

$$U_P = \frac{\sum T_{\text{prefill\_service}}}{W}, \qquad U_D = \frac{\sum T_{\text{decode\_busy}}}{W}$$

$$B = \frac{\min(U_P, U_D)}{\max(U_P, U_D)} \quad (0.0 \le B \le 1.0)$$

* $B \to 1.0$: Ideal balanced workload; both GPUs operate near full saturation.
* $B < 0.20$: Highly imbalanced; indicates stranded GPU capacity (e.g., idle prefill worker during decode-heavy workloads).

#### 4. SLO-Qualified Goodput Gain
Computes the interactive goodput multiplier achieved by P/D over DP=2:

$$\text{Goodput Gain} = \frac{G_{\text{qualified, P/D}}}{G_{\text{qualified, DP=2}}}$$

#### 5. Energy-Qualified Efficiency
Measures the energy cost per compliant token using physical sysfs hwmon sensors ($250\text{ ms}$ sampling):

$$E_{\text{node}} = \int_0^W [P_{\text{R9700, 0}}(t) + P_{\text{R9700, 1}}(t)]\,dt \quad (\text{Joules})$$

$$\text{Energy Per Qualified Token} = \frac{E_{\text{node}}}{\sum_{i \in \text{Passed}} O_i} \quad (\text{Joules / Qualified Token})$$

---

## 7. Explicit Pass / Fail Acceptance Criteria

Success must be evaluated against predefined numeric gates:

| Evaluation Criterion | P/D Acceptance Target | Technical & Business Rationale |
|---|---|---|
| **Decode Isolation Efficiency** | $\ge 90\%$ of isolated decode rate | Confirms decode worker operates free of prefill stalls. |
| **P/D KV Handoff Latency ($p95$)** | $< 50\text{ ms}$ (stretch $< 100\text{ ms}$) | Ensures software control plane and transfer do not dominate TTFT. |
| **$p95$ ITL Under Cold 8K Bursts** | $\ge 2\times$ lower than DP=2 Cache-Affine | Proves interactive latency protection against prompt stalls. |
| **$p99$ ITL Under Cold 8K Bursts** | $\le 250\text{ ms}$ (Standard Tier Limit) | Guarantees elimination of multi-hundred-millisecond outliers. |
| **SLO-Qualified Goodput Gain** | $\ge 1.20\times$ DP=2 Goodput | Justifies the added operational complexity of disaggregated routing. |
| **Raw Output Throughput** | Report honestly (no mandatory P/D win) | DP=2 is expected to lead in raw volume under light/warm loads. |
| **Prefill Worker Utilization** | $> 40\%$ (or explicitly justified) | Identifies whether the 1P1D topology strands prefill capacity. |
| **Energy / Qualified Output Token** | $\le 1.15\times$ DP=2 Energy Rate | Proves second card delivers proportional interactive value per Joule. |
| **Continuous Pipeline Stability** | 60-minute soak without leak or crash | Confirms robustness of KV memory manager and ZMQ control plane. |

---

## 8. Technical Runtime & Container Readiness Audit

An audit of the production container (`local/vllm-mxfp4:gfx1201`) was conducted to establish exact runtime dependency status across the inference stack:

```text
================================================================================
CONTAINER DEPENDENCY AUDIT (vLLM 0.27.1 / ROCm 7.14)
================================================================================
• PyTorch ROCm Backend   : 2.11.0+rocm7.14               [INSTALLED / OPERATIONAL]
• AITER Attention Backend: ROCM_AITER_UNIFIED_ATTN       [INSTALLED / OPERATIONAL]
• MXFP4 W4A8 FP8-WMMA    : radiance_mxfp4_fp8.so         [INSTALLED / OPERATIONAL]
• Fast Serializer        : msgspec                       [INSTALLED / OPERATIONAL]
• Network Control Plane  : pyzmq                         [INSTALLED / OPERATIONAL]
• MessagePack Serializer : msgpack (1.2.2)               [INSTALLED / OPERATIONAL]
• MoRI-IO Native Engine  : mori.io C++ backend           [MISSING: No module named 'mori']
================================================================================
```

### 8.1 Six-Stage Connector Readiness Classification

To avoid conflating registry visibility or Python class importability with operational multi-GPU transport capability, connector qualification follows a 6-stage lifecycle model:

$$\text{REGISTERED} \longrightarrow \text{PYTHON\_IMPORTABLE} \longrightarrow \text{NATIVE\_RUNTIME\_MISSING} \longrightarrow \text{RUNTIME\_READY} \longrightarrow \text{TRANSFER\_VALIDATED} \longrightarrow \text{PERF\_VALIDATED}$$

| Stage | Lifecycle State | Verification Criteria | Container Status (`local/vllm-mxfp4:gfx1201`) |
| :---: | :--- | :--- | :--- |
| **1** | `REGISTERED` | Class name registered in `KVConnectorFactory._registry`. | `MoRIIOConnector`, `SimpleCPUOffloadConnector`, `ExampleConnector`, `LMCacheConnectorV1`, `NixlConnector`, `MooncakeConnector` |
| **2** | `PYTHON_IMPORTABLE` | Python module loads without `ModuleNotFoundError`. | All registered connectors import cleanly. |
| **3** | `NATIVE_RUNTIME_MISSING` | Python class exists, but native C++ shared library (`.so`) is absent. | **`MoRIIOConnector`** (`mori.io` native C++ absent $\implies$ `[NOT_READY]` for live P/D). |
| **4** | `RUNTIME_READY` | Python module and all required native runtimes are present and linked. | **`SimpleCPUOffloadConnector`**, **`ExampleConnector`** (PyTorch CPU DMA; zero external native C++ dependency). |
| **5** | `TRANSFER_VALIDATED` | Producer/consumer smoke test completes; KV blocks accepted without prompt recomputation. | Pending hardware arrival (Gate 2). |
| **6** | `PERF_VALIDATED` | Measured cross-GPU handoff latency satisfies PCIe 5.0 bounds ($p95 < 50\text{ ms}$). | Pending dual R9700 physical benchmark (Phase 5). |

### 8.2 KV Handoff Transport Architectures: Direct P2P vs. Host-Staged Fallback

When deploying Prefill/Decode Disaggregation, the physical KV transport mechanism dictates end-to-end handoff latency:

1. **Direct Peer-to-Peer PCIe DMA (`MoRIIOConnector`):**
   * Operates via direct GPU-to-GPU PCIe peer-to-peer DMA over the host PCIe 5.0 root complex:
     $$T_{\text{direct P2P}} = T_{\text{serialize}} + T_{\text{control plane}} + T_{\text{buffer ready}} + T_{\text{data movement}} + T_{\text{decode admission}}$$
   * Ideal payload-copy floor for 8,192 tokens of Qwen3.8-27B FP8 KV cache is **5.16 ms** (at empirical 52.0 GB/s PCIe 5.0 x16). Total handoff is expected at 15–35 ms.
   * **Gating blocker:** Requires AMD's native `mori.io` C++ library to be compiled into the ROCm container.

2. **Host-Staged Shared-Memory Prototype (`/dev/shm` & `SimpleCPUOffloadConnector`):**
   * When native P2P transport is unavailable, a host-staged shared-memory IPC harness (`torch.multiprocessing`, POSIX `/dev/shm`, or `SimpleCPUOffloadConnector`) is used for correctness, choreography, and lifecycle validation.
   * **Physical Transport Reality:** POSIX `/dev/shm` is host DRAM, not GPU VRAM. Transferring KV cache between GPUs via host shared memory requires two separate PCIe bus traversals:
     $$T_{\text{shm handoff}} = T_{\text{D2H}} + T_{\text{metadata}} + T_{\text{shm sync}} + T_{\text{H2D}} + T_{\text{decode admission}}$$
     * Step 1: Device-to-Host DMA transfer (GPU 0 VRAM $\to$ Host RAM pinned buffer).
     * Step 2: Inter-process POSIX synchronization fence and memory barrier.
     * Step 3: Host-to-Device DMA transfer (Host RAM pinned buffer $\to$ GPU 1 VRAM).
   * **Methodological Scope:** The `/dev/shm` mechanism is explicitly defined as a **correctness, lifecycle, and router validation harness**. It must **not** be characterized as "zero-copy" or assumed to achieve sub-8 ms handoff without empirical measurement of the dual DMA hops.

### 8.3 Production Router & Load Balancing Architecture

Both [`benchmark/pd_router.py`](file:///home/amd/workspace/coder/benchmark/pd_router.py) and [`benchmark/dp_router.py`](file:///home/amd/workspace/coder/benchmark/dp_router.py) have been hardened for production stability:

1. **Persistent Connection Pool Management:**
   * Downstream vLLM engines are accessed via a persistent `httpx.AsyncClient` managed in the FastAPI application `lifespan`.
   * Connection limits: `httpx.Limits(max_keepalive_connections=50, max_connections=200)`.
   * Timeout configuration: `connect=5.0s, pool=5.0s, write=30.0s, read=None` (unbounded read timeout allows long-duration autoregressive generation streams to proceed without premature connection abort).

2. **Application-Level Idle-Stream Watchdog:**
   * Active generation streams are guarded by an application-level idle chunk watchdog (`IDLE_STREAM_TIMEOUT_S = 60.0`). If an upstream decode worker halts token emission for $>60\text{ s}$, the watchdog cleanly tears down the stream, emits an SSE error frame, and updates telemetry counters.

3. **Defensive Telemetry Finalization:**
   * Request and stream completion metrics are finalized inside defensive `finally` blocks, ensuring that client disconnects (`asyncio.CancelledError`, `GeneratorExit`) or upstream exceptions never leave uncollected metrics or mask primary exceptions.
   * Real-time metrics and connection pool health are exposed via `/metrics/pd` and `/metrics/dp`.

4. **Namespaced Cache-Affine DP Routing:**
   * To prevent plain round-robin from alternating multi-turn agent conversations across GPUs (which destroys prefix cache hit rates by causing ~50% artificial cache misses), `dp_router.py` implements CRC32 cache-affine replica pinning using namespaced session keys:
     $$\text{Session Key} = \text{tenant\_id} : \text{user\_id} : \text{conversation\_id}$$
   * Extraction precedence: (1) HTTP headers (`x-tenant-id`, `x-user-id`, `x-session-id`, `session-id`), (2) JSON body metadata, (3) prompt-derived SHA-256 hash fallback (`default:default:{hash}`).
   * Observability metrics track affinity hits, affinity misses, active streams, and per-replica request distributions.

---

## 9. Step-by-Step Execution Plan

```
 Phase 1: Freeze Baseline Calibration
    │
    ▼
 Phase 2: Trace Generation & Parameterization
    │
    ▼
 Phase 3: Single-Card Capacity & Interference Emulation
    │
    ▼
 Phase 4: Container Update & Multi-GPU Environment Verification
    │
    ▼
 Phase 5: Micro-Handoff Functional Validation (256B to 16KB)
    │
    ▼
 Phase 6: Live Two-Card A/B Benchmark Execution
    │
    ▼
 Phase 7: 60-Minute Soak & Memory Stability Test
    │
    ▼
 Phase 8: Analysis, Energy Accounting & Artifact Publication
```

### Phase 1: Freeze Baseline Calibration
* Run [`benchmark/bench_phases.py`](file:///home/amd/workspace/coder/benchmark/bench_phases.py) with `--mode isolated-prefill`, `--mode isolated-decode`, and `--mode contention-jitter`.
* Confirm single-card figures: Cold 8K prefill ~2.78s, Decode TPOT ~29.35ms (34.07 tok/s), J3 retention factor $\eta = 0.335$.

### Phase 2: Deterministic Trace Generation
* Generate standardized `.jsonl` traces in `_results/traces/` representing the five workload families (Decode-Dominant, Balanced, Cold Long-Context, Warm Coding, and Ingest-Heavy RAG).

### Phase 3: Single-Card Counterfactual Emulation
* Execute [`benchmark/pd_capacity_emulator.py`](file:///home/amd/workspace/coder/benchmark/pd_capacity_emulator.py) across all five workload traces.
* Map the exact arrival rate $\lambda^*$ where $\eta_{\text{collocated}} < 0.5$, identifying the theoretical crossover threshold where P/D exceeds DP=2 raw capacity.

### Phase 4: Container Update & Hardware Check
* Install `msgpack` into the container image:
  ```bash
  docker exec -it rocm-mxfp4-server pip install msgpack
  ```
* Upon arrival of the second R9700 GPU, run PCIe topology inspection:
  ```bash
  python3 scripts/inspect_dual_gpu.py
  ```
* Verify PCIe Gen 5.0 x16 link negotiation (32 GT/s) and peer-to-peer DMA support.

### Phase 5: Micro-Handoff Functional Validation
* Start prefill (port 8100) and decode (port 8200) engines.
* Execute unit handoff transfers of incremental prompt sizes: 256 tokens, 1,024 tokens, 4,096 tokens, 8,192 tokens, and 16,384 tokens.
* Measure and log raw handoff latency ($T_{\text{kv\_handoff}}$). Confirm $p95 < 50\text{ ms}$.

### Phase 6: Live Two-Card A/B Benchmark Execution
* Replay the deterministic traces across:
  1. `docker-compose.dp2.yml` (DP=2 Round-Robin & Cache-Affine)
  2. `docker-compose.pd.yml` (P/D 1P1D)
  3. `docker-compose.tp2.yml` (TP=2 baseline for contexts > 32 GB)
* Collect client logs, engine metrics, and continuous 250ms hwmon power logs.

### Phase 7: 60-Minute Soak & Stability Test
* Run continuous sustained load under Workload Family 3 (Cold Long-Context) for 60 minutes.
* Monitor GPU VRAM fragmentation, ZMQ socket leaks, and KV page release correctness.

### Phase 8: Comprehensive Analysis & Publication
* Calculate $E_{\text{decode isolation}}$, $E_{\text{recovery}}$, Phase Balance $B$, Qualified Goodput Gain, and Joules/Qualified Token.
* Compile final report into `docs/DUAL_R9700_EMPIRICAL_EVALUATION.md` and commit all JSON traces to version control.
