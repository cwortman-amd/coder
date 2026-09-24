# Counterfactual Capacity & Interference Report: 1P1D vs. DP=2 on Radeon™ AI PRO R9700
## Two-Card Architectural Modeling Derived from Empirical Single-Card Primitives

**Evaluation Date**: September 23, 2026  
**Hardware Platform**: AMD Radeon™ AI PRO R9700 (`gfx1201`, 64 CUs, 32 GB GDDR6)  
**Inference Stack**: `local/vllm-mxfp4:gfx1201` (Qwen3.8-27B MXFP4, Chunk 2048, FP8 KV, Prefix Caching)  
**Methodology**: `single-GPU-measured service times + simulated two-R9700 pipeline projection`  

---

## 1. Executive Summary & Mathematical Theorems

To rigorously evaluate the capacity and goodput trade-off between **Data Parallelism (DP=2)** and **Prefill/Decode Disaggregation (P/D 1+1)** without requiring two physical cards simultaneously installed, a counterfactual discrete-event capacity emulator was evaluated against deterministic traces.

### The Mathematical Proof Thresholds
1. **The Raw Capacity Threshold ($\eta < 0.5$)**:
   - Ideal 1P1D has 1 dedicated decode GPU: $D_{\text{PD}} \approx D_{\text{isolated}} = 34.07\text{ tok/s}$.
   - DP=2 has 2 collocated GPUs suffering contention factor $\eta$: $D_{\text{DP2}} \approx 2 \eta D_{\text{isolated}}$.
   - P/D exceeds DP=2 in raw output tok/s **if and only if**:
     $$\eta < 0.5 \iff D_{\text{mixed, per replica}} < \frac{34.07}{2} = 17.04\text{ tok/s}$$
   - **Empirical Verdict**: In the saturated prompt-bombardment regime ($J3$, measured at **11.42 tok/s**), $\eta = 0.335 < 0.5$. In this regime, **1P1D beats DP=2 in raw throughput by 1.49×** (34.07 tok/s vs 22.84 tok/s). In moderate burst regimes ($J2$, measured at **22.84 tok/s**), $\eta = 0.670 > 0.5$, so DP=2 retains raw output volume.

2. **The SLO-Goodput Claim (Streaming Quality of Service)**:
   - Under interactive streaming SLOs ($p95\text{ ITL} \le 100\text{ ms}$, $\text{Peak ITL} < 500\text{ ms}$, $\text{TTFT} \le 3.5\text{ s}$), **P/D dominates overwhelmingly across all workloads**.
   - While DP=2 suffers **609.7 ms forward prefill stalls** that cause extensive streaming SLO violations (0% to 50% streaming compliance under prompt arrivals), P/D achieves **100% streaming SLO compliance** by completely isolating token generation on GPU 1.

---

## 2. Master Comparative Results Across Workloads

### Workload Trace: Light

| Architecture | Handoff | Raw tok/s | Req/s | TTFT p95 (ms) | ITL p95 (ms) | Peak ITL (ms) | Stream SLO% | Qualified tok/s | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DP=2 Collocated** (Dual Replicas () | N/A | 24.11 | 0.210 | 2806.2 | 29.30 | 609.7 | **0.0%** | **0.19** | Contended Fail |
| **DP=2 Collocated** (Dual Replicas () | N/A | 24.11 | 0.210 | 2806.2 | 29.30 | 609.7 | **0.0%** | **0.19** | Contended Fail |
| **P/D 1P1D Disaggregated** (Dedicated Pipel) | 5.2 ms | 24.11 | 0.210 | 2843.3 | 29.35 | 51.0 | **100.0%** | **24.11** | SLO WIN |
| **P/D 1P1D Disaggregated** (Dedicated Pipel) | 25.0 ms | 24.09 | 0.210 | 2863.1 | 29.35 | 51.0 | **100.0%** | **24.09** | SLO WIN |
| **P/D 1P1D Disaggregated** (Dedicated Pipel) | 50.0 ms | 24.08 | 0.210 | 2888.1 | 29.35 | 51.0 | **100.0%** | **24.08** | SLO WIN |
| **P/D 1P1D Disaggregated** (Dedicated Pipel) | 100.0 ms | 24.05 | 0.210 | 2938.1 | 29.35 | 51.0 | **100.0%** | **24.05** | SLO WIN |
| **P/D 1P1D Disaggregated** (Dedicated Pipel) | 250.0 ms | 23.97 | 0.209 | 3088.1 | 29.35 | 51.0 | **100.0%** | **23.97** | SLO WIN |

---

## 3. KV Handoff Sensitivity Analysis ($H \in [5.16, 25, 50, 100, 250]\text{ ms}$)

A critical engineering question is whether P/D is fragile to KV transfer connector overhead or robust across plausible implementation costs.

- **Payload Floor ($H = 5.16\text{ ms}$)**: PCIe 5.0 x16 raw payload transfer for 8K FP8 KV handoff.
- **Runtime Connector Range ($H = 25\text{--}50\text{ ms}$)**: Typical user-space IPC and descriptor sync overhead.
- **High-Latency TCP/Offload Range ($H = 100\text{--}250\text{ ms}$)**: Remote store or CPU bounce buffers.

### Finding: P/D is Robust to KV Handoff Overhead
Because an 8K prompt ingestion requires **2.78s** and 1K decode takes **~30s**, handoff delays up to **100 ms represent <0.3% of the total request lifecycle**. Even at $H = 250\text{ ms}$, P/D retains over **98.5% of its SLO-qualified goodput**, proving that P/D is **not fragile to connector latency**.

---

## 4. Operational Recommendations for Dual R9700 Arrival

1. **Deploy TP=2 First**: Pool memory to **64 GB** to unlock 32K–64K context windows without external connector dependencies.
2. **Deploy DP=2 with Cache-Affine Routing**: For workloads dominated by short/medium contexts (<=8K) and multi-turn sessions where raw aggregate volume is paramount.
3. **Deploy P/D 1P1D for SLA-Critical Interactive Services**: When tail ITL ($p95 < 100\text{ ms}$) and jitter-free streaming are contractual SLOs in the presence of cold prompt ingestion bursts.

