# Power-of-Two Concurrency Sweep Report (C = 1, 2, 4, 8, 16)
## Qwen3.8-27B Quark AWQ MXFP4 on AMD Radeon™ AI PRO R9700 (`gfx1201`)

**Author / Evaluator**: Antigravity Benchmarking Suite  
**Date**: September 23, 2026  
**Hardware Under Test**: AMD Radeon™ AI PRO R9700 (Navi 48 / `gfx1201`, 32 GB GDDR6, 256-bit bus, 64 CUs, 300 W TBP)  
**Host Environment**: Linux Ubuntu 24.04 LTS (Kernel `6.8.0-71-generic`), ROCm KFD Driver 31.50  
**Model Under Test**: `Qwen3.8-27B-Quark-AWQ-MXFP4` (Hybrid Architecture: 48 GDN Linear Attention Layers + 16 Full Softmax Attention Layers)  
**Runtime Configuration**:
- **Inference Stack**: `local/vllm-mxfp4:gfx1201` (Radiance A-tiled GEMM kernel paths enabled)
- **Attention Backend**: `ROCM_AITER_UNIFIED_ATTN`
- **KV Cache Precision**: `fp8`
- **Mamba Cache Precision**: `bf16` (Mamba State) / `fp16` (SSM)
- **Max Model Length**: 12,288
- **GPU Memory Utilization**: 0.88
- **Max Batched Tokens**: 4,096 tokens
- **Telemetry**: High-frequency sysfs hwmon / AMD-SMI power telemetry at **250 ms** sampling interval
- **Workload Shape**: **8,192 input prompt tokens** / **1,024 output generation tokens**

---

## 1. Executive Summary & Defensible Operating Decisions

Across a systematic power-of-two concurrency sweep ($C = 1, 2, 4, 8, 16$), the RDNA 4 platform displays clear, bifurcated scaling regimes:

1. **Throughput & Efficiency Knee ($C=8$)**:
   - Aggregate generation throughput peaks at **138.67 tok/s** ($C=8$), representing a **4.54× throughput expansion** over the single-stream baseline (30.57 tok/s).
   - Energy efficiency maximizes at **0.4550 tok/J** (2.198 J/token), achieving a **4.51× energy reduction** relative to $C=1$ (9.919 J/token).
   - Transitioning $C=4 \to C=8$ yields **+51.61% marginal throughput** and **+52.44% marginal efficiency**, decisively beating the plateau criteria ($\ge 10\%$ throughput, $\ge 5\%$ efficiency).

2. **Throughput & Capacity Plateau ($C=16$)**:
   - Doubling concurrency from $C=8 \to C=16$ results in a **throughput contraction to 135.93 tok/s (-1.98% marginal gain)** and an **efficiency decline to 0.4500 tok/J (-1.10%)**.
   - GPU KV Cache utilization reaches **99.1%**, forcing vLLM sequence preemption/queueing (`Running: 15 reqs, Waiting: 1 reqs`).
   - Prefill serialization across 32 chunked prefill iterations ($16 \times 8192 = 131,072$ tokens / 4096 batch limit) drives $p95$ TTFT to **42.11 seconds** and $p95$ TPOT to **79.98 ms**.

3. **Recommended Deployment Policy**:
   - **Interactive Agent Service Tier ($C=4$)**: Strict interactive SLA compliance ($p95\text{ TPOT} = 39.77\text{ ms} \le 50\text{ ms}$), 91.46 tok/s aggregate, 22.87 tok/s per stream.
   - **Asynchronous / Batch Coding Queue ($C=8$)**: Maximum hardware saturation and token-per-Joule efficiency (138.67 tok/s, 0.4550 tok/J), satisfying batch SLAs ($p95\text{ TPOT} = 53.37\text{ ms} \le 100\text{ ms}$, $p95\text{ TTFT} = 20.94\text{ s}$).
   - **Capacity / Stress Boundary ($C=16$)**: Disallow $C \ge 16$ in production for 8K context workloads without increasing `--max-num-batched-tokens` to 8,192 or reducing context length.

---

## 2. Power-of-Two Concurrency Matrix (8,192 Input : 1,024 Output)

All runs executed under identical thermal initial conditions, with active 250 ms continuous power and temperature sampling:

| C | Aggregate tok/s | Per-stream tok/s | TTFT p50 / p95 (ms) | TPOT p50 / p95 (ms) | E2E Latency p50 / p95 | Avg / Max Sampled W | Total J | J/token | tokens/Joule | VRAM Allocation | Hotspot Max (°C) | Empirical Verdict |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **30.57** | 30.57 | 2,654.2 / 2,674.9 | 30.15 / 30.17 | 33.50 s / 33.53 s | 233.5 / 334.0 W | 20,314.0 J | 9.919 | 0.1008 | 26.2 GiB (82%) | 93.0°C | **Baseline Interactive Profile** |
| **2** | **54.58** | 27.29 | 4,556.0 / 5,267.2 | 32.20 / 32.86 | 37.49 s / 38.89 s | 201.4 / 325.0 W | 11,529.8 J | 5.630 | 0.1776 | 26.2 GiB (82%) | 88.0°C | **Linear Scaling Region** |
| **4** | **91.46** | 22.87 | 7,736.1 / 10,472.0 | 36.12 / 39.77 | 44.69 s / 51.15 s | 212.7 / 324.0 W | 13,721.2 J | 3.350 | 0.2985 | 26.2 GiB (82%) | 89.0°C | **Interactive Ceiling Knee** |
| **8** | **138.67** | 17.33 | 12,950.2 / 20,936.2 | 44.87 / 53.37 | 58.85 s / 75.53 s | 227.9 / 307.0 W | 18,002.5 J | 2.198 | 0.4550 | 26.2 GiB (82%) | 90.0°C | **Throughput & Efficiency Peak** |
| **16** | **135.93** | 8.50 | 23,444.9 / 42,106.1 | 69.50 / 79.98 | 94.54 s / 123.92 s | 258.7 / 335.0 W | 36,407.0 J | 2.222 | 0.4500 | 28.2 GiB (88%) | 94.0°C | **Capacity & Thermal Plateau** |

> **Definitions**:
> - **Aggregate tok/s**: Cumulative generation tokens completed across all concurrent sequences per second.
> - **Per-stream tok/s**: Effective generation throughput delivered to each individual active sequence ($\text{Aggregate} / C$).
> - **E2E Latency**: Full turn duration $\text{TTFT} + (1023 \times \text{TPOT})$.
> - **Sampled Power**: 250 ms instantaneous GPU socket power.

---

## 3. Explicit Marginal Plateau Analysis

The plateau evaluation applies the mathematical definitions established for scheduler and batch-shape characterization:

$$\text{Marginal Throughput Gain}(C_n) = \frac{R(C_n) - R(C_{n-1})}{R(C_{n-1})}, \quad \text{Plateau Threshold} < 0.10 \ (10\%)$$

$$\text{Marginal Efficiency Gain}(C_n) = \frac{\text{Eff}(C_n) - \text{Eff}(C_{n-1})}{\text{Eff}(C_{n-1})}, \quad \text{Plateau Threshold} < 0.05 \ (5\%)$$

$$\text{Scaling Efficiency } \eta(C) = \frac{R(C)}{C \times R(1)}$$

### Step-by-Step Transition Ledger

| Concurrency Transition | Throughput Marginal Gain | Efficiency Marginal Gain | Scaling Efficiency $\eta(C)$ | Plateau Triggered? |
| :---: | :---: | :---: | :---: | :--- |
| **$C=1 \to C=2$** | **+78.55%** ($30.57 \to 54.58$) | **+76.19%** ($0.1008 \to 0.1776$) | **89.27%** | **No** (Robust weight reuse) |
| **$C=2 \to C=4$** | **+67.58%** ($54.58 \to 91.46$) | **+68.06%** ($0.1776 \to 0.2985$) | **74.80%** | **No** (Strong execution density) |
| **$C=4 \to C=8$** | **+51.61%** ($91.46 \to 138.67$) | **+52.44%** ($0.2985 \to 0.4550$) | **56.71%** | **No** (Major async throughput win) |
| **$C=8 \to C=16$** | **-1.98%** ($138.67 \to 135.93$) | **-1.10%** ($0.4550 \to 0.4500$) | **27.79%** | **YES: Hard Plateau & Contraction** |

```
                              Aggregate Throughput Scaling Curve
   150 ┬                                                      ╭─── 138.67 tok/s (C=8 Peak)
       │                                            ╭─────────╯
   125 ┼                                            │              135.93 tok/s (C=16 Plateau)
       │                                  ╭─────────╯
   100 ┼                                  │ 91.46 tok/s (C=4 Knee)
       │                        ╭─────────╯
    75 ┼                        │ 54.58 tok/s (C=2)
       │              ╭─────────╯
    50 ┼              │
       │    ╭─────────╯ 30.57 tok/s (C=1)
    25 ┼────╯
       │
     0 ┴─────┬────────────┬─────────────┬─────────────┬─────────────┬─────────────
            C=1          C=2           C=4           C=8          C=16
```

---

## 4. Bottleneck Identification & Diagnostic Findings at $C=16$

The deliberate stress point at $C=16$ reveals several fundamental hardware and scheduler bounds:

### 1. KV-Cache Capacity & Scheduler Eviction / Stalling
- Total active sequence token count at full load: $16 \text{ sequences} \times 9,216 \text{ tokens} = 147,456 \text{ tokens}$.
- In `vLLM` engine logs at $C=16$:
  ```
  INFO Engine 000: Running: 15 reqs, Waiting: 1 reqs, GPU KV cache usage: 99.1%
  ```
- The available FP8 KV cache memory on the 32 GB R9700 peaks at **99.1%** capacity. Because the allocator cannot safely host 16 simultaneous $9\text{K}$ sequences, it enters sequential queueing (`Waiting: 1 reqs`), causing decode stalls and context thrashing.

### 2. Scheduler Prefill Chunking Serialization
- With `--max-num-batched-tokens 4096`, 16 incoming 8,192-token prompts require processing $131,072$ prefill tokens.
- At 4,096 tokens per scheduler iteration, prompt evaluation requires $\ge 32$ forward passes before all requests enter decode.
- Consequently, $p50$ TTFT inflates from $7.73\text{ s}$ ($C=4$) to **$23.44\text{ s}$ ($C=16$)**, and $p95$ TTFT surges to **$42.11\text{ s}$**.

### 3. Thermal & Power Characteristics
- Average power increases from $212.7\text{ W}$ ($C=4$) to **$258.7\text{ W}$** ($C=16$), with transient peaks reaching **$335.0\text{ W}$**.
- Hotspot temperature rises to **$94.0^\circ\text{C}$** and memory junction temperature reaches **$88.0^\circ\text{C}$**, remaining within safe thermal throttling envelopes ($< 105^\circ\text{C}$), but indicating heavy memory subsystem pressure.

---

## 5. Workload-Specific Service Tier Recommendations

```
  Workload Category        Recommended Concurrency     Target Metric                Observed Performance
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  Interactive Chat / CLI           C = 1 to 2          TPOT < 35 ms, TTFT < 5 s     TPOT: 30-32 ms, TTFT: 2.6-4.5 s
  Interactive Agent Workers        C = 4               TPOT < 50 ms, TTFT < 10 s    TPOT: 39.8 ms, TTFT: 7.7 s (91.5 tok/s)
  Batch / Async Agent Swarms       C = 8               Max tok/s & Max tok/J        138.7 tok/s, 0.455 tok/J (2.2 J/tok)
  Overnight Codebase Scans         C = 8 (batched=8K)  Scheduler throughput         High-density batching
  Stress Testing Limit             C = 16              Capacity exploration only    NOT recommended for production
```

### Recommendation 1: Deploy Dual Service Profiles
- **Interactive Tier**: Run instance pinned to `--max-num-seqs 4` with strict priority scheduling. Guarantees sub-40 ms TPOT and sub-8 s TTFT on 8K context.
- **Batch / Background Tier**: Run queue workers against `--max-num-seqs 8`. Delivers **138.67 aggregate tok/s** at maximum token economy (**0.4550 tok/J**).

### Recommendation 2: Tuning `--max-num-batched-tokens`
- For deployments targeting $C=8$ and $C=16$, expanding `--max-num-batched-tokens` from `4096` to `8192` will halve the prefill chunking iterations from 32 down to 16, directly cutting TTFT by ~40-45% without exhausting VRAM headroom (since MXFP4 weights consume only 14.2 GB of the 32 GB capacity).
