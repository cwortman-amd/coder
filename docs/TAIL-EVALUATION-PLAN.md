---
type: Evaluation Plan
title: Enterprise LLM Tail Latency & Agent Serving Evaluation Plan
description: 'Document ID: EVAL-TAIL-AGENT-2026.1 Target Hardware: Single & Dual AMD
  Radeon™ AI PRO R9700 (32 GB GDDR6) / AMD Instinct™ MI350P (288 GB HBM3e).'
tags:
- evaluation
- tail-latency
- agentx
status: stable
---

# Enterprise LLM Tail Latency & Agent Serving Evaluation Plan

**Document ID:** `EVAL-TAIL-AGENT-2026.1`  
**Target Hardware:** Single & Dual AMD Radeon™ AI PRO R9700 (32 GB GDDR6) / AMD Instinct™ MI350P (288 GB HBM3e)  
**Software Baseline:** ROCm 7.14, vLLM `0.27.1` (`local/vllm-mxfp4:gfx1201`), Inference Perf / aiperf  
**Served Model:** `Qwen3.8-27B-Quark-AWQ-MXFP4` (W4A8 FP8-WMMA, FP8 KV Cache)  
**Execution Companion Tools:**  
* [`scripts/request_event_schema.py`](/scripts/request_event_schema.py)
* [`scripts/bench_agent_chain.py`](/scripts/bench_agent_chain.py)
* [`scripts/bench_prefill_decode_interference.py`](/scripts/bench_prefill_decode_interference.py)
* [`scripts/bench_open_loop_sweep.py`](/scripts/bench_open_loop_sweep.py)
* [`scripts/bench_cold_start_probe.py`](/scripts/bench_cold_start_probe.py)
* [`scripts/analyze_tail_metrics.py`](/scripts/analyze_tail_metrics.py)
* [`scripts/plot_tail_distributions.py`](/scripts/plot_tail_distributions.py)

---

## 1. Architectural Strategy: Two Linked Benchmark Tracks

A rigorous enterprise evaluation cannot rely on either synthetic compute benchmarks alone or uninstrumented user traces alone. We establish **two linked benchmark tracks**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 TRACK 1: CONTROLLED INFERENCE-SERVER EXPERIMENTS            │
│                 "Isolates the causes of tail latency — Tells you WHY"       │
│  • Single-call closed-loop concurrency sweeps (C=1, 5, 20, ...)             │
│  • Open-loop rate sweeps (Poisson & paced arrivals from low load to overload)│
│  • Timed prefill-burst injection onto active decoders (1K, 4K, 8K prompts)  │
│  • Memory perturbation A/B tests (cold start, KV thrashing, chunk size)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Correlates With
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 TRACK 2: END-TO-END AGENTX TASK REPLAYS                     │
│                 "Evaluates user-facing impact — Tells you IF IT MATTERS"    │
│  • Real Claude Code multi-turn agent traces (semianalysis_cc_traces)        │
│  • Synthetic controlled-step DAGs (serial tool chains vs. parallel fan-out) │
│  • Task-level completion time, critical-path waterfalls, & deadline misses │
│  • Dual resampling baselines (independent i.i.d. draws vs block correlation)│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Unified Request & Event Telemetry Schema

To eliminate measurement ambiguity across all experiments, every test client and trace recorder adheres to [`scripts/request_event_schema.py`](/scripts/request_event_schema.py).

### 2.1 Distinct Measurement Boundaries

We strictly isolate three measurement boundaries and **never pool them into the same percentile distribution**:

1. **Scheduled Arrival $\to$ First Output:** Captures open-loop client-side queueing, load-generator lag, network transport, server queueing, and prefill execution.
2. **Actual Send $\to$ First Output:** Standard client-experienced TTFT. Measures from the moment HTTP write begins until the first token chunk arrives over the wire.
3. **Server Admission $\to$ First Output:** Server-side engine prefill execution duration. Requires server spans (`prefill_duration_ms`). Client-side TTFT includes network, proxy, and batching queue overhead; server spans isolate engine GEMM time.

### 2.2 Event Field Dictionary

| Category | Field Name | Type | Description |
|---|---|---|---|
| **Identity** | `request_id` | `str` | Unique UUID for client request |
| | `task_id` | `str` | High-level agent trajectory / session ID |
| | `turn_index` | `int` | Sequential step index within agent trajectory |
| | `replica_id` | `str` | Specific GPU / worker container instance |
| **Timestamps** | `scheduled_send_ns` | `int` | Intended dispatch timestamp (open-loop target) |
| | `actual_send_ns` | `int` | Timestamp socket write actually began (detects client lag) |
| | `first_streamed_output_ns` | `int` | Client-observed TTFT (first reasoning or content chunk) |
| | `first_answer_token_ns` | `int` | Client-observed TTFAT (first visible user-facing token) |
| | `token_timestamps_ns` | `List[int]` | Nanosecond timestamps for **every individual token event** |
| | `completion_or_timeout_ns` | `int` | Stream closure, error, or cancellation timestamp |
| **Accounting** | `prompt_tokens` | `int` | Tokens processed during prefill phase |
| | `completion_tokens` | `int` | Total tokens generated during decode phase |
| | `reasoning_tokens` | `int` | Internal reasoning / `<think>` tokens |
| | `answer_tokens` | `int` | Visible user-facing output tokens |
| **Configuration** | `serving_configuration` | `str` | Engine flags (e.g. `--max-num-batched-tokens 2048`, P/D 1P1D) |
| | `prompt_length_bucket` | `str` | Tagged length bracket (`short_100`, `long_8192`) |
| | `output_length_bucket` | `str` | Tagged generation bracket (`short_64`, `long_512`) |
| | `offered_load_tag` | `str` | Offered load cell (`qps_5.0`, `concurrency_16`) |
| | `cache_state` | `str` | `cold_miss`, `warm_hit`, `partial_evicted` |
| **Status** | `status` | `str` | `completed`, `timeout`, `cancelled`, `error` |
| **Server Spans** | `queue_duration_ms` | `float` | Server-side time in queue before worker dispatch |
| | `prefill_duration_ms` | `float` | Server-side prompt GEMM execution time |
| | `decode_duration_ms` | `float` | Server-side token generation execution time |
| | `kv_transfer_duration_ms` | `float` | (For P/D) Time spent transferring KV tensors across PCIe/network |

### 2.3 Explicit Metric Weighting & Calculation Rules

1. **Uncompressed ITL Distribution (Token-Weighted):**  
   We **never** substitute request-average TPOT for individual token arrival gaps. A request with a nominal 25 ms TPOT can experience a 500 ms pause when an incoming prefill burst runs, causing a conspicuous stream freeze that disappears in the average. The ITL distribution is compiled by pooling all raw token intervals across all requests.
2. **Per-Request Decode Rate Calculation:**  
   Decode rate is strictly evaluated when $M > 1$ using token arrival timestamps:
   $$\text{Decode Rate} = \frac{M - 1}{t_{\text{last\_token}} - t_{\text{first\_token}}}$$
   Evaluating $(M / t)$ or relying on trailing usage metadata chunks introduces end-of-stream socket flush artifacts and fencepost errors.
3. **Request-Level TTFT & TPOT (Request-Weighted):**  
   TTFT, TTFAT, and request-average TPOT are calculated once per request and aggregated with linear interpolation percentiles (p50, p90, p95, p99).
4. **Explicit Error & Timeout Accounting:**  
   Timeouts, errors, and client cancellations are tracked as discrete first-class metrics (`timeout_requests`, `error_requests`), never silently dropped from latency CDFs.

---

## 3. The 7-Experiment Matrix

| # | Experiment Name & Objective | Workload & Methodological Controls | Decisive Datapoints & Figures | Companion Tool |
|---|---|---|---|---|
| **1** | **Concurrency and Queueing**<br>*Determine how p50, p95, and p99 TTFT and completion latency change as offered work approaches and exceeds sustainable capacity.* | • Closed-loop sweep: Concurrency $C=1, 5, 20$, and levels around the observed throughput knee.<br>• Open-loop sweep: Scheduled Poisson/paced arrivals from low load to overload.<br>• Replication: Hundreds of completed requests for exploratory p95; several thousand across repeated runs before quantitative p99 claims. | **Plot:** Load-vs-latency percentile curves; TTFT and completion CDFs; achieved throughput and deadline-qualified Goodput vs. offered load.<br>**Decisive Finding:** Identify the saturation knee where p50 stays flat (<150ms) while p99 blows out by 5x+ or deadline-miss rate surges. | [`bench_open_loop_sweep.py`](/scripts/bench_open_loop_sweep.py) |
| **2** | **TTFT vs. Generation Latency**<br>*Isolate engine prefill launch time from ongoing autoregressive decode speed, and quantify the crossover.* | • Fixed input $\times$ output grid (e.g. 100/1000 prompt tokens $\times$ 32/256 decode tokens) at matched load.<br>• Request waterfall showing client dispatch, TTFT, each ITL gap, and stream finish.<br>• Correct crossover math ($M \approx 21$ tokens). | **Plot:** TTFT vs. input tokens; TPOT and worst ITL vs. active decodes; Request waterfalls; Latency trade-off curves across generation length $M$.<br>**Decisive Finding:** Separates a slow engine launch from a frozen or slow token stream, establishing empirical crossover points. | [`bench_agent_chain.py`](/scripts/bench_agent_chain.py) |
| **3** | **Cold Starts and Idle Intervals**<br>*Quantify post-idle TTFT penalties and evaluate bimodal latency distributions.* | • Cool-down probe with varied idle intervals (e.g. 0s, 30s, 60s, 120s, 300s).<br>• Measure client TTFT, TCP connect time, TLS handshake time, and first byte delivery.<br>• Label findings descriptively ("bimodal TTFT", "post-idle TTFT penalty") rather than diagnostic leaps unless confirmed by server/container logs. | **Plot:** TTFT vs. idle duration; distribution histograms testing for multimodality; network connection vs engine prefill breakdown.<br>**Decisive Finding:** Quantifies the latency penalty of scale-from-zero or idle serverless workers without speculative diagnoses. | [`bench_cold_start_probe.py`](/scripts/bench_cold_start_probe.py) |
| **4** | **Provider Consistency & Multi-Tenancy**<br>*Measure latency stability and jitter across endpoints under identical workloads.* | • Send identical, interleaved requests from the same client VM to local AMD GPU endpoint vs. hosted APIs over multi-hour windows.<br>• Compute coefficient of variation (CV = $\sigma / \mu$), interquartile range (IQR), and p99:p50 ratio.<br>• Explicitly distinguish customer-experienced endpoint performance from bare-silicon roofline. | **Plot:** Time-series latency traces; per-provider CDFs; CV and p99:p50 dispersion plots.<br>**Decisive Finding:** Quantifies multi-tenant jitter and tail variance under matched prompt/output conditions. | [`analyze_tail_metrics.py`](/scripts/analyze_tail_metrics.py) |
| **5** | **Agent-Chain Compounding & Task Durations**<br>*Determine how single-call tail latency propagates through sequential and parallel multi-turn workflows.* | • Replay real Claude Code multi-turn agent traces (`semianalysis_cc_traces`) and synthetic DAGs ($N=10\dots 30$).<br>• Sequential model calls add: $T = \sum T_{\text{LLM}} + \sum T_{\text{tool}} + T_{\text{orchestration}}$.<br>• Evaluate dual resampling baselines: independent i.i.d. draws vs. block/time-correlated resampling. | **Plot:** Complete task p50/p95/p99 duration; task deadline-miss rate; empirical vs. theoretical risk ($1 - (1-p)^n$); single-call vs. whole-task spread.<br>**Decisive Finding:** Demonstrates that sequential chains add with correlated queueing, and parallel fan-out is bound by $\max(t_i)$. | [`bench_agent_chain.py`](/scripts/bench_agent_chain.py) / [`run_agentx_tail_sweep.py`](/scripts/run_agentx_tail_sweep.py) |
| **6** | **Prefill/Decode Interference**<br>*Test the hypothesis that incoming prefill bursts degrade active decode streams, and evaluate disaggregation.* | • Steady decoders (e.g. 4 active streams $\times$ 128 tokens).<br>• Timed long-prompt injections (1K, 4K, 8K tokens).<br>• Compare collocated chunked prefill (default vs tuned chunk sizes) vs. P/D 1P1D disaggregation at equal accelerator budget. | **Plot:** Aligned timeline of injected prefill vs. active decode ITL intervals; peak per-request ITL stall; goodput retention under burst load.<br>**Decisive Finding:** Proves whether isolating prefill eliminates decode freezes without sacrificing overall throughput. | [`bench_prefill_decode_interference.py`](/scripts/bench_prefill_decode_interference.py) |
| **7** | **SLOs vs. Raw Tokens/s**<br>*Evaluate cluster capacity using deadline-qualified Goodput rather than raw peak throughput.* | • Sweep offered load across candidate configurations.<br>• Apply pre-registered interactive SLOs: TTFT $\le 1.0\text{s}$, worst ITL $\le 100\text{ms}$, task duration $\le 30\text{s}$.<br>• Goodput counts tokens only from requests that met all defined SLO thresholds. | **Plot:** Achieved raw tokens/s vs. SLO-qualified Goodput tokens/s on the same offered load axis.<br>**Decisive Finding:** Yields an actionable capacity knee based on user-acceptable latency rather than saturated engine throughput. | [`bench_open_loop_sweep.py`](/scripts/bench_open_loop_sweep.py) |

---

## 4. Methodological Foundations & Mathematical Rigor

### 4.1 The Exact TTFT vs. Decode Speed Crossover Derivation

When evaluating trade-offs between a fast-TTFT/slow-decode engine (Engine A) and a slow-TTFT/fast-decode engine (Engine B), total generation latency for $M$ output tokens is modeled as:
$$T = \text{TTFT} + (M - 1) \cdot \overline{\text{ITL}}$$

Consider a representative scenario:
* **Engine A (Fast TTFT, Slower Decode):** $\text{TTFT}_A = 200\text{ ms} = 0.2\text{ s}$, Decode Rate $= 20\text{ tok/s} \implies \overline{\text{ITL}}_A = 0.05\text{ s/tok}$.
* **Engine B (Slower TTFT, Fast Decode):** $\text{TTFT}_B = 1000\text{ ms} = 1.0\text{ s}$, Decode Rate $= 100\text{ tok/s} \implies \overline{\text{ITL}}_B = 0.01\text{ s/tok}$.

Setting $T_A = T_B$:
$$0.2 + (M - 1) \cdot 0.05 = 1.0 + (M - 1) \cdot 0.01$$
$$(M - 1) \cdot (0.05 - 0.01) = 1.0 - 0.2$$
$$(M - 1) \cdot 0.04 = 0.8 \implies M - 1 = 20 \implies M = 21\text{ tokens}$$

> [!NOTE]
> * Informal claims of $M \approx 15$ tokens typically stem from omitting the $-1$ token fencepost ($M$ vs $M-1$) or dividing by an incorrect rate delta.
> * At $M < 21$ tokens (e.g. classification, binary decisions, short agent action tags), Engine A is strictly faster.
> * At $M > 21$ tokens (e.g. code generation, chain-of-thought planning), Engine B's decode advantage dominates.
> * In real production workloads, variance in ITL, chunked prefill stalls, and network packetization shift this boundary; therefore, crossover points must be reported as **empirical measurements across the $(N, M)$ grid**, not assumed as static constants.

### 4.2 Compounding Outlier Risk vs. Workflow Duration

1. **Independent Outlier Encounter Risk:**
   $$P(\ge 1\text{ outlier in } n\text{ independent calls}) = 1 - (1 - p)^n$$
   For $n=10$ calls with $p=0.01$ (p99 threshold), $P = 1 - (0.99)^{10} \approx 9.56\%$.  
   For $n=10$ calls with $p=0.05$ (p95 threshold), $P = 1 - (0.95)^{10} \approx 40.13\%$.
2. **What This Math Does NOT Say:**
   * It is **not** a formula for the workflow's p99 latency.
   * **Sequential calls add:** $T_{\text{task}} = \sum_{i=1}^n T_{\text{LLM}, i} + \sum_{j=1}^m T_{\text{tool}, j} + T_{\text{orchestration}}$. The task latency distribution is the convolution of the individual response time distributions.
   * **Parallel calls wait for the slowest:** $T_{\text{scatter}} = \max(t_1, t_2, \dots, t_k)$. As Dean & Barroso demonstrate in *The Tail at Scale*, if a user request triggers 100 parallel sub-requests each with a 1% chance of taking $>1\text{s}$, over 63% of user requests will take $>1\text{s}$.
   * **Queueing and KV memory pressure are correlated:** Calls within the same agent session often hit the same server/replica within a narrow time window, meaning high latency on turn $i$ increases the probability of high latency on turn $i+1$ ($\text{Cov}(t_i, t_j) > 0$).

### 4.3 Two Statistical Resampling Baselines for Agent Tasks

To rigorously evaluate how tail latency affects complete multi-turn agent tasks without making invalid independence assumptions, [`scripts/bench_agent_chain.py`](/scripts/bench_agent_chain.py) computes **two distinct resampling baselines**:

1. **Resampling Baseline 1: Independent i.i.d. Draws:**
   * Randomly samples $N$ call durations from the single-call baseline distribution with replacement ($B = 5,000$ iterations).
   * Models an idealized system where every call is completely uncorrelated with previous turns.
2. **Resampling Baseline 2: Block / Time-Preserving Resampling:**
   * Resamples contiguous temporal blocks of calls or preserves turn ordering from empirical sessions.
   * Preserves real-world temporal autocorrelation, server queue state, and context accumulation effects.
3. **Contrast with Empirical Task Replays:**
   Comparing Empirical Task Duration against Baseline 1 and Baseline 2 reveals the exact magnitude of latency penalty caused by **correlated congestion and session context growth**.

---

## 5. Five Rules for Defensible Evidence

1. **Pilot First, Then Size the Tail Run:**  
   A 50-request run can identify trace bugs, but provides zero statistical confidence on p99 (tail resolution $\approx 2\%$). A defensible p99 estimate requires hundreds of requests per condition for exploratory runs, and **several thousand requests across repeated runs** before making quantitative p99 claims. For agent workflows, resample and size at the **task level**, not just the individual token level.
2. **Validate the Load Generator (Detect Coordinated Omission):**  
   Record both `scheduled_send_ns` and `actual_send_ns` on every request. If `client_send_lag_p95_ms` exceeds 5 ms, the load generator is saturated and the run is invalid. Never assume an asynchronous or threaded client is keeping pace without verifying send lag.
3. **Avoid Diagnostic Leaps from Latency Shapes:**  
   A bimodal histogram or high coefficient of variation describes the shape of the data; it does **not** prove GPU queueing, cold starts, or KV eviction. Attribute latency spikes to specific server mechanisms only when verified by server-side spans (`queue_duration_ms`, `prefill_duration_ms`) or container lifecycle events.
4. **Define a Fair Prefill/Decode (P/D) Comparison:**  
   Compare collocated vs. disaggregated serving at **equal total accelerator budgets** (e.g. 2× R9700 GPUs: collocated DP=2 vs disaggregated 1P1D), matched precision (MXFP4), matched context window, and evaluate at **both equal offered load and equal achieved goodput**.
5. **Pre-Register Interactive SLOs:**  
   Pre-register latency SLO thresholds (e.g. TTFT $\le 1.0\text{s}$, worst ITL $\le 100\text{ms}$, task $\le 30\text{s}$) before inspecting benchmark data to prevent post-hoc threshold manipulation.

---

## 6. Execution Roadmap

```
Phase 1: Instrumentation & Load Generator Validation
  ├── Validate request_event_schema.py and client lag telemetry (p95 lag < 5ms)
  └── Run smoke tests across local vLLM endpoint (Qwen3.8-27B-Quark-AWQ-MXFP4)
Phase 2: Concurrency & Open-Loop Load Surface (Experiments 1, 2, 7)
  ├── Closed-loop concurrency sweeps (C=1, 5, 20)
  ├── Open-loop Poisson sweeps (λ = 1, 3, 6, 12 req/s)
  └── Map the goodput knee vs. raw throughput cliff
Phase 3: Timed Prefill/Decode Interference (Experiment 6)
  ├── Steady decode baseline (4 streams × 128 tokens)
  ├── Inject 1K, 4K, and 8K prefill bursts
  └── Measure peak active-decode stall (ms) and evaluate disaggregation
Phase 4: Multi-Turn Agent Task Replay & Dual Resampling (Experiment 5)
  ├── Replay multi-turn agent chains (M=30 chains, N=10 calls)
  ├── Compute empirical task duration vs. Baseline 1 (i.i.d.) vs. Baseline 2 (block-preserving)
  └── Contrast with theoretical exceedance reference (1 - (1-p)^n)
Phase 5: Cold Starts & Provider Consistency (Experiments 3, 4)
  ├── Probe cool-down idle intervals (0s, 30s, 60s, 120s, 300s)
  └── Interleave identical prompts between local AMD GPU and hosted endpoints
```
