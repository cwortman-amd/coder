---
type: Technical Report
title: The Token Freeze That Throughput Charts Miss
description: 'A coding agent can deliver an impressive number of tokens per second
  and still feel broken. The failure often happens between tokens: a response is streaming
  smoothly, a new long prompt arrives, and the existing...'
tags:
- technical-report
- qwen
- tail
- pdd
status: stable
---

# The Token Freeze That Throughput Charts Miss

### What Qwen3.8-27B MXFP4 on AMD GPUs taught us about prefill/decode disaggregation

A coding agent can deliver an impressive number of tokens per second and still feel broken. The failure often happens between tokens: a response is streaming smoothly, a new long prompt arrives, and the existing response briefly stops.

That is the question behind our evaluation of **prefill/decode disaggregation**, or P/D. Rather than ask only, “How fast is this GPU?”, we asked: **Can we protect a user’s active stream when other users submit long prompts—and can we do it without making first-token latency, capacity, or cost worse?**

The traffic-shape framing echoes [Bidit Pakrashi’s discussion of llm-d](https://www.linkedin.com/pulse/scaling-llm-inference-traffic-shape-llm-d-gpu-kernels-bidit-pakrashi-uj7mc/). The focus on tail latency is equally important: as [DigitalOcean’s p50-versus-p99 discussion](https://www.digitalocean.com/community/tutorials/p50-vs-p99-latency-llm-inference) notes, a typical request does not describe what happens to the unlucky request in a loaded agent workflow. The article below uses **our measurements**, not performance claims from either post.

## The shape of a request

Our anchor workload is Qwen3.8-27B Quark AWQ MXFP4 with an **8,192-token input and a 1,024-token output**. Those numbers describe two very different demands on a serving system.

Prefill reads the incoming prompt and builds the state needed for generation. Decode then emits output autoregressively, one step after another. In a collocated vLLM instance, both phases share a GPU and scheduler. A newly admitted prefill can interrupt the cadence of requests that are already streaming. Systems such as llm-d separate the phases so prefill and decode workers can be routed, scheduled, and scaled independently; they identify avoidance of long-prefill interference as a quality-of-service benefit. [llm-d](https://llm-d.ai/docs/dev/architecture/advanced/disaggregation)

Our single-Radeon AI PRO R9700 phase tests made that interference visible. The victim stream is 1,024 in / 256 out. The injected prompts are cold 8,192-token prefills. The server was running `--max-num-batched-tokens 4096`, so each 8K prompt arrives as two 4K chunks.

| Single-R9700 condition | Observed decode output rate | Peak gap between tokens |
|---|---:|---:|
| Isolated decode, no injected prefills | 34.07 tok/s | 32 ms |
| One cold 8K prefill every 5 s | 31.22 tok/s | 1,354 ms |
| One cold 8K prefill every 1 s | 22.84 tok/s | 1,360 ms |
| Sustained prefill bombardment | 11.42 tok/s | 1,366 ms |

These are **single-GPU interference tests**, not measured throughput for two collocated DP replicas. They show why isolation is worth testing; they cannot, by themselves, prove that a two-GPU P/D service beats cache-affine DP=2 on completed requests.

The user-visible issue is also more specific than “throughput fell.” On the light burst, output rate only slips from 34.07 to 31.22 tok/s, while one token gap reaches **1.35 seconds**. ITL p95 on that same run is still 30 ms. The median and even p95 describe a healthy stream. The pause is in the maximum.

Later chunk-size and Poisson-arrival experiments produced different tails under different settings. With a 2,048-token chunk, the recorded cold-burst stall fell from 1,108.5 ms to **609.7 ms**. With prefix caching on, the continuous-burst peak at that chunk size was **253.9 ms**. We therefore treat a token freeze as a *workload- and scheduler-dependent event*, not as a claim that every cold prompt imposes a 1.35-second stall.

## Empirical tail latency distributions on Radeon AI PRO R9700

To substantiate these dynamics with report-grade statistical resolution, we swept power-of-two concurrencies ($C = 1, 2, 4, 8, 16$) with 100 requests per cell ($N = 100$, 6,400 token samples per run) for Qwen3.8-27B MXFP4 on a dedicated R9700. Raw detailed traces are archived under `reports/results/qwen3.8-27b-mxfp4/latency/r9700/`, with normalized distributions and histograms in `reports/profiling/qwen3.8-27b-mxfp4-latency.json`.

| Workload | C | TTFT p50 | TTFT p90 | TTFT p95 | TTFT p99 | TPOT p50 | TPOT p95 | ITL p50 | ITL p95 | ITL p99 | Max ITL | Interactive SLO (TTFT ≤ 3s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **1k:64** | **C1** | 481 ms | 484 ms | 485 ms | 487 ms | 30.1 ms | 30.1 ms | 30.0 ms | 31.1 ms | 32.0 ms | 36.7 ms | 100% Pass (Sub-second) |
| **1k:64** | **C2** | 864 ms | 918 ms | 919 ms | 921 ms | 31.8 ms | 32.6 ms | 31.0 ms | 32.2 ms | 35.4 ms | 124 ms | 100% Pass (Sub-second) |
| **1k:64** | **C4** | 1,589 ms | 1,602 ms | 1,603 ms | 1,604 ms | 32.2 ms | 36.3 ms | 32.2 ms | 33.4 ms | 35.9 ms | 292 ms | 100% Pass (≤ 3.0s) |
| **1k:64** | **C8** | 5,159 ms | 5,183 ms | 5,184 ms | 5,185 ms | 32.9 ms | 36.3 ms | 32.2 ms | 33.5 ms | 75.0 ms | 293 ms | Queue Cut-Off (5.2s TTFT) |
| **1k:64** | **C16** | 12,383 ms | 12,432 ms | 12,440 ms | 12,449 ms | 33.0 ms | 36.4 ms | 32.2 ms | 33.6 ms | 76.7 ms | 292 ms | Queue Cut-Off (12.4s TTFT) |
| **8k:64** | **C1** | 3,152 ms | 3,165 ms | 3,169 ms | 3,171 ms | 30.5 ms | 30.5 ms | 30.5 ms | 31.7 ms | 32.6 ms | 39.5 ms | Marginal (3.15s TTFT) |
| **8k:64** | **C2** | 5,611 ms | 6,148 ms | 6,155 ms | 6,158 ms | 39.5 ms | 47.1 ms | 31.9 ms | 33.2 ms | 328 ms | 700 ms | Fail (5.6s TTFT, 700ms ITL) |
| **8k:64** | **C4** | 8,173 ms | 9,696 ms | 9,715 ms | 10,501 ms | 102 ms | 146 ms | 33.9 ms | 1,183 ms | 1,501 ms | 1,550 ms | Severe Contention (1.55s stall) |
| **8k:64** | **C8** | 20,585 ms | 21,731 ms | 21,745 ms | 24,944 ms | 122 ms | 146 ms | 33.9 ms | 1,267 ms | 1,509 ms | 1,550 ms | Severe Contention (20.6s TTFT) |
| **8k:64** | **C16** | 49,733 ms | 50,880 ms | 50,913 ms | 54,016 ms | 122 ms | 146 ms | 33.9 ms | 1,266 ms | 1,508 ms | 1,552 ms | Severe Contention (49.7s TTFT) |

Two publication-grade distribution plots visualize these empirical profiles:
* **TTFT Distribution Histograms** (`reports/figures/latency/01_ttft_tail_histogram.png`): Demonstrates the transition from prompt execution scaling ($C \le 4$) to discrete queueing delays beyond the `--max-num-seqs 4` scheduler threshold.
* **ITL Tail & CDF Histograms** (`reports/figures/latency/02_itl_tail_histogram.png`): Proves the bimodal distribution of decoding tokens, where 90%+ remain at the nominal 30–33 ms rate while concurrent 8k prefills inject recurring 1,180–1,552 ms freezes in the p95/p99 tail.
* **Master Evaluation Dashboard** (`reports/figures/latency/03_tail_latency_master_dashboard.png`): 4-panel synthesis of TTFT, ITL tail distributions, and SLO compliance states.

Beyond synthetic $S:O$ pairs, we evaluated real-world multi-turn Claude Code agent traces (`semianalysis_cc_traces_weka_062126`) across concurrencies $C = 1 \dots 32$ with a 65,536-token context window (detailed in [`docs/AGENTX-TAIL.md`](AGENTX-TAIL.md)). That sweep revealed an insidious **concurrency cliff**: at $C=8$, aggregate throughput reaches its nominal peak (17.2 tok/s) while p95 TTFT slips to **51.8 seconds** and 363 token pauses exceed 1.0 second. At $C \ge 16$, KV cache saturation (>80%) triggers a catastrophic eviction cascade: prefix hit rate collapses from 93% to **0%**, scheduler queues balloon to 13.7 waiting requests, TTFT p50 explodes to **5.5 minutes**, and output throughput collapses by 59%.

**Suggested presentation visual:** Show a timestamped streaming-token trace, with cold-prefill arrivals marked above it. Put a raw tok/s figure beside the trace, not in place of it. The audience should be able to see why an acceptable aggregate rate can coexist with an unacceptable pause.

## First, improve the collocated baseline

Disaggregation should compete against a well-configured single engine—not against avoidable prompt recomputation or a poor scheduler setting.

On our R9700 configuration, enabling prefix caching changed the first-token experience for repeated 8K prompts:

| Reusable prefix | New suffix | Measured TTFT | Speedup over cold |
|---:|---:|---:|---:|
| 0% | 8K | 2.777 s | 1.00× |
| 50% | 4K | 1.758 s | 1.58× |
| 75% | 2K | 0.914 s | 3.04× |
| 100% | 0K | 0.301 s | 9.24× |

For a multi-turn coding agent, this is consequential: a stable system prompt, tool definitions, and unchanged earlier context need not be digested from scratch every turn. Prefix reuse also changes the economics of P/D. If most requests are warm, there is less prefill work to isolate, and a **cache-affine pair of full serving replicas** becomes a stronger alternative.

We also swept the collocated `--max-num-batched-tokens` setting. In that experiment, a 2,048-token setting retained **2,688.7 prompt tok/s**, versus **2,881.8** at 4,096, while the burst-decode rate in that sweep stayed at **33.22 tok/s**. Smaller was not uniformly better: at 512 tokens, measured prefill rate fell to **1,923.0 tok/s** and burst decode fell to **3.45 tok/s**. The lesson is not “2K is universally optimal”; it is **tune the collocated baseline before crediting P/D with a gain that scheduler and cache changes could already provide**.

**Suggested presentation visual:** Place cold, 75%-warm, and fully warm TTFT bars next to a chunk-size trade-off plot. This explains why traffic shape—including prefix-hit distribution—is as important as nominal requests per second. llm-d’s routing design similarly treats cache state and predicted latency as serving inputs, rather than relying on generic round-robin placement. [llm-d](https://llm-d.ai/docs/architecture)

## Moving KV across PCIe

P/D creates a new requirement: after a prefill GPU computes the prompt state, a decode GPU must receive usable state before producing output. Our current transport investigation uses **two AMD Instinct MI350P PCIe cards in one node**. It is a separate hardware campaign from the single-R9700 phase tests; its timings must not be presented as R9700 transport results.

These MI350Ps sit behind separate host bridges on different CPU sockets of a dual-socket EPYC 9015. Both slots negotiated PCIe Gen5 x16 (32.0 GT/s), but the route between them traverses the inter-socket topology—not a shared PCIe switch. `rocm-smi` reports the link as PCIE, 3 hops, weight 72. That distinction showed up in both directional and concurrent-transfer tests.

The matched **256 MiB transport benchmark**, GPU 0 to GPU 1, found:

| Transport, forward direction | Payload shape | p50 | p95 | p99 |
|---|---|---:|---:|---:|
| Direct two-process HIP IPC | One contiguous buffer | 5.72 ms | 56.8 ms | 58.6 ms |
| UCX `rocm_ipc` | One descriptor | 9.70 ms | 110 ms | 161 ms |
| UCX `rocm_ipc` | 256 descriptors | 59.0 ms | 124 ms | 216 ms |

The one-descriptor median difference is **3.98 ms**. Yet the more important warning is the **256-descriptor result**: the selected transport remained `rocm_ipc`, but its median rose to 59.0 ms and throughput at that median fell to **4.55 GB/s**. The loss was associated with descriptor fragmentation, not a switch back to the previously observed host-fragment path.

Cold setup also matters. The first UCX transfer took **114 ms**, compared with 9.70 ms steady-state p50. Keeping endpoints, registrations, and IPC mappings alive is therefore part of a serving design—not merely a microbenchmark trick. Both HIP IPC and UCX showed substantial cross-socket tails, so a connector that improves p50 alone cannot promise jitter-free P/D. Direction matters too: the same native copy GPU 1 to GPU 0, which is the serving direction, is **12.7 ms** at p50 (**21.2 GB/s**), with p95 **175 ms** and p99 **222 ms**.

This motivated a **native HIP-IPC NIXL backend** rather than an entirely separate vLLM KV connector. The design retains NIXL’s transfer lifecycle while reusing registrations and mappings, coalescing safe adjacent regions, posting peer copies on persistent streams, and bounding in-flight cross-socket traffic. NIXL’s backend interface provides the registration, prepared-transfer, posting, completion, and cleanup hooks needed for such an implementation. [github](https://github.com/ai-dynamo/nixl/blob/main/docs/BackendGuide.md)

On the same 256 MiB buffer, that backend keeps the 256-descriptor case at **5.56 ms** p50 by collapsing contiguous ranges into one prepared copy. A Qwen-shaped layout does not collapse that far. Sixteen attention regions and 272 descriptors, 884 MiB, coalesce to 16 copies at **20.5 ms** p50. The same bytes split into 272 independent allocations take **77.9 ms**.

That work has now reached a functional milestone: **a real Qwen3.8 P→D handoff completes**. On the short prompt, 208 descriptors coalesced into 64 copies. On a cold 8K prompt, 240 descriptors coalesced into 96 copies and moved **699,203,584 bytes**. The decoder queried 8,257 external prefix-cache tokens and hit 8,256: one prompt token was not imported. The remaining gates are decisive: compare generated token IDs with a single-GPU control, prove the decoder did not silently recompute missing state, and treat the waterfall below as transport-plus-admission time rather than a qualified user latency. An external-cache-hit log line is useful evidence, but it is not a completed correctness qualification. Greedy completion IDs still diverge from the single-GPU control, at completion index 2 on an 81-token prompt and index 5 on an 8,246-token prompt.

The one-token 8K waterfall, eight unique cold prompts, serial, GPU 1 prefill to GPU 0 decode:

| Interval | p50 | p95 | p99 |
|---|---:|---:|---:|
| Prefill HTTP | 854 ms | 860 ms | 862 ms |
| NIXL post | 24.8 ms | 44.0 ms | 44.5 ms |
| NIXL transfer, post through completion | 100 ms | 235 ms | 265 ms |
| Client arrival through the one-token response | 1,189 ms | 1,443 ms | 1,447 ms |

Failed transfers were zero. One earlier sample of the same payload completed the copy in 21.5 ms, so 100 ms is the median of this run, not the best observed copy. Four of these prompts issued together moved client p50 from 1,189 ms to **3,233 ms**, almost entirely in prefill (p50 **3,012 ms**). The four transfers summed to 299 ms of NIXL time.

Four serial streaming prompts at 8,192 in / 1,024 out on that same pair produced **72.49 tok/s** aggregate, ITL **10.58 ms** p50 and **10.64 ms** p95, and TTFT **1.47 s** p50. TTFT p95 on those four requests is **4.48 s**, because the first request took 5.0 s. Per-request maximum token gaps were 309 ms, 1,315 ms, 11 ms, and 918 ms. Median decode on this path is smooth. The tail is not gone.

**Suggested presentation visual:** Use a handoff waterfall with *prefill completion → connector preparation → PCIe transfer → decoder state usable → decode admission → first token*, filled with the table above. Label the HIP/UCX rows **transport measurements** and the waterfall **one-token serving handoff**. Leave the equal-card DP=2 comparison blank. It has not been run.

## What would prove the benefit?

The strongest claim is not “P/D makes each token generate faster.” A dedicated decode GPU still has to execute the model. On the R9700, isolated single-stream decode is already the ceiling, about **34 tok/s** and **29.3 ms** per token. The claim is that **phase isolation preserves its decode cadence under cold-prompt load**, potentially increasing the number of requests completed within a declared first-token and streaming-latency objective.

For two cards, the fair comparison is:

| Architecture | What both cards do | Possible advantage |
|---|---|---|
| Cache-affine DP=2 | Each hosts a full model and handles both phases for its assigned sessions | Both cards can decode; warm sessions retain local prefixes |
| 1P1D | One card prefills; one card decodes after KV transfer | Cold prefills do not execute on the decode GPU |

P/D is **not guaranteed to win raw output tok/s**: DP=2 has two decode-capable GPUs. It can win useful capacity when prefill interference causes enough tail-latency failures, or even win raw completed throughput in a sufficiently severe regime—but those outcomes require the **same complete requests, arrival trace, cache states, GPU count, and stable queues** on both sides. The earlier single-R9700 victim-stream bombardment is a motivation for that test, not its substitute. The single-GPU emulator that projects a J3 win for 1P1D is the same kind of evidence: a hypothesis drawn from measured service times, drawn dashed until two cards run the trace.

Our publication test will report, at each offered request rate:

- **Raw capacity:** completed requests/s and generated output tok/s.
- **User experience:** TTFT, per-request TPOT, individual-token ITL tails, and worst gaps.
- **Qualified goodput:** completed requests/s meeting an explicitly labeled, *project-defined* SLO.
- **Operational behavior:** prefill/decode queue growth, KV-transfer p95/p99, failures, cache reuse, and power.
- **Economics:** GPU-hours or energy per SLO-qualified completed request.

We will plot **raw throughput and qualified goodput separately**. We will also sweep multiple TTFT/TPOT thresholds rather than attributing an invented threshold—such as 3.5 seconds TTFT and 25 ms TPOT—to a coding-agent provider. A strict SLO can make an architecture’s qualified goodput approach zero while it still emits raw tokens; the audience needs to see both quantities and the pass/fail definition. Work on disaggregated serving likewise evaluates capacity against *joint* TTFT and output-latency constraints, not peak token rate in isolation. [usenix](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)

**Suggested presentation visual:** Lead with a paired **DP=2 versus 1P1D goodput-versus-offered-load curve**. Use solid marks only for measured two-card serving results; show any current emulator projections as dashed marks. Under it, show an ITL-tail plot explaining *why* either architecture passes or fails. The visual should be allowed to show DP=2 winning where prefixes are warm or traffic is light—that makes a measured P/D win under cold bursts more credible.

## Measured AgentX load on one MI350P

The fixed-length interference tests above explain the mechanism. The 30 September AgentX replay measures the same collocated GPU under recorded agent dependencies. Six 15-minute cells, C=1 through C=128, all completed with zero request errors.

Output throughput rises from 11.59 tok/s at C=1 to **46.80 tok/s at C=32**, then falls to **4.01 tok/s at C=64**. Gaps above one second rise from none at C=1 to 4.59 per 1,000 intervals at C=16 and 23.39 per 1,000 at C=32. At C=64, 949 of every 1,000 observed intervals exceed one second, median TTFT is 42 s, and no AgentX conversation finishes. C=128 adds queue depth—48 waiting requests on average—without restoring throughput or completing a conversation.

This is the collocated baseline a later DP=2 or 1P1D comparison has to beat at the same offered conversations. It is not yet that comparison: one GPU, seven traces that fit in 65,536 tokens, and too few finished sessions for a stable task-completion tail. The table, sample limits, and reproduction commands are in [MI350P-AGENTX.md](MI350P-AGENTX.md).

## The architectural takeaway

Multiple PCIe GPUs in one node give us a practical way to **test independent prefill and decode pools without requiring an external network**. Our MI350P results now show a functional HIP-IPC/NIXL handoff path and demonstrate why payload layout, cross-socket topology, and tail behavior matter as much as nominal PCIe link speed. The one-token 8K waterfall is measured: transfer p99 is 265 ms and client p99 is 1.45 s. They do **not yet show a sustained improvement in TTFT tail latency**. The four-request streaming cell still has a 4.5 s TTFT p95 and a 1.3 s token gap, completion IDs do not match the single-GPU control, and the equal-card-count DP comparison has not been run.

The user-experience hypothesis is precise:

> **When cold, long prompts arrive during active generation, a dedicated decode GPU should avoid the prefill-induced streaming pauses seen in collocated service—provided KV handoff, decoder admission, and queueing do not create a new tail-latency problem.**

That is the full P/D story for our Qwen3.8-27B MXFP4 evaluation: **measured interference, validated single-GPU mitigations, a working cross-socket PCIe handoff, and a clearly defined test for whether isolation delivers more useful service capacity.** The next result should be a measured TTFT/ITL/goodput comparison—not another theoretical PCIe bandwidth estimate.
