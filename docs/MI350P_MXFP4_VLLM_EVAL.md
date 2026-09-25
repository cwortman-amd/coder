# MI350P vLLM Quark MXFP4 evaluation (reproducible)

**Date:** 25 September 2026  
**SKU:** 1× AMD Instinct MI350P PCIe (`gfx950`, PCI `0x75a8`, 144 GB HBM3E, 600 W cap), GPU 0 only  
**Target:** [amd/Qwen3.8-27B-Quark-AWQ-MXFP4](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4)  
**Draft (optional):** [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (`dflash_config.block_size=8`)  
**Engine:** vLLM **0.30.0** ROCm (`vllm/vllm-openai-rocm:latest`)  
**Kernel confirmation:** `AiterMxfp4LinearKernel`; decoder attention auto-selects **ROCM_ATTN**

This is the measured MXFP4 serving campaign. Do not mix it with the earlier FP8/SGLang numbers in `MI350P_EVAL.md` (24 Sep) or with public NVIDIA 454 tok/s / Helix 167.8 tok/s C1 claims.

Lab JSON and logs: `_results/priority_eval/`, `_results/profiling/`, `_results/quality/`.

---

## Headline

The HF Quark MXFP4 **non-speculative** configuration is the frozen general-purpose production baseline. **Leave `VLLM_ROCM_USE_AITER` unset and keep stock gfx950 `M_LEQ_8` (including `NUM_KSPLIT=4`).** Isolated eager M=1 GEMM wins (global ASM; `NUM_KSPLIT=1`) **regressed or failed to improve** unprofiled graph-enabled serving. Promotion gate for every later kernel change: unprofiled C1/C8 vs **79.35 ± 0.57 / 553.58 ± 2.97**, with graph state recorded.

DFLASH-3 is the strongest optional C1 throughput cell (**113.75 tok/s**) but **fails both promotion gates for a general interactive profile**: only **101/116 (87.1%)** exact greedy sequences, TTFT p95 **58.6 → 127.5 ms**, and token-arrival p95 **12.07 → 18.84 ms** in a matched 24-request C1 stream run. It remains an opt-in long-output throughput profile, not a default product improvement.

| Area | Validated conclusion |
|---|---|
| Default production profile | **Frozen** published HF vLLM Quark AWQ MXFP4; AITER unset; stock `M_LEQ_8` |
| Kernel-change promotion gate | Unprofiled graph-enabled C1/C8 vs **79.35 ± 0.57 / 553.58 ± 2.97**; not eager GEMM |
| Global `VLLM_ROCM_USE_AITER=1` | Isolated ASM ~2.2× faster; serving **C1 71.40 (−10%)**, **C8 519 (−6%)** — do not enable |
| `NUM_KSPLIT=1` `M_LEQ_8` overlay | Isolated GEMM ~2×; serving **C1 75.93 (−4.5%)**, C8 noise — do not promote |
| C1 vs Babel Read | **79.53 tok/s ≈ 40%** of 197 tok/s conditional roof (1.53 TB/s if one full 17.91 GiB stream/step). Unused 60% is **not** assigned to a single operator |
| DFlash2-7 throughput crossover | Between **C5 and C6** |
| DFlash2-7 throughput benefit | C1 through C5; strongest at C1 |
| DFlash2-7 throughput regression | C6 and above; unsuitable for C8 saturated serving |
| DFLASH-3 matched stream (C1, n=24) | Output **82.56 → 97.41 tok/s (+18%)**; TTFT p95 **58.6 → 127.5 ms**; ITL p95 **12.07 → 18.84 ms** |
| DFLASH-3 quality | **101/116 (87.1%)** exact; same 15 mismatch IDs as DFLASH-7; JSON/tool **15/15** |
| Deterministic control stability | Bit-stable with identical batch shape and process conditions |
| MTP | Blocked by Quark-packed MTP tensor / vLLM loader mismatch |
| DFlash2-8 failure | Invalid configuration / GPU fault, **not** a graph-capture defect |
| Quantization quality | MXFP4 production-sampling **30/32 (93.75%)** on a 32-item suite; **MXFP4 vs FP8/BF16 still unmeasured** |
| C1 limiter | Babel unused 60% still **not** assigned without unprofiled serving HBM counters. Profiler-on captured CUDA: `_gemm_afp4wfp4` **~40%** at **22.4 µs**; quant 9%; split-K reduce 7%; GDN packed 4%. Live EngineCore rocprof attach **blocked**. Not PPT, thermal, peak MXFP4, or SKU HBM (Babel Read **3793 GB/s**) |

---

## 1. Production profiles

### Default (general purpose)

Non-speculative HF recipe. **Do not** set `VLLM_ROCM_USE_AITER` or overlay AITER Triton `DEFAULT.json`. Use for general chat, low-TTFT interactive, structured output / tool calls, mixed traffic, C4+ and all C6–C32 throughput.

Validated 1K/1K `ignore_eos` same-process 3×: **79.35 ± 0.57 C1 tok/s**, **553.58 ± 2.97 C8 aggregate tok/s**. Published HF-recipe capture: 79.15 C1 / 544.78 C8.

### Conditional DFlash (optional)

Same MXFP4 target + DFlash2, **`num_speculative_tokens=3`** (8 is invalid). Use only after request classification or explicit product selection: C1–C3, long expected outputs, throughput-oriented generation, **non-critical** text where altered token paths are acceptable.

Do **not** use by default for tool calls / strict reproducibility, first-token SLOs, C6+ serving, C8 saturation, or bit-identical benchmarks. JSON/tool 15/15 is encouraging but too small to upgrade the general policy.

```text
C1–C3:
  Route to DFLASH-3 only when longer output throughput is more important
  than TTFT and p95 token-arrival latency.

C4–C5:
  Default to non-speculative control unless customer workload testing
  proves a net user-experience win.

C6+:
  Route to non-speculative Quark AWQ MXFP4 control.

C8:
  Never compare DFLASH-3 as a saturated-throughput profile.
```

Avoid a hardcoded static threshold if arrivals vary. Route on current active decode sequences, queue depth, output-length expectation, and latency class.

### Three objectives (do not collapse into one tok/s)

| Objective | Best candidate | Evidence |
|---|---|---|
| Maximum C1 long-output throughput | DFLASH-3 (conditional) | 113.75 tok/s at 1K/1K; fails equality and interactive-latency gates |
| Best initial response time | Non-speculative control | TTFT p50/p95/p99 |
| Best smooth interactive streaming | Non-speculative unless DFlash p95 ITL meets SLO | Token-arrival p50/p95/p99 |
| Maximum aggregate capacity | Non-speculative control | C6–C32 aggregate tok/s and tail latency |

External-facing sentence:

> On the evaluated MI350P Quark MXFP4 stack, DFLASH-3 increased C1 output throughput but matched only 101/116 sequential-control greedy outputs and, in a matched 24-request stream run, increased TTFT p95 from 58.6 to 127.5 ms and token-arrival p95 from 12.07 to 18.84 ms. It is therefore an opt-in low-concurrency, long-output profile—not a default interactive-chat improvement.

---

## 2. Throughput, graphs, saturation

### Control repeatability (1K/1K `ignore_eos`)

| | r1 | r2 | r3 | mean | pstdev | CV |
|---|---:|---:|---:|---:|---:|---:|
| C1 tok/s | 78.55 | 79.72 | 79.79 | **79.35** | 0.57 | 0.72% |
| C8 tok/s | 549.39 | 555.40 | 555.95 | **553.58** | 2.97 | 0.54% |

### Graph / compile

Default **O2 = `FULL_AND_PIECEWISE`**. O3 is the same. O1 (piecewise only) loses ~5% at C8. `--enforce-eager` is a negative control (**13.88 / 108.52** tok/s), about **0.18×**. Never ship eager as a performance config.

### Saturation (HF recipe, auto `max-num-seqs`, batched tokens 8192)

C8 is a reporting point, not the peak: C16 **904**, C32 **1504**, C64 **2017** tok/s (still rising). `--max-num-seqs 32` caps C64 at ~1477. `--max-num-batched-tokens 16384` did not beat 8192 on 1K decode. No C64 power/energy capture — do not invent J/tok from C1/C8.

---

## 3. DFlash2 crossover, latency, acceptance, DFLASH-8

Matched 1K/1K (one active batch per concurrency):

| Concurrency | DFlash2-7 vs MXFP4 control | Recommendation |
|---:|---|---|
| C1 | **+30.3%** (78.80 → 102.68) | Eligible for a controlled low-QPS throughput profile |
| C2 | **+17.4%** | Eligible, subject to TTFT/p95-ITL SLO |
| C3 | **+10.3%** | Potentially useful if output-length weighted |
| C4–C5 | **+4.7–5.9%** | Marginal; choose on latency/quality, not headline tok/s |
| C6 | **−21.3%** | Do not use DFlash2-7 |
| C8 | **−24.2%** | Use non-speculative MXFP4 control |

Earlier DFLASH-4 C1 was **81.48** (not competitive). DFLASH-7 C8 later matched **422** tok/s vs control **557**.

This matches dense-model expectation: C1–C3 is bandwidth/launch limited so skipping target verifies helps; C4–C5 reuses quantized weights over a larger batch; C6+ drafter/verify/state/scheduling exceeds the target passes saved.

### Latency (1K in / 256 out, client token-arrival)

DFlash2-7 worsens C1 TTFT p95 (**76 → 244 ms**) and p95 ITL (**12.05 → 20.57 ms**). The completed matched n=24 DFLASH-3 run confirms the same trade-off with tighter estimates: output throughput **82.56 → 97.41 tok/s (+18%)**, TTFT p95 **58.6 → 127.5 ms (2.18×)**, token ITL p95 **12.07 → 18.84 ms (+56%)**, and chunk ITL p95 **12.07 → 18.89 ms (+57%)**. Mean ITL improves (**11.95 → 9.80 ms**) only because accepted tokens arrive in bursts. Call DFlash a **long-generation throughput profile**, not an “interactive profile.” Artifacts: `_results/priority_eval/dflash3_validation/`.

### Depth 1–5 (25 Sep; `scripts/dflash_depth_sweep.sh`)

| Depth | C1 | vs ctrl | C5 | C6 | C8 | C1 TTFT p95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 85.96 | +9.1% | 285 (−10%) | 337 | 420 | 137 |
| 2 | 96.71 | +22.7% | 343 | 399 (−6.5%) | 496 | 136 |
| **3** | **113.75** | **+44.3%** | **354** | 417 (−2.4%) | 511 (−8.3%) | **136** |
| 4 | 84.29 | +7.0% | 347 | 330 | 429 | 145 |
| 5 | 112.58 | +42.9% | 345 | 406 | 516 (−7.3%) | **135** |
| 7 (prior) | 102.68 | +30.3% | 337 | 336 (−21%) | 422 (−24%) | 244 |

Crossover stays **C5/C6**. Optional DFlash profile, if used: **depth 3**, not 7. Depth 1 loses at C5. Depth 4 C1 is a one-run weak cell (matches earlier 81.48); repeat before drawing a geometry lesson. DFLASH-3 greedy equality is now **101/116 (87.1%)**, so it is not bit-identical. JSON: `_results/priority_eval/dflash_depth/`, `_results/quality/compare_baseline_vs_dflash3.json`.

### Acceptance

DFLASH-7: **~17.8% C1 / ~17.4% C8**; positions **2–6 zero**. Shorter depths accept on **all requested positions** in the mixed C1–C8 window (depth 3: pos 0–2). C8 loss remains overhead, not acceptance collapse.

### DFLASH-8

Failure **persists with `--enforce-eager`**: not graph capture, replay, or graph-pool allocation. `dflash_config.block_size=8` supports **seven** draft positions; `num_speculative_tokens=8` is invalid (nine-position query). Result: GPU fault / EngineCore SIGABRT. Safe contract: **`num_speculative_tokens ≤ 7`**. Treat DFLASH-8 as an invalid test point. Worth filing upstream: host should `ValueError` before kernel launch instead of SIGABRT.

### Native MTP

**Not evaluated.** `method=mtp` → `Qwen3_5MTP.load_weights` shape assert (Quark-packed MTP vs vLLM 0.30 column-parallel). Do not report MTP as slow or “incompatible with MI350P.” File as a loader defect.

---

## 4. Quality (greedy vs same MXFP4 target)

Settings: `temperature=0`, `ignore_eos=false`, 116 text-only prompts. Details: `_results/quality/QUALITY.md`.

| Comparison | Exact greedy token IDs | Conclusion |
|---|---:|---|
| MXFP4 control, restart vs same-process repeat | 116/116, 100% | Bit-stable when process state and batch shape match |
| JSON/tool subset, control vs DFlash | 15/15 | No observed difference on this structured subset |
| Control C1 vs control C8, subset | 21/24, 87.5% | Batch shape alone can alter greedy output |
| Control vs DFLASH-3 | **101/116, 87.1%** | Same 15 mismatch IDs as DFLASH-7; not bit-identical |
| Control vs DFLASH-7 | 101/116, 87.1% | DFlash is not bit-identical to sequential C1 control |
| Control vs DFLASH-4 | 102/116, 87.9% | Same caveat at smaller draft depth |
| MXFP4 production sampling (32-item, T=1.0/top_p=0.95/top_k=20) | **30/32 (93.75%)** scored | Compact math 6/8; JSON/code/multilingual/reasoning 8/8 or 4/4. Not vs FP8/BF16 |

DFlash mismatch rate \(1-101/116=12.9\%\) is close to non-spec C1-vs-C8 \(1-21/24=12.5\%\). That does **not** prove DFlash has no independent effect (different subsets, small *n*). It supports that a large share of divergence can come from GEMM shapes, FP reduction order, graphs, verify batch shape, MXFP4 numerics, and greedy near-ties. Remaining diffs are **alternate generated strings**, not tokenizer-only; semantic similarity is not functional equivalence.

**Not proven:** DFlash lossless in the bit-exact greedy sense; MXFP4 vs FP8/BF16 on the same suite (FP8 is cached; exclusive GPU required); DFlash production-sampling task scores; long-context or multimodal quality; native MTP quality.

Report wording:

> On a 116-prompt text-only deterministic corpus, DFLASH-3 and DFLASH-7 each matched the sequential C1 control on 101/116 prompts and had the same 15 mismatch IDs (though DFLASH-3 and DFLASH-7 differed from each other on 8 prompts). The non-speculative control itself matched between C1 and C8 on 21/24 prompts. These results do not establish practical bit-exact losslessness for DFlash under the tested vLLM 0.30 ROCm/MXFP4 implementation. Structured JSON/tool outputs matched on the 15-prompt subset.

---

## 5. Roofline and power (P1–P4 telemetry + RVS BABEL)

C1 MXFP4 decode is **not** compute-, thermal-, or PPT-limited, and the SKU **can** stream HBM: RVS BABEL Read is **3793 GB/s (92.6% of 4096 GB/s spec)**. Unprofiled control C1 **79.53 tok/s** is **40% of the Babel Read roof** (~197 tok/s) under a *conditional* 17.91 GiB full-weight-stream-per-step model (~1.53 TB/s implied). That roof is useful; it does **not** assign the unused 60% to pack/unpack, gate GEMM, or any other single operator.

**Promotion gate:** unprofiled, graph-enabled end-to-end C1/C8 (and latency when relevant). Eager isolated GEMM is diagnostic only. Isolated ASM and `NUM_KSPLIT=1` both improved M=1 GEMM wall time and **did not** improve serving. Lab: `_results/profiling/C1_BABEL_ROOFLINE.md`, `_results/profiling/mxfp4_gemm/GEMM.md`.

### Device facts

| Property | Value | Interpretation |
|---|---:|---|
| HBM3E peak | **4,096 GB/s** (`amd-smi`) | Hardware specification |
| Babel Read (measured) | **3,793 GB/s** median | Practical streaming roof; 2 GiB arrays, GPU 0 |
| HBM clock | 2,000 MHz | Full memory-clock state |
| Board power cap | 600 W | No apparent PPT constraint |
| Observed power | **~350–410 W** | 58–68% of cap |
| GFX clock | **~2.19 GHz** | Sustained |
| GFX activity | 100% C1 control | Busy, not proof of matrix/HBM pipeline saturation |
| Compute units | 128 | MI350P CDNA 4 |
| MXFP4 peak | ~4.6 PFLOPS | Far above C1 dense decode FLOP rate |
| Loaded MXFP4 target | **17.91 GiB** | Conservative full-weight-stream bound |
| On-disk target | 18.44 GiB | Do not use for runtime roofline |

### C1 memory roof (ideal, not expected production)

\(W = 17.91\,\mathrm{GiB} \times 1.073741824 = 19.231\,\mathrm{GB}\).  
\(T_\mathrm{C1,HBM-ideal} = 4096 / 19.231 \approx \mathbf{213}\) tok/s.

Conditional **practical** roof using Babel Read: \(3793 / 19.231 \approx \mathbf{197}\) tok/s. Unprofiled C1 **79.53 / 197 ≈ 40%**.

This assumes all of that streaming bandwidth goes to target weights and ignores scales, activations, residual/norm, GDN/attention state, KV, LM-head/sampling, kernel boundaries, controller inefficiency, and scheduler overhead.

### Compute roof (order of magnitude)

\(2 \times 27\times10^9 \approx 54\times10^9\) FLOP/token → \(4.6\times10^{15}/54\times10^9 \approx 85{,}000\) tok/s. Compute is not the first-order C1 limiter. Matrix-unit idle time is **not** proven without counters.

### Measured lanes (1K/1K; power = GFX≥95% mean from `amd-smi monitor`)

| Lane | Output rate | Board power | Energy/output token | Implied target-weight traffic | HBM peak fraction |
|---|---:|---:|---:|---:|---:|
| Control C1 (P1) | 79.73 tok/s | 400 W | **5.0 J/token** | ~1.53 TB/s | **37%** of spec |
| DFlash-7 C1 (P2) | 102.47 tok/s | 349 W | **3.4 J/token** (−32%) | Not one full target stream/token | N/A |
| Control C8 (P3) | 558.61 agg. tok/s | 410 W | **0.73 J/token** | ~1.34 TB/s (one pass / batch step) | **33%** |
| DFlash-7 C8 (P4) | 416.71 agg. tok/s | 380 W | 0.91 J/token | Extra draft/verify traffic | N/A |
| Control C64 | 2017 agg. tok/s | **not sampled** | **do not estimate** | Weight reuse + KV/packing | N/A |

`amd-smi MEM%` ~31% at C1 would be ~1.27 TB/s **if** it were linear GB/s. It is a **busy indicator**, not a calibrated HBM counter. Same ballpark as 1.53 TB/s, not proof of exact bandwidth.

C8 does **not** stream eight independent 17.91 GiB copies. Weights are reused across the active batch. Ratios (label them separately):

| Ratio | Value | Meaning |
|---|---:|---|
| \(558.61 / (8 \times 79.73)\) | **87.5%** | C8 vs linear scaling of **measured** C1 |
| \(558.61 / (8 \times 212.98)\) | **32.8%** | C8 vs naïve 8× ideal C1 HBM roof |
| \(\sim 1.34 / 4.096\) | **32.7%** | Inferred batch-step weight traffic vs HBM spec |

C64 \(2017/(64\times 79.73)=39.5\%\) is **not** “39.5% GPU efficiency”: arithmetic intensity, KV/GDN state, packing, and kernel shapes have changed. No clean HBM plateau is demonstrated.

### Isolated MXFP4 GEMM vs serving (25 Sep)

Eager M=1 GEMM (`scripts/bench_mxfp4_gemm.sh`, `bench_mxfp4_tile.py`) and graph-enabled `vllm serve` **respond differently** to the same knobs. Triton `_gemm_afp4wfp4` GPU ~**22 µs**; isolated wall ~**101 µs** (quant + split-K reduce + launch). That wall/kernel gap is why eager operator time cannot replace captured serving.

| Configuration | Isolated M=1 | Serving C1 | Serving C8 | Decision |
|---|---|---:|---:|---|
| HF control; `VLLM_ROCM_USE_AITER` unset; stock `M_LEQ_8` | Production Triton path | **79.53** (best unprofiled); campaign **79.35 ± 0.57** | **553.58 ± 2.97** | **Keep / freeze** |
| `VLLM_ROCM_USE_AITER=1` | ASM wall ~46 µs vs Triton ~101 µs | **71.40 (−10%)** | **519 (−6%)** | Do not enable globally (does **not** prove ASM kernels are inherently worse) |
| `NUM_KSPLIT=1` overlay on `M_LEQ_8` | Eager gate ~0.032 vs ~0.066 ms default (bit-exact) | **75.93 (−4.5%)** | **555.86** (noise) | Do not promote; default split-K is **not** a proven tile defect |

Do not optimize only `mlp.gate_proj`. A gate-only isolated win can lose end-to-end when a global setting changes dispatch. Record graph/compile state on every serving run.

### vLLM-native captured mini-decode (25 Sep)

Standalone `torch.cuda.CUDAGraph` around AITER Triton **faulted**. Exclusive in-process vLLM O2 `FULL_AND_PIECEWISE` (`scripts/bench_vllm_captured_decode.sh --exclusive`) completed with torch profiler on: M=1 **60.92 tok/s** (profiler perturbation, not 79.35). CUDA ranking from `profiler_out_0.txt`: `_gemm_afp4wfp4` **39.7%** at **22.4 µs** mean (same as isolated kernel), quant **8.9%**, split-K reduce **6.9%**, `wvSplitK` **5.9%**, GDN packed **4.1%**. Down-proj and attention projections did not correlate. Report: `_results/profiling/vllm_captured/SUMMARY.md`. Live EngineCore `rocprofv3 --pid` still has no `rocp-bg-attach` thread (`_results/profiling/enginecore_attach/ATTACH.md`).

This does **not** assign the unused Babel 60% to GEMM: it is a profiler-on CUDA-time share, not unprofiled serving HBM. Promotion gate remains **79.35 ± 0.57 / 553.58 ± 2.97**.

### Isolated GDN decode (25 Sep)

`scripts/bench_gdn_decode.py` / `_results/profiling/gdn_decode/GDN.md`. 48 linear-attention layers. Packed FLA: conv **24 µs** + recurrent **29 µs** wall, **1.16 + 1.59 µs** GPU. ×48 = **2.56 ms eager / 0.13 ms GPU** per token. Missing `fused_gdn_decode_post_conv_mtp` and AITER KDA (Qwen3-Next only) do **not** by themselves explain C1 vs Babel; they also do not get a share of the unused 60% without a serving timeline.

Report-safe C1 wording:

> The MI350P sustained 3,793 GB/s median on RVS BABEL Read (2 GiB arrays; 93% of the 4,096 GB/s spec). Unprofiled Qwen3.8-27B C1 control was 79.53 tok/s (campaign mean 79.35 ± 0.57), equivalent to about 1.53 TB/s if each decode step reads the full 17.91 GiB loaded weight set once, or about 40% of that Babel Read roof (197 tok/s) and 37% of spec (213 tok/s). Isolated M=1 MXFP4 GEMM and end-to-end serving responded differently to ASM and split-K changes. A profiler-on vLLM captured decode ranked `_gemm_afp4wfp4` at about 40% of CUDA time (22 µs mean); that is not an unprofiled serving-HBM assignment of the unused Babel 60%. Live EngineCore rocprof attach remains blocked. Board power remained approximately 350–410 W under a 600 W cap. rocprofv3 wrapping `vllm serve` produced parent HIP init only (C1 60–65 tok/s under wrap is profiler perturbation, not a campaign point).

Report-safe C8 wording:

> At C8, aggregate throughput reached 558.61 tok/s while the estimated batch-step weight stream remained approximately 1.34 TB/s. This is expected because eight active sequences reuse target weights in batched decode GEMMs. The result is 87.5% of linear scaling from measured C1 throughput, while still below a simplified HBM roof that excludes state, attention, scaling, packing, and scheduler overhead.

---

## 6. Reproducible setup

### Fingerprints (25 Sep 2026)

```text
vLLM:
  0.30.0

Container image digest:
  sha256:2e7da1ad1c66836802072588adea75f9f4991da5f9545b4318e91d422c22ce6a

GPU architecture:
  gfx950

Detected devices:
  2 × device ID 0x75a8   (tests: HIP_VISIBLE_DEVICES=0)

Checkpoint provenance:
  19.8 GiB monolithic artifact hashed
  39-shard staged checkpoint hashed
```

| Item | Value |
|---|---|
| Host kernel | `6.8.0-142-generic` |
| Driver (rocm-smi) | 6.19.14.31400000 |
| Host RAM | ~32 GiB — use shards, do not mmap the 19 GiB monolith plus GPU BAR |
| Monolith | `models/Qwen3.8-27B-Quark-AWQ-MXFP4/model.safetensors` 19 798 196 184 B, sha256 `be1d745bc7312fdf1486059ec57cdeb514cc4d1aa06528c6677a0ebc0a0e1272` |
| Shards | 39 files; index sha256 `d0f1b25f2efefad0852e2d100dd0ac9bc2fcd0d343acb73c2418552b40fed98f` |
| DFlash2 draft | `models/Qwen3.8-27B-DFlash2-sharded/` (12 shards, 3.58 GiB), `block_size=8` |

Full lists: `_results/priority_eval/baseline_fingerprint.txt`, `mxfp4_monolith.sha256`, `mxfp4_sharded.sha256`.

### Environment (inside the container)

```text
HIP_VISIBLE_DEVICES=0
PYTORCH_ROCM_ARCH=gfx950
GPU_ARCHS=gfx950
SAFETENSORS_FAST_GPU=1
HIP_FORCE_DEV_KERNARG=1
```

Do not export an empty `HSA_OVERRIDE_GFX_VERSION`. Bind **127.0.0.1** only. Sequential 27B servers only on this ~32 GiB host.

### Launch

```bash
./scripts/launch_vllm_mxfp4.sh
```

```text
vllm serve /models/Qwen3.8-27B-Quark-AWQ-MXFP4-sharded
  --served-model-name awq
  --trust-remote-code
  --tensor-parallel-size 1
  --max-model-len 16384
  --host 127.0.0.1 --port 8000
  --kv-cache-memory-bytes 103223724237
```

`--optimization-level` default is **2**. No `-O` alias. Prefix caching and chunked prefill (`max_num_batched_tokens=8192`) are already on.

Optional DFLASH-3 (never 8):

```bash
VLLM_SERVED_NAME=awq-mxfp4-dflash2 ./scripts/launch_vllm_mxfp4.sh \
  --speculative-config '{"method":"dflash","model":"/models/Qwen3.8-27B-DFlash2-sharded","num_speculative_tokens":3}'
```

### Throughput harness (speed lane)

```bash
python3 scripts/bench_openai_chat.py \
  --base-url http://127.0.0.1:8000/v1 --model awq \
  --input-len 1024 --output-len 1024 \
  --num-prompts 4 --concurrency 1 --timeout 900 \
  --out _results/priority_eval/run.json
```

Client: `ignore_eos=true`, `temperature=0`, `enable_thinking=false`. Warm a C8 shape before quoting C8. Streaming: `scripts/bench_openai_stream.py`.

### Quality harness

```bash
python3 scripts/quality_corpus.py --out _results/quality/corpus.jsonl
python3 scripts/eval_greedy_tokens.py \
  --corpus _results/quality/corpus.jsonl \
  --out _results/quality/baseline_greedy.json \
  --model awq --profile mxfp4_control
python3 scripts/compare_greedy_runs.py \
  --baseline _results/quality/baseline_greedy.json \
  --candidate _results/quality/dflash7_greedy.json \
  --out _results/quality/compare_baseline_vs_dflash7.json
```

Greedy: `temperature=0`, `top_p=1`, `top_k=-1`, `seed=1234`, **`ignore_eos=false`**.

### Telemetry

```bash
./scripts/profile_capture.sh _results/profiling/control_c1 \
  --model awq --concurrency 1 --num-prompts 4 --skip-attach
./scripts/memory_bandwidth.sh --gpu 0 --weight-gib 17.91 \
  --out _results/profiling/babel/c1_roof
bash scripts/bench_mxfp4_gemm.sh   # isolated MXFP4 GEMM + rocprof; restores :8000
```

rocprofv3 **cannot attach** to `VLLM::EngineCore` on this image (no `rocp-bg-attach`). Launch-under-wrapper (`scripts/profile_enginecore_launch.sh`) is healthy at C1 **60–65 tok/s** (profiler overhead; 1K/128) but traces **only parent HIP init** — no `KERNEL_DISPATCH` CSV. `VLLM_ENABLE_V1_MULTIPROCESSING=0` still forks EngineCore. How-to and Babel roof: `_results/profiling/PROFILE.md`, `_results/profiling/C1_BABEL_ROOFLINE.md`.

---

## 7. What not to claim

- MXFP4 is not “universally 4.9× FP8.” Earlier FP8 on this host was a different stack.
- C8/C64 aggregate tok/s is not C1 user throughput and is not Helix 167.8 or NVIDIA 454.
- DFlash2 does not fail because of quality; it fails as a **C8 throughput** and **interactive-latency** profile.
- DFlash effective tok/s is not a raw per-token target decode rate (verification frequency changes).
- Native MTP has no performance or quality number until the Quark loader is fixed.
- Graph wrapping with rocprofv3 is not a valid MXFP4 performance point.
- Do not treat an eager M=1 GEMM speedup as a serving win. ASM (~2× isolated) and `NUM_KSPLIT=1` (~2× isolated) both **slowed or failed to beat** graph-enabled C1.
- Do not assign the unused ~60% of Babel Read to pack/unpack, gate projection, or GDN from eager GEMM or from a **profiler-on** captured CUDA share. `_gemm_afp4wfp4` ~40% of profiled CUDA time is not a serving-HBM proof.
- Do not treat wrapped-rocprof C1 (60–65 tok/s) as a campaign point; it is profiler perturbation. Another `vllm serve` parent wrapper is not useful (HIP init only).
- 4.096 TB/s HBM is a **spec**; Babel Read **3793 GB/s** is the practical streaming measurement. `MEM%` is not calibrated GB/s; 17.91 GiB is a **conditional** full-weight-stream bound, not proof every tensor is read once per step.
- No C64 energy result.
- Compute peak does not prove matrix units are idle without counters.

---

## 8. Next work (priority order)

Standing policy: freeze HF control (`VLLM_ROCM_USE_AITER` unset, stock `M_LEQ_8`). Gate on unprofiled **79.35 ± 0.57 C1** / **553.58 ± 2.97 C8**. DFLASH-3 stays opt-in long-output only.

1. **Exclusive-GPU FP8** production-sampling vs the MXFP4 30/32 pass (`scripts/launch_vllm_fp8.sh`, then restore MXFP4). BF16 weights are not local.
2. **EngineCore spawn-exec** rocprof wrapper (restart) for `KERNEL_DISPATCH`. Do not wrap `vllm serve`. Live attach remains blocked.
3. **Mid-batch** power/ITL/KV scrape on C32–C128 (after-batch samples are cooldown; KV gauge is 0 after the batch).
4. **File upstream** from `_results/upstream/ISSUES.md` (`gh` is not installed here): Quark MTP loader; DFlash invalid-depth abort; rocprofiler-sdk EngineCore attach; Qwen3.8 a4w4 shape tune.
5. Shape-keyed MXFP4 dispatch only if it beats the frozen control on unprofiled serving—not eager GEMM, not profiler-on capture tok/s.

LM-head/sampling at C1 is **not** a measured 79→197 tok/s explanation: `generation_config.json` vs greedy T=0 is **81.98 vs 82.78** tok/s at 1K/256. Isolated notes: `_results/profiling/mxfp4_gemm/GEMM.md`, `_results/profiling/gdn_decode/GDN.md`.
