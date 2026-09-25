# MI350P MXFP4 control + graph/batch ablations (2026-09-25)

Canonical write-up for reports: [`docs/MI350P_MXFP4_VLLM_EVAL.md`](../../docs/MI350P_MXFP4_VLLM_EVAL.md). This file is the lab ledger (JSON paths, per-run notes).

**Production conclusion:** freeze the published HF Quark MXFP4 recipe (`VLLM_ROCM_USE_AITER` unset, stock `M_LEQ_8`). Do not enable global AITER ASM or a `NUM_KSPLIT=1` overlay: both improved isolated M=1 GEMM and **did not** improve graph-enabled serving (C1 **71.40** / **75.93** vs **79.53**). Promotion gate is unprofiled C1/C8 vs **79.35 ± 0.57 / 553.58 ± 2.97**. Babel Read **3793 GB/s**; C1 **40%** of that *conditional* full-weight roof — unused 60% is **not** assigned to one operator. **DFLASH-3** is the strongest optional C1 throughput cell (**113.75 tok/s**) but fails strict equality (**101/116**) and interactive latency (TTFT p95 **2.18×**, ITL p95 **+56%**); opt-in long-output only. Details: `_results/profiling/C1_BABEL_ROOFLINE.md`, `_results/profiling/mxfp4_gemm/GEMM.md`.

Immutable control remains the published HF vLLM recipe on [amd/Qwen3.8-27B-Quark-AWQ-MXFP4](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4): **79.15 C1 / 544.78 C8** (`vllm_hf_mxfp4_c*.json`). Quark MXFP4 + `AiterMxfp4LinearKernel` is the frozen target path. Do not treat NVIDIA 454 tok/s or Helix 167.8 as matched C1 claims.

Repeatability on the same process (3×, 1024/1024 `ignore_eos`):

| | r1 | r2 | r3 | mean | pstdev |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 tok/s | 78.55 | 79.72 | 79.79 | **79.35** | 0.57 |
| C8 tok/s | 549.39 | 555.40 | 555.95 | **553.58** | 2.97 |

vLLM 0.30 ROCm: `--optimization-level` (no `-O` alias). Default **O2** = `FULL_AND_PIECEWISE`. Decoder attention auto-overrides to **ROCM_ATTN** (candidates: ROCM_AITER_UNIFIED_ATTN, TRITON_ATTN). Env already on: `HIP_FORCE_DEV_KERNARG=1`, `SAFETENSORS_FAST_GPU=1`. Prefix caching and chunked prefill (`max_num_batched_tokens=8192`) are already enabled by this engine default.

Pinned KV after graph profile: `--kv-cache-memory-bytes 103223724237` (96.13 GiB; skip startup estimate). Logs: `log_graph_*.txt`.

## Graph / compile sweep (1K/1K)

One extra flag per restart. GRAPH-DEFAULT ≡ O2 (HF recipe).

| Test | Mode | C1 tok/s | C8 tok/s | vs control C1 | vs control C8 |
| --- | --- | ---: | ---: | ---: | ---: |
| GRAPH-DEFAULT | FULL_AND_PIECEWISE | 79.16 | 550.15 | 1.00× | 1.01× |
| GRAPH-O1 | PIECEWISE | 77.58 | 521.18 | 0.98× | 0.96× |
| GRAPH-O2 | default (not re-run) | — | — | same as DEFAULT | same as DEFAULT |
| GRAPH-O3 | FULL_AND_PIECEWISE | 79.32 | 550.31 | 1.00× | 1.01× |
| GRAPH-EAGER | `--enforce-eager` | **13.88** | **108.52** | 0.18× | 0.20× |

O3 is O2 within noise. O1 loses ~5% at C8. Eager is the negative control: graphs are worth **~5.7× C1** and **~5.1× C8** on this MXFP4 path. Do not ship `--enforce-eager`.

GRAPH-DEFAULT 8K/1K (discriminator): **C1 39.19** / **C8 287.08** output tok/s (prefill-heavy; not comparable to 1K decode).

## Client concurrency on HF recipe (1K/1K)

Server unchanged (auto `max-num-seqs`, batched tokens 8192). C8 is **not** the throughput peak.

| C | output tok/s | mean latency s | vs C8 |
| ---: | ---: | ---: | ---: |
| 1 | 79.43 | 12.89 | 0.14× |
| 2 | 149.14 | 13.73 | 0.27× |
| 4 | 292.80 | 13.98 | 0.53× |
| 8 | 554.57 | 14.77 | 1.00× |
| 12 | 703.87 | 17.45 | 1.27× |
| 16 | 903.65 | 18.12 | 1.63× |
| 24 | 1231.29 | 19.95 | 2.22× |
| 32 | 1504.33 | 21.77 | 2.71× |
| 48 | 1652.13 | 29.71 | 2.98× |
| 64 | **2017.49** | 32.43 | 3.64× |

No plateau by C64. Aggregate still rises while mean latency ~2.5× C1. Interactive SLO and saturated tok/s diverge here; C8 is a reporting point, not saturation.

## `max-num-seqs` (at C8 / C32 / C64)

| Server `max-num-seqs` | C8 | C32 | C64 |
| --- | ---: | ---: | ---: |
| auto (HF default) | 554.57 | 1504.33 | **2017.49** |
| 32 | 556.94 (warm) | 1438.29 | 1477.16 |
| 64 | 520.25 (under-warm) | 1491.96 | 1961.34 |

`--max-num-seqs 32` caps C64 ~1.48k (queue behind 32 slots). Do not force 32 on a throughput profile. Auto/default wins; 64 is close.

## `max-num-batched-tokens`

Default 8192 vs `--max-num-batched-tokens 16384` (auto seqs, pinned KV):

| | C32 1K | C64 1K | 8K C8 out tok/s |
| --- | ---: | ---: | ---: |
| 8192 (default) | 1504.33 | 2017.49 | 287.08 |
| 16384 | 1451.43 | 1987.29 | 289.97 |

16384 is noise / slightly worse on 1K decode. Keep 8192 until a real long-prefill mix is measured.

## Native MTP — did not load

`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` resolved `Qwen3_5MTP` then died in `qwen3_5_mtp.py` `assert self.data.shape == loaded_weight.shape`. Log: `mtp_load_failure.txt`.

## DFlash2 — C1 win, C8 loss

| Test | C1 | C4 | C8 | vs control C1 | vs control C8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DFLASH-4 | 81.48 | — | — | 1.03× | — |
| **DFLASH-7** | **102.24** | 317.60 | 387.83 | **1.29×** | **0.71×** |
| DFLASH-8 | load fail | — | — | EngineCore SIGABRT | |

DFLASH-7 clears the C1 15% gate and fails the C8 5% floor (387.83 < 517.54).

### Matched crossover sweep

Fresh-process, matched 1K/1K runs used one active batch at each concurrency
(four sequential requests at C1):

| C | Baseline tok/s | DFlash2-7 tok/s | DFlash delta |
| ---: | ---: | ---: | ---: |
| 1 | 78.80 | **102.68** | **+30.3%** |
| 2 | 150.07 | **176.12** | **+17.4%** |
| 3 | 222.39 | **245.36** | **+10.3%** |
| 4 | 293.51 | **307.29** | +4.7% |
| 5 | 318.04 | **336.65** | +5.9% |
| 6 | **426.95** | 336.09 | **−21.3%** |
| 8 | **557.14** | 422.30 | **−24.2%** |

The throughput crossover is between **C5 and C6** for this corpus. The useful
promotion boundary is more conservative: DFlash has a material throughput win
at C1–C3, only a small win at C4–C5, and a large loss from C6.

### Streaming latency (1K input / 256 output)

Eight requests per point, `stream_interval=1`, with returned token IDs. Token
ITL records token arrival at the client; accepted speculative tokens delivered
in one SSE chunk have zero interval.

| C | Baseline TTFT p95 ms | DFlash TTFT p95 ms | Baseline mean / p95 ITL ms | DFlash mean / p95 ITL ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 75.7 | 244.4 | 11.83 / 12.05 | 11.06 / 20.57 |
| 2 | 125.4 | 276.1 | 12.39 / 12.61 | 12.66 / 22.93 |
| 3 | 159.9 | 401.6 | 12.53 / 12.79 | 12.66 / 24.08 |
| 4 | 270.8 | 552.2 | 12.73 / 12.92 | 13.51 / 24.94 |
| 5 | 316.7 | 2453.4 | 13.96 / 14.97 | 15.42 / 29.63 |
| 6 | 369.0 | 762.4 | 12.89 / 13.24 | 16.91 / 34.78 |
| 8 | 461.4 | 979.0 | 15.12 / 13.50 | 19.27 / 36.22 |

This changes the operational interpretation: DFlash2-7 improves **long-output
C1 completion throughput**, but it does not improve TTFT or p95 token-arrival
cadence. At C1 its mean token interval improves slightly through burst delivery,
while p95 ITL is worse. Call it the low-concurrency, long-generation profile,
not an unconditional latency/interactive winner.

### DFLASH-3 promotion gates (matched C1 n=24 + 116 greedy)

| Metric | Control | DFLASH-3 | Result |
|---|---:|---:|---|
| 1K/256 stream output tok/s | 82.56 | **97.41** | **+18.0%** |
| TTFT p95 ms | **58.55** | 127.45 | **2.18× worse** |
| token ITL mean ms | 11.95 | **9.80** | burst-delivery mean improves |
| token ITL p95 ms | **12.07** | 18.84 | **+56.1% worse** |
| chunk ITL p95 ms | **12.07** | 18.89 | **+56.6% worse** |
| exact greedy token IDs | 116/116 self-repeat | **101/116 vs control** | strict equality fails |

DFLASH-3 and DFLASH-7 have the same 15 mismatch IDs versus control, but differ
from each other on 8 prompts. JSON/tool remains **15/15 exact**. Therefore
DFLASH-3 is the preferred **opt-in long-output** depth, but it is not a better
general interactive profile. Summary: `dflash3_validation/VALIDATION.md`.
Artifacts: `dflash3_validation/`,
`../quality/compare_baseline_vs_dflash3.json`.

### Acceptance telemetry

The vLLM metrics counters after the crossover run recorded 15,136 drafts,
105,952 proposed draft tokens, and 18,144 accepted draft tokens (17.12%).
Accepted positions were 10,880 at position 0, 7,264 at position 1, and zero at
positions 2–6. Runtime logs reported mean acceptance length around 2.04–2.39;
this directly confirms that the metric includes the target token plus accepted
draft tokens. The available vLLM counters do not expose a full 0–7 acceptance
histogram or separate target/drafter forward timing.

### DFlash2-8 diagnosis

The failure is **not graph capture specific**. A fresh default run and an
`--enforce-eager` run both terminated during startup with EngineCore `-6` and
`HSA_STATUS_ERROR_EXCEPTION` (0x1016). Eager explicitly disabled torch.compile
and CUDA graphs before reproducing the same hardware exception.

The checkpoint declares `dflash_config.block_size=8`; DFlash uses one anchor
plus mask positions, so this artifact supports seven speculative tokens.
`num_speculative_tokens=8` requests nine query positions. vLLM 0.30 does not
reject that mismatch at argument validation and instead reaches a GPU fault.
Treat seven as the checkpoint maximum and file an upstream validation bug.
Logs: `dflash8/default.log`, `dflash8/eager.log`.

### Deterministic quality smoke test

Baseline and DFlash2-7 produced exact matching content and token IDs on all six
temperature-zero smoke cases (format following, arithmetic, Python, JSON,
translation, and factual QA). This validates the tested deterministic lane,
not general task quality. Artifacts: `quality/baseline.json`,
`quality/dflash7.json`, and `quality/comparison.json`.

## Profiles

| Profile | Settings | Evidence |
| --- | --- | --- |
| Saturated | HF recipe, graphs on (O2), auto seqs, batched 8192, optional pinned KV bytes | C8 554.57; C64 2017.49 |
| Lowest TTFT / smoothest stream | HF recipe, no speculation | Better TTFT and p95 ITL at C1–C8 |
| Low-concurrency long output | **DFLASH-3** on same target; opt-in C1–C3 only | C1 113.75; equality 101/116; TTFT/ITL p95 regress |

Do not add `--enforce-eager`, `--max-num-seqs 32`, `--optimization-level 1`, `VLLM_ROCM_USE_AITER=1`, or a `NUM_KSPLIT=1` DEFAULT overlay to the saturated command.

Live: HF recipe `awq` on `127.0.0.1:8000` with `--kv-cache-memory-bytes 103223724237`.

## Next (frozen control; serving is the gate)

1. Keep HF control unchanged. Gate kernel changes on unprofiled C1/C8 vs **79.35 ± 0.57 / 553.58 ± 2.97**, not eager GEMM or profiler-on capture tok/s (M=1 **60.92**).
2. Exclusive-GPU FP8 sampling vs MXFP4 30/32. EngineCore spawn-exec for `KERNEL_DISPATCH` (live attach blocked). Mid-batch C32–C128 power/KV.
3. File `_results/upstream/ISSUES.md`. DFLASH-3 stays opt-in long-output only.
4. Captured vLLM O2 replay and Babel `MiBytes/sec` parser are **done**: `_results/profiling/vllm_captured/SUMMARY.md`, `scripts/parse_rvs_babel.py`. 1K/256 C32–C128: `_results/priority_eval/saturation_c32_c128/SATURATION.md`.

Canonical: `docs/MI350P_MXFP4_VLLM_EVAL.md` §5/§8.

## Profiling enablement

How-to and blockers: `_results/profiling/PROFILE.md`. Capture helper: `scripts/profile_capture.sh`. P1–P4 Prometheus+telemetry: `_results/profiling/P1_P4_METRICS.md`.

rocprofv3 attach cannot instrument `VLLM::EngineCore` on this ROCm image (`rocp-bg-attach` never starts). Launch-under-`rocprofv3` also failed to emit kernel CSVs (parent HIP init only). Practical HBM: Babel Read **3793 GB/s**. Unprofiled C1 **79.53 tok/s**.

## Speculative quality vs MXFP4 target

Greedy token-ID equality (`temperature=0`, `ignore_eos=false`, 116 text-only prompts):

- MXFP4 control is **100%** self-consistent across restart and a same-process repeat.
- DFLASH-3 and DFLASH-7 each match the control on **101/116 (87.1%)**, with the same 15 mismatch IDs; DFLASH-4 matches **102/116 (87.9%)**.
- JSON/tool prompts are **15/15** exact.
- A C8 concurrent subset of the **same non-speculative** control already diverges on 3/24 IDs that also appear in the DFlash mismatch set.

Do not claim 100% lossless speculation on this path. Do not treat 17.8% draft acceptance as a quality failure. MTP is still a load-path blocker, not a quality result. MXFP4 production sampling scored **30/32**; MXFP4 versus BF16/FP8 was not measured. Details: `_results/quality/QUALITY.md`.
