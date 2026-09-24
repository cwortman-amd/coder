# Single Instinct MI350P — Engine × Quantization Report

**Scope:** one discrete GPU, **no** prefill/decode disaggregation, no TP=2 / DP=2.  
**SKU:** AMD Instinct™ MI350P PCIe (`gfx950`, Device ID `0x75a8`, 144 GB HBM3E, 600 W cap).  
**Date:** 24 September 2026  
**Host:** Linux 6.8.0-138-generic, AMD EPYC 9015, HIP device 0 = `0000:8b:00.0`.

This document analyses **llama.cpp**, **SGLang**, and **vLLM** for **Qwen3.8-27B FP8** and **MXFP4** on a *single* MI350P. Cells that were not executed on this card are marked **not measured**. Numbers copied from the Radeon AI PRO R9700 (`gfx1201`, 32 GB, 300 W) campaign are confined to §9 and must not be treated as MI350P results.

---

## 1. Executive summary

| Question | Answer on this MI350P |
| --- | --- |
| What actually served? | **vLLM 0.30.0** (`vllm/vllm-openai-rocm:latest`) · **`Qwen/Qwen3.8-27B-FP8`** · GPU 0 only |
| Isolated decode | **26.2–35.6 tok/s** at 256 output tokens (D1–D4); best **35.63 tok/s** at 1024-token prompt |
| Isolated prefill | P1–P4 **PASS**; 8K first-token **~103 ms**; 12K/16K **HTTP 400** (`max-model-len=9600`) |
| vLLM MXFP4 | **Not measured** — `local/vllm-mxfp4:gfx950` is not on the host |
| llama.cpp FP8 / MXFP4 | **Not measured** — no `local/llama.cpp:rocm10-gfx950`; GGUF in this repo is **Q4_K_M**, not FP8/MXFP4 |
| SGLang FP8 / MXFP4 | **Not measured** — `coder-sglang:latest` is not on the host |
| Mixed `vllm bench serve` matrix | **Not completed** on this SKU after the FP8 server was brought up |
| P/D | Out of scope (see `DUAL_MI450P_REPORT.md`) |

**Architectural takeaway (MI350P, not R9700):** 144 GB HBM3E removes the 32 GB “FP8 barely fits” constraint that made MXFP4 mandatory on the R9700. On this card FP8 already leaves ~144 − 27.5 ≈ **116 GB** after weights; a 32 GiB KV reservation still yielded **~337k tokens** of cache. MXFP4 remains interesting for **decode bandwidth and energy**, not for “will it load.” llama.cpp and SGLang still need **gfx950-native images** before any ranking vs vLLM is valid here.

---

## 2. Comparison matrix (intended vs measured)

Formats in this codebase:

| Label | Checkpoint | Typical engine |
| --- | --- | --- |
| **FP8** | `Qwen/Qwen3.8-27B-FP8` (Hugging Face, W8A8) | vLLM, SGLang |
| **MXFP4** | `Qwen3.8-27B-Quark-AWQ-MXFP4` (Radiance / Quark W4A8) | vLLM-MXFP4 (`--quantization quark`) |
| **Q4_K_M** (not FP8/MXFP4) | `Qwen3.8-27B-Q4_K_M.gguf` | llama.cpp |

llama.cpp does **not** load the HF FP8 or Quark MXFP4 safetensors as-is. A fair “llama.cpp FP8” cell would require an FP8 GGUF (or `--file` path this tree does not ship). A fair “llama.cpp MXFP4” cell would require an MXFP4 GGUF or a HIP kernel path that does not exist in `docker-compose.gguf.yml`.

### 2.1 Single-MI350P status grid

| Engine | FP8 | MXFP4 | Notes for this host |
| --- | --- | --- | --- |
| **vLLM** | **Measured** (phase isolation, 24 Sep 12:21–12:22) | **Blocked** — image `local/vllm-mxfp4:gfx950` missing; compose defaults to gfx1201 Radiance flags (`RADIANCE_USE_R4D`) that `lib/gpu_profile.sh` turns **off** for `mi350p` | `.env` pins `VLLM_IMAGE=vllm/vllm-openai-rocm:latest` (0.30), not the ROCm 10 py3.14 tag |
| **SGLang** | **Blocked** — `coder-sglang:latest` missing | **Blocked** — same image; no Quark MXFP4 launch path in `docker-compose.sglang.yml` | Historical R9700 runs marked SGLang **UNAVAILABLE** |
| **llama.cpp** | **Blocked** — no gfx950 HIP image; no FP8 GGUF in `models/` | **Blocked** — no MXFP4 GGUF; image tag `local/llama.cpp:rocm10-gfx950` missing | GGUF compose is `-ngl 999 -fa on` Q4_K_M only |

Docker images present when this report was written: `vllm/vllm-openai-rocm:latest`, `ghcr.io/anomalyco/opencode:latest`. No MXFP4, llama.cpp, or SGLang images.

---

## 3. Platform: why MI350P changes the engine ranking vs R9700

| Dimension | MI350P (this host) | R9700 (prior campaign) | Implication for 27B |
| --- | --- | --- | --- |
| ISA | CDNA 4 `gfx950` | RDNA 4 `gfx1201` | Different AITER/Triton binaries; do not reuse `HSA_OVERRIDE_GFX_VERSION=12.0.1` |
| Memory | 144 GB HBM3E | 32 GB GDDR6 | FP8 27B (~27.5 GiB) is comfortable; MXFP4 is not required to “fit” |
| LDS | 160 KB/workgroup (AITER tables) | 64 KB | Stock AITER unified attn that OOM’d R9700 at 66 KB **can** fit here |
| TDP | 600 W cap, ~100 W idle | 300 W TBP | Energy/token must be re-measured; R9700 192–253 W figures do not transfer |
| Profile | `RADIANCE_USE_R4D=0` | `RADIANCE_USE_R4D=1` | RDNA4 Radiance attention packing is explicitly disabled on Instinct |
| Compile | Prefer piecewise CUDA graphs | `cudagraph_mode: NONE` (allocator) | `.env` still forced 9600 ctx + quoted `NONE` for 0.30 stability |

Empty `HSA_OVERRIDE_GFX_VERSION=` **breaks** `rocminfo` / AITER on this card. `GPU_ARCHS=gfx950` is required so AITER does not depend on a nested `rocminfo`. See `DUAL_MI450P_REPORT.md` §3 for the serve bring-up log.

---

## 4. vLLM × FP8 — measured on MI350P GPU 0

**Stack:** vLLM 0.30.0 · overlay `patches/qwen3_5_vllm030.py` (`ignore_unexpected_prefixes` for `visual.` / `mtp.`) · `max-model-len 9600` · `kv-cache-memory-bytes` 32 GiB · `cudagraph_mode: NONE` · `HIP_VISIBLE_DEVICES=0`.

**Weights:** 27.48 GiB in ~23 s. After load: ~143.7 GiB free, 32 GiB reserved for KV → **337,515** cache tokens, **35.16×** concurrency at 9600 tokens/request.

### 4.1 Decode-only (collocated, 256 output tokens, 1 trial)

Artifact: `_results/phases/phase_profile_summary_20260924_122130.json`

| Case | Prompt | Output | Decode tok/s | TPOT p50 (ms) | ITL p50 / p95 / p99 (ms) | TTFT p50 (ms) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| D1 | 128 | 256 | 26.16 | 38.22 | 28.38 / 56.70 / 87.77 | 129.59 |
| D2 | 1024 | 256 | **35.63** | 28.07 | 28.28 / 28.81 / 29.06 | 316.37 |
| D3 | 4096 | 256 | 33.38 | 29.96 | 30.15 / 30.37 / 30.52 | 755.38 |
| D4 | 8192 | 256 | 26.98 | 37.06 | 37.35 / 37.56 / 37.66 | 924.85 |

Steady decode is **~27–36 tok/s** at this output length. D1 has a fat ITL tail (p99 88 ms). D4 is slower than D2 because 8K prefill is collocated on the same engine (this is **not** P/D).

Power columns are **0 W**: the login user is not in group `render`, so host `amd-smi` telemetry failed.

### 4.2 Prefill-only (`max_tokens=1`, 2 measured trials + warmup)

Artifact: `_results/phases/phase_profile_summary_20260924_122210.json`

Use **engine prefill seconds** and **TTFT**, not client `prompt_tok_s` (prefix cache + repeating synthetic prompt inflates tok/s).

| Case | Input | Status | Engine prefill (s) | TTFT p50 (ms) | TTFT p95 (ms) |
| --- | ---: | --- | ---: | ---: | ---: |
| P1 | 128 | PASS | 0.056 | 58.21 | 59.63 |
| P2 | 1024 | PASS | 0.065 | 67.37 | 67.73 |
| P3 | 4096 | PASS | 0.063 | 66.04 | 67.10 |
| P4 | 8192 | PASS | 0.097 | **102.55** | 103.15 |
| P5 | 12288 | BLOCKED | — | HTTP 400, max context 9600 | |
| P6 | 16384 | BLOCKED | — | HTTP 400, max context 9600 | |

Prefill wall time is almost flat from 128–4096 tokens (~56–67 ms) then steps at 8K (~100 ms). That is **not** the 2.6 s 8K TTFT seen on R9700 MXFP4 in `docs/SINGLE_R9700_PHASE_PROFILING.md` — different quant, engine, context config, and likely cache state. Do not compute a “MI350P vs R9700 prefill speedup” from these two tables without a matched cold-cache `vllm bench serve` run.

### 4.3 What was *not* collected for vLLM FP8

- `throughput.sh` default matrix **8192:1024, 1024:8192, 1024:1024** at C=1 and C>1  
- 1024-token decode isolation (suite default OSL; this run used 256)  
- Contention / prefix-cache suites in `bench_phases.py` (`concurrency`, `contention`, `prefix`)  
- Joules/token  

Until those land, do not claim an 8K:1K “agent” decode rate for MI350P FP8. The 26.98 tok/s D4 figure is 8192-in / **256-out**, not 1024-out.

---

## 5. vLLM × MXFP4 — analysis without a gfx950 image

**Intended stack:** `MXFP4_IMAGE=local/vllm-mxfp4:gfx950`, `docker-compose.mxfp4.yml`, Quark `--quantization quark`, `--kv-cache-dtype fp8`, AITER unified attention.

**Why it did not run:** the only MXFP4 image this repo documents for Instinct is a **local** tag that was never built here. The gfx1201 Radiance image (`local/vllm-mxfp4:gfx1201`) is the wrong ISA. `apply_gpu_profile` for `mi350p` sets `RADIANCE_USE_R4D=0` because R4D is an RDNA4 path.

**What MXFP4 would change on 144 GB (hypothesis, not data):**

1. **Capacity:** Weight footprint on R9700 was ~14–19 GB vs ~27–32 GB FP8. On MI350P both fit; MXFP4’s advantage is **more KV / longer context / higher `max-num-seqs`**, not “loads at all.”
2. **Decode:** R9700 single-stream MXFP4 held **~33–34 tok/s** independent of context (1024 OSL). MI350P FP8 already sits in the same **27–36 tok/s** band at 256 OSL. MXFP4 on CDNA 4 could go either way (better GEMM vs missing R4D).
3. **Prefill:** Radiance + AITER on R9700 delivered ~3.1–3.3k prompt tok/s cold. MI350P has more LDS and HBM; a gfx950 MXFP4 build is required before quoting a rate.
4. **Accuracy (other SKU):** On R9700, MXFP4 matched FP8 at **80% GPQA diamond (n=5)** and was the only quant to land a valid SWE-bench Lite patch in that 5-problem sample. That accuracy claim is **not** re-validated on MI350P.

Fill command (after the image exists):

```bash
export GPU_PROFILE=mi350p
# apply_gpu_profile sets MXFP4_IMAGE=local/vllm-mxfp4:gfx950
./throughput.sh -e mxfp4 --test-cases 8192:1024,1024:8192,1024:1024
./accuracy.sh --diamond   # against the MXFP4 server
```

---

## 6. llama.cpp — FP8 / MXFP4 mapping and gaps

`docker-compose.gguf.yml` runs `llama-server` with:

- `-m /models/${MODEL_FILE}` default **`Qwen3.8-27B-Q4_K_M.gguf`**
- `-ngl 999 -fa on -c MAX_MODEL_LEN -np 1`

There is **no FP8 or MXFP4 GGUF** in `models/` on this host (the directory is empty of weight files in the workspace snapshot). Profile image `local/llama.cpp:rocm10-gfx950` is absent; the R9700 tag is `local/llama.cpp:rocm10-gfx1201` / `rocm7-gfx1201`.

| Requested cell | What llama.cpp can actually run | MI350P status |
| --- | --- | --- |
| llama.cpp × FP8 | Needs an FP8 GGUF or a CUDA/HIP FP8 backend this compose does not enable | Not measured |
| llama.cpp × MXFP4 | Needs MXFP4 GGUF + kernels | Not measured |
| llama.cpp × Q4_K_M | Supported by compose | Not measured on gfx950; R9700 only (§9) |

**CDNA-4 notes:** HIP builds must target `gfx950`. A gfx1201 binary with `HSA_OVERRIDE_GFX_VERSION` is the wrong approach (that override is what broke vLLM here when left empty, and 12.0.1 is RDNA4). Flash attention (`-fa on`) must be confirmed on Instinct; if FA is CPU-fallback, decode collapses (R9700 showed both **75 tok/s** smoke and **2.2 tok/s** “all-engine” smokes — treat llama.cpp numbers as invalid until `rocm-smi` shows GPU% during generate).

---

## 7. SGLang — FP8 / MXFP4 mapping and gaps

`docker-compose.sglang.yml` launches `python3 -m sglang.launch_server` with `--model-path` defaulting to `MODEL_NAME` (FP8 HF id). Optional `--load-format gguf` is for GGUF, not Quark MXFP4.

| Requested cell | Path | MI350P status |
| --- | --- | --- |
| SGLang × FP8 | HF `Qwen/Qwen3.8-27B-FP8`, `--tp 1` | Image missing |
| SGLang × MXFP4 | Not wired; would need Quark/SGLang MXFP4 support plus weights | Not available in this compose |

Prior R9700 `throughput.sh -e sglang` slices recorded **UNAVAILABLE** (image or tokenizer mismatch). `SGLANG_TOKENIZER_PATH` in `.env` still points at `Qwen/Qwen2.5-0.5B-Instruct`, which is the wrong tokenizer family for Qwen3.8 and will bias TTFT/tok counts if left unchanged.

---

## 8. Cross-engine analysis (what a completed MI350P matrix should decide)

Once all six cells have matched `vllm bench serve` traces (same ISL/OSL, concurrency, tokenizer, `temperature=0`), rank engines on:

1. **Decode tok/s** at 1024:1024 and 8192:1024 (agent-like).  
2. **TTFT** at 8192 (prefill).  
3. **Decode tok/s** at 1024:8192 (generation-heavy).  
4. **GPQA / SWE-bench** (n much larger than 5 before claiming accuracy).  
5. **J/token** from in-container `amd-smi` (host user lacks `/dev/kfd`).

**Working hypotheses for MI350P (to be confirmed, not cited as results):**

- **vLLM FP8** is the only production-ready path on this machine today; 144 GB makes long KV the default optimization (raise `MAX_MODEL_LEN` from 9600).  
- **vLLM MXFP4** is the primary *quant* competitor, not llama.cpp FP8 (which is a category error unless a GGUF exists). Expect MXFP4 to matter more at high concurrency than at C=1 if HBM bandwidth, not weight size, dominates.  
- **llama.cpp Q4_K_M** can still win raw tok/s on short prompts (seen on R9700) while losing GPQA (60% vs 80% on n=5). Do not put Q4_K_M in the FP8 or MXFP4 columns.  
- **SGLang** is unproven on both SKUs in this repo; it is a RadixAttention / prefix-cache story, not a quant story, until FP8 serving works.

```
                 Single-MI350P measurement coverage
  ┌──────────────────┬─────────────┬─────────────┐
  │                  │    FP8      │   MXFP4     │
  ├──────────────────┼─────────────┼─────────────┤
  │ vLLM             │ ████ PHASE  │ ░░░░░ NONE  │
  │ SGLang           │ ░░░░░ NONE  │ ░░░░░ NONE  │
  │ llama.cpp        │ ░░░░░ NONE  │ ░░░░░ NONE  │
  └──────────────────┴─────────────┴─────────────┘
  PHASE = isolated prefill/decode only (not full I:O throughput matrix)
```

---

## 9. R9700 campaign (different card — reference only)

These figures are from `docs/COMPREHENSIVE_BENCHMARK_REPORT.md`, `docs/QUANTIZATION_ACCURACY_COMPARISON_REPORT.md`, `docs/SINGLE_R9700_PHASE_PROFILING.md`, and `_results/throughput/`. Hardware: **Radeon AI PRO R9700, 32 GB, 300 W, kernel 6.8.0-71**. Peak power traces cap at **300 W**. They answer “what did we already learn about engines and quants,” not “what is MI350P.”

### 9.1 Throughput (selected)

| Engine / quant | Shape | Output tok/s | Notes |
| --- | --- | ---: | --- |
| vLLM FP8 ROCm 10 | 8192:1024 C=1 | **13.10** | Official container baseline |
| vLLM FP8 optimized | 8192:1024 C=1 | **13.12** | FP16 KV, no epilogue autotune |
| vLLM MXFP4 Radiance | 8192:1024 C=1 | **30.59** | ~2.3× FP8 |
| vLLM MXFP4 | 8192:1024 C=4 | **91.61** | 210.9 W avg, 3.30 J/tok |
| llama.cpp Q4_K_M | 8192:1024 C=1 | **23.48** | 2 prompts; TPOT 35.17 ms |
| llama.cpp Q4_K_M | 128:64 smoke | **74.98** | Tiny slice; not comparable to 8K |
| llama.cpp (other smoke) | 128:64 | **2.20** | Likely CPU / mis-device; discard |
| SGLang | 128:64 | UNAVAILABLE | Image/tokenizer |
| vLLM FP8 `bench serve` | 8192:1024 C=1 | 9.01–9.09 | Earlier 0.30-class serve on labeled R9700 reports |

### 9.2 Accuracy (n=5 diamond / 5 SWE-Lite) — R9700

| | GPQA | SWE-bench valid patches | SWE tok/s | GPQA tok/s |
| --- | ---: | ---: | ---: | ---: |
| vLLM FP8 | 80% (4/5) | 0/5 | 12.02 | 17.02 |
| vLLM MXFP4 | 80% (4/5) | **1/5** | 19.11 | 19.17 |
| llama.cpp Q4_K_M | 60% (3/5) | 0/5 | 28.76 | 28.76 |

MXFP4 matching FP8 on this tiny GPQA sample is the strongest *quant* result in the repo. It has **not** been repeated on MI350P.

### 9.3 Why R9700 MXFP4 looked mandatory

FP8 used **~31.6 GB / 32 GB**. CUDA graphs were turned off; KV was squeezed to 1 GB. MXFP4 dropped weights to ~19 GB and restored graphs. **That VRAM crisis does not exist on 144 GB MI350P.** Copying the R9700 “always ship MXFP4” conclusion onto Instinct without new measurements would be incorrect.

---

## 10. Reproduce — fill the empty cells on this machine

```bash
cd /home/amd/workspace/coder
set -a && . ./.env && set +a
. ./lib/gpu_profile.sh
export GPU_PROFILE=auto
apply_gpu_profile
unset HSA_OVERRIDE_GFX_VERSION

# A) vLLM FP8 (already proven)
docker compose -f docker-compose.yml up -d --force-recreate --no-deps inference
# wait for :8000/health
./throughput.sh -e vllm --test-cases 8192:1024,1024:8192,1024:1024 -c 1
python3 benchmark/bench_phases.py --mode decode --model Qwen/Qwen3.8-27B-FP8 \
  --output-tokens 1024 --trials 2

# B) vLLM MXFP4 — only after docker images | grep vllm-mxfp4:gfx950
./throughput.sh -e mxfp4 --test-cases 8192:1024,1024:8192,1024:1024 -c 1

# C) llama.cpp — only after local/llama.cpp:rocm10-gfx950 and a GGUF under models/
#     Q4_K_M is the supported file; label results Q4_K_M, not FP8/MXFP4
./throughput.sh -e llama.cpp --test-cases 8192:1024,1024:1024 -c 1

# D) SGLang — only after coder-sglang:latest; fix SGLANG_TOKENIZER_PATH to Qwen3.8
./throughput.sh -e sglang --test-cases 8192:1024,1024:1024 -c 1
```

Keep `HIP_VISIBLE_DEVICES=0` so GPU 1 is idle (single-card study).

---

## 11. Conclusions

1. **On this MI350P, the only completed single-GPU engine×quant cell is vLLM × FP8**, characterized with isolated decode and prefill, not the full throughput matrix.  
2. **MXFP4, llama.cpp, and SGLang have no gfx950 runtime here**; ranking them on MI350P would be fiction.  
3. **llama.cpp cannot occupy the FP8 or MXFP4 columns** without different weight files than this repo’s GGUF default.  
4. **R9700 results remain useful as a 32 GB / RDNA4 baseline:** MXFP4 ≈ 2.3× vLLM FP8 decode, llama.cpp Q4_K_M fast but weaker GPQA, SGLang not stood up. They are the wrong memory and ISA to predict Instinct.  
5. Next measurement that actually changes decisions: **vLLM FP8 vs vLLM MXFP4** on GPU 0 with identical 8192:1024 / 1024:8192 / 1024:1024 traces and in-container power, then llama.cpp Q4_K_M as a third *format*, not as “FP8.”

---

## 12. Artifacts

| Path | Role |
| --- | --- |
| `_results/phases/phase_profile_summary_20260924_122130.json` | MI350P vLLM FP8 decode-only |
| `_results/phases/phase_profile_summary_20260924_122210.json` | MI350P vLLM FP8 prefill-only |
| `docker-compose.yml` | vLLM FP8 single GPU |
| `docker-compose.mxfp4.yml` | vLLM MXFP4 (needs gfx950 image) |
| `docker-compose.gguf.yml` | llama.cpp Q4_K_M |
| `docker-compose.sglang.yml` | SGLang FP8 HF |
| `lib/gpu_profile.sh` | `mi350p` vs `r9700` image/arch defaults |
| `docs/QUANTIZATION_ACCURACY_COMPARISON_REPORT.md` | R9700 FP8 vs MXFP4 vs Q4_K_M accuracy |
| `docs/COMPREHENSIVE_BENCHMARK_REPORT.md` | R9700 engine/quant throughput |
| `DUAL_MI450P_REPORT.md` | Dual-card P/D (out of scope here) |
