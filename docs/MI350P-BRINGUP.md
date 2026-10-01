---
type: Technical Report
title: MI350P bring-up
description: Earlier single-GPU notes from 24 Sep. The current Quark MXFP4 serving
  record is MI350P.md. Dual-GPU prefill/decode is MI350P-PD.md.
tags:
- technical-report
- mi350p
- bringup
status: stable
---

# MI350P bring-up

Earlier single-GPU notes from 24 Sep. The current Quark MXFP4 serving record is [MI350P.md](MI350P.md). Dual-GPU prefill/decode is [MI350P-PD.md](MI350P-PD.md).

## Engine and quantization matrix

## Single Instinct MI350P — Engine × Quantization Report

> **25 Sep 2026 update.** vLLM × Quark MXFP4 **has been measured** on stock `vllm/vllm-openai-rocm:latest`. Canonical: [`MI350P.md`](MI350P.md) — freeze HF control **79.35 C1 / 553.58 C8**. Babel Read **3793 GB/s**; C1 **40%** of that conditional roof. Isolated GEMM knobs that won M=1 **lost serving**. This 24 Sep report remains the **FP8 bring-up**.

**Scope:** one discrete GPU, **no** prefill/decode disaggregation, no TP=2 / DP=2.  
**SKU:** AMD Instinct™ MI350P PCIe (`gfx950`, Device ID `0x75a8`, 144 GB HBM3E, 600 W cap).  
**Date:** 24 September 2026 (FP8) / 25 September 2026 (MXFP4 pointer)  
**Host:** Linux 6.8.0-138-generic (later 6.8.0-142-generic), AMD EPYC 9015, HIP device 0 = `0000:8b:00.0`.

This document analyses **llama.cpp**, **SGLang**, and **vLLM** for **Qwen3.8-27B FP8** and **MXFP4** on a *single* MI350P. Cells that were not executed on this card are marked **not measured**. Numbers copied from the Radeon AI PRO R9700 (`gfx1201`, 32 GB, 300 W) campaign are confined to §9 and must not be treated as MI350P results.

---

### 1. Executive summary

| Question | Answer on this MI350P |
| --- | --- |
| What actually served? | **vLLM 0.30.0** (`vllm/vllm-openai-rocm:latest`) · GPU 0. FP8 on 24 Sep; **Quark MXFP4** on 25 Sep |
| Isolated decode (FP8) | **26.2–35.6 tok/s** at 256 output tokens (D1–D4); best **35.63 tok/s** at 1024-token prompt |
| Isolated prefill (FP8) | P1–P4 **PASS**; 8K first-token **~103 ms**; 12K/16K **HTTP 400** (`max-model-len=9600`) |
| vLLM MXFP4 | **Measured** — HF recipe, `AiterMxfp4LinearKernel`; **79.35 C1 / 553.58 C8** (1K/1K). See `MI350P.md` |
| llama.cpp FP8 / MXFP4 | **Not measured** — no `local/llama.cpp:rocm10-gfx950`; GGUF in this repo is **Q4_K_M**, not FP8/MXFP4 |
| SGLang FP8 / MXFP4 | **Not measured** — `coder-sglang:latest` is not on the host |
| Mixed `vllm bench serve` matrix | **Not completed** on this SKU after the FP8 server was brought up |
| P/D | Out of scope (see `MI350P-PD.md`) |

**Architectural takeaway (MI350P, not R9700):** 144 GB HBM3E removes the 32 GB “FP8 barely fits” constraint that made MXFP4 mandatory on the R9700. On this card FP8 already leaves ~144 − 27.5 ≈ **116 GB** after weights; a 32 GiB KV reservation still yielded **~337k tokens** of cache. MXFP4 remains interesting for **decode bandwidth and energy**, not for “will it load.” llama.cpp and SGLang still need **gfx950-native images** before any ranking vs vLLM is valid here.

---

### 2. Comparison matrix (intended vs measured)

Formats in this codebase:

| Label | Checkpoint | Typical engine |
| --- | --- | --- |
| **FP8** | `Qwen/Qwen3.8-27B-FP8` (Hugging Face, W8A8) | vLLM, SGLang |
| **MXFP4** | `Qwen3.8-27B-Quark-AWQ-MXFP4` (Radiance / Quark W4A8) | vLLM-MXFP4 (`--quantization quark`) |
| **Q4_K_M** (not FP8/MXFP4) | `Qwen3.8-27B-Q4_K_M.gguf` | llama.cpp |

llama.cpp does **not** load the HF FP8 or Quark MXFP4 safetensors as-is. A fair “llama.cpp FP8” cell would require an FP8 GGUF (or `--file` path this tree does not ship). A fair “llama.cpp MXFP4” cell would require an MXFP4 GGUF or a HIP kernel path that does not exist in `docker-compose.gguf.yml`.

#### 2.1 Single-MI350P status grid

| Engine | FP8 | MXFP4 | Notes for this host |
| --- | --- | --- | --- |
| **vLLM** | **Measured** (phase isolation, 24 Sep) | **Measured 25 Sep** on stock `vllm/vllm-openai-rocm:latest` + sharded Quark AWQ MXFP4 (`AiterMxfp4LinearKernel`). Dedicated `local/vllm-mxfp4:gfx950` / Radiance compose was **not** required | `.env` still pins `VLLM_IMAGE=vllm/vllm-openai-rocm:latest` (0.30). Launch: `./scripts/launch_vllm_mxfp4.sh` |
| **SGLang** | **Blocked** — `coder-sglang:latest` missing | **Blocked** — same image; no Quark MXFP4 launch path in `docker-compose.sglang.yml` | Historical R9700 runs marked SGLang **UNAVAILABLE** |
| **llama.cpp** | **Blocked** — no gfx950 HIP image; no FP8 GGUF in `models/` | **Blocked** — no MXFP4 GGUF; image tag `local/llama.cpp:rocm10-gfx950` missing | GGUF compose is `-ngl 999 -fa on` Q4_K_M only |

Docker images as of 24 Sep: `vllm/vllm-openai-rocm:latest`, `ghcr.io/anomalyco/opencode:latest`. MXFP4 later served on that same vLLM image; llama.cpp and SGLang images still absent.

---

### 3. Platform: why MI350P changes the engine ranking vs R9700

| Dimension | MI350P (this host) | R9700 (prior campaign) | Implication for 27B |
| --- | --- | --- | --- |
| ISA | CDNA 4 `gfx950` | RDNA 4 `gfx1201` | Different AITER/Triton binaries; do not reuse `HSA_OVERRIDE_GFX_VERSION=12.0.1` |
| Memory | 144 GB HBM3E | 32 GB GDDR6 | FP8 27B (~27.5 GiB) is comfortable; MXFP4 is not required to “fit” |
| LDS | 160 KB/workgroup (AITER tables) | 64 KB | Stock AITER unified attn that OOM’d R9700 at 66 KB **can** fit here |
| TDP | 600 W cap, ~100 W idle | 300 W TBP | Energy/token must be re-measured; R9700 192–253 W figures do not transfer |
| Profile | `RADIANCE_USE_R4D=0` | `RADIANCE_USE_R4D=1` | RDNA4 Radiance attention packing is explicitly disabled on Instinct |
| Compile | Prefer piecewise CUDA graphs | `cudagraph_mode: NONE` (allocator) | `.env` still forced 9600 ctx + quoted `NONE` for 0.30 stability |

Empty `HSA_OVERRIDE_GFX_VERSION=` **breaks** `rocminfo` / AITER on this card. `GPU_ARCHS=gfx950` is required so AITER does not depend on a nested `rocminfo`. See `MI350P-PD.md` §3 for the serve bring-up log.

---

### 4. vLLM × FP8 — measured on MI350P GPU 0

**Stack:** vLLM 0.30.0 · overlay `patches/qwen3_5_vllm030.py` (`ignore_unexpected_prefixes` for `visual.` / `mtp.`) · `max-model-len 9600` · `kv-cache-memory-bytes` 32 GiB · `cudagraph_mode: NONE` · `HIP_VISIBLE_DEVICES=0`.

**Weights:** 27.48 GiB in ~23 s. After load: ~143.7 GiB free, 32 GiB reserved for KV → **337,515** cache tokens, **35.16×** concurrency at 9600 tokens/request.

#### 4.1 Decode-only (collocated, 256 output tokens, 1 trial)

Artifact: `_results/phases/phase_profile_summary_20260924_122130.json`

| Case | Prompt | Output | Decode tok/s | TPOT p50 (ms) | ITL p50 / p95 / p99 (ms) | TTFT p50 (ms) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| D1 | 128 | 256 | 26.16 | 38.22 | 28.38 / 56.70 / 87.77 | 129.59 |
| D2 | 1024 | 256 | **35.63** | 28.07 | 28.28 / 28.81 / 29.06 | 316.37 |
| D3 | 4096 | 256 | 33.38 | 29.96 | 30.15 / 30.37 / 30.52 | 755.38 |
| D4 | 8192 | 256 | 26.98 | 37.06 | 37.35 / 37.56 / 37.66 | 924.85 |

Steady decode is **~27–36 tok/s** at this output length. D1 has a fat ITL tail (p99 88 ms). D4 is slower than D2 because 8K prefill is collocated on the same engine (this is **not** P/D).

Power columns are **0 W**: the login user is not in group `render`, so host `amd-smi` telemetry failed.

#### 4.2 Prefill-only (`max_tokens=1`, 2 measured trials + warmup)

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

Prefill wall time is almost flat from 128–4096 tokens (~56–67 ms) then steps at 8K (~100 ms). That is **not** the 2.6 s 8K TTFT seen on R9700 MXFP4 in [R9700.md](R9700.md) — different quant, engine, context config, and likely cache state. Do not compute a “MI350P vs R9700 prefill speedup” from these two tables without a matched cold-cache `vllm bench serve` run.

#### 4.3 What was *not* collected for vLLM FP8

- `throughput.sh` default matrix **8192:1024, 1024:8192, 1024:1024** at C=1 and C>1  
- 1024-token decode isolation (suite default OSL; this run used 256)  
- Contention / prefix-cache suites in `bench_phases.py` (`concurrency`, `contention`, `prefix`)  
- Joules/token  

Until those land, do not claim an 8K:1K “agent” decode rate for MI350P FP8. The 26.98 tok/s D4 figure is 8192-in / **256-out**, not 1024-out.

---

### 5. vLLM × MXFP4 — measured 25 Sep (not the Radiance compose)

The 24 Sep hypothesis below assumed a missing `local/vllm-mxfp4:gfx950` image. Production serving used **stock** `vllm/vllm-openai-rocm:latest` + 39 shards of `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`. Logs confirm **`AiterMxfp4LinearKernel`**; decoder attention is **ROCM_ATTN**. Reproduce: [`MI350P.md`](MI350P.md).

Headline 1K/1K `ignore_eos` (same-process 3×): **79.35 C1 / 553.58 C8** tok/s. C64 still rising (**2017** tok/s). The 28 Sep shape sweep (8192/1024, 1024/8192, and DFlash-3 at C1–C32) is priced in [TCO.md](TCO.md). DFLASH-7 wins C1–C3 long output only (~102 tok/s C1) and loses from C6; greedy token-ID match vs the same MXFP4 target is **87.1%**, not 100%. Native MTP did not load (`Qwen3_5MTP.load_weights` shape assert).

`docker-compose.mxfp4.yml` / Radiance R4D remains the **wrong** Instinct path (`RADIANCE_USE_R4D=0` on `mi350p`). Do not mix R9700 30.59 tok/s MXFP4 figures with these MI350P numbers.

**Capacity (still true):** both FP8 and MXFP4 fit on 144 GB; MXFP4’s value here is decode bandwidth, not “will it load.” Task quality vs FP8/BF16 (GPQA/SWE-bench) is **still unmeasured** on this SKU.

```bash
./scripts/launch_vllm_mxfp4.sh
python3 scripts/bench_openai_chat.py --base-url http://127.0.0.1:8000/v1 --model awq \
  --input-len 1024 --output-len 1024 --num-prompts 4 --concurrency 1 --timeout 900
```

---

### 6. llama.cpp — FP8 / MXFP4 mapping and gaps

`docker-compose.gguf.yml` runs `llama-server` with:

- `-m /models/${MODEL_FILE}` default **`Qwen3.8-27B-Q4_K_M.gguf`**
- `-ngl 999 -fa on -c MAX_MODEL_LEN -np 1`

There is **no FP8 or MXFP4 GGUF** in `models/` on this host (the directory is empty of weight files in the workspace snapshot). Profile image `local/llama.cpp:rocm10-gfx950` is absent; the R9700 tag is `local/llama.cpp:rocm10-gfx1201` / `rocm7-gfx1201`.

| Requested cell | What llama.cpp can actually run | MI350P status |
| --- | --- | --- |
| llama.cpp × FP8 | Needs an FP8 GGUF or a CUDA/HIP FP8 backend this compose does not enable | Not measured |
| llama.cpp × MXFP4 | Needs MXFP4 GGUF + kernels | Not measured |
| llama.cpp × Q4_K_M | Supported by compose | Not measured on gfx950; R9700 only (§9) |

**CDNA-4 notes:** HIP builds must target `gfx950`. A gfx1201 binary with `HSA_OVERRIDE_GFX_VERSION` is the wrong approach (that override is what broke vLLM here when left empty, and 12.0.1 is RDNA4). Flash attention (`-fa on`) must be confirmed on Instinct; if FA is CPU-fallback, decode collapses (R9700 showed both **75 tok/s** smoke and **2.2 tok/s** “all-engine” smokes — treat llama.cpp numbers as invalid until `amd-smi metric -u` shows GFX activity during generate).

---

### 7. SGLang — FP8 / MXFP4 mapping and gaps

`docker-compose.sglang.yml` launches `python3 -m sglang.launch_server` with `--model-path` defaulting to `MODEL_NAME` (FP8 HF id). Optional `--load-format gguf` is for GGUF, not Quark MXFP4.

| Requested cell | Path | MI350P status |
| --- | --- | --- |
| SGLang × FP8 | HF `Qwen/Qwen3.8-27B-FP8`, `--tp 1` | Image missing |
| SGLang × MXFP4 | Not wired; would need Quark/SGLang MXFP4 support plus weights | Not available in this compose |

Prior R9700 `throughput.sh -e sglang` slices recorded **UNAVAILABLE** (image or tokenizer mismatch). `SGLANG_TOKENIZER_PATH` in `.env` still points at `Qwen/Qwen2.5-0.5B-Instruct`, which is the wrong tokenizer family for Qwen3.8 and will bias TTFT/tok counts if left unchanged.

---

### 8. Cross-engine analysis (what a completed MI350P matrix should decide)

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

### 9. R9700 campaign (different card — reference only)

These figures are from [R9700.md](R9700.md) and `_results/throughput/`. Hardware: **Radeon AI PRO R9700, 32 GB, 300 W, kernel 6.8.0-71**. Peak power traces cap at **300 W**. They answer “what did we already learn about engines and quants,” not “what is MI350P.”

#### 9.1 Throughput (selected)

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

#### 9.2 Accuracy (n=5 diamond / 5 SWE-Lite) — R9700

| | GPQA | SWE-bench valid patches | SWE tok/s | GPQA tok/s |
| --- | ---: | ---: | ---: | ---: |
| vLLM FP8 | 80% (4/5) | 0/5 | 12.02 | 17.02 |
| vLLM MXFP4 | 80% (4/5) | **1/5** | 19.11 | 19.17 |
| llama.cpp Q4_K_M | 60% (3/5) | 0/5 | 28.76 | 28.76 |

MXFP4 matching FP8 on this tiny GPQA sample is the strongest *quant* result in the repo. It has **not** been repeated on MI350P.

#### 9.3 Why R9700 MXFP4 looked mandatory

FP8 used **~31.6 GB / 32 GB**. CUDA graphs were turned off; KV was squeezed to 1 GB. MXFP4 dropped weights to ~19 GB and restored graphs. **That VRAM crisis does not exist on 144 GB MI350P.** Copying the R9700 “always ship MXFP4” conclusion onto Instinct without new measurements would be incorrect.

---

### 10. Reproduce — fill the empty cells on this machine

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

### 11. Conclusions

1. **Completed cells on this MI350P:** vLLM × FP8 (24 Sep, isolated 256-out phases) and **vLLM × Quark MXFP4** (25 Sep, 1K/1K control + graph/batch/DFlash). See `MI350P.md`.  
2. **llama.cpp and SGLang still have no gfx950 runtime here**; ranking them on MI350P would be fiction.  
3. **llama.cpp cannot occupy the FP8 or MXFP4 columns** without different weight files than this repo’s GGUF default.  
4. **R9700 results remain useful as a 32 GB / RDNA4 baseline** and must not be mixed with Instinct MXFP4 79/554 tok/s.  
5. Remaining decision data: matched **MXFP4 vs FP8/BF16 task quality** (MXFP4 sampling 30/32 done; FP8 exclusive not run), attention-backend at 8K, Quark MTP loader, SGLang MXFP4.

---

### 12. Artifacts

| Path | Role |
| --- | --- |
| [MI350P.md](MI350P.md) | Canonical 25 Sep MXFP4 vLLM eval |
| `_results/priority_eval/` | MXFP4 control JSON, fingerprints, graph/batch/DFlash logs |
| `_results/quality/` | Greedy token-ID quality vs DFlash |
| `_results/phases/phase_profile_summary_20260924_122130.json` | MI350P vLLM FP8 decode-only |
| `_results/phases/phase_profile_summary_20260924_122210.json` | MI350P vLLM FP8 prefill-only |
| `docker-compose.yml` | vLLM FP8 single GPU |
| `docker-compose.mxfp4.yml` | Radiance compose (R9700); **not** the MI350P MXFP4 path — use `scripts/launch_vllm_mxfp4.sh` |
| `docker-compose.gguf.yml` | llama.cpp Q4_K_M |
| `docker-compose.sglang.yml` | SGLang FP8 HF |
| `lib/gpu_profile.sh` | `mi350p` vs `r9700` image/arch defaults |
| [R9700.md](R9700.md) | R9700 accuracy, engines, and phase profiles |
| `MI350P-PD.md` | Dual-card P/D (out of scope here) |

## FP8 lane versus RTX PRO 6000

## MI350P Evaluation Report vs RTX PRO 6000

> **25 Sep 2026 — vLLM Quark MXFP4 is measured.** Canonical: [`MI350P.md`](MI350P.md). **Freeze HF control** (AITER unset, stock `M_LEQ_8`): **79.35 C1 / 553.58 C8**. Babel Read **3793 GB/s**; C1 is **40%** of that *conditional* roof — do not assign the rest to one kernel. Isolated ASM / ksplit=1 GEMM wins **regressed serving**. This 24 Sep document remains the **FP8 / competitive-lane** note.

**Status:** Active empirical campaign. Completed results are explicitly marked **measured**; future work is marked **planned** or **projected**.

**Thesis:** RTX PRO 6000 public numbers are a **feature and measurement blueprint**, not a single apples-to-apples tok/s claim. The target MI350P stack is **SGLang + ROCm/AITER + MXFP4 + native MTP**, followed by DFlash2 only if it improves median and p95.

**Our card:** 1 × [AMD Instinct MI350P PCIe](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350p.html) (`gfx950`, 144 GB HBM3E, **4 TB/s** HBM, **4.6 PFLOPS** peak MXFP4, 600 W).  
**Competitor:** 1 × NVIDIA RTX PRO 6000 Blackwell (96 GB GDDR7, ~1.8 TB/s).  
**Date:** 24 September 2026.

Primary stack (presales path): **SGLang + AITER + Quark MXFP4 + native MTP**.  
Cross-check: **vLLM ROCm** (already serving FP8 on this host). DFlash2 is a later experiment, not the first bet.

---

### Executive result

The initial 35.63 tok/s result was substantially software-bound. Enabling piecewise graph capture and AITER raised the matched 1K-context decode case to **56.79 tok/s (+59.4%)**. AITER supplied most of the gain; graph capture alone reached 40.13 tok/s.

Native Qwen MTP was then enabled successfully and showed **excellent acceptance**—96.0% at depth 1, 88.4% at depth 2, and 81.9% at depth 3—but it **reduced practical throughput** at 1K–8K context. Runtime logs identify two ROCm gaps:

1. `fused_gdn_decode_post_conv_mtp` is not built, so MTP GDN decode falls back to Triton.
2. `ROCM_ATTN` cannot perform fused multi-step draft decode, so attention metadata is rebuilt between draft steps.

This is a kernel/runtime overhead failure, **not an acceptance failure**. On the Helix-comparable 1K-input lane, depth-1 MTP delivered 46.51 tok/s versus 56.79 tok/s without speculation. Deeper drafts became progressively slower.

The production-style no-speculation sweep (1K input / 1K output) sustained **51.48 tok/s at C1**, **193.28 tok/s at C4**, and **371.15 tok/s at C8**. Mean TPOT remained tightly bounded at 18.91–20.50 ms while p99 ITL rose from 39.30 to 62.20 ms.

The AMD-qualified MI350P SGLang image has now been downloaded:

```text
rocm/sgl-dev:v0.5.15.post1-ubuntu24.04-py3.14-rocm10.0.0
sha256:13171073ebf66fee0a1473d43ae8080cc7f8500fa213ddf1507f158fbb6dd63a
```

**SGLang + MXFP4 is still unmeasured.** vLLM + Quark MXFP4 on this host **is** measured (25 Sep); see [`MI350P.md`](MI350P.md).

#### Measured scorecard

| Experiment | D1 128 ctx | D2 1K ctx | D3 4K ctx | D4 8K ctx | Verdict |
|---|---:|---:|---:|---:|---|
| FP8 baseline | 26.16 | 35.63 | 33.38 | 26.98 | Software-bound starting point |
| + piecewise graphs | 31.44 | 40.13 | 33.03 | 26.77 | Helps short context only |
| + graphs + AITER | **45.60** | **56.79** | **43.46** | **33.14** | Best decode configuration |
| + MTP depth 1 | **50.96** | 46.51 | 33.09 | 24.08 | Helps only 128-token case |
| + MTP depth 2 | 46.74 | 43.76 | 31.68 | 23.05 | Slower |
| + MTP depth 3 | 43.80 | 41.49 | 30.22 | 22.16 | Slowest |

All values are output tok/s, 256 output tokens, one measured trial per cell after server warmup.

#### MTP acceptance and cost

| Draft depth | Draft tokens | Accepted | Acceptance | Accepted / draft step | D2 tok/s | D2 delta vs no-spec |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 521 | 500 | **96.0%** | 0.96 | 46.51 | **−18.1%** |
| 2 | 742 | 656 | **88.4%** | 1.77 | 43.76 | **−22.9%** |
| 3 | 891 | 730 | **81.9%** | 2.46 | 41.49 | **−26.9%** |

Do not advertise native MTP on this vLLM/ROCm build as a speed feature. It is functionally correct and highly accepted, but the missing fused kernels erase the saved target passes.

---

### 1. Competitive baseline — separate lanes, not one number

Public RTX PRO 6000 figures mix models, quants, speculation, context, and measurement definitions. Treat each row as its own **target lane**.

| Source | Platform / software | Configuration | Reported result | AMD comparison use |
|---|---|---|---:|---|
| [HelixML](https://helix.ml/blog/chasing-454-toks-qwen38-rtx-pro-6000) | RTX PRO 6000 **Server Edition**, SGLang | Qwen3.8-27B NVFP4, DFlash2, production-oriented | **167.8 tok/s median** after `torch.compile`; 149.8 before | **First C1 target** for a production MI350P profile |
| HelixML | RTX 6000 **workstation** SKU | NVFP4 W4A4, DFlash2 block 16, FP8 KV, +6000 mem clock | Claimed **454 tok/s** peak; Helix estimates **335** best cell without workstation-only factors | Do **not** position against 454 as a standard SKU claim |
| [NVIDIA forum Flash-Next](https://forums.developer.nvidia.com/t/optimized-qwen3-8-flash-next-on-1x-rtx-pro-6000-171-tok-s-524k-and-hicache-nixl-persistence/381722) | RTX PRO 6000 96 GB, custom SGLang | Qwen3.8 **Flash-Next** NVFP4, MTP, FP8 KV, 524K YaRN | **171.09 tok/s C1**; **427.54 tok/s C4** aggregate | Long-context hybrid / MTP reference; **distinct model** from dense 27B |
| NVIDIA forum, online FP8 | same | Flash-Next, online FP8, native MTP | 207.12 short C1; **195.63 @128K**; **172.64 @~490K** | MI350P must publish a **decode-vs-context curve**, not empty-KV only |
| [HF discussion #101](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/101) | RTX PRO 6000, Memra | NVFP4 + Q5_K, speculative | 260 tok/s spec p50, 0.156 s TTFT, native 262K | Community claim; use only after recipe/prompt/acceptance are disclosed |
| [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) | RTX 5090 / Blackwell examples | NVFP4, FP8 KV, native MTP | Flags and observability, not a PRO 6000 score | Reproducibility reference for MTP metrics |
| Reddit LocalLLaMA | vLLM, BF16/FP8, multimodal | Extracted command incomplete | — | **Do not use** until raw output and command are verified |

HelixML’s analysis is the credibility filter: Server Edition vBIOS reports memory-clock offset range **`[0, 0]`**, and their DFlash2 path could not use a fully quantized W4A4 `lm_head`. Their best cell was 335 tok/s; 454 ≈ 335 × ~1.1 (smaller W4A4) × ~1.2 (workstation OC). [HelixML](https://helix.ml/blog/chasing-454-toks-qwen38-rtx-pro-6000)

The NVIDIA forum’s **62,040.60 tok/s** is **not cold prefill**. It is restored-prefix throughput after restart (~489,856 of 489,879 tokens restored). Track it as **cache rehydration**, never vs cold prefill. [NVIDIA forum](https://forums.developer.nvidia.com/t/optimized-qwen3-8-flash-next-on-1x-rtx-pro-6000-171-tok-s-524k-and-hicache-nixl-persistence/381722)

#### Four lanes we will publish

| Lane | Required workload | Primary metric | Competitive target |
|---|---|---|---|
| Interactive short-context | ISL 1K, OSL 1K, C=1 | C1 tok/s / TPOT, TTFT | **167.8 tok/s** Helix production median; 260 p50 only if fully reproduced |
| Saturated decode | ISL 1K, OSL 1K, C=1,4,8,16 | Aggregate tok/s, p95 ITL | 427.54 C4 is Flash-Next, **not** dense-27B parity |
| Long-context decode | KV depth 128K and ~490K, OSL 1K, C=1 | tok/s at each depth and drop from short | 195.63 @128K, 172.64 @~490K (Flash-Next online FP8) |
| Long-context prefill | Cold 64K and 262K; optional 524K YaRN | Input tok/s, TTFT | 10,103.70 @64K, 7,872.15 @~490K (custom Flash-Next) |

---

### 2. Where this host actually is (vLLM FP8, not the target stack)

This section is **vLLM 0.30 ROCm × Qwen3.8-27B-FP8** (24 Sep), GPU 0, `max-model-len=9600`. The AMD ROCm 10 SGLang 0.5.15 image is local but has not served the model. Quark MXFP4 later served on **stock** `vllm/vllm-openai-rocm:latest` (no `local/vllm-mxfp4:gfx950` image required); numbers are in [`MI350P.md`](MI350P.md).

Dense FP8 weights ≈ **30.5 GB per decode step**. Unspeculated roofline at 4 TB/s is ~131 tok/s; at RTX’s 1.8 TB/s ~59 tok/s. NVIDIA’s 167–260 therefore **cannot** be dense-FP8 bandwidth — it is speculation + 4-bit weights.

#### Knob attribution (identical ISL/OSL, 256 output tokens, 24 Sep)

| Case | Baseline | CUDA graphs only | Graphs + AITER | Graphs | AITER extra |
| --- | ---: | ---: | ---: | ---: | ---: |
| D1 128 ctx | 26.16 | 31.44 | **45.60** | +20% | +45% |
| D2 1024 ctx | 35.63 | 40.13 | **56.79** | +13% | +42% |
| D3 4096 ctx | 33.38 | 33.03 | **43.46** | −1% | +32% |
| D4 8192 ctx | 26.98 | 26.77 | **33.14** | −1% | +24% |

AITER dominates decode; graphs only help short context. AITER **regressed** 8K TTFT (923 → 1466 ms) because every a8w8 GEMM shape logged **untuned default** (`a8w8_blockscale_tuned_gemm.csv` miss). Keep AITER on for decode; treat prefill regression as a GEMM-tune item.

Effective weight-read bandwidth at D2: **1.09 → 1.73 TB/s** (still ~43% of 4 TB/s peak).

#### Production-style 1K / 1K concurrency sweep

Unlike the 256-output phase probes, this uses `vllm bench serve` with 1K requested input and 1K output:

| C | Prompts | Output tok/s | Mean TPOT | Mean / p99 ITL | Mean / median / p99 TTFT |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | **51.48** | 18.91 ms | 19.82 / 39.30 ms | 543.71 / 444.47 / 808.54 ms |
| 4 | 20 | **193.28** | 19.85 ms | 20.95 / 60.21 ms | 881.56 / 1114.86 / 1263.13 ms |
| 8 | 24 | **371.15** | 20.50 ms | 21.36 / 62.20 ms | 1100.22 / 1295.34 / 1649.31 ms |

Artifacts: `_results/mi350p_eval/mi350p_fp8_1024_1024_c{1,4,8}.json`.

The C1 number is the most comparable measured MI350P result to Helix’s production lane, but it is still **FP8, no speculation**, versus NVFP4 + DFlash2. The remaining C1 gap to 167.8 tok/s is 3.26×.

At aggregate load, MI350P FP8 already reaches **371.15 tok/s at C8** while holding mean TPOT to 20.50 ms. The NVIDIA Flash-Next reference is **427.54 tok/s at C4**, but it uses a different model, NVFP4, and MTP; this is an upper reference, not parity. Our C4 result is 193.28 tok/s.

#### Evaluation run ledger

| Run | Configuration | Outcome |
|---|---|---|
| Bring-up | Stock vLLM FP8 | Failed initially on vision weights, empty `HSA_OVERRIDE_GFX_VERSION`, and graph config quoting |
| Baseline | FP8, ROCM_ATTN, graphs off | Healthy; D2 35.63 tok/s |
| Graph isolation | FP8, piecewise graphs, AITER off | D2 40.13 (+12.6%); no gain at 4K/8K |
| AITER isolation | FP8, piecewise graphs, AITER on | D2 56.79; best decode; prefill TTFT regressed |
| MTP-3 | AITER + graphs + MTP depth 3 | 81.9% acceptance; D2 41.49 (−26.9%) |
| MTP-1 | AITER + graphs + MTP depth 1 | 96.0% acceptance; D2 46.51 (−18.1%); D1 improved |
| MTP-2 | AITER + graphs + MTP depth 2 | 88.4% acceptance; D2 43.76 (−22.9%) |
| Production C1 | AITER + graphs, 1K/1K, five prompts | 51.48 output tok/s; 18.91 ms mean TPOT |
| C4/C8 | Same 1K/1K workload | **Complete:** 193.28 / 371.15 output tok/s |
| SGLang image | Pull AMD ROCm 10 SGLang 0.5.15 | **Complete**, digest recorded above |
| SGLang serve | FP8 then MXFP4 | Not started |
| MXFP4 (vLLM) | Quark AWQ MXFP4 + `AiterMxfp4LinearKernel` | **Complete 25 Sep:** 79.35 C1 / 553.58 C8; see `MI350P.md` |

#### Reproducibility limitations

- Phase cells use **one trial**, so they are directional tuning data, not publication statistics.
- Power is 0 W in JSON because host user `amd` is not in `render`; energy/token is unavailable.
- Prefix caching is enabled and synthetic prompts repeat; prefill `prompt_tok_s` is not a cold-prefill FLOP rate.
- Current max context is 9,600; 128K/262K lanes remain unmeasured.
- The vLLM target reports `ROCM_ATTN` and uses AITER IR/GEMM operations; “AITER on” is not equivalent to all-attention kernels selecting an AITER backend.

#### Native MTP loader behavior

The FP8 checkpoint contains **22 `mtp.*` tensors**. The target-model loader intentionally ignores them:

```290:292:patches/qwen3_5_vllm030.py
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=["visual.", "model.visual.", "mtp."],
        )
```

This is **not a repo bug**. When speculative decoding is enabled, vLLM instantiates a separate `Qwen3_5MTP` draft model and loads the `mtp.*` tensors through `qwen3_5_mtp.py`; the target must continue to map them out. Removing the target skip would reintroduce an unexpected-weight failure.

Native MTP was enabled with:

```bash
--speculative-config='{"method":"mtp","num_speculative_tokens":N}'
```

Depths 1, 2, and 3 all loaded and emitted non-zero `spec_decode_num_{draft,accepted}_tokens_total` metrics. The negative performance result is caused by unfused ROCm execution, not failed loading.

---

### 3. Transferable optimizations (NVIDIA → MI350P)

#### 3.1 Native MTP first, DFlash2 second — vLLM result complete

[vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B):

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The metric that matters is **accepted draft length / acceptance ratio**, not a higher tok/s banner. We read `vllm:spec_decode_num_{accepted,draft}_tokens_total`; all three draft depths were genuinely active.

AMD already showed SGLang MTP/NextN on Instinct (DeepSeek V3: 1.25–2.11× random, 1.36–1.80× ShareGPT; **gain falls with concurrency**). That is not a Qwen3.8 promise, but it justifies **C1 vs aggregate as separate lanes**. [ROCm MTP blog](https://rocm.blogs.amd.com/software-tools-optimization/mtp/README.html)

**vLLM verdict:** native MTP is not competitive in this ROCm 10 / vLLM 0.30 build despite high acceptance. Keep it disabled for production until `fused_gdn_decode_post_conv_mtp` and fused ROCm multi-step draft attention are available.

**SGLang priority remains valid:** AMD’s NextN path is a different implementation and should be tested in the newly downloaded MI350-qualified SGLang image. Native MTP still precedes DFlash2 there because it uses the in-checkpoint head and gives the cleanest matched comparison.

HelixML: DFlash2 block 12/16 lifted a **best-case code** cell 214 → 332 tok/s, but **median barely moved** and prose got slower. Use DFlash2 only if it improves median and p95. Recipe (vLLM ≥ 0.28):

```bash
--speculative-config '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'
```

| Experiment | Draft config | Sweep | Decision |
|---|---|---|---|
| Baseline | None | — | C1 TPOT, C4 aggregate, quality |
| Native MTP (vLLM) | Measured 1, 2, 3 | 4/5 cancelled after monotonic slowdown | Disabled pending fused ROCm kernels |
| Native MTP (SGLang NextN) | 3 tokens | 2, 3, 4, 5 | Highest quality-preserving C1 and long-ctx gain |
| DFlash2 | 7 tokens initially | Block/draft range supported on ROCm | **Median**, not best prompt |
| Acceptance | MTP and DFlash2 | Code, prose, tools, reasoning, multimodal | Reject large class regressions |
| Concurrency | Winner | C1–C16 | Quantify speculative fade as batch grows |

Presales report must include: mean accepted tokens/step, acceptance %, C1 tok/s and TPOT, C4/C8 aggregate, p50/p95/p99 TTFT and ITL, quality vs FP8/BF16, **by prompt class**.

#### 3.2 MXFP4 as the hardware advantage

NVIDIA path is **NVFP4**. MI350P path is **OCP MXFP4 via AMD Quark**. Do not load NVFP4 artifacts on Instinct. Confirm logs show **native MXFP4 GEMMs**, not dequant-to-BF16 fallback.

Lessons from NVIDIA, not “use NVFP4”:

- Compress weights so decode is not HBM-capacity bound.
- Keep `lm_head`, embeddings, norms, attention-sensitive ops, vision in higher precision **if** needed for speculation/accuracy.
- Measure whether a quantized `lm_head` breaks the drafter (HelixML’s W4A4 blocker).
- Compare pure MXFP4 vs mixed W4A8 / selected BF16 layers. AMD’s 2.4T Qwen3.8 MXFP4 recipe leaves `lm_head`, embeddings, attention, linear-attention, and some MTP parts in BF16 while quantizing expensive MoE experts — instructive even though 27B is dense. [AMD Day-0 Qwen3.8](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-8-on-amd-instinct-gpus.html)

vLLM’s NVIDIA recipe: **MXFP4 does not load on NVIDIA**; they use NVFP4 instead. That asymmetry is ours to spend. [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)

#### 3.3 FP8 KV as capacity, not speed

HelixML: FP8 KV ~doubled cache (644K tok/GPU) but **no short-context speedup** and **~9% loss at 82K**, blamed on draft acceptance, not “FP8 is slow.” [HelixML](https://helix.ml/blog/chasing-454-toks-qwen38-rtx-pro-6000)

NVIDIA Flash-Next: 524,288 context, **824,384** GPU KV tokens with FP8 target/MTP KV.

| KV mode | Use when | Risk |
|---|---|---|
| BF16 KV | Quality and baseline decode | Less capacity |
| FP8 KV | Density, 262K–524K, high C | Acceptance, long-ctx quality, TPOT |
| Mixed / tiered | Product SKUs | Operational complexity |

On **144 GB**, BF16 KV at 262K native may fit where a 96 GB card needs FP8. That is a competitive story **if measured**. This host currently pins `MAX_MODEL_LEN=9600` in `.env` — that is a software cap, not HBM.

R9700 history: FP8 KV produced non-power-of-2 attention pages (`attn_block_size=800`) and Triton fallback. This MI350P run already uses **block size 784** to match mamba pages. A/B prefix-cache on/off and page-size 1 vs 16 before claiming FP8 KV.

#### 3.4 Graph / compile → ROCm equivalents

HelixML’s practical win: `--enable-torch-compile` **149.8 → 167.8 tok/s median (+12%)**, +10–15% at long context; ~8 min cold compile; graph capture needed `--mem-fraction-static 0.85`. [HelixML](https://helix.ml/blog/chasing-454-toks-qwen38-rtx-pro-6000)

MI350P experiment:

- ROCm SGLang image matched to CDNA 4 / MI350.
- `--attention-backend aiter`; confirm fused paths in logs.
- Eager vs decode-only graphs vs piecewise/full.
- Cache compile artifacts across restarts.
- Sweep `mem-fraction-static` (0.80 / 0.85 / 0.90), not “max it.”
- Text-only vs multimodal graph shapes separately.
- Report **startup compile cost** and **steady-state** tok/s.

AMD SGLang Qwen3.8 MXFP4 reference uses `--attention-backend aiter`, `--page-size 1`, `--chunked-prefill-size 16384`, `--mem-fraction-static 0.9` — published for **MI355X TP=8 2.4T MoE**. Start from those *knobs*, not that topology. MI350P has half the HBM of MI355X. [AMD Day-0](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-8-on-amd-instinct-gpus.html)

#### 3.5 Hybrid attention, not generic Transformer KV

Qwen3.8-27B: 64 layers, **16 full attention**, 48 linear/GDN, vision tower, MTP, **262,144 native**, YaRN to 1M. [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)

- Full-attention layers dominate KV pressure.
- Linear/GDN/Mamba state must be in cache and persistence tests.
- Sample performance **over cache depth**.
- Prefix reuse is **not KV-only**. NVIDIA persistence stores packed target/MTP KV, **GDN, PLE, compressed QSA**. [NVIDIA forum](https://forums.developer.nvidia.com/t/optimized-qwen3-8-flash-next-on-1x-rtx-pro-6000-171-tok-s-524k-and-hicache-nixl-persistence/381722)

#### 3.6 Prefix cache vs persistent restore

| Capability | Immediate MI350P action | Competitive value |
|---|---|---|
| In-memory radix / prefix cache | Enable; agentic repeated prefixes | TTFT and amortized input cost |
| Host-memory cache tier | ROCm hierarchical cache if available | Prefix set beyond HBM |
| NVMe persistence | After hybrid-state correctness | Agentic/RAG narrative, **not** a C1 speed claim |
| Restart restoration | Restored-prefix rate, exactness, post-restart TTFT | Functional compete with HiCache/NIXL |

**Labels only:** cold prefill tok/s · warm prefix-hit TTFT · **effective cache restore throughput**. Never mix them.

#### 3.7 What not to copy

| Their finding | Why not |
|---|---|
| +6000 memory OC | Server Edition offset range `[0,0]`; we have 600 W / clocks, different mechanism |
| FP8 KV for speed | Capacity feature; ~9% loss at 82K in Helix config |
| NVFP4 checkpoints | Instinct uses MXFP4 |
| Relaxed acceptance | ~4% Helix gain, real quality cost |
| 454 tok/s | Best cell × W4A4 × workstation OC vs ~149–168 median |
| 62k tok/s restore | Not cold prefill |
| Flash-Next 171 / 427 | Different model than dense 27B |

---

### 4. Test matrix

#### Controls

| Control | Requirement |
|---|---|
| Model | Qwen3.8-27B **text path first**; multimodal separate |
| Precision baseline | Official FP8 (this host) then BF16 if needed |
| Competitive precision | RTX: NVFP4; MI350P: **Quark MXFP4** |
| KV | BF16 and FP8 |
| Engines | **SGLang primary**; vLLM ROCm cross-check |
| Speculation | None, native MTP, DFlash2 **if** ROCm supports this arch/checkpoint |
| Context | 1K, 8K, 32K, 128K, 262K; optional 524K YaRN |
| Output | 1K decode lane; 128/512/1K service sweeps |
| Concurrency | 1, 2, 4, 8, 16; 32/64 for saturation |
| Generation | Greedy microbench; production sampling separately |
| Repeats | ≥5 steady-state samples; **median and spread** |
| Warmup | Explicit compile/graph stabilization |
| Power | Card limit; log clocks/thermals (host `render` group required) |
| Host | CPU, NUMA, RAM, PCIe domains, OS, driver, **container digest** |

This host: EPYC 9015, two MI350P on **PCI domains 0000 and 0001**, GPU 0 = `8b:00.0`. Single-card tests use `HIP_VISIBLE_DEVICES=0` only.

#### Required cases

| ID | Purpose | ISL / OSL | Context | C |
|---|---|---:|---|---:|
| D1 | Empty-context interactive | 1K / 1K | Cold | 1 |
| D2 | Agentic code/prose | 4K / 1K | Cold | 1 |
| D3 | Mid-context decode | 128K / 1K | Preloaded | 1 |
| D4 | Deep-context decode | 262K / 1K | Preloaded | 1 |
| D5 | YaRN extension | ~490K / 1K | Preloaded | 1 |
| T1 | Aggregate scaling | 1K / 1K | Cold | 1,2,4,8,16 |
| P1 | Cold prefill | 64K / 1 | Cold | 1 |
| P2 | Native max prefill | 262K / 1 | Cold | 1 |
| P3 | Extended prefill | ~490K / 1 | Cold | 1 |
| C1 | Prefix cache | Repeated 32K–128K prefix | Cold then warm | 1,4,8 |
| C2 | Cache restoration | Persist / restart / replay | Restored | 1 |
| M1 | Multimodal | Image + 1K text / 512 | Cold | 1,4 |

---

### 5. Reproducible NVIDIA baseline (capture sheet, not a ROCm claim)

```bash
export MODEL=Inferact/Qwen3.8-27B-NVFP4
export MAX_LEN=262144

vllm serve "${MODEL}" \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_LEN}" \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

[vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)

DFlash2 lane:

```bash
export MODEL=Qwen/Qwen3.8-27B
export DRAFTER=incoai/Qwen3.8-27B-DFlash2

vllm serve "${MODEL}" \
  --tensor-parallel-size 1 --max-model-len 262144 \
  --kv-cache-dtype fp8 --reasoning-parser qwen3 \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":7}"
```

Helix-style SGLang capture must record: workstation vs **Server Edition**, driver/CUDA/SGLang/FlashInfer + **digest**, NVFP4 checkpoint + `lm_head` dense vs quantized, DFlash2 block 8–16, BF16 vs FP8 target/draft KV, `torch.compile`, graph + mem fraction, power limit + NVML clock-offset range, prompt class, acceptance **by class**.

---

### 6. MI350P recipes (this repo)

#### 6.1 SGLang MXFP4 baseline (primary vehicle)

`docker-compose.sglang.yml` today is R9700-shaped (`coder-sglang:latest`, no AITER flags, `SGLANG_TOOL_PARSER=qwen25`). The qualified CDNA 4 image is now downloaded but not yet wired into compose. Qwen3 parsers are still required. Conceptual launch (validate flags in the container):

```bash
export HIP_VISIBLE_DEVICES=0
export ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
# never export empty HSA_OVERRIDE_GFX_VERSION on gfx950

python3 -m sglang.launch_server \
  --model-path /models/Qwen3.8-27B-Quark-MXFP4 \
  --served-model-name Qwen3.8-27B-MXFP4 \
  --host 0.0.0.0 --port 30000 --tp-size 1 \
  --attention-backend aiter \
  --kv-cache-dtype auto \
  --page-size 1 \
  --chunked-prefill-size 16384 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --enable-metrics
```

Knobs from AMD’s published SGLang MXFP4 reference; **do not copy MI355X TP=8 or 0.9 mem fraction blindly**. [AMD Day-0](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-8-on-amd-instinct-gpus.html)

#### 6.2 Native MTP on SGLang (AMD NextN pattern)

```bash
--speculative-algorithm NEXTN \
--speculative-num-steps 2 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 3
```

[ROCm MTP](https://rocm.blogs.amd.com/software-tools-optimization/mtp/README.html)

Staged sweep (do not factorial all at once):

```text
speculative-num-draft-tokens = 2, 3, 4, 5
chunked-prefill-size         = 4096, 8192, 16384, 32768
mem-fraction-static          = 0.80, 0.85, 0.90
kv-cache-dtype               = bf16, fp8
page-size                    = 1, 16
```

1. Non-speculative BF16-KV and FP8-KV baselines.  
2. MTP depth at C1 for short, 128K, 262K.  
3. One MTP depth × concurrency.  
4. Prefill chunk independently.  
5. Graph/compile modes.  
6. **DFlash2 only after MTP is stable.**

#### 6.3 vLLM ROCm cross-check (measured on this host)

```bash
export HIP_VISIBLE_DEVICES=0
export VLLM_ROCM_USE_AITER=1
export GPU_ARCHS=gfx950
# piecewise graphs; do not leave cudagraph_mode NONE

vllm serve Qwen/Qwen3.8-27B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Add MXFP4 (`/models/...Quark-MXFP4`) only after the weights and gfx950 runtime exist. The target overlay should **continue ignoring `mtp.`** because the separate `Qwen3_5MTP` draft loader owns those weights. Parsers today are still `--tool-call-parser hermes` in `docker-compose.yml` — switch before accuracy or agent benches.

---

### 7. How MI350P can win

**Defendable claim (not “beat 454”):**

> A single MI350P runs Qwen3.8-27B in hardware-native MXFP4 with 144 GB HBM, sustained long-context capacity, native speculative decoding, and reproducible **median/p95** latency and throughput that exceed an RTX PRO 6000 **Server Edition** under matched model, quality, context, and power.

Convert 144 vs 96 GB into:

- BF16 KV where NVIDIA needs FP8 KV for the same native 262K / concurrency  
- More simultaneous 262K sessions  
- Larger warm-prefix set  
- Quality on tools / multimodal / long reasoning  
- No workstation-only memory OC  
- Enterprise PCIe: ECC HBM3E, RAS, SR-IOV, fixed envelope. [MI350P](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350p.html)

**Quality-per-dollar lanes:** MXFP4 + BF16 KV (quality) · MXFP4 + FP8 KV (density) · YaRN 524K labeled **extended** · agentic cache (hit rate, warm TTFT, restore). Enable static YaRN only when long prompts are real. [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)

If C1 does not beat NVIDIA’s **best cell**, still win on C4–C16 aggregate, concurrent 262K sessions at a p95 ITL SLO, mixed-workload tokens, HBM-resident cache, cost per sustained token, rack density. Do not compare a greedy short-code max to an enterprise median.

---

### 8. Acceptance criteria

| Category | Minimum |
|---|---|
| Functional | Text, image, tools, reasoning parser, long-context complete reliably |
| Precision | MXFP4 logs show native kernels; no silent BF16/FP8 fallback on critical GEMMs |
| Quality | Predeclared tolerance vs BF16/FP8 across code, reasoning, tools, multimodal |
| MTP | Acceptance telemetry shows real draft use — not an enabled flag |
| C1 | **Median** C1 tok/s exceeds the **matched** RTX PRO 6000 Server Edition baseline |
| Long-context | 128K and 262K decode with quality and p95 ITL, not only short C1 |
| Throughput | Aggregate tok/s and p95 at C1/C4/C8/C16 |
| Capacity | Concurrent 262K sessions for BF16-KV and FP8-KV |
| Reproducibility | Digest, commits, hashes, commands, seeds, topology, power, clocks, raw JSON/CSV |
| Honesty | Cold prefill, warm prefix hit, restored-prefix are **three metrics** |

---

### 9. Sequencing

1. Reproduce public NVIDIA lanes on RTX PRO 6000 with **fixed** workloads (vLLM MTP, then Helix DFlash2).  
2. MI350P correctness: FP8 (or BF16), BF16 KV, **no speculation**, text-only. *(FP8 vLLM already running; raise max-model-len.)*  
3. Native MTP vLLM sweep. **Complete:** high acceptance, negative throughput due unfused ROCm paths. Repeat on SGLang NextN.  
4. Quark MXFP4 + native kernels. **vLLM path complete 25 Sep** (`MI350P.md`). SGLang MXFP4 still pending.  
5. KV precision sweep (memory vs acceptance).  
6. Hybrid long context at 128K / 262K **before** YaRN.  
7. DFlash2: measured on vLLM MXFP4 — C1–C3 long-output only; C6+ use non-spec. Not lossless vs greedy target (87.1% token-ID).  
8. Graphs/compile and static HBM after model/draft/KV are stable. *(O2 graphs measured on MXFP4; eager is 0.18×.)*  
9. Prefix cache and optional persistence as **separate** agentic capabilities.  
10. Publish matched report: peak · median · long-context · aggregate.

Do not relabel 24 Sep **vLLM FP8** cells as the MXFP4 campaign. Native MTP on Quark MXFP4 is a **loader** blocker, not a perf result.

---

### 10. Immediate repo actions

1. Wire `rocm/sgl-dev:v0.5.15.post1-ubuntu24.04-py3.14-rocm10.0.0` into a MI350P-specific SGLang compose profile.  
2. Serve FP8 in SGLang first; verify AITER, graphs, parsers, and NextN acceptance before introducing MXFP4.  
3. Quark Qwen3.8-27B MXFP4 on vLLM: **done** (`AiterMxfp4LinearKernel`). Remaining: SGLang MXFP4, attention-backend 8K, MTP loader.  
4. Keep vLLM native MTP disabled for production until fused GDN MTP and multi-step ROCm attention exist.  
5. `--reasoning-parser qwen3`, `--tool-call-parser qwen3_xml`; drop hermes.  
6. Un-pin `.env` `MAX_MODEL_LEN=9600`; use 128K then 262144 for long-context lanes.  
7. Tune AITER a8w8 shapes (decode M=1..24, prefill M=2048+).  
8. `usermod -aG render amd` for host power/Joule metrics.  
9. Add C2/C16 and ≥5 full sweep repetitions before publishing; C1/C4/C8 first-pass data is complete.

---

### 11. References

- [Chasing a 454 tok/s tweet — HelixML](https://helix.ml/blog/chasing-454-toks-qwen38-rtx-pro-6000)  
- [Flash-Next on 1× RTX PRO 6000 — NVIDIA Forums](https://forums.developer.nvidia.com/t/optimized-qwen3-8-flash-next-on-1x-rtx-pro-6000-171-tok-s-524k-and-hicache-nixl-persistence/381722)  
- [HF Qwen3.8-27B discussion #101](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/101)  
- [vLLM recipe Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)  
- [Instinct MI350P](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350p.html)  
- [AMD Day-0 Qwen3.8 on Instinct](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-8-on-amd-instinct-gpus.html)  
- [ROCm SGLang MTP / NextN](https://rocm.blogs.amd.com/software-tools-optimization/mtp/README.html)  
- Local: `MI350P-BRINGUP.md`, `MI350P-PD.md`, `_results/phases/`, `_results/phases_tuned/`, `_results/phases_graphs_only/`

