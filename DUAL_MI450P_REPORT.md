# Dual Instinct MI350P PCIe — Phase Isolation and P/D Report

**Filename:** `DUAL_MI450P_REPORT.md` (requested). The cards on this host are **AMD Instinct MI350P PCIe**, not an MI450 SKU.

| Field | Value |
| --- | --- |
| Date | 24 September 2026 |
| Host | Linux 6.8.0-138-generic, AMD EPYC 9015 (8 cores) |
| GPUs | 2 × AMD Instinct MI350P PCIe, Device ID `0x75a8`, ISA `gfx950` |
| PCI BDF | GPU 0 `0000:8b:00.0` (KFD node 2) · GPU 1 `0001:c7:00.0` (KFD node 3) |
| Power cap | 600 W per GPU (idle ~94–100 W in `rocm-smi`) |
| Model | `Qwen/Qwen3.8-27B-FP8` (27.48 GiB weights) |
| Runtime | `vllm/vllm-openai-rocm:latest` · vLLM **0.30.0** · in-image ROCm 7.2.3 userspace / host KFD 6.19.14 |
| Profile | `GPU_PROFILE=mi350p` · `PYTORCH_ROCM_ARCH=gfx950` · `GPU_ARCHS=gfx950` |
| Context | `--max-model-len 9600` · `--kv-cache-memory-bytes 34359738368` (32 GiB reserved) |

This report covers the **decode-only**, **prefill-only**, and **dual-GPU prefill/decode** runs executed on this machine. It is not an R9700 (`gfx1201`) result set.

---

## 1. Executive summary

1. **Homogeneous dual-card preflight passed.** `inspect_dual_gpu.py --check-preflight` reports two matching MI350P devices once detection uses `rocm-smi` / `lspci` (host `rocminfo` fails for the unprivileged user: `/dev/kfd` is `root:render`).
2. **Collocated serving on GPU 0 is healthy** after weight-load and HSA bring-up fixes. Isolated decode at 256 output tokens peaked at **35.63 tok/s** (D2, 1024-token prompt). Isolated 8K prefill first-token latency was **~103 ms** (P4).
3. **True KV-disaggregated P/D did not complete.** Prefill on `:8100` and decode on `:8200` both served the model. The live router on `:8000` received HTTP 200 from prefill but **no `prompt_token_ids`**, so it could not hand KV to decode. Stock vLLM 0.30 in this image does not expose the remote-decode KV connector path the router expects.
4. **Split-phase occupancy on two cards did run:** GPU 0 prefill suite and GPU 1 decode suite in parallel. GPU 1 decode started cold (D1 **1.69 tok/s**) and recovered to **16.56 tok/s** at D4 versus **26.98 tok/s** on collocated GPU 0.
5. **P5/P6 (12K/16K prefill) are blocked** by the configured 9600-token context, not by HBM capacity. The engine reported ~337k KV tokens / **35.16×** concurrency at 9600 once 32 GiB KV was reserved.

---

## 2. Hardware and software topology

```
  EPYC 9015
       │
       ├── PCI domain 0000 : GPU 0  MI350P  8b:00.0  HIP=0  (collocated, then PD-prefill)
       └── PCI domain 0001 : GPU 1  MI350P  c7:00.0  HIP=1  (PD-decode)
```

The two cards sit on **different PCI domains**. P2P KV copies, if enabled later, must be measured rather than assumed.

| Component | Collocated (phase benches) | Dual P/D |
| --- | --- | --- |
| Compose | `docker-compose.yml` | `docker-compose.pd.vllm.yml` |
| GPU 0 | `HIP_VISIBLE_DEVICES=0`, port **8000** | port **8100**, prefill-biased batch (`--max-num-batched-tokens 8192`, `--max-num-seqs 8`) |
| GPU 1 | unused | port **8200**, decode-biased batch (`2048` batched tokens, `--max-num-seqs 16`) |
| Router | n/a | `benchmark/pd_router.py` on **8000**, live mode |
| Overlay | `patches/qwen3_5_vllm030.py` → in-image `qwen3_5.py` (Python 3.12) | same |

MXFP4 (`docker-compose.pd.yml` / `local/vllm-mxfp4:gfx950`) was **not** used: that image is not present on the host. All numbers below are **FP8 vLLM 0.30**.

---

## 3. Bring-up issues that blocked serving (and the fixes)

These are required to reproduce the numbers, not optional polish.

| Symptom | Cause | Fix used on this host |
| --- | --- | --- |
| `ValueError: no module or parameter named 'visual'` | Checkpoint has vision tensors; `Qwen3_5ForCausalLM` does not | Overlay `patches/qwen3_5_vllm030.py` with `ignore_unexpected_prefixes=["visual.", "model.visual.", "mtp."]` (vLLM 0.30 API; `skip_prefixes` does not exist) |
| Overlay had no effect / `ImportError: _is_shared_expert_fse_compatible` | First bind-mount targeted Python 3.14; full `qwen3_5_rocm10.py` is a newer tree than 0.30 | Mount the **0.30-derived** patch at `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py` |
| `rocminfo` exit 8 / `HSA_STATUS_ERROR_OUT_OF_RESOURCES` inside the container | Compose exported **empty** `HSA_OVERRIDE_GFX_VERSION=` on MI350P; empty override breaks HSA | Omit the variable on gfx950 (`unset` in `lib/gpu_profile.sh` for `mi350p`) |
| AITER `Get GPU arch from rocminfo failed` | `get_gfx_runtime()` always shells out to `rocminfo` unless architecture is explicit | Set `GPU_ARCHS=gfx950` (compose now mirrors `PYTORCH_ROCM_ARCH`) |
| Decode engine: `No CUDA GPUs are available` on HIP=1 | `HIP_VISIBLE_DEVICES=1` **and** `ROCR_VISIBLE_DEVICES=1` double-filtered the device list | HIP only; do not set ROCR to the same index |
| Dual-card preflight counted 0 GPUs | `rocminfo` requires `render` on `/dev/kfd`; user `amd` is not in that group | `gpu_profile.py` falls back to `rocm-smi` / `lspci` (`1002:75a8`) |
| Compilation JSON parse error | Unquoted `{cudagraph_mode:...}` eaten by Compose YAML | `--compilation-config='{"cudagraph_mode":"NONE"}'` |

Host user groups: `amd` is in `docker` but **not** `render` (GID 993). Docker `group_add` 44/993 is enough for the containers. Host `amd-smi` / energy collectors from the login user still see permission denied; **all power fields in the JSON are 0.0 W**.

---

## 4. Decode-only (collocated GPU 0, port 8000)

**Command:** `benchmark/bench_phases.py --mode decode --trials 1 --output-tokens 256 --model Qwen/Qwen3.8-27B-FP8`  
**Artifact:** `_results/phases/phase_profile_summary_20260924_122130.json`

Max tokens per request is 256 (not the suite’s printed “1024” banner). One trial per case.

| Case | Prompt | Output | Decode tok/s | TPOT p50 (ms) | ITL p50 / p95 / p99 (ms) | TTFT p50 (ms) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| D1 | 128 | 256 | **26.16** | 38.22 | 28.38 / 56.70 / 87.77 | 129.59 |
| D2 | 1024 | 256 | **35.63** | 28.07 | 28.28 / 28.81 / 29.06 | 316.37 |
| D3 | 4096 | 256 | **33.38** | 29.96 | 30.15 / 30.37 / 30.52 | 755.38 |
| D4 | 8192 | 256 | **26.98** | 37.06 | 37.35 / 37.56 / 37.66 | 924.85 |

**Read-through:** Steady-state decode on a warm GPU 0 is about **27–36 tok/s** at 256-token generations. Short-context D1 has a heavier ITL tail (p99 88 ms). Long-context D4 pays ~37 ms/token and ~925 ms TTFT for the 8K prefill that precedes decode. These are **collocated** numbers (prefill + decode on the same engine).

---

## 5. Prefill-only (collocated GPU 0, port 8000)

**Command:** `benchmark/bench_phases.py --mode prefill --prefill-trials 2`  
**Artifact:** `_results/phases/phase_profile_summary_20260924_122210.json`

Each case uses `max_tokens=1`. Engine-reported prefill time (`engine_prefill_s`) is the trustworthy capacity signal. Client `prompt_tok_s` is **not** a clean FLOP rate: prefix caching plus a repeating synthetic prompt make P2–P4 tok/s look unrealistically high.

| Case | Input | Status | Engine prefill (s) | TTFT p50 (ms) | TTFT p95 (ms) | Client prompt tok/s (do not treat as peak FLOPs) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| P1 | 128 | PASS | 0.056 | 58.21 | 59.63 | 2198.9 |
| P2 | 1024 | PASS | 0.065 | 67.37 | 67.73 | 15200.3 |
| P3 | 4096 | PASS | 0.063 | 66.04 | 67.10 | 62024.7 |
| P4 | 8192 | PASS | 0.097 | **102.55** | 103.15 | 79882.5 |
| P5 | 12288 | BLOCKED (HTTP 400) | — | — | — | max context 9600 |
| P6 | 16384 | BLOCKED (HTTP 400) | — | — | — | max context 9600 |

**Read-through:** From 128 through 4096 tokens, wall-clock prefill stays ~56–67 ms. 8192 tokens steps to ~100 ms. Raising `--max-model-len` (HBM is not the limiter at 144 GB) is required before P5/P6 are meaningful.

---

## 6. Dual-GPU prefill / decode

### 6.1 Stack

| Service | Container | Device | Port | Health |
| --- | --- | --- | --- | --- |
| Prefill | `rocm-vllm-pd-prefill` | HIP 0 | 8100 | healthy (~2 min after GPU 0 compile cache) |
| Decode | `rocm-vllm-pd-decode` | HIP 1 | 8200 | healthy (~3 min; first compile on GPU 1) |
| Router | `rocm-vllm-pd-router` | CPU | 8000 | live disaggregation mode |

### 6.2 Live KV handoff (failed)

Request: `POST /v1/chat/completions` on the router with a 16-token cap.

| Step | Result |
| --- | --- |
| Prefill `POST :8100/v1/chat/completions` | HTTP **200** |
| Router extracts `prompt_token_ids` / KV transfer params | **Missing** |
| Decode `:8200` | never invoked |
| Client | `Prefill engine did not return prompt_token_ids; cannot hand off KV to decode` |

The router still injects `extra_body.return_token_ids` and `kv_transfer_params.do_remote_decode`. This image’s OpenAI server ignores those fields. Until a connector (MoRI-IO / NIXL / CPU offload) is registered **and** the API returns token ids + KV handles, `./bench_dual_gpu.sh -m pd` cannot measure true 1P1D E2E.

### 6.3 Concurrent phase occupancy (succeeded)

Prefill suite on **:8100** and decode suite on **:8200** started together.

**Prefill GPU 0** — `_results/phases_pd_prefill/phase_profile_summary_20260924_123225.json`

| Case | Engine prefill (s) | TTFT p50 (ms) |
| --- | ---: | ---: |
| P1 | 0.055 | 56.80 |
| P2 | 0.064 | 66.57 |
| P3 | 0.061 | 64.47 |
| P4 | 0.096 | 100.67 |

Matches collocated prefill within a few milliseconds. GPU 0 prefill was **not** disturbed by GPU 1 decode (the prefill suite finished in ~6 s, before decode D1 completed).

**Decode GPU 1** — `_results/phases_pd_decode/phase_profile_summary_20260924_123225.json`

| Case | Decode tok/s | TPOT p50 (ms) | ITL p95 (ms) | TTFT p50 (ms) |
| --- | ---: | ---: | ---: | ---: |
| D1 | **1.69** | 591.57 | 1229.21 | 851.88 |
| D2 | 2.15 | 464.72 | 892.07 | 2658.11 |
| D3 | 4.41 | 226.63 | 589.15 | 3382.41 |
| D4 | **16.56** | 60.39 | 140.08 | 1858.26 |

D1–D3 are **cold / first-graph** behavior on GPU 1 (compile cache was populated for GPU 0 earlier; GPU 1 paid it during the decode suite). D4 is the first number that approaches collocated decode, still **~0.61×** GPU 0’s D4 (16.56 vs 26.98 tok/s) with a worse ITL tail.

### 6.4 Collocated vs split-phase (D4 / P4)

| Metric | GPU 0 collocated | GPU 0 PD-prefill | GPU 1 PD-decode |
| --- | ---: | ---: | ---: |
| P4 TTFT p50 | 102.55 ms | 100.67 ms | — |
| D4 decode tok/s | 26.98 | — | 16.56 |
| D4 TPOT p50 | 37.06 ms | — | 60.39 ms |
| D4 ITL p95 | 37.56 ms | — | 140.08 ms |

Do not treat GPU 1 D1–D3 as a steady-state MI350P decode floor. Re-run decode-only on `:8200` after warmup for a fair GPU 0 vs GPU 1 comparison.

---

## 7. Serving config used (GPU 0 collocated, after healthy start)

From engine logs:

- Weights: **27.48 GiB**, load ~23.4 s
- Compile range (1, 2048), Dynamo ~11 s, graph compile ~57 s, warmup ~50 s
- Free memory after load ~143.66 GiB; **32 GiB reserved for KV**
- KV size: **337,515 tokens**; max concurrency at 9600 tokens/request: **35.16×**
- Attention page aligned to mamba page (block size 784 tokens)

`--max-model-len 9600` is inherited from `.env` and was **not** raised to the MI350P profile default (32k). That is why P5/P6 400.

---

## 8. How to reproduce

```bash
cd /home/amd/workspace/coder
set -a && . ./.env && set +a
. ./lib/gpu_profile.sh
export GPU_PROFILE=auto
apply_gpu_profile
unset HSA_OVERRIDE_GFX_VERSION   # never export this empty on gfx950

# 1) Collocated GPU 0
docker compose up -d --force-recreate --no-deps inference
# wait until curl -sf http://127.0.0.1:8000/health

docker exec -w /workspace rocm-inference-server python3 benchmark/bench_phases.py \
  --url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B-FP8 \
  --mode decode --trials 1 --output-tokens 256 --results-dir /workspace/_results/phases

docker exec -w /workspace rocm-inference-server python3 benchmark/bench_phases.py \
  --url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B-FP8 \
  --mode prefill --prefill-trials 2 --results-dir /workspace/_results/phases

# 2) Dual GPU (stops port 8000 collocated server)
docker compose stop inference
docker compose -f docker-compose.pd.vllm.yml up -d --force-recreate --no-deps prefill decode
# wait :8100 and :8200 /health
docker compose -f docker-compose.pd.vllm.yml up -d --no-deps router
```

Preflight (works without `/dev/kfd` for the login user):

```bash
python3 inspect_dual_gpu.py --check-preflight --gpu-profile auto
```

---

## 9. Caveats

- **SKU name:** measured hardware is **MI350P PCIe (`75a8` / gfx950)**. Cross-card A/B vs R9700 needs a second host with `GPU_PROFILE=r9700`.
- **Quantization:** FP8 dense vLLM only. Quark MXFP4 / llama.cpp / SGLang were not part of this run.
- **Prefill tok/s columns** in `bench_phases.py` overstate ingestion rate when prefix cache hits; use `engine_prefill_s` and TTFT.
- **Decode length:** 256 tokens, 1 trial. Not a 1024-OSL throughput sweep (`throughput.sh` was not re-run here).
- **Energy:** 0 W / 0 J in JSON. Add the user to `render` or collect telemetry from inside the container / as root.
- **P/D:** two independent vLLM replicas, not a production KV fabric. Router live mode is blocked on token-id / connector support.
- **NUMA / P2P:** GPUs are on distinct PCI domains (`0000` vs `0001`). Handoff latency modeling in `inspect_dual_gpu.py` remains analytical until a connector exists.

---

## 10. Recommended follow-ups

1. Warm up GPU 1 (`:8200`) then repeat decode-only D1–D4 for a fair dual-card decode comparison.
2. Raise `MAX_MODEL_LEN` (and re-compile) so P5/P6 exercise HBM instead of the 9600 cap.
3. Enable a real KV connector and re-test `pd_router` until `/metrics/pd` records prefill TTFT, handoff ms, and decode TPOT.
4. Run `./throughput.sh -e vllm -c 1 --test-cases 8192:1024,1024:8192,1024:1024` on collocated MI350P for mixed-workload tok/s comparable to the R9700 reports in `docs/`.
5. `sudo usermod -aG render amd` (then re-login) so host `rocminfo` / `collect_amd_power.py` work.
6. Build or pull `local/vllm-mxfp4:gfx950` if MXFP4 vs FP8 on the same two cards is required.

---

## 11. Artifact index

| Path | Contents |
| --- | --- |
| `_results/phases/phase_profile_summary_20260924_122130.json` | Collocated decode-only |
| `_results/phases/phase_profile_summary_20260924_122210.json` | Collocated prefill-only |
| `_results/phases_pd_prefill/phase_profile_summary_20260924_123225.json` | Dual-GPU prefill on GPU 0 |
| `_results/phases_pd_decode/phase_profile_summary_20260924_123225.json` | Dual-GPU decode on GPU 1 |
| `docker-compose.yml` | Collocated vLLM |
| `docker-compose.pd.vllm.yml` | FP8 1P1D compose |
| `patches/qwen3_5_vllm030.py` | Visual/MTP ignore overlay for vLLM 0.30 |
| `gpu_profile.py` / `lib/gpu_profile.sh` | r9700 vs mi350p detection |
