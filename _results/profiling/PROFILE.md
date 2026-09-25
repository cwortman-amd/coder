# Profiling Qwen3.8-27B MXFP4 on MI350P

Campaign summary: [`docs/MI350P_MXFP4_VLLM_EVAL.md`](../../docs/MI350P_MXFP4_VLLM_EVAL.md).

Use three layers, in this order: **vLLM `/metrics`**, **rocprofv3 timeline**, **ROCm Compute Profiler counters**. Add **amd-smi / pidstat / vmstat / iostat** on every candidate.

## Unblock EngineCore traces (still blocked)

**KERNEL_DISPATCH was not obtained.** Detail: [`enginecore_attach/ATTACH.md`](enginecore_attach/ATTACH.md).

**Live attach** (`rocprofv3 --pid` / `--attach`) **exists** on SDK **1.3.2** in `vllm/vllm-openai-rocm:latest`. There is **no** `rocp-bg-attach` binary; that name is a **thread** the attacher looks up in `/proc/<pid>/task`. `docker exec rocprofv3 --pid 277` against EngineCore **does not crash** the server: it exits 1 *before* ptrace (`rocp-bg-attach` thread not found). Tiny `max_tokens=8` decode stayed healthy.

`ROCP_TOOL_ATTACH=1` is on the **API parent only**. EngineCore (`VLLM_WORKER_MULTIPROC_METHOD=spawn`) does **not** inherit it, does not map `librocprofiler-sdk-attach.so`, and never starts the attach thread. Parent also has `/dev/kfd`; **decode maps belong to EngineCore**.

**Do not wrap `vllm serve` again** expecting kernel CSV. `scripts/profile_enginecore_launch.sh` remains a **profiler-perturbation diagnostic** (1K/128: **60.46 tok/s** default MP, **65.26 tok/s** with `VLLM_ENABLE_V1_MULTIPROCESSING=0`; parent HIP init only, **no KERNEL_DISPATCH**). Unprofiled C1 **79.53**. Artifacts: `_results/profiling/c1_rocprof_launch/`, `_results/profiling/c1_rocprof_mp0/`.

**EngineCore-only exec wrapper is possible** via `multiprocessing.spawn.set_executable` (rocprofv3 as parent of the EngineCore Python child). That requires a **new spawn / server restart**. Emit with `./scripts/profile_enginecore_attach.sh --emit-exec-wrapper scripts/enginecore_rocprof_exec.sh`. Never use that file as docker `--entrypoint`.

```bash
./scripts/profile_enginecore_attach.sh --identify-only
# refused attach (safe): ./scripts/profile_enginecore_attach.sh --force
# perturbation check only (displaces :8000):
./scripts/profile_enginecore_launch.sh --concurrency 1 --warmup 8 --num-prompts 8
```

Production `:8000` container has **no `SYS_PTRACE`**. Do **not** gdb / extra `LD_PRELOAD` the live EngineCore.

Open `rocprof/*.pftrace` in [Perfetto](https://ui.perfetto.dev) only after a real kernel CSV exists. Rank:

```bash
python3 scripts/aggregate_kernel_csv.py \
  --input-dir _results/profiling/launch_<stamp>/rocprof \
  --out _results/profiling/launch_<stamp>/kernel_top.json
```

## Scripts

| Script | Role |
|---|---|
| `scripts/profile_enginecore_attach.sh` | EngineCore-only `rocprofv3 --pid`; never wraps `vllm serve`. Blocked without `rocp-bg-attach` thread |
| `scripts/enginecore_rocprof_exec.sh` | Spawn executable: rocprofv3 parent of EngineCore Python only (restart required) |
| `scripts/profile_enginecore_launch.sh` | rocprofv3 wraps `vllm serve` (port 8001). **Boots; HIP init only; 60–65 tok/s diagnostic** |
| `scripts/bench_gdn_decode.py` | Isolated FLA GDN packed decode + rocprof. Report: `gdn_decode/GDN.md` |
| `scripts/launch_vllm_mxfp4.sh` | Unprofiled HF recipe on :8000; `VLLM_ROCPROF_DIR=...` is the older wrap (EngineCore still often untraced) |
| `scripts/profile_capture.sh` | Metrics+telemetry on an already-running server; `--skip-attach` |
| `scripts/profile_server.sh` | Host-native `rocprofv3 -- vllm serve` (no Docker) |
| `scripts/summarize_vllm_metrics.py` | Prometheus deltas |
| `scripts/aggregate_kernel_csv.py` | Top kernels once a kernel CSV exists |

## Layer 1 — vLLM metrics (working)

```bash
curl -s http://127.0.0.1:8000/metrics | grep -E 'time_to_first|inter_token|e2e|queue|spec_decode|kv_cache|running|waiting'
./scripts/profile_capture.sh _results/profiling/control_c1 \
  --model awq --concurrency 1 --num-prompts 4 --skip-attach
```

This build does **not** export `vllm:time_to_first_token_seconds` as a separate family name in some scrapes; it does export TTFT/ITL/e2e histograms, queue/prefill/decode times, KV usage, and DFlash draft/accept counters. No `spec_decode_efficiency` on 0.30 ROCm.

## Layer 2 — rocprofv3 attach (still blocked)

```text
rocprofv3 --pid <EngineCore> --attach-duration-msec …
```

Tried `docker exec rocprofv3 --pid 277` (EngineCore). Refused: no `rocp-bg-attach`
thread (`enginecore_attach/ATTACH.md`). Host `rocprofv3` is also 1.3.2; we did
not ptrace the live EngineCore from the host. Do **not** wrap `vllm serve`.
Helper: `scripts/profile_enginecore_attach.sh`.

## Layer 3 — rocprof-compute (after a named hot kernel)

Do not counter-profile a full vLLM server. After kernel names exist, filter to one MXFP4 GEMM. Discover `gfx950` counters locally; do not reuse MI300 names.

## System telemetry

Healthy unprofiled decode: GPU ~100% GFX, ~2.2 GHz, ~390–400 W of 600 W, MCLK 2000 MHz. `amd-smi monitor -w` can hang; prefer a single `amd-smi metric` sample.

## Matrix

| ID | Command | Status |
|---|---|---|
| P1 | Control C1 1K/1K | Metrics+telemetry; **no kernel CSV** |
| P2 | DFlash-7 C1 | Metrics+telemetry |
| P3 | Control C8 1K/1K | Metrics+telemetry |
| P4 | DFlash-7 C8 | Metrics+telemetry |
| Launch-under-profiler | Control C1 1K/128 | Wrapper boots; **HIP init only**, no kernel CSV; **~60 tok/s** diagnostic |
| Launch + MP=0 | Same | EngineCore still forks; **65.26 tok/s**; no kernel CSV |
| EngineCore `--pid` attach | Tiny decode max_tokens=8 | **Safe refuse**; **no KERNEL_DISPATCH**; healthy through attach |
| Unprofiled C1 best | Control 1K/1K | **79.53 tok/s** (`c1_best/`) |
| RVS BABEL | GPU 0, 2 GiB arrays | Read **3793 GB/s**; `_results/profiling/C1_BABEL_ROOFLINE.md` |
