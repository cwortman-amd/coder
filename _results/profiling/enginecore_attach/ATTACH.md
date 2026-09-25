# EngineCore attach experiment (25 Sep 2026)

Goal: GPU-owner kernel timeline **without** wrapping `vllm serve`.

## Result

**KERNEL_DISPATCH obtained: no.** Compatible dynamic attach was attempted against live `VLLM::EngineCore`. rocprofv3 refused **before ptrace** and did **not** crash EngineCore:

```text
Could not find 'rocp-bg-attach' thread in /proc/277/task
Cannot attach to process 277: 'rocp-bg-attach' thread not found.
Start the target with ROCP_TOOL_ATTACH=1, or use a rocprofiler-register
build configured with ROCPROFILER_REGISTER_BUILD_DEFAULT_ATTACHMENT=ON.
```

Tiny decode (`max_tokens=8`) succeeded (~0.47 s, 8 completion tokens). `/health` stayed HTTP 200 through the attach. No `rocprof/` CSV.

`ROCP_TOOL_ATTACH=1` is on the **API parent only**. EngineCore spawn (`VLLM_WORKER_MULTIPROC_METHOD=spawn`) does not have that env, does not map `librocprofiler-sdk-attach.so`, and never starts a `rocp-bg-attach` **thread**. There is no `rocp-bg-attach` **binary** in SDK 1.3.2 (`rocprofv3-attach` / `rocprof-attach` exist).

Do **not** wrap `vllm serve` again (parent HIP init only; C1 **60–65 tok/s** profiler perturbation vs unprofiled **79.53**).

## PIDs at attach time

| Role | Host PID | nsPID | `/dev/kfd` | Decode maps |
|---|---|---|---|---|
| `vllm serve` | 511098 | 1 | yes | small |
| resource_tracker | 511909 | 276 | no | no |
| **`VLLM::EngineCore`** | **511910** | **277** | **yes** | **yes** |

Container: `rocm-inference-server` (`vllm/vllm-openai-rocm:latest`), rocprofiler-sdk **1.3.2**, ROCm **7.2.3**. No `SYS_PTRACE`. `yama.ptrace_scope=1`. `PTRACE_SEIZE` from `docker exec` → EPERM. `VLLM_ROCM_USE_AITER` unset.

## rocprofv3 flags (`rocprofv3 --help`)

`--pid` / `--attach`, `--attach-duration-msec`, `--attach-children` (default true), `--attach-sync-output`, `--process-sync` (launch finalize). Example in help: `rocprofv3 --attach 1234 --attach-duration-msec 10`.

## Commands tried

```bash
docker exec -i rocm-inference-server python3 -c 'PTRACE_SEIZE 277'  # EPERM
python3 scripts/bench_openai_chat.py --base-url http://127.0.0.1:8000/v1 --model awq \
  --input-len 32 --output-len 8 --num-prompts 1 --concurrency 1 --timeout 60 \
  --out _results/profiling/enginecore_attach/tiny_decode.json
docker exec rocm-inference-server rocprofv3 \
  --pid 277 --attach-children=false --attach-duration-msec 2500 --attach-sync-output \
  --kernel-trace --hip-runtime-trace \
  --output-format csv pftrace \
  --output-directory /results/profiling/enginecore_attach/rocprof \
  --output-file attach_try
# exit 1; EngineCore still Sl; health 200
```

Not tried (would interrupt GPU work): host `--pid 511910`, gdb, live `LD_PRELOAD`, privileged ptrace sidecar.

## EngineCore-only exec wrapper (not installed)

Possible via `multiprocessing.spawn.set_executable` so rocprofv3 is parent of the EngineCore Python child. Requires **restart**. Emit:

```bash
./scripts/profile_enginecore_attach.sh --emit-exec-wrapper scripts/enginecore_rocprof_exec.sh
```

Never use that file as docker `--entrypoint`.

## After the experiment

Attach: **17:09:37Z**, healthy. At **17:12:41Z** API `[shutdown]` / EngineCore SIGTERM; container **exit 137**. That matches a later `scripts/bench_vllm_captured_decode.sh --exclusive` (`vllm-captured-decode`, EngineCore ~24 GiB on GPU 0). Restore `launch_vllm_mxfp4.sh` (AITER unset) then failed: free **125.38/143.98 GiB** vs 0.92 util. That exclusive script restores `:8000` on its own EXIT. This attach work did **not** `docker stop` the exclusive bench.

## Remaining blockers

1. No `rocp-bg-attach` thread in EngineCore → attach cannot start.
2. `ROCP_TOOL_ATTACH=1` never reaches EngineCore environ before HIP init.
3. KERNEL_DISPATCH still needs EngineCore started with attach support **or** spawn exec wrapper (both restart).
4. Production container has no `SYS_PTRACE` (not needed for this refusal).

Helper: `scripts/profile_enginecore_attach.sh` (`--identify-only`; `--force` reproduces the refusal).
