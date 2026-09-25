# Captured MXFP4 mini-decode attempt (25 Sep 2026)

Goal: graph-replay real Qwen3.8-27B MXFP4 shapes under stock Triton and
shape-local split-K variants, as an intermediate screen before serving.

Result: **blocked in the standalone AITER/Triton path**.

- Capturing `dynamic_mxfp4_quant` + `gemm_afp4wfp4` caused an HSA GPU memory
  access fault before the first result.
- Precomputing quantized activations and capturing only
  `gemm_afp4wfp4` (+ split-K reduction) caused the same fault.
- The HF server was restored after both attempts and is healthy.
- No timing from these failed captures is valid.

This does not mean vLLM graph replay is broken: the production server uses
vLLM's compile/capture machinery successfully, and eager mode is ~0.18×.
It means a raw `torch.cuda.CUDAGraph` wrapper around these standalone Triton
entry points is not a valid reproduction route on this image.

Next valid routes:

1. **vLLM-native captured mini-decode** (`scripts/bench_vllm_captured_decode.sh`):
   uses `LLM` + O2 `FULL_AND_PIECEWISE` + vLLM's CUDAGraph/static pools, not a
   raw `torch.cuda.CUDAGraph` around AITER Triton. Safe default is preflight
   only (refuses a second 27B while `:8000` is healthy). `--exclusive` stops
   the frozen server, runs M=1 and M=8 graph-replay windows, ranks
   gate/up/down/attention/LM-head from capture traces if ROCm exposes them,
   then restores `scripts/launch_vllm_mxfp4.sh`. Intermediate screen only.
2. Obtain an EngineCore-native GPU timeline or compatible dynamic attach
   (`_results/profiling/enginecore_attach/ATTACH.md`). Live attach still has
   no `rocp-bg-attach` thread.
3. Continue to use unprofiled graph-enabled serving as the promotion gate.

Standalone Triton reproducer (invalid timings): `scripts/bench_mxfp4_captured.py`.

## vLLM-native captured decode route

Use `scripts/bench_vllm_captured_decode.py`, via:

```bash
# Safe check only; docker-execs into the live container and loads no model.
scripts/bench_vllm_captured_decode.sh

# Required for the actual run on this host. Stops and restores :8000.
scripts/bench_vllm_captured_decode.sh --exclusive
```

This route constructs an in-process vLLM `LLM` with compile mode 3
(`VLLM_COMPILE`, the v0.30 O2 path), `FULL_AND_PIECEWISE`, and capture sizes
1 and 8. It therefore uses `CUDAGraphWrapper`, vLLM's static runner buffers,
forward-context dispatch, and the platform global graph pool. It does **not**
wrap AITER/Triton in a standalone graph.

The running server cannot coexist with this experiment: only about 19 GiB is
free while serving, but logs show 17.91 GiB of model weights and 6.84 GiB of
captured graphs, before workspaces and KV cache. The Python preflight refuses
to load a second engine. **Exclusive run completed 25 Sep 2026** (M=1 60.92
tok/s profiler-on; `_gemm_afp4wfp4` 39.7% CUDA at 22.4 µs). Report:
`_results/profiling/vllm_captured/SUMMARY.md`. The frozen server was restored
with `scripts/launch_vllm_mxfp4.sh` (`VLLM_ROCM_USE_AITER` unset, stock
`M_LEQ_8`).

Artifacts are written under `_results/profiling/vllm_captured/`:

- `summary.json` and `SUMMARY.md`
- vLLM-managed torch-profiler capture/replay traces
- M=1 and M=8 steady decode windows derived after first-token prefill
- best-effort projection ranking for MLP gate/up/down, attention projections,
  and LM-head

vLLM's built-in layerwise NVTX hooks are not a replay-time per-module timer:
compiled/full-graph replay bypasses Python module hooks. vLLM 0.30 has Proton
CUDA-graph attribution, but explicitly rejects it on ROCm (NVIDIA CUDA only).
The script therefore correlates capture-time operator shapes/external IDs to
GPU events and separately times actual graph-backed decode. If ROCm torch
profiler omits that correlation, the ranking is reported unavailable rather
than substituting CPU launch time or claiming a gate-only result. Gate and up
may be physically fused as `gate_up_proj`; when so, both are reported as
shares of the same fused measurement.

