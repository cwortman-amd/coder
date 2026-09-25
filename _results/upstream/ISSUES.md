# Upstream issues to file (measured on vLLM 0.30 ROCm + MI350P)

These are lab-ready reports. File against the listed projects; do not treat this folder as a substitute for the tracker.

## 1. vLLM — Quark MTP loader shape mismatch

**Project:** [vllm-project/vllm](https://github.com/vllm-project/vllm)  
**Title:** Qwen3.5/Qwen3.8 Quark MXFP4 MTP weights fail `assert self.data.shape == loaded_weight.shape` in `Qwen3_5MTP.load_weights`

Quark-packed MTP tensors on [amd/Qwen3.8-27B-Quark-AWQ-MXFP4](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) do not match the column-parallel parameter layout in `vllm/model_executor/models/qwen3_5_mtp.py` on vLLM 0.30 ROCm (`AiterMxfp4LinearKernel`). `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` dies at load regardless of depth.

Lab log: `_results/priority_eval/mtp_load_failure.txt`.

Ask: unpack / reshape Quark MTP tensors before the shape assert, or reject with a clear `ValueError` naming the expected vs packed shapes.

## 2. vLLM — DFlash invalid depth should abort on the host

**Project:** vllm-project/vllm  
**Title:** DFlash `num_speculative_tokens=8` GPU-faults / SIGABRT instead of validating `block_size`

`incoai/Qwen3.8-27B-DFlash2` has `dflash_config.block_size=8` (seven draft positions). `num_speculative_tokens=8` is a nine-position query. Result on MI350P: EngineCore SIGABRT / GPU fault, including with `--enforce-eager`. Depth 7 loads.

Ask: validate `num_speculative_tokens <= block_size - 1` (or equivalent) before kernel launch.

Logs: `_results/priority_eval/dflash8/`.

## 3. rocprofiler-sdk — EngineCore child is untraced

**Project:** [ROCm/rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk) (and vLLM multiprocessing)  
**Title:** rocprofv3 on `vllm serve` records parent HIP init only; no `KERNEL_DISPATCH`; `rocp-bg-attach` absent in ROCm 7.14 / SDK 1.3.2 image

HIP work lives in `VLLM::EngineCore`. `VLLM_ENABLE_V1_MULTIPROCESSING=0` still forks. Launch-under-rocprofv3 yields ~5 KB pftrace (`hipDriverGetVersion` / `hipGetProcAddress`). Attach needs a `rocp-bg-attach` thread this `vllm/vllm-openai-rocm:latest` image never starts.

Ask: document a supported EngineCore exec or dynamic-attach path for vLLM V1 multiprocessing; expose attach in the container image.

Lab: `_results/profiling/PROFILE.md`, `c1_rocprof_launch/`, `c1_rocprof_mp0/`.

## 4. AITER — missing Qwen3.8 a4w4 / MXFP4 shape tune

**Project:** [ROCm/aiter](https://github.com/ROCm/aiter)  
**Title:** gfx950 `gemm_afp4wfp4` DEFAULT.json has no Qwen3.8-27B decode-shape tune

Stock `M_LEQ_8` (`NUM_KSPLIT=4`, `BLOCK_SIZE_K=512`) is the frozen serving path. Isolated M=1 GEMM is faster with `NUM_KSPLIT=1` or global ASM, but unprofiled vLLM 0.30 serving **regressed** (C1 75.93 / 71.40 vs 79.53). Need a **shape-keyed** Qwen3.8 decode tune (gate/up/down/attn/LM-head, M=1 and M=8), not a global switch.

Do not treat fused GDN CUDA as a C1 control justification (spec-decode path).

Lab: `_results/profiling/mxfp4_gemm/GEMM.md`.
