---
type: Technical Report
title: Qwen3.8-Flash-Next on two MI350P GPUs
description: Separate from the frozen dense-27B MXFP4 control (MI350P.md). The published
  AMD recipe is 4× MI355X, TP=4. Nothing below is a validated.
tags:
- technical-report
- flash
- next
status: stable
---

# Qwen3.8-Flash-Next on two MI350P GPUs

Separate from the frozen dense-27B MXFP4 control (`MI350P.md`).
The published AMD recipe is **4× MI355X, TP=4**. Nothing below is a validated
single-MI350P or two-MI350P performance result.

Measured on this host, 27 Sep 2026:

| Resource | Measurement |
|---|---|
| GPUs | 2× Instinct MI350P, gfx950, device 0x75a8. `rocm-smi` VRAM **154,602,045,440 B** each (144 GiB class) |
| Host RAM | **30.7 GiB** total. With the 27B server up: **4.3 GiB** available, container RSS **5.7 GiB**, swap free **51 GiB** |
| Disk | **529 GiB** free |
| Local Flash-Next weights | absent at the start of this track |
| Installed server | `vllm/vllm-openai-rocm:latest`, vLLM **0.30.0**. `Qwen4ExpForConditionalGeneration` dispatches to `vllm/models/qwen4_exp/amd/` on ROCm |

## Checkpoint, not a half-byte estimate

Hugging Face `Qwen/Qwen3.8-Flash-Next-FP8` parameter counts:

| Dtype | Parameters | Bytes |
|---|---:|---:|
| F8_E4M3 | 174,512,783,360 | 162.5 GiB |
| BF16 | 5,487,198,064 | 10.2 GiB |
| Total | 179,999,981,459 | **172.8 GiB** |

`modules_to_not_convert` keeps the LM head, token embeddings, hyper-connections,
GDN projections, router, and shared experts in higher precision. The 51B
n-gram table matches `ngram_vocab_size_base=20,000,000` × `ple_embed_dim=2560`.
The file layout is **131 safetensors shards**, mostly 1.0–1.6 GiB each.
`176B × 0.5 byte` is not a single-card fit proof.

`ple_layer_ids` is `[2]`. The text stack is 48 layers, three
`linear_attention` then one `full_attention`. `num_experts=512`,
`num_experts_per_tok=10`, `moe_intermediate_size=640`. Native context is
262,144. MTP is one extra full-attention layer and stays off for bring-up.
QSA indexer budget is 2048 blocks.

## Topology

| Candidate | This machine |
|---|---|
| Official FP8, one GPU | Checkpoint alone is 172.8 GiB. It does not fit in 144 GiB |
| Official FP8, TP=2 | Weights can split near 86 GiB/GPU **if** the n-gram table is vocab-parallel. At `--gpu-memory-utilization 0.85` each card exposes about 122 GiB, leaving on the order of 30 GiB for KV, GDN/QSA state, and graphs at 16K. Not yet loaded |
| N-gram CPU offload | `EngramConfig` in this image accepts the offload path only when `current_platform.is_cuda()`. The AMD PLE module is GPU-resident. Host RAM is also below the recipe’s ≥51 GB plus headroom. `VLLM_PLE_CPU_OFFLOAD=0` on the experimental launch |
| Single-card MXFP4 | Not started. Quantizing every current FP8 parameter to 4-bit and keeping the measured BF16 tensors would be about 91 GiB plus scales, before state and graphs. Leaving the 51B table at FP8 keeps the checkpoint well above a comfortable 144 GiB budget. Fit is unproven until a real load |

The dense-27B rule “leave `VLLM_ROCM_USE_AITER` unset” does not transfer.
The experimental launch sets `VLLM_ROCM_USE_AITER=1` and
`VLLM_ROCM_USE_AITER_MOE=0`, matching the published ROCm recipe, and then
has to measure those flags on gfx950. `max-num-seqs` is **16** for the first
load so GDN state does not hide a weight-capacity failure. The MI355X recipe’s
256 sequences is not the first probe.

## Two roofs

Resident capacity is every weight, the n-gram table, MTP, KV/GDN/QSA state,
and workspaces. Decode traffic per token is shared-weight reads, the experts
actually selected, n-gram fetches, QSA/KV and GDN-state traffic, and TP=2
collectives. The “6B activated” figure is not an HBM byte count. Babel Read
**3793 GB/s** remains the streaming-bandwidth measurement from the dense-27B
campaign. It is the roof for contiguous expert-weight reads, not for QSA
gathers, n-gram lookups, or GDN recurrences.

## What this image already contains

- AMD model path: `vllm/models/qwen4_exp/amd/model.py`.
- GPU-resident PLE: `amd/ple_layer.py`. CPU n-gram offload lives under `nvidia/`.
- QSA: Triton kernels in `amd/ops/qsa.py` (`_qsa_mqa_paged_kernel`), plus
  `amd/indexer_qsa.py`. That is not the NVIDIA FlashInfer QSA-to-XQA path.
  No gfx950 timing exists yet.
- AITER MoE is a flag (`VLLM_ROCM_USE_AITER_MOE`), left at 0 until a trace
  compares it with the default grouped GEMM on these expert shapes.

## Stages

| Stage | State |
|---|---|
| 0. Capacity, TP=2, 16K, text-only, MTP off | Download started separately. Serve script refuses a partial checkpoint. Pass when both GPUs load with host RSS and HBM recorded and the machine is not swapping |
| 1. Correctness | One deterministic generation against the recipe’s GDN/QSA prompt, then task checks |
| 2. Attribution | EngineCore timeline only after stage 1. Same spawn-exec route as the 27B trace; profiler-on tok/s stay out of the table |
| 3–4. Kernels and serving | Per-shape MoE, then QSA index versus selected-block attention, then GDN and n-gram locality. Promote only unprofiled repeated runs |
| 5. MXFP4 TP=1 | Separate deliverable after FP8 is a reference. Routed experts first. Do not silently dequantize |
| 6. MTP | Off until the target is correct. The recipe’s 4×H100 test got slower with MTP |
| 7. Long context | 128K and 262K after 16K. 524K YaRN is its own run |

Scorecards stay separate: one MI350P MXFP4 versus one RTX PRO 6000, and the
best multi-GPU FP8 result on each side with per-GPU rate and collective cost.
A TP=2 aggregate is not a one-card result.

## Commands

```bash
./scripts/download_flash_next_fp8.sh
./scripts/launch_vllm_flash_next_fp8.sh
# restore the dense control when this track is idle
./scripts/launch_vllm_mxfp4.sh
```

Recipe: [Qwen3.8-Flash-Next](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next).
The NVIDIA 171 / 207 tok/s figures are a different stack and are not targets
for this TP=2 load.
