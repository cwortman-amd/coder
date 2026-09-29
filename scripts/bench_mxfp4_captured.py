#!/usr/bin/env python3
"""Capture-aware MXFP4 shape benchmark using real Qwen3.8-27B weights.

This is an intermediate screen, not a serving promotion test. It compares the
stock gfx950 Triton config with shape-local variants under HIP graph replay.
Quantization is precomputed: allocating dynamic_mxfp4_quant inside capture
faults this ROCm/AITER build, which is recorded separately.
"""
from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import torch
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from safetensors import safe_open

BABEL_READ_GBS = 3793.4139104873675
SHARD = "model-00007-of-00039.safetensors"

LAYERS = {
    "mlp.gate_proj": (
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight_scale",
    ),
    "mlp.up_proj": (
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight_scale",
    ),
    "mlp.down_proj": (
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight_scale",
    ),
    "linear_attn.in_proj_qkv": (
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale",
    ),
    "linear_attn.out_proj": (
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight_scale",
    ),
}

BASE_CONFIG = {
    "BLOCK_SIZE_M": 8,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 32,
    "cache_modifier": ".cg",
    "NUM_KSPLIT": 4,
}


def load_pair(model_dir: Path, keys: tuple[str, str]) -> tuple[torch.Tensor, torch.Tensor]:
    with safe_open(str(model_dir / SHARD), framework="pt", device="cpu") as handle:
        return handle.get_tensor(keys[0]), handle.get_tensor(keys[1])


def variants() -> list[tuple[str, dict | None]]:
    no_split = deepcopy(BASE_CONFIG)
    no_split["NUM_KSPLIT"] = 1
    no_split_bk1024 = deepcopy(no_split)
    no_split_bk1024["BLOCK_SIZE_K"] = 1024
    split2_bk1024 = deepcopy(BASE_CONFIG)
    split2_bk1024["NUM_KSPLIT"] = 2
    split2_bk1024["BLOCK_SIZE_K"] = 1024
    return [
        ("stock", None),
        ("ksplit1_bk512", no_split),
        ("ksplit1_bk1024", no_split_bk1024),
        ("ksplit2_bk1024", split2_bk1024),
    ]


def replay_ms(graph: torch.cuda.CUDAGraph, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", default="/models/Qwen3.8-27B-Quark-AWQ-MXFP4-sharded"
    )
    parser.add_argument(
        "--out", default="/results/profiling/mxfp4_captured/summary.json"
    )
    parser.add_argument("--ms", default="1,8")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    model_dir = Path(args.model_dir)
    rows: list[dict] = []

    for layer_name, keys in LAYERS.items():
        weight_cpu, scale_cpu = load_pair(model_dir, keys)
        weight = weight_cpu.cuda()
        # Checkpoint scales are [N, K/32]; production Triton passes .T.
        weight_scale = scale_cpu.cuda().T.contiguous()
        n, packed_k = weight.shape
        k = packed_k * 2
        bytes_read = weight.numel() * weight.element_size()
        bytes_read += weight_scale.numel() * weight_scale.element_size()

        for m in (int(value) for value in args.ms.split(",") if value):
            static_x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            reference: torch.Tensor | None = None

            for variant, config in variants():
                # Compile and allocate outside capture. Quantization allocation
                # is not graph-safe in this image, so replay only the GEMM
                # (including split-K reduction when selected).
                xq, xs = dynamic_mxfp4_quant(static_x)
                static_y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                for _ in range(5):
                    output = gemm_afp4wfp4(
                        xq,
                        weight,
                        xs,
                        weight_scale,
                        torch.bfloat16,
                        static_y,
                        config=config,
                    )
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured_output = gemm_afp4wfp4(
                        xq,
                        weight,
                        xs,
                        weight_scale,
                        torch.bfloat16,
                        static_y,
                        config=config,
                    )

                graph.replay()
                torch.cuda.synchronize()
                current = captured_output.clone()
                if reference is None:
                    reference = current
                    max_abs = 0.0
                else:
                    max_abs = (current.float() - reference.float()).abs().max().item()

                elapsed_ms = replay_ms(graph, args.warmup, args.iters)
                weight_gbs = bytes_read / (elapsed_ms / 1e3) / 1e9
                row = {
                    "layer": layer_name,
                    "m": m,
                    "n": n,
                    "k": k,
                    "variant": variant,
                    "capture_scope": "prequantized_gemm_and_splitk_reduce",
                    "captured_ms": elapsed_ms,
                    "weight_scale_bytes": bytes_read,
                    "weight_gbs": weight_gbs,
                    "vs_babel": weight_gbs / BABEL_READ_GBS,
                    "max_abs_vs_stock": max_abs,
                    "config": config,
                }
                rows.append(row)
                print(
                    f"{layer_name:26s} M={m:2d} {variant:17s} "
                    f"{elapsed_ms * 1e3:7.2f} us {weight_gbs:7.1f} GB/s "
                    f"{100 * weight_gbs / BABEL_READ_GBS:5.1f}% Babel "
                    f"max|d|={max_abs:.4f}",
                    flush=True,
                )
                del graph, captured_output, current, output, xq, xs, static_y
                torch.cuda.empty_cache()

        del weight, weight_scale, weight_cpu, scale_cpu
        torch.cuda.empty_cache()

    winners = []
    for layer_name in LAYERS:
        for m in (int(value) for value in args.ms.split(",") if value):
            candidates = [r for r in rows if r["layer"] == layer_name and r["m"] == m]
            winners.append(min(candidates, key=lambda row: row["captured_ms"]))

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "note": (
                    "HIP graph replay of prequantized GEMM+reduce only; "
                    "dynamic quant allocation faulted capture; serving is promotion gate"
                ),
                "babel_read_gbs": BABEL_READ_GBS,
                "winners": winners,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
