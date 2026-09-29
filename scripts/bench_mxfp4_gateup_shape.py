#!/usr/bin/env python3
"""Compare Triton tiles for fused gate-up only: N=34816, K=5120, M=1.

Stock dispatch for this shape is M_LEQ_8 (requested NUM_KSPLIT=4), which
get_splitk reduces to 2. Other projections are not timed. VLLM_ROCM_USE_AITER
must stay unset; ASM is a process-wide flag, not a shape key.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from vllm.model_executor.kernels.linear.mxfp4.aiter import AiterMxfp4LinearKernel
from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearLayerConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic

SHARD = "model-00007-of-00039.safetensors"
STOCK = {
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


def load_pair(model_dir: Path, stem: str):
    path = model_dir / SHARD
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(f"{stem}.weight"), handle.get_tensor(f"{stem}.weight_scale")


class Packed(nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(scale, requires_grad=False)


def time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def candidates() -> list[tuple[str, dict | None]]:
    rows: list[tuple[str, dict | None]] = [
        ("library_default", None),
        ("stock_m_leq_8", deepcopy(STOCK)),
    ]
    for ksplit in (1, 2):
        cfg = deepcopy(STOCK)
        cfg["NUM_KSPLIT"] = ksplit
        rows.append((f"ks{ksplit}_bn64_bk512_w4", cfg))
    for block_n in (32, 128, 256):
        cfg = deepcopy(STOCK)
        cfg["NUM_KSPLIT"] = 1
        cfg["BLOCK_SIZE_N"] = block_n
        rows.append((f"ks1_bn{block_n}_bk512_w4", cfg))
    for block_k in (256, 1024):
        cfg = deepcopy(STOCK)
        cfg["NUM_KSPLIT"] = 1
        cfg["BLOCK_SIZE_K"] = block_k
        rows.append((f"ks1_bn64_bk{block_k}_w4", cfg))
    for warps in (2, 8):
        cfg = deepcopy(STOCK)
        cfg["NUM_KSPLIT"] = 1
        cfg["num_warps"] = warps
        rows.append((f"ks1_bn64_bk512_w{warps}", cfg))
    return rows


def load_projection(model_dir: Path, name: str):
    stem = "model.language_model.layers.0.mlp"
    if name == "gate_up":
        gate_w, gate_s = load_pair(model_dir, f"{stem}.gate_proj")
        up_w, up_s = load_pair(model_dir, f"{stem}.up_proj")
        weight = torch.cat([gate_w, up_w], dim=0).contiguous()
        scale = torch.cat([gate_s, up_s], dim=0).contiguous()
        expected = (34816, 5120)
    elif name == "down":
        weight, scale = load_pair(model_dir, f"{stem}.down_proj")
        expected = (5120, 17408)
    else:
        raise SystemExit(f"unknown projection {name}")
    return weight, scale, expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", choices=("gate_up", "down"), default="gate_up")
    parser.add_argument("--model-dir", default="/models/Qwen3.8-27B-Quark-AWQ-MXFP4-sharded")
    parser.add_argument("--out", default="/results/profiling/mxfp4_gateup/shape_sweep.json")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--m", type=int, default=1)
    args = parser.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    kernel = AiterMxfp4LinearKernel(MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic))
    if kernel.use_asm_gemm:
        raise SystemExit("VLLM_ROCM_USE_AITER is set; this sweep is Triton-only")

    weight, scale, expected = load_projection(Path(args.model_dir), args.projection)
    layer = Packed(weight.cuda(), scale.cuda())
    kernel.process_weights_after_loading(layer)
    n, packed_k = layer.weight.shape
    logical_k = packed_k * 2
    if (n, logical_k) != expected:
        raise SystemExit(f"unexpected {args.projection} shape N={n} K={logical_k}")

    x = torch.randn(args.m, logical_k, device="cuda", dtype=torch.bfloat16)
    x_q, x_s = dynamic_mxfp4_quant(x)
    y_ref = torch.empty(args.m, n, device="cuda", dtype=torch.bfloat16)
    gemm_afp4wfp4(
        x_q, layer.weight, x_s, layer.weight_scale.T, torch.bfloat16, y_ref, None
    )
    print(f"{args.projection} N={n} K={logical_k} M={args.m}", flush=True)

    rows = []
    for name, cfg in candidates():
        y = torch.empty(args.m, n, device="cuda", dtype=torch.bfloat16)

        def run(y=y, cfg=cfg):
            gemm_afp4wfp4(
                x_q, layer.weight, x_s, layer.weight_scale.T, torch.bfloat16, y, cfg
            )

        try:
            run()
            torch.cuda.synchronize()
            max_abs = (y.float() - y_ref.float()).abs().max().item()
            ms = time_ms(run, args.warmup, args.iters)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"config": name, "ok": False, "error": str(exc), "cfg": cfg})
            continue
        rec = {
            "config": name,
            "ok": True,
            "ms": ms,
            "max_abs_vs_library_default": max_abs,
            "cfg": cfg,
        }
        rows.append(rec)
        print(f"  {name:24s} {ms:8.4f} ms  max|d|={max_abs:.5f}", flush=True)

    ok = [row for row in rows if row.get("ok")]
    ok.sort(key=lambda row: row["ms"])
    baseline = next(row["ms"] for row in rows if row["config"] == "library_default" and row.get("ok"))
    payload = {
        "shape": {"M": args.m, "N": n, "K": logical_k, "projection": args.projection},
        "library_default_ms": baseline,
        "ranked": [
            {**row, "vs_library_default": baseline / row["ms"]}
            for row in ok
        ],
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    if ok:
        best = ok[0]
        print(
            f"best {best['config']} {best['ms']:.4f} ms "
            f"({baseline / best['ms']:.3f}x vs library default)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
