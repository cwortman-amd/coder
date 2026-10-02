#!/usr/bin/env python3
"""Triton MXFP4 tile sweeps.

main_gateup times one fused projection. main_tile times the layer grid.
The old script names remain the entry points.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mxfp4_eager import STOCK, load_tensors, time_ms  # noqa: E402

from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from vllm.model_executor.kernels.linear.mxfp4.aiter import AiterMxfp4LinearKernel
from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearLayerConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic

def load_pair(model_dir: Path, stem: str):
    return load_tensors(model_dir, f"{stem}.weight", f"{stem}.weight_scale")


class Packed(nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(scale, requires_grad=False)


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


def main_gateup() -> int:
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


BABEL = 3793.4139104873675
BASE = STOCK


def load(model_dir: Path, wkey: str, skey: str):
    return load_tensors(model_dir, wkey, skey)


def configs() -> list[tuple[str, dict | None]]:
    rows: list[tuple[str, dict | None]] = [("library_default", None)]
    # Production M_LEQ_8 is bm8 / ksplit4 / bk512. Vary the decode-relevant axes.
    for bm in (1, 2, 4, 8, 16):
        cfg = deepcopy(BASE)
        cfg["BLOCK_SIZE_M"] = bm
        cfg["NUM_KSPLIT"] = 1
        cfg["BLOCK_SIZE_K"] = 512
        rows.append((f"bm{bm}_ks1_bk512", cfg))
    for ksplit in (1, 2, 4):
        for bk in (256, 512, 1024):
            if ksplit == 1 and bk == 512:
                continue
            cfg = deepcopy(BASE)
            cfg["BLOCK_SIZE_M"] = 8
            cfg["NUM_KSPLIT"] = ksplit
            cfg["BLOCK_SIZE_K"] = bk
            rows.append((f"bm8_ks{ksplit}_bk{bk}", cfg))
    return rows


def main_tile() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="/models/Qwen3.8-27B-Quark-AWQ-MXFP4-sharded")
    p.add_argument("--out", default="/results/profiling/mxfp4_gemm/tile_sweep.json")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--m", type=int, default=1)
    args = p.parse_args()
    torch.set_default_dtype(torch.bfloat16)

    kernel = AiterMxfp4LinearKernel(
        MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic)
    )
    assert not kernel.use_asm_gemm, "tile sweep is Triton-only; unset VLLM_ROCM_USE_AITER"

    layers = {
        "mlp.gate_proj": (
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight_scale",
        ),
        "mlp.down_proj": (
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight_scale",
        ),
        "linear_attn.out_proj": (
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight_scale",
        ),
    }

    model_dir = Path(args.model_dir)
    out_rows = []
    for lname, (wk, sk) in layers.items():
        w, s = load(model_dir, wk, sk)
        layer = Packed(w.cuda(), s.cuda())
        kernel.process_weights_after_loading(layer)
        n, packed_k = layer.weight.shape
        k = packed_k * 2
        x = torch.randn(args.m, k, device="cuda", dtype=torch.bfloat16)
        x_q, x_s = dynamic_mxfp4_quant(x)
        w_bytes = int(layer.weight.numel() * layer.weight.element_size())
        s_bytes = int(layer.weight_scale.numel() * layer.weight_scale.element_size())
        y_ref = torch.empty(args.m, n, device="cuda", dtype=torch.bfloat16)
        gemm_afp4wfp4(
            x_q, layer.weight, x_s, layer.weight_scale.T, torch.bfloat16, y_ref, None
        )
        print(f"layer={lname} N={n} K={k} W+S={(w_bytes+s_bytes)/1e9:.4f} GB", flush=True)
        for name, cfg in configs():
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
                out_rows.append(
                    {"layer": lname, "config": name, "ok": False, "error": str(exc)}
                )
                continue
            gbs = (w_bytes + s_bytes) / (ms / 1e3) / 1e9
            rec = {
                "layer": lname,
                "config": name,
                "ok": True,
                "ms": ms,
                "weight_gbs": gbs,
                "vs_babel": gbs / BABEL,
                "max_abs_vs_default": max_abs,
                "cfg": cfg,
            }
            out_rows.append(rec)
            print(
                f"  {name:22s} {ms:7.3f} ms  {gbs:7.1f} GB/s  "
                f"{100*gbs/BABEL:5.1f}% Babel  max|d|={max_abs:.4f}",
                flush=True,
            )
        del layer, x, x_q, x_s, y_ref
        torch.cuda.empty_cache()

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    # rank M=1 gate by ms
    ok = [r for r in out_rows if r.get("ok") and r["layer"] == "mlp.gate_proj"]
    ok.sort(key=lambda r: r["ms"])
    payload = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "m": args.m,
        "best_gate": ok[:8],
        "rows": out_rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print("best gate", [(r["config"], round(r["ms"], 3), round(r["weight_gbs"], 1)) for r in ok[:8]])
    print("wrote", path)
    return 0
