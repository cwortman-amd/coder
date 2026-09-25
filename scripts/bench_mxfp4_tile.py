#!/usr/bin/env python3
"""Sweep Triton gemm_afp4wfp4 tiles at M=1 (BLOCK_M / NUM_KSPLIT).

Default gfx950 config is DEFAULT.json M_LEQ_8: BLOCK_M=8, NUM_KSPLIT=4
(get_splitk typically yields 2 + a reduce kernel).
"""
from __future__ import annotations

import argparse
import json
import time
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

BABEL = 3793.4139104873675
SHARD = "model-00007-of-00039.safetensors"
BASE = {
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


def load(model_dir: Path, wkey: str, skey: str):
    path = model_dir / SHARD
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(wkey), handle.get_tensor(skey)


class Packed(nn.Module):
    def __init__(self, w, s):
        super().__init__()
        self.weight = nn.Parameter(w, requires_grad=False)
        self.weight_scale = nn.Parameter(s, requires_grad=False)


def time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True)
    b = torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


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


def main() -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
