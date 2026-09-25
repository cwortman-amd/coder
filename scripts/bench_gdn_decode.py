#!/usr/bin/env python3
"""Isolated Qwen3.8-27B GDN decode ops (FLA packed path used in production).

AITER fused GDN decode is Qwen3-Next interleaved-GQA only; this checkpoint
is Qwen3.5 non-interleaved, so VLLM_ROCM_USE_AITER=1 does not replace this path.
fused_gdn_decode_post_conv_mtp is not built on the ROCm image.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)

BABEL = 3793.4139104873675
H_K = 16
H_V = 48
D_K = 128
D_V = 128
CONV_W = 4
QKV = H_K * D_K * 2 + H_V * D_V  # 10240
N_GDN_LAYERS = 48  # 64 layers, full_attention every 4th


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/results/profiling/gdn_decode/summary.json")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--ms", default="1,2,4,8")
    args = p.parse_args()
    device = "cuda"
    rows = []
    print(
        json.dumps(
            {
                "qkv_dim": QKV,
                "h_k": H_K,
                "h_v": H_V,
                "d_k": D_K,
                "d_v": D_V,
                "gdn_layers": N_GDN_LAYERS,
            }
        ),
        flush=True,
    )

    for m in [int(x) for x in args.ms.split(",") if x]:
        mixed = torch.randn(m, QKV, device=device, dtype=torch.bfloat16)
        conv_state = torch.zeros(
            args.slots, QKV, CONV_W - 1, device=device, dtype=torch.bfloat16
        )
        conv_w = torch.randn(QKV, CONV_W, device=device, dtype=torch.bfloat16)
        conv_b = torch.zeros(QKV, device=device, dtype=torch.bfloat16)
        idx = torch.arange(m, device=device, dtype=torch.int32) % args.slots
        a = torch.randn(m, H_V, device=device, dtype=torch.bfloat16)
        b = torch.randn(m, H_V, device=device, dtype=torch.bfloat16)
        a_log = torch.randn(H_V, device=device, dtype=torch.float32)
        dt_bias = torch.randn(H_V, device=device, dtype=torch.float32)
        state = torch.zeros(
            args.slots, H_V, D_V, D_K, device=device, dtype=torch.float32
        )
        out = torch.empty(m, 1, H_V, D_V, device=device, dtype=torch.bfloat16)

        def conv():
            return causal_conv1d_update(
                mixed,
                conv_state,
                conv_w,
                conv_b,
                "silu",
                conv_state_indices=idx,
                validate_data=False,
            )

        def rec(mixed_qkv=None):
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mixed_qkv if mixed_qkv is not None else mixed,
                a=a,
                b=b,
                A_log=a_log,
                dt_bias=dt_bias,
                scale=D_K**-0.5,
                initial_state=state,
                out=out,
                ssm_state_indices=idx,
                use_qk_l2norm_in_kernel=True,
            )

        try:
            mixed_after = conv()
            rec(mixed_after.contiguous())
            ms_conv = time_ms(conv, args.warmup, args.iters)

            def rec_fixed(x=mixed_after.contiguous()):
                rec(x)

            ms_rec = time_ms(rec_fixed, args.warmup, args.iters)

            def both():
                y = conv()
                rec(y.contiguous() if not y.is_contiguous() else y)

            ms_both = time_ms(both, args.warmup, args.iters)
        except Exception as exc:  # noqa: BLE001
            print(f"M={m} FAIL {type(exc).__name__}: {exc}", flush=True)
            rows.append({"m": m, "ok": False, "error": str(exc)})
            continue

        tok_ms = ms_both * N_GDN_LAYERS
        rec_tok = {
            "m": m,
            "ok": True,
            "conv_ms": ms_conv,
            "recurrent_ms": ms_rec,
            "pair_ms": ms_both,
            "pair_48_layers_ms": tok_ms,
            "c1_tok_s_if_gdn_only": 1000.0 / tok_ms if m == 1 else None,
        }
        rows.append(rec_tok)
        print(
            f"M={m} conv={ms_conv:.3f} ms  rec={ms_rec:.3f} ms  "
            f"pair={ms_both:.3f} ms  x48={tok_ms:.2f} ms/tok "
            f"({1000/tok_ms if m==1 else 0:.1f} tok/s if GDN-only)",
            flush=True,
        )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "note": "FLA packed decode; not fused CUDA; not AITER Qwen3-Next path",
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
