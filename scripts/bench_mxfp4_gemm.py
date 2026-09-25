#!/usr/bin/env python3
"""Isolated AITER MXFP4 GEMM vs Babel Read roof (no vLLM EngineCore).

Loads real Qwen3.8-27B layer-0 packed weights and times
torch.ops.vllm.gemm_with_dynamic_quant at decode-like M.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from vllm.model_executor.kernels.linear.mxfp4.aiter import AiterMxfp4LinearKernel
from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearLayerConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic

BABEL_READ_GBS = 3793.4139104873675
SPEC_GBS = 4096.0

LAYERS = {
    "mlp.gate_proj": {
        "weight": "model.language_model.layers.0.mlp.gate_proj.weight",
        "scale": "model.language_model.layers.0.mlp.gate_proj.weight_scale",
        "shard": "model-00007-of-00039.safetensors",
    },
    "mlp.up_proj": {
        "weight": "model.language_model.layers.0.mlp.up_proj.weight",
        "scale": "model.language_model.layers.0.mlp.up_proj.weight_scale",
        "shard": "model-00007-of-00039.safetensors",
    },
    "mlp.down_proj": {
        "weight": "model.language_model.layers.0.mlp.down_proj.weight",
        "scale": "model.language_model.layers.0.mlp.down_proj.weight_scale",
        "shard": "model-00007-of-00039.safetensors",
    },
    "linear_attn.out_proj": {
        "weight": "model.language_model.layers.0.linear_attn.out_proj.weight",
        "scale": "model.language_model.layers.0.linear_attn.out_proj.weight_scale",
        "shard": "model-00007-of-00039.safetensors",
    },
    "linear_attn.in_proj_qkv": {
        "weight": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "scale": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale",
        "shard": "model-00007-of-00039.safetensors",
    },
}


class PackedLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(scale, requires_grad=False)


def load_pair(model_dir: Path, spec: dict[str, str]) -> tuple[torch.Tensor, torch.Tensor]:
    path = model_dir / spec["shard"]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(spec["weight"])
        scale = handle.get_tensor(spec["scale"])
    return weight, scale


def time_op(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # milliseconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/models/Qwen3.8-27B-Quark-AWQ-MXFP4-sharded",
    )
    parser.add_argument("--out", default="/results/profiling/mxfp4_gemm/summary.json")
    parser.add_argument("--ms", default="1,2,4,8,16,32,64")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--layers",
        default="mlp.gate_proj,mlp.up_proj,mlp.down_proj,linear_attn.out_proj,linear_attn.in_proj_qkv",
    )
    args = parser.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    if not torch.cuda.is_available():
        raise SystemExit("cuda/hip not available")
    cfg = MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic)
    kernel = AiterMxfp4LinearKernel(cfg)
    print(
        json.dumps(
            {
                "use_asm_gemm": kernel.use_asm_gemm,
                "out_dtype": str(kernel.out_dtype),
                "device": str(torch.cuda.get_device_name(0)),
            }
        ),
        flush=True,
    )

    model_dir = Path(args.model_dir)
    ms = [int(x) for x in args.ms.split(",") if x]
    rows: list[dict] = []

    for name in args.layers.split(","):
        name = name.strip()
        spec = LAYERS[name]
        weight_cpu, scale_cpu = load_pair(model_dir, spec)
        layer = PackedLinear(weight_cpu.cuda(), scale_cpu.cuda())
        kernel.process_weights_after_loading(layer)
        weight = layer.weight.data
        scale = layer.weight_scale.data
        n, packed_k = weight.shape[0], weight.shape[1]
        # packed uint8: two FP4 values per byte along K
        k = packed_k * 2
        w_bytes = int(weight.numel() * weight.element_size())
        s_bytes = int(scale.numel() * scale.element_size())
        print(
            f"layer={name} weight={tuple(weight.shape)}/{weight.dtype} "
            f"scale={tuple(scale.shape)}/{scale.dtype} N={n} K={k} "
            f"W+S={ (w_bytes + s_bytes) / 1e9:.4f} GB asm={kernel.use_asm_gemm}",
            flush=True,
        )

        for m in ms:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            x_bytes = int(x.numel() * x.element_size())
            y_bytes = m * n * 2  # bf16 out

            def run() -> None:
                kernel.apply_weights(layer, x)

            # one correctness-ish launch
            y = kernel.apply_weights(layer, x)
            assert y.shape == (m, n), (y.shape, m, n)
            ms_iter = time_op(run, args.warmup if m <= 8 else max(20, args.warmup // 2), args.iters)
            s_iter = ms_iter / 1e3
            weight_gbs = (w_bytes + s_bytes) / s_iter / 1e9
            traffic_gbs = (w_bytes + s_bytes + x_bytes + y_bytes) / s_iter / 1e9
            tok_equiv = BABEL_READ_GBS / ((w_bytes + s_bytes) / 1e9) if m == 1 else None
            rows.append(
                {
                    "layer": name,
                    "m": m,
                    "n": n,
                    "k": k,
                    "weight_shape": list(weight.shape),
                    "scale_shape": list(scale.shape),
                    "ms": ms_iter,
                    "weight_scale_bytes": w_bytes + s_bytes,
                    "x_bytes": x_bytes,
                    "y_bytes": y_bytes,
                    "weight_gbs": weight_gbs,
                    "traffic_gbs": traffic_gbs,
                    "weight_gbs_over_babel": weight_gbs / BABEL_READ_GBS,
                    "weight_gbs_over_spec": weight_gbs / SPEC_GBS,
                }
            )
            print(
                f"  M={m:3d} {ms_iter:8.3f} ms  weight-stream {weight_gbs:7.1f} GB/s "
                f"({100 * weight_gbs / BABEL_READ_GBS:5.1f}% Babel Read) "
                f"all-traffic {traffic_gbs:7.1f} GB/s",
                flush=True,
            )
            del y, x
            torch.cuda.empty_cache()
        del layer, weight, scale, weight_cpu, scale_cpu
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "use_asm_gemm": bool(kernel.use_asm_gemm),
        "babel_read_gbs": BABEL_READ_GBS,
        "spec_gbs": SPEC_GBS,
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
