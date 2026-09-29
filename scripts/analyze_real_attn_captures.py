#!/usr/bin/env python3
"""Compare real per-layer captures from three attention backends."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


BACKENDS = ("stock", "triton", "aiter")


def load_capture(directory: Path) -> list[dict]:
    return [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted(directory.glob("layer_*.pt"))
    ]


def reference(payload: dict) -> torch.Tensor:
    query = payload["query"].double()
    key = payload["key"].double()
    value = payload["value"].double()
    tokens, query_heads, head_size = query.shape
    kv_heads = key.shape[1]
    group = query_heads // kv_heads
    output = torch.empty_like(query)
    mask = torch.ones((tokens, tokens), dtype=torch.bool).triu(diagonal=1)
    for head in range(query_heads):
        kv_head = head // group
        scores = torch.matmul(query[:, head], key[:, kv_head].transpose(0, 1))
        scores.mul_(float(payload["scale"])).masked_fill_(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        output[:, head] = torch.matmul(probs, value[:, kv_head])
    return output


def output(payload: dict) -> torch.Tensor:
    value = payload.get("backend_output")
    if value is None:
        value = payload["stock_output"]
    return value


def metrics(left: torch.Tensor, right: torch.Tensor) -> dict:
    left = left.double()
    right = right.double()
    diff = (left - right).abs()
    last = diff[-1]
    return {
        "exact": bool(torch.equal(left, right)),
        "max_abs_all": diff.max().item(),
        "mean_abs_all": diff.mean().item(),
        "max_abs_last": last.max().item(),
        "mean_abs_last": last.mean().item(),
        "rmse_last": last.square().mean().sqrt().item(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True)
    parser.add_argument("--triton", required=True)
    parser.add_argument("--aiter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    captures = {
        "stock": load_capture(Path(args.stock)),
        "triton": load_capture(Path(args.triton)),
        "aiter": load_capture(Path(args.aiter)),
    }
    if {len(rows) for rows in captures.values()} != {16}:
        raise SystemExit({name: len(rows) for name, rows in captures.items()})

    result_rows = []
    for index in range(16):
        layer = {name: captures[name][index] for name in BACKENDS}
        row = {
            "capture_index": index,
            "model_layer_index": layer["stock"]["model_layer_index"],
            "inputs": {},
            "outputs": {},
            "output_vs_own_fp64": {},
        }
        for tensor_name in ("query", "key", "value"):
            row["inputs"][tensor_name] = {
                "triton_vs_stock": metrics(
                    layer["triton"][tensor_name], layer["stock"][tensor_name]
                ),
                "aiter_vs_stock": metrics(
                    layer["aiter"][tensor_name], layer["stock"][tensor_name]
                ),
                "triton_vs_aiter": metrics(
                    layer["triton"][tensor_name], layer["aiter"][tensor_name]
                ),
            }
        row["outputs"] = {
            "triton_vs_stock": metrics(output(layer["triton"]), output(layer["stock"])),
            "aiter_vs_stock": metrics(output(layer["aiter"]), output(layer["stock"])),
            "triton_vs_aiter": metrics(output(layer["triton"]), output(layer["aiter"])),
        }
        for name in BACKENDS:
            row["output_vs_own_fp64"][name] = metrics(
                output(layer[name]), reference(layer[name])
            )
        result_rows.append(row)
        q = row["inputs"]["query"]
        o = row["outputs"]
        ref = row["output_vs_own_fp64"]
        print(
            f"layer={row['model_layer_index']:02d} "
            f"Q(T-S)={q['triton_vs_stock']['max_abs_last']:.6g} "
            f"O(T-S)={o['triton_vs_stock']['max_abs_last']:.6g} "
            f"O(A-S)={o['aiter_vs_stock']['max_abs_last']:.6g} "
            f"ref_rmse S/T/A={ref['stock']['rmse_last']:.6g}/"
            f"{ref['triton']['rmse_last']:.6g}/{ref['aiter']['rmse_last']:.6g}",
            flush=True,
        )
    result = {
        "prompt": "reasoning_00",
        "prompt_tokens": 44,
        "stock_first_token": 20,
        "unified_first_token": 9764,
        "reference": "CPU FP64 causal GQA on each backend's own captured Q/K/V",
        "rows": result_rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(out_path), "layers": len(result_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
