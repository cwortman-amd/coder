#!/usr/bin/env python3
"""Replay real stock Q/K/V through both unified kernels and a FP64 reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aiter.ops.triton.attention.unified_attention import (
    unified_attention as aiter_unified_attention,
)
from vllm.v1.attention.ops.triton_unified_attention import (
    unified_attention as triton_unified_attention,
)


PAGES = {"triton": 784, "aiter": 832}


def pack(key: torch.Tensor, value: torch.Tensor, page: int):
    tokens, kv_heads, head_size = key.shape
    blocks = (tokens + page - 1) // page
    cache = torch.zeros(
        (blocks, kv_heads, page, 2 * head_size),
        dtype=key.dtype,
        device=key.device,
    )
    for block in range(blocks):
        start = block * page
        stop = min(start + page, tokens)
        width = stop - start
        cache[block, :, :width, :head_size] = key[start:stop].permute(1, 0, 2)
        cache[block, :, :width, head_size:] = value[start:stop].permute(1, 0, 2)
    viewed = cache.transpose(1, 2)
    key_cache, value_cache = viewed.split(head_size, dim=-1)
    table = torch.arange(blocks, dtype=torch.int32, device=key.device).view(1, blocks)
    return key_cache, value_cache, table


def reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """CPU FP64 causal GQA attention."""
    query = query.double()
    key = key.double()
    value = value.double()
    tokens, query_heads, head_size = query.shape
    kv_heads = key.shape[1]
    group = query_heads // kv_heads
    out = torch.empty((tokens, query_heads, head_size), dtype=torch.float64)
    for position in range(tokens):
        for head in range(query_heads):
            kv_head = head // group
            scores = torch.mv(key[: position + 1, kv_head], query[position, head])
            probs = torch.softmax(scores * scale, dim=0)
            out[position, head] = torch.mv(
                value[: position + 1, kv_head].transpose(0, 1),
                probs,
            )
    return out


def replay(name: str, payload: dict, device: torch.device) -> torch.Tensor:
    query = payload["query"].to(device)
    key = payload["key"].to(device)
    value = payload["value"].to(device)
    key_cache, value_cache, table = pack(key, value, PAGES[name])
    tokens, query_heads, head_size = query.shape
    kv_heads = key.shape[1]
    output = torch.empty_like(query)
    cu = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([tokens], dtype=torch.int32, device=device)
    scale = float(payload["scale"])
    k_descale = torch.ones((1, kv_heads), dtype=torch.float32, device=device)
    v_descale = torch.ones((1, kv_heads), dtype=torch.float32, device=device)
    if name == "triton":
        triton_unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu,
            max_seqlen_q=tokens,
            seqused_k=seq_lens,
            max_seqlen_k=tokens,
            softmax_scale=scale,
            causal=True,
            window_size=tuple(payload["sliding_window"]),
            block_table=table,
            softcap=float(payload["logits_soft_cap"]),
            q_descale=None,
            k_descale=k_descale,
            v_descale=v_descale,
        )
    else:
        aiter_unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu,
            max_seqlen_q=tokens,
            seqused_k=seq_lens,
            max_seqlen_k=tokens,
            softmax_scale=scale,
            causal=True,
            window_size=tuple(payload["sliding_window"]),
            block_table=table,
            softcap=float(payload["logits_soft_cap"]),
            q_descale=None,
            k_descale=k_descale,
            v_descale=v_descale,
        )
    torch.cuda.synchronize()
    return output.cpu()


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.double()
    expected = expected.double()
    diff = (actual - expected).abs()
    last = diff[-1]
    actual_last = actual[-1].flatten()
    expected_last = expected[-1].flatten()
    cosine = torch.nn.functional.cosine_similarity(
        actual_last.unsqueeze(0),
        expected_last.unsqueeze(0),
    ).item()
    return {
        "max_abs_all": diff.max().item(),
        "mean_abs_all": diff.mean().item(),
        "max_abs_last": last.max().item(),
        "mean_abs_last": last.mean().item(),
        "rmse_last": last.square().mean().sqrt().item(),
        "cosine_last": cosine,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    rows = []
    for path in sorted(Path(args.capture_dir).glob("layer_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ref = reference(
            payload["query"],
            payload["key"],
            payload["value"],
            float(payload["scale"]),
        )
        stock = payload["stock_output"]
        triton = replay("triton", payload, device)
        aiter = replay("aiter", payload, device)
        row = {
            "file": path.name,
            "capture_index": payload["capture_index"],
            "model_layer_index": payload["model_layer_index"],
            "tokens": payload["num_actual_tokens"],
            "query_shape": list(payload["query"].shape),
            "query_stride": list(payload["query_stride"]),
            "key_stride": list(payload["key_stride"]),
            "value_stride": list(payload["value_stride"]),
            "stock_vs_fp64": metrics(stock, ref),
            "triton_vs_fp64": metrics(triton, ref),
            "aiter_vs_fp64": metrics(aiter, ref),
            "triton_vs_stock": metrics(triton, stock),
            "aiter_vs_stock": metrics(aiter, stock),
            "triton_vs_aiter": metrics(triton, aiter),
        }
        rows.append(row)
        print(
            f"layer={row['model_layer_index']:02d} "
            f"last max stock={row['stock_vs_fp64']['max_abs_last']:.6g} "
            f"triton={row['triton_vs_fp64']['max_abs_last']:.6g} "
            f"aiter={row['aiter_vs_fp64']['max_abs_last']:.6g} "
            f"t-a={row['triton_vs_aiter']['max_abs_last']:.6g}",
            flush=True,
        )
    result = {
        "reference": "CPU FP64 causal GQA attention on captured stock Q/K/V",
        "triton_page": PAGES["triton"],
        "aiter_page": PAGES["aiter"],
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(out), "layers": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
