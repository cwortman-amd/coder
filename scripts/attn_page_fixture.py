#!/usr/bin/env python3
"""Compare paged decode attention against one FP32 reference.

Packs the same logical K/V into the cache layout both backends consume:
[num_blocks, num_kv_heads, page, 2 * head_size], then the transpose-and-split
view used by TRITON_ATTN and ROCM_AITER_UNIFIED_ATTN. Page size is 784 for
the Triton path and 832 for AITER unified, matching this hybrid model's
startup logs. The reference attends over the logical token sequence, so a
page-boundary indexing error shows up as a large miss on the spike cases.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from aiter.ops.triton.attention.unified_attention import (
    unified_attention as aiter_unified_attention,
)
from vllm.v1.attention.ops.triton_unified_attention import (
    unified_attention as triton_unified_attention,
)

HQ = 24
HKV = 4
HD = 256
SCALE = 1.0 / math.sqrt(HD)
PAGE_TRITON = 784
PAGE_AITER = 832
SEQ_THRESHOLD_3D = 32  # nearest CUDA-graph capture size to 128 / 4 KV heads
SEGMENTS = 16
WINDOW = (-1, -1)


def pack_cache(keys: list[torch.Tensor], values: list[torch.Tensor], page: int):
    """keys/values: each [S, HKV, HD] bf16 on device."""
    lengths = [k.shape[0] for k in keys]
    blocks_per = [(length + page - 1) // page for length in lengths]
    total_blocks = sum(blocks_per)
    cache = torch.zeros(
        (total_blocks, HKV, page, 2 * HD),
        dtype=torch.bfloat16,
        device=keys[0].device,
    )
    table = torch.zeros((len(keys), max(blocks_per)), dtype=torch.int32, device=keys[0].device)
    cursor = 0
    for seq, (key, value, nblocks) in enumerate(zip(keys, values, blocks_per)):
        table[seq, :nblocks] = torch.arange(cursor, cursor + nblocks, device=key.device, dtype=torch.int32)
        flat_k = key.reshape(-1, HKV, HD)
        flat_v = value.reshape(-1, HKV, HD)
        for block in range(nblocks):
            start = block * page
            stop = min(start + page, key.shape[0])
            width = stop - start
            cache[cursor + block, :, :width, :HD] = flat_k[start:stop].permute(1, 0, 2)
            cache[cursor + block, :, :width, HD:] = flat_v[start:stop].permute(1, 0, 2)
        cursor += nblocks
    viewed = cache.transpose(1, 2)
    key_cache, value_cache = viewed.split(HD, dim=-1)
    return key_cache, value_cache, table, cache.stride()


def reference(q: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor]) -> torch.Tensor:
    """q: [B, HQ, HD]. Decode attends over every cached token."""
    outs = []
    for index, (key, value) in enumerate(zip(keys, values)):
        # key [S, HKV, HD] -> scores [HQ, S]
        group = HQ // HKV
        key_f = key.float()
        value_f = value.float()
        query = q[index].float()
        scores = torch.empty((HQ, key.shape[0]), dtype=torch.float32, device=q.device)
        for head in range(HQ):
            kv = head // group
            scores[head] = torch.matmul(key_f[:, kv, :], query[head]) * SCALE
        weights = torch.softmax(scores, dim=-1)
        out = torch.zeros((HQ, HD), dtype=torch.float32, device=q.device)
        for head in range(HQ):
            kv = head // group
            out[head] = torch.matmul(weights[head], value_f[:, kv, :])
        outs.append(out)
    return torch.stack(outs, dim=0)


def make_spike(lengths: list[int], position_fn, device: torch.device):
    keys, values = [], []
    queries = torch.zeros((len(lengths), HQ, HD), dtype=torch.bfloat16, device=device)
    queries[..., 0] = 1
    for length in lengths:
        pos = position_fn(length)
        key = torch.zeros((length, HKV, HD), dtype=torch.bfloat16, device=device)
        value = torch.zeros((length, HKV, HD), dtype=torch.bfloat16, device=device)
        key[pos, :, 0] = 300
        value[pos, :, :] = 1
        value[pos, :, 1] = torch.arange(HKV, device=device, dtype=torch.float32).to(torch.bfloat16)
        keys.append(key)
        values.append(value)
    return queries, keys, values


def make_random(lengths: list[int], device: torch.device, seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    keys, values = [], []
    queries = torch.randn((len(lengths), HQ, HD), generator=g, dtype=torch.float32).to(torch.bfloat16).to(device)
    for length in lengths:
        key = torch.randn((length, HKV, HD), generator=g, dtype=torch.float32).to(torch.bfloat16).to(device)
        value = torch.randn((length, HKV, HD), generator=g, dtype=torch.float32).to(torch.bfloat16).to(device)
        keys.append(key)
        values.append(value)
    return queries, keys, values


def stats(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float((diff / denom).max()),
    }


def run_backend(name: str, fn, q, key_cache, value_cache, table, lengths, repeats: int) -> tuple[torch.Tensor, dict]:
    batch = q.shape[0]
    cu = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    seqused = torch.tensor(lengths, dtype=torch.int32, device=q.device)
    out = torch.empty((batch, HQ, HD), dtype=torch.bfloat16, device=q.device)
    k_descale = torch.ones((batch, HKV), dtype=torch.float32, device=q.device)
    v_descale = torch.ones((batch, HKV), dtype=torch.float32, device=q.device)
    seg_out = torch.empty((SEQ_THRESHOLD_3D, HQ, SEGMENTS, HD), dtype=torch.float32, device=q.device)
    seg_max = torch.empty((SEQ_THRESHOLD_3D, HQ, SEGMENTS), dtype=torch.float32, device=q.device)
    seg_sum = torch.empty((SEQ_THRESHOLD_3D, HQ, SEGMENTS), dtype=torch.float32, device=q.device)

    def launch():
        if name == "triton":
            triton_unified_attention(
                q=q,
                k=key_cache,
                v=value_cache,
                out=out,
                cu_seqlens_q=cu,
                max_seqlen_q=1,
                seqused_k=seqused,
                max_seqlen_k=max(lengths),
                softmax_scale=SCALE,
                causal=True,
                window_size=WINDOW,
                block_table=table,
                softcap=0.0,
                q_descale=None,
                k_descale=k_descale,
                v_descale=v_descale,
                seq_threshold_3D=SEQ_THRESHOLD_3D,
                num_par_softmax_segments=SEGMENTS,
                softmax_segm_output=seg_out,
                softmax_segm_max=seg_max,
                softmax_segm_expsum=seg_sum,
            )
        else:
            aiter_unified_attention(
                q=q,
                k=key_cache,
                v=value_cache,
                out=out,
                cu_seqlens_q=cu,
                max_seqlen_q=1,
                seqused_k=seqused,
                max_seqlen_k=max(lengths),
                softmax_scale=SCALE,
                causal=True,
                window_size=WINDOW,
                block_table=table,
                softcap=0.0,
                q_descale=None,
                k_descale=k_descale,
                v_descale=v_descale,
            )

    for _ in range(2):
        launch()
    torch.cuda.synchronize()
    graph_ok = False
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            launch()
        torch.cuda.synchronize()
        graph_ok = True
    except Exception as exc:  # noqa: BLE001
        graph_error = str(exc).splitlines()[-1][:200]
    else:
        graph_error = None

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    if graph_ok:
        for _ in range(repeats):
            graph.replay()
    else:
        for _ in range(repeats):
            launch()
    end.record()
    end.synchronize()
    micros = start.elapsed_time(end) * 1000.0 / repeats
    copied = out.detach().clone()
    return copied, {
        "eager_or_graph": "graph" if graph_ok else "eager",
        "us_per_call": micros,
        "graph_error": graph_error,
    }


def evaluate(kind: str, lengths: list[int], queries, keys, values, repeats: int) -> dict:
    expected = reference(queries, keys, values)
    row = {
        "kind": kind,
        "batch": len(lengths),
        "lengths": lengths,
        "max_length": max(lengths),
    }
    outputs = {}
    for name, page in (("triton", PAGE_TRITON), ("aiter", PAGE_AITER)):
        key_cache, value_cache, table, strides = pack_cache(keys, values, page)
        actual, timing = run_backend(name, None, queries, key_cache, value_cache, table, lengths, repeats)
        outputs[name] = actual
        row[name] = {
            "page": page,
            "cache_stride_blocks_heads_page_dim": list(strides),
            "key_cache_shape": list(key_cache.shape),
            "key_cache_stride": list(key_cache.stride()),
            **timing,
            **stats(actual, expected),
        }
    row["triton_vs_aiter"] = stats(outputs["triton"], outputs["aiter"].float())
    # stats() casts both to float; aiter tensor is bf16. Fix by passing float via stats which casts.
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    device = torch.device("cuda")
    cases = []

    def add_spike(lengths: list[int], label: str, position_fn):
        queries, keys, values = make_spike(lengths, position_fn, device)
        cases.append(evaluate(f"spike:{label}", lengths, queries, keys, values, args.repeats))
        last = cases[-1]
        print(
            f"{last['kind']} B={last['batch']} S={last['lengths']} "
            f"triton max={last['triton']['max_abs']:.3e} {last['triton']['us_per_call']:.1f}us "
            f"aiter max={last['aiter']['max_abs']:.3e} {last['aiter']['us_per_call']:.1f}us",
            flush=True,
        )

    def add_random(lengths: list[int], seed: int):
        queries, keys, values = make_random(lengths, device, seed)
        cases.append(evaluate("random", lengths, queries, keys, values, args.repeats))
        last = cases[-1]
        print(
            f"random B={last['batch']} S={last['lengths']} "
            f"triton max={last['triton']['max_abs']:.3e} aiter max={last['aiter']['max_abs']:.3e}",
            flush=True,
        )

    c1_lengths = [32, 44, 128, 256, 512, 513, 783, 784, 785, 831, 832, 833, 1024, 2048, 3072, 8192, 16384]
    for length in c1_lengths:
        add_spike([length], "last", lambda n: n - 1)
        if length > PAGE_TRITON:
            add_spike([length], "before_784", lambda n: PAGE_TRITON - 1)
            add_spike([length], "at_784", lambda n: PAGE_TRITON)
        if length > PAGE_AITER:
            add_spike([length], "before_832", lambda n: PAGE_AITER - 1)
            add_spike([length], "at_832", lambda n: PAGE_AITER)

    for length in (1024, 8192):
        add_random([length], seed=length)
        add_spike([length] * 4, "last", lambda n: n - 1)
        add_spike([length] * 8, "last", lambda n: n - 1)
        add_random([length] * 4, seed=length + 4)
        add_random([length] * 8, seed=length + 8)

    add_spike([783, 784, 832, 2048], "last", lambda n: n - 1)
    add_random([512, 1024, 4096, 8192], seed=7)

    payload = {
        "heads_q": HQ,
        "heads_kv": HKV,
        "head_dim": HD,
        "scale": SCALE,
        "page_triton": PAGE_TRITON,
        "page_aiter": PAGE_AITER,
        "seq_threshold_3d": SEQ_THRESHOLD_3D,
        "segments": SEGMENTS,
        "dtype": "bfloat16",
        "reference": "fp32 softmax over the logical token sequence",
        "cases": cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    worst_t = max(cases, key=lambda row: row["triton"]["max_abs"])
    worst_a = max(cases, key=lambda row: row["aiter"]["max_abs"])
    print(json.dumps({
        "out": str(out),
        "n": len(cases),
        "worst_triton": [worst_t["kind"], worst_t["lengths"], worst_t["triton"]["max_abs"]],
        "worst_aiter": [worst_a["kind"], worst_a["lengths"], worst_a["aiter"]["max_abs"]],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
