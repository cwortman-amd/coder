"""Shared eager timer and stock MXFP4 tile for the gate-up and tile sweeps."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

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


def load_tensors(
    model_dir: Path,
    weight_key: str,
    scale_key: str,
    shard: str = SHARD,
):
    path = model_dir / shard
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(weight_key), handle.get_tensor(scale_key)


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
