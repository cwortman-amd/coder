"""Kernel short names shared by the all-reduce attribution scripts."""

from __future__ import annotations


def short_name(name: str) -> str:
    lowered = name.lower()
    if "cross_device_reduce" in lowered:
        return "custom_allreduce"
    if "gemm_afp4" in lowered or "dynamic_mxfp4" in lowered:
        return "mxfp4"
    if "nccldevkernel" in lowered or "nccl" in lowered or "rccl" in lowered:
        return "rccl"
    if "triton" in lowered:
        return "triton"
    if any(
        token in lowered
        for token in (
            "paged_attention",
            "reshape_and_cache",
            "gated_delta",
            "causal_conv",
        )
    ):
        return "attention"
    return name.split("(")[0][-80:]
