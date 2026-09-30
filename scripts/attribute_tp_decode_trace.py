#!/usr/bin/env python3
"""Attribute one steady decode step from a gated rocprofv3 kernel trace.

The step marker is ``_topk_topp_kernel``, which fired once per generated
token on the TP=1 window. Intervals longer than 30 ms are the prefill
boundaries and are excluded. Category times are sums of kernel durations
inside the step. On the TP=1 trace those sums matched the step wall clock,
so the kernels were effectively serialized.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def category(name: str) -> str:
    lowered = name.lower()
    if "cross_device_reduce" in lowered:
        return "custom_allreduce"
    if "nccldevkernel" in lowered or "nccl" in lowered or "rccl" in lowered:
        return "rccl_kernel"
    if "gemm_afp4" in lowered or "dynamic_mxfp4" in lowered:
        return "mxfp4"
    if "copybuffer" in lowered or "memcpy" in lowered:
        return "copy"
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
    return "other"


def load_kernels(path: Path) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    int(row["Start_Timestamp"]),
                    int(row["End_Timestamp"]),
                    row["Kernel_Name"],
                )
            )
    rows.sort()
    return rows


def steady_steps(
    rows: list[tuple[int, int, str]],
) -> list[tuple[float, Counter[str], Counter[str]]]:
    markers = [(start, end) for start, end, name in rows if "_topk_topp_kernel" in name]
    deltas = [
        (markers[index + 1][1] - markers[index][1]) / 1e6
        for index in range(len(markers) - 1)
    ]
    ordered = sorted(deltas)
    low = ordered[len(ordered) // 4]
    high = ordered[(len(ordered) * 3) // 4]
    steps: list[tuple[float, Counter[str], Counter[str]]] = []
    for index in range(len(markers) - 1):
        begin = markers[index][1]
        finish = markers[index + 1][1]
        wall_ms = (finish - begin) / 1e6
        if wall_ms < low or wall_ms > high:
            continue
        durations: Counter[str] = Counter()
        counts: Counter[str] = Counter()
        for start, end, name in rows:
            if begin <= start and end <= finish:
                kind = category(name)
                durations[kind] += end - start
                counts[kind] += 1
        steps.append((wall_ms, durations, counts))
    return steps


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def summarize(trace: Path) -> dict[str, object]:
    kernel_files = sorted(trace.glob("*_kernel_trace.csv"))
    ranks = []
    for kernel_file in kernel_files:
        rows = load_kernels(kernel_file)
        if not rows:
            continue
        steps = steady_steps(rows)
        if not steps:
            ranks.append({"file": kernel_file.name, "steady_steps": 0})
            continue
        walls = [wall for wall, _, _ in steps]
        entry: dict[str, object] = {
            "file": kernel_file.name,
            "kernels": len(rows),
            "span_ms": (rows[-1][1] - rows[0][0]) / 1e6,
            "steady_steps": len(steps),
            "wall_ms_median": median(walls),
            "categories": {},
        }
        categories: dict[str, dict[str, float]] = {}
        for kind in (
            "custom_allreduce",
            "rccl_kernel",
            "mxfp4",
            "attention",
            "copy",
            "other",
        ):
            times = [durations[kind] / 1e6 for _, durations, _ in steps]
            counts = [float(counts[kind]) for _, _, counts in steps]
            categories[kind] = {
                "ms_median": median(times),
                "calls_median": median(counts),
            }
        entry["categories"] = categories
        ranks.append(entry)
    return {"ranks": ranks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.trace_dir)
    text = json.dumps(summary, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
