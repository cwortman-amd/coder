#!/usr/bin/env python3
"""Clip a rocprof kernel CSV to a recorded UTC wall-clock window.

rocprof kernel timestamps use CLOCK_MONOTONIC nanoseconds on this host. The
script samples realtime-minus-monotonic to map ISO UTC marker files into that
domain, writes a gzip-compressed row-level trace, and emits an aggregate JSON.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def classify(name: str) -> str:
    lower = name.lower()
    if "_dynamic_mxfp4_quant" in lower:
        return "mxfp4_quant"
    if "_gemm_afp4wfp4" in lower:
        return "mxfp4_gemm_and_reduce"
    if "wvsplitk" in lower:
        return "large_splitk_likely_lm_head"
    if any(
        token in lower
        for token in ("paged_attention", "fmha", "unified_attention", "reshape_and_cache")
    ):
        return "attention_and_kv"
    if any(
        token in lower
        for token in (
            "delta_rule",
            "causal_conv",
            "mamba",
            "chunk_scaled",
            "chunk_local",
            "recompute_w_u",
            "post_conv",
        )
    ):
        return "gdn_and_state"
    if any(token in lower for token in ("gumbel", "argmax", "scatter_num_accepted")):
        return "sampling"
    if any(token in lower for token in ("copybuffer", "memcpy", "memset")):
        return "memory_copy"
    if lower.startswith("triton_"):
        return "compiled_elementwise_norm"
    return "other"


def parse_utc(path: Path) -> int:
    text = path.read_text(encoding="utf-8").strip()
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return int(value.astimezone(timezone.utc).timestamp() * 1e9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--end", type=Path, required=True)
    parser.add_argument("--out-trace", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--padding-ms", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    realtime_minus_monotonic = time.time_ns() - time.monotonic_ns()
    padding_ns = int(args.padding_ms * 1e6)
    start_ns = parse_utc(args.start) - realtime_minus_monotonic - padding_ns
    end_ns = parse_utc(args.end) - realtime_minus_monotonic + padding_ns

    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "total_ns": 0, "min_ns": 2**63 - 1, "max_ns": 0}
    )
    durations_by_kernel: dict[str, list[int]] = defaultdict(list)
    selected = 0
    first_ns: int | None = None
    last_ns: int | None = None

    args.out_trace.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.input.open(newline="", encoding="utf-8") as source,
        gzip.open(args.out_trace, "wt", newline="", encoding="utf-8") as target,
    ):
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise SystemExit("input trace has no CSV header")
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            try:
                row_start = int(row["Start_Timestamp"])
                row_end = int(row["End_Timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            if row_end < start_ns or row_start > end_ns:
                continue

            writer.writerow(row)
            selected += 1
            first_ns = row_start if first_ns is None else min(first_ns, row_start)
            last_ns = row_end if last_ns is None else max(last_ns, row_end)
            duration = max(row_end - row_start, 0)
            name = row.get("Kernel_Name") or "<unknown>"
            stats = grouped[name]
            stats["calls"] += 1
            stats["total_ns"] += duration
            stats["min_ns"] = min(stats["min_ns"], duration)
            stats["max_ns"] = max(stats["max_ns"], duration)
            durations_by_kernel[name].append(duration)

    total_ns = sum(stats["total_ns"] for stats in grouped.values())
    robust_total_ns = 0
    for name, values in durations_by_kernel.items():
        values.sort()
        stats = grouped[name]

        def percentile(q: float) -> int:
            return values[min(int((len(values) - 1) * q), len(values) - 1)]

        stats["p50_ns"] = percentile(0.50)
        stats["p95_ns"] = percentile(0.95)
        stats["p99_ns"] = percentile(0.99)
        stats["winsorized_p99_total_ns"] = sum(
            min(value, stats["p99_ns"]) for value in values
        )
        robust_total_ns += stats["winsorized_p99_total_ns"]

    ranked = sorted(
        grouped.items(),
        key=lambda item: item[1]["winsorized_p99_total_ns"],
        reverse=True,
    )
    top = []
    for name, stats in ranked[: args.top]:
        top.append(
            {
                "kernel": name,
                **stats,
                "mean_ns": stats["total_ns"] / stats["calls"],
                "share": stats["total_ns"] / total_ns if total_ns else 0,
                "winsorized_p99_share": (
                    stats["winsorized_p99_total_ns"] / robust_total_ns
                    if robust_total_ns
                    else 0
                ),
            }
        )

    categories: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "total_ns": 0, "winsorized_p99_total_ns": 0}
    )
    for name, stats in grouped.items():
        category = categories[classify(name)]
        category["calls"] += stats["calls"]
        category["total_ns"] += stats["total_ns"]
        category["winsorized_p99_total_ns"] += stats["winsorized_p99_total_ns"]
    category_table = {
        name: {
            **stats,
            "share": stats["total_ns"] / total_ns if total_ns else 0,
            "winsorized_p99_share": (
                stats["winsorized_p99_total_ns"] / robust_total_ns
                if robust_total_ns
                else 0
            ),
        }
        for name, stats in sorted(
            categories.items(),
            key=lambda item: item[1]["winsorized_p99_total_ns"],
            reverse=True,
        )
    }

    summary = {
        "source": str(args.input),
        "trace": str(args.out_trace),
        "wall_start_utc": args.start.read_text(encoding="utf-8").strip(),
        "wall_end_utc": args.end.read_text(encoding="utf-8").strip(),
        "realtime_minus_monotonic_ns": realtime_minus_monotonic,
        "window_start_monotonic_ns": start_ns,
        "window_end_monotonic_ns": end_ns,
        "first_selected_timestamp_ns": first_ns,
        "last_selected_timestamp_ns": last_ns,
        "selected_dispatches": selected,
        "unique_kernels": len(grouped),
        "summed_kernel_duration_ns": total_ns,
        "winsorized_p99_summed_kernel_duration_ns": robust_total_ns,
        "note": (
            "Summed kernel duration can overlap across queues and is a ranking "
            "weight, not necessarily wall time. Ranking and robust share "
            "winsorize each kernel's top 1% dispatch durations at its p99 to "
            "suppress profiler/stop outliers while preserving KV-length effects."
        ),
        "categories": category_table,
        "top_kernels": top,
    }
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected_dispatches": selected,
                "unique_kernels": len(grouped),
                "summed_kernel_duration_ns": total_ns,
                "out_trace": str(args.out_trace),
                "out_summary": str(args.out_summary),
            }
        )
    )
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
