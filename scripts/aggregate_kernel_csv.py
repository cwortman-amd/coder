#!/usr/bin/env python3
"""Rank rocprofv3 kernel CSV rows by total GPU duration."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def classify(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("dflash", "draft", "selector")):
        return "dflash_draft_or_select"
    if any(token in lower for token in ("mxfp4", "fp4", "aiter", "gemm", "gemm_")):
        return "mxfp4_or_gemm"
    if any(token in lower for token in ("attn", "attention", "gdn", "mamba", "delta")):
        return "attention_or_state"
    if any(token in lower for token in ("softmax", "sample", "logit", "lm_head", "argmax")):
        return "logits_or_sample"
    if any(token in lower for token in ("copy", "memcpy", "memset", "alloc")):
        return "memory_ops"
    return "other"


def pick(fieldnames: list[str], *needles: str) -> str | None:
    lowered = {name.lower(): name for name in fieldnames}
    for needle in needles:
        for key, orig in lowered.items():
            if needle in key:
                return orig
    return None


def load_kernel_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    files = sorted(Path(args.input_dir).rglob("*.csv"))
    kernel_files = [path for path in files if "kernel" in path.name.lower()]
    if not kernel_files:
        kernel_files = files
    if not kernel_files:
        raise SystemExit(f"no CSV files under {args.input_dir}")

    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "sum_ns": 0.0, "max_ns": 0.0})
    groups: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "sum_ns": 0.0})
    columns: list[str] = []
    used_files: list[str] = []

    for path in kernel_files:
        fieldnames, rows = load_kernel_rows(path)
        if not rows:
            continue
        name_col = pick(fieldnames, "kernel_name", "kernelname", "name")
        dur_col = pick(fieldnames, "duration")
        if not name_col or not dur_col:
            continue
        columns = fieldnames
        used_files.append(str(path))
        for row in rows:
            name = row.get(name_col) or "<unknown>"
            try:
                duration = float(row.get(dur_col) or 0.0)
            except ValueError:
                continue
            entry = grouped[name]
            entry["count"] += 1
            entry["sum_ns"] += duration
            entry["max_ns"] = max(entry["max_ns"], duration)
            bucket = groups[classify(name)]
            bucket["count"] += 1
            bucket["sum_ns"] += duration

    ranked = sorted(grouped.items(), key=lambda item: item[1]["sum_ns"], reverse=True)
    total = sum(item[1]["sum_ns"] for item in ranked) or 1.0
    top = []
    for name, stats in ranked[: args.top]:
        top.append(
            {
                "kernel": name,
                "group": classify(name),
                "count": int(stats["count"]),
                "sum_ns": stats["sum_ns"],
                "mean_ns": stats["sum_ns"] / stats["count"] if stats["count"] else 0,
                "max_ns": stats["max_ns"],
                "share": stats["sum_ns"] / total,
            }
        )
    group_table = {
        name: {
            "count": int(stats["count"]),
            "sum_ns": stats["sum_ns"],
            "share": stats["sum_ns"] / total,
        }
        for name, stats in sorted(groups.items(), key=lambda item: item[1]["sum_ns"], reverse=True)
    }
    result = {
        "files": used_files,
        "columns": columns,
        "total_duration_ns": total if ranked else 0,
        "groups": group_table,
        "top_kernels": top,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"files": used_files, "kernels": len(ranked), "groups": {k: round(v["share"], 4) for k, v in group_table.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
