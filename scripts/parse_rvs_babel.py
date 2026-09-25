#!/usr/bin/env python3
"""Parse RVS BABEL logs. Prefers the MiBytes/sec table (mibibytes:true)."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

OPS = ("Copy", "Mul", "Add", "Triad", "Dot", "Read", "Write")
MIB_TO_GBS = (1024**2) / 1e9  # MiB/s → decimal GB/s

TABLE_ROW = re.compile(
    r"^\s*\d+\s+("
    + "|".join(OPS)
    + r")\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s*$"
)
UNIT_ROW = re.compile(
    r"\b("
    + "|".join(OPS)
    + r")\b\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(MiBytes/sec|MiB/s|GiB/s|GB/s|GB/sec|GiB/sec)",
    re.I,
)


def to_gbs(val: float, unit: str) -> float:
    u = unit.lower().replace("sec", "s").replace("bytes", "b")
    if u.startswith("mib") or u.startswith("miby"):
        return val * MIB_TO_GBS
    if u.startswith("gib"):
        return val * 1.073741824
    return val


def parse_text(text: str) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {op: [] for op in OPS}
    for line in text.splitlines():
        m = TABLE_ROW.match(line.replace("\r", ""))
        if m:
            op = m.group(1).title()
            # Columns: Function, MiBytes/sec, Max, Min, Avg. Use Avg across kernel iters.
            found[op].append(to_gbs(float(m.group(5)), "MiB/s"))
            continue
        m = UNIT_ROW.search(line)
        if m:
            op = m.group(1).title()
            found[op].append(to_gbs(float(m.group(2)), m.group(3)))
    return found


def summarize(xs: list[float]) -> dict | None:
    if not xs:
        return None
    return {
        "n": len(xs),
        "min_gbs": min(xs),
        "median_gbs": statistics.median(xs),
        "max_gbs": max(xs),
        "mean_gbs": statistics.mean(xs),
        "samples_gbs": xs,
    }


def roof_tok_s(gbs: float | None, weight_gib: float) -> float | None:
    if not gbs:
        return None
    bytes_per_token = weight_gib * (1024**3)
    return (gbs * 1e9) / bytes_per_token


def collect_text(paths: list[Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.glob("rvs*.log")) + sorted(path.glob("babel*.log")) + sorted(
                path.glob("rvs_rep*.json")
            ):
                chunks.append(child.read_text(errors="replace"))
        elif path.is_file():
            chunks.append(path.read_text(errors="replace"))
    return "\n".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weight-gib", type=float, default=17.91)
    parser.add_argument("--spec-gbs", type=float, default=4096.0)
    parser.add_argument("--array-size", type=int, default=268435456)
    parser.add_argument("--name", default="AMD Instinct MI350P")
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--num-iter", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    text = collect_text(args.inputs)
    found = parse_text(text)
    ops_sum = {k: summarize(v) for k, v in found.items()}
    read = ops_sum.get("Read") or {}
    copy = ops_sum.get("Copy") or {}
    triad = ops_sum.get("Triad") or {}
    ws_gib = args.array_size * 8 / (1024**3)
    result = {
        "gpu_name": args.name,
        "arch": args.arch,
        "array_size_elements": args.array_size,
        "approx_one_array_gib": ws_gib,
        "approx_triad_working_set_gib": 3 * ws_gib,
        "num_iter": args.num_iter,
        "repeats": args.repeats,
        "bandwidth_decimal_gbs": ops_sum,
        "spec_peak_gbs": args.spec_gbs,
        "assumed_weight_gib_per_token": args.weight_gib,
        "conditional_decode_tok_s": {
            "from_read_median": roof_tok_s(read.get("median_gbs"), args.weight_gib),
            "from_copy_median": roof_tok_s(copy.get("median_gbs"), args.weight_gib),
            "from_triad_median": roof_tok_s(triad.get("median_gbs"), args.weight_gib),
        },
        "parser": "MiBytes/sec table Avg column, fallback unit regex; never treat pebb/pbqt as this roof",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines: list[str] = []
    if not any(ops_sum[k] for k in OPS):
        lines.append(
            "ERROR: no Copy/Read/Triad bandwidth parsed. Expected RVS `MiBytes/sec` table or GiB/s lines."
        )
        rc = 1
    else:

        def fmt(op: str) -> str:
            s = ops_sum[op]
            if not s:
                return f"{op}: (not parsed)"
            return (
                f"{op}: median {s['median_gbs']:.1f} GB/s "
                f"(min {s['min_gbs']:.1f}, max {s['max_gbs']:.1f}, n={s['n']})"
            )

        lines.extend([fmt("Read"), fmt("Copy"), fmt("Triad")])
        read_med = read.get("median_gbs") if read else None
        if args.spec_gbs and read_med:
            lines.append(f"Read / spec {args.spec_gbs:.0f} GB/s = {100.0 * read_med / args.spec_gbs:.1f}%")
        if read_med:
            lines.append(
                f"RVS BABEL Read sustained {read_med:.1f} GB/s on {args.name} {args.arch}, "
                f"using {ws_gib:.2f} GiB per array (~{3 * ws_gib:.2f} GiB triad set), "
                f"{args.repeats} repetitions, {args.num_iter} kernel iterations."
            )
            lines.append(
                f"Conditional C1 decode roof (Read / {args.weight_gib} GiB/token) ≈ "
                f"{roof_tok_s(read_med, args.weight_gib):.1f} tok/s. This is synthetic bandwidth ÷ assumed "
                f"full-weight stream, not measured vLLM DRAM traffic."
            )
        rc = 0
    (args.out_dir / "REPORT.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out_dir / 'summary.json'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
