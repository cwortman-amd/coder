#!/usr/bin/env python3
"""Split custom-allreduce launch time into arrival skew and synchronized work.

The two worker kernel traces share rocprof timestamps. Each
``cross_device_reduce`` launch is paired with the peer launch whose start is
nearest. Skew is the wait from the earlier start until the later start. The
synchronized portion is the time from that later start until both launches
have ended. Preceding kernels are the last kernel on the late rank that
ended at or before its own all-reduce start.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rocprof_trace import short_name  # noqa: E402


def load(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "agent": row["Agent_Id"].split()[-1],
                    "start": int(row["Start_Timestamp"]),
                    "end": int(row["End_Timestamp"]),
                    "name": row["Kernel_Name"],
                }
            )
    rows.sort(key=lambda item: (item["start"], item["end"]))
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "median_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "sum_ms": sum(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--early-pid", type=int, default=416)
    parser.add_argument("--late-pid", type=int, default=417)
    args = parser.parse_args()
    early_rows = load(args.trace_dir / f"decode-{args.early_pid}_kernel_trace.csv")
    late_rows = load(args.trace_dir / f"decode-{args.late_pid}_kernel_trace.csv")
    early_ar = [row for row in early_rows if "cross_device_reduce" in row["name"]]
    late_ar = [row for row in late_rows if "cross_device_reduce" in row["name"]]
    late_starts = [row["start"] for row in late_ar]

    # The two launches of one collective finish the end barrier together.
    # Pair in end-time order. A start-time match binds the short peer launch
    # to the next collective and hides the spin.
    last_other: dict | None = None
    preceding_for: dict[int, dict | None] = {}
    for row in late_rows:
        if "cross_device_reduce" in row["name"]:
            preceding_for[id(row)] = last_other
        else:
            last_other = row
    early_by_end = sorted(early_ar, key=lambda row: row["end"])
    late_by_end = sorted(late_ar, key=lambda row: row["end"])
    pairs = []
    unmatched = 0
    for early, late in zip(early_by_end, late_by_end, strict=False):
        end_gap = abs(early["end"] - late["end"])
        if end_gap > 5_000_000:
            unmatched += 1
            continue
        later_start = max(early["start"], late["start"])
        both_end = max(early["end"], late["end"])
        preceding = preceding_for.get(id(late))
        pairs.append(
            {
                "early_start": early["start"],
                "late_start": late["start"],
                "early_ms": (early["end"] - early["start"]) / 1e6,
                "late_ms": (late["end"] - late["start"]) / 1e6,
                "skew_ms": (late["start"] - early["start"]) / 1e6,
                "sync_ms": (both_end - later_start) / 1e6,
                "end_gap_ms": end_gap / 1e6,
                "preceding": short_name(preceding["name"]) if preceding else "",
                "preceding_ms": (
                    (preceding["end"] - preceding["start"]) / 1e6 if preceding else 0.0
                ),
                "preceding_full": preceding["name"] if preceding else "",
            }
        )

    # Long samples are the spins. The central half of all launch durations
    # is a few tens of microseconds and is the synchronized tail, not the wait.
    spinning = [pair for pair in pairs if pair["early_ms"] >= 1.0]
    late_starts = [row["start"] for row in late_rows]

    def covering(pair: dict) -> dict | None:
        """Late-rank kernel with the most overlap of the arrival wait."""
        begin = min(pair["early_start"], pair["late_start"])
        finish = max(pair["early_start"], pair["late_start"])
        if finish <= begin:
            return None
        # Kernels that start before the wait ends. Look back far enough to
        # include a kernel that started before the wait and is still running.
        right = bisect.bisect_left(late_starts, int(finish))
        best = None
        best_overlap = 0
        index = right - 1
        while index >= 0 and late_rows[index]["start"] > begin - 3_000_000_000:
            row = late_rows[index]
            overlap = min(row["end"], finish) - max(row["start"], begin)
            if overlap > best_overlap and "cross_device_reduce" not in row["name"]:
                best_overlap = overlap
                best = row
            if row["end"] < begin and (begin - row["start"]) > 3_000_000_000:
                break
            index -= 1
        return best

    for pair in spinning:
        cover = covering(pair)
        pair["cover"] = short_name(cover["name"]) if cover else ""
        pair["cover_ms"] = (
            (cover["end"] - cover["start"]) / 1e6 if cover else 0.0
        )
        pair["cover_full"] = cover["name"] if cover else ""
    band = [pair for pair in spinning if 20.0 <= pair["skew_ms"] <= 150.0]
    worst = sorted(band, key=lambda pair: pair["skew_ms"], reverse=True)[:8]
    markers = [row["end"] for row in early_rows if "_topk_topp_kernel" in row["name"]]
    ordered_pairs = sorted(pairs, key=lambda pair: pair["early_start"])
    cursor = 0
    token_skew: list[float] = []
    token_sync: list[float] = []
    token_walls: list[float] = []
    for begin, finish in zip(markers, markers[1:]):
        wall_ms = (finish - begin) / 1e6
        if wall_ms < 30 or wall_ms > 2000:
            continue
        while cursor < len(ordered_pairs) and ordered_pairs[cursor]["early_start"] < begin:
            cursor += 1
        chunk: list[dict] = []
        scan = cursor
        while scan < len(ordered_pairs) and ordered_pairs[scan]["early_start"] < finish:
            chunk.append(ordered_pairs[scan])
            scan += 1
        cursor = scan
        if len(chunk) < 100:
            continue
        token_walls.append(wall_ms)
        token_skew.append(sum(pair["skew_ms"] for pair in chunk))
        token_sync.append(sum(pair["sync_ms"] for pair in chunk))
    low = 1.0
    high = max((pair["early_ms"] for pair in spinning), default=0.0)

    def counts(rows: list[dict]) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            key = short_name(row["preceding"])
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items(), key=lambda item: -item[1]))

    report = {
        "early_pid": args.early_pid,
        "late_pid": args.late_pid,
        "early_launches": len(early_ar),
        "late_launches": len(late_ar),
        "paired": len(pairs),
        "unmatched_end_gap_over_5ms": unmatched,
        "spin_threshold_ms": low,
        "max_early_ms": high,
        "all": {
            "skew_ms": summarize([pair["skew_ms"] for pair in pairs]),
            "sync_ms": summarize([pair["sync_ms"] for pair in pairs]),
            "end_gap_ms": summarize([pair["end_gap_ms"] for pair in pairs]),
        },
        "per_token_topk_window": {
            "tokens": len(token_walls),
            "wall_ms": summarize(token_walls),
            "skew_sum_ms": summarize(token_skew),
            "sync_sum_ms": summarize(token_sync),
        },
        "spinning_early_at_least_1ms": {
            "skew_ms": summarize([pair["skew_ms"] for pair in spinning]),
            "sync_ms": summarize([pair["sync_ms"] for pair in spinning]),
            "end_gap_ms": summarize([pair["end_gap_ms"] for pair in spinning]),
            "preceding": counts(spinning),
        },
        "skew_20_to_150ms": {
            "skew_ms": summarize([pair["skew_ms"] for pair in band]),
            "sync_ms": summarize([pair["sync_ms"] for pair in band]),
            "covering_kernel": counts(
                [{"preceding": pair["cover"]} for pair in band]
            ),
        },
        "worst_spins": [
            {
                "skew_ms": round(pair["skew_ms"], 3),
                "sync_ms": round(pair["sync_ms"], 3),
                "early_ms": round(pair["early_ms"], 3),
                "late_ms": round(pair["late_ms"], 3),
                "preceding": pair["preceding"],
                "preceding_ms": round(pair["preceding_ms"], 3),
                "cover": pair["cover"],
                "cover_ms": round(pair["cover_ms"], 3),
                "cover_full": pair["cover_full"][:200],
            }
            for pair in worst
        ],
    }
    destination = args.trace_dir / "custom_ar_skew.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
