#!/usr/bin/env python3
"""Convert a rocprof kernel CSV(.gz) to a Perfetto-compatible trace JSON(.gz)."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import TextIO


def open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    base_ns: int | None = None
    with open_text(args.input, "r") as source, open_text(args.output, "w") as target:
        reader = csv.DictReader(source)
        target.write('{"displayTimeUnit":"ns","traceEvents":[\n')
        first = True
        for row in reader:
            try:
                start_ns = int(row["Start_Timestamp"])
                end_ns = int(row["End_Timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            if base_ns is None:
                base_ns = start_ns
            event = {
                "name": row.get("Kernel_Name") or "<unknown>",
                "cat": "KERNEL_DISPATCH",
                "ph": "X",
                # Chrome/Perfetto trace-event timestamps are microseconds.
                "ts": (start_ns - base_ns) / 1000,
                "dur": max(end_ns - start_ns, 0) / 1000,
                "pid": row.get("Agent_Id") or "GPU",
                "tid": f"queue-{row.get('Queue_Id', 'unknown')}",
                "args": {
                    "dispatch_id": row.get("Dispatch_Id"),
                    "stream_id": row.get("Stream_Id"),
                    "thread_id": row.get("Thread_Id"),
                    "grid": [
                        row.get("Grid_Size_X"),
                        row.get("Grid_Size_Y"),
                        row.get("Grid_Size_Z"),
                    ],
                    "workgroup": [
                        row.get("Workgroup_Size_X"),
                        row.get("Workgroup_Size_Y"),
                        row.get("Workgroup_Size_Z"),
                    ],
                },
            }
            if not first:
                target.write(",\n")
            json.dump(event, target, separators=(",", ":"))
            first = False
            count += 1
        target.write("\n]}\n")
    print(json.dumps({"events": count, "output": str(args.output)}))
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
