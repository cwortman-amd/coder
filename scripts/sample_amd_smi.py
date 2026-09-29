#!/usr/bin/env python3
"""Sample amd-smi activity, power, and clocks to JSONL until stopped."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def first(pattern: str, text: str, cast=float):
    match = re.search(pattern, text, re.MULTILINE)
    return cast(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as target:
        while True:
            started = time.time()
            result = subprocess.run(
                ["amd-smi", "metric", "-g", str(args.gpu)],
                capture_output=True,
                text=True,
                check=False,
            )
            text = result.stdout
            record = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "unix_s": started,
                # UMC_ACTIVITY is a device busy percentage, not calibrated GB/s.
                "gfx_activity_pct": first(r"^\s*GFX_ACTIVITY:\s*(\d+)\s*%", text),
                "umc_activity_pct": first(r"^\s*UMC_ACTIVITY:\s*(\d+)\s*%", text),
                "socket_power_w": first(r"^\s*SOCKET_POWER:\s*(\d+)\s*W", text),
                "gfx_clock_mhz": first(
                    r"^\s*GFX_0:\s*\n\s*CLK:\s*(\d+)\s*MHz", text
                ),
                "mem_clock_mhz": first(
                    r"^\s*MEM_0:\s*\n\s*CLK:\s*(\d+)\s*MHz", text
                ),
                "returncode": result.returncode,
            }
            target.write(json.dumps(record) + "\n")
            target.flush()
            time.sleep(max(args.interval - (time.time() - started), 0.0))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
