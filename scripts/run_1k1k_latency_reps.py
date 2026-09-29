#!/usr/bin/env python3
"""Three unprofiled 1K/1K streaming reps at C1/C4/C8.

Calls scripts/bench_openai_stream.py. Samples vLLM running/waiting gauges
during each measured run. Warmup output is excluded from the summary.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def gauges(base: str) -> dict[str, float]:
    url = base.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    wanted = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:kv_cache_usage_perc": "kv_perc",
    }
    found: dict[str, float] = {}
    with urllib.request.urlopen(f"{url}/metrics", timeout=2) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line or line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name in wanted:
                found[wanted[name]] = float(line.rsplit(" ", 1)[-1])
    return found


def sample_loop(base: str, stop: threading.Event, sink: list[dict]) -> None:
    while not stop.is_set():
        try:
            row = gauges(base)
            row["t"] = time.time()
            sink.append(row)
        except Exception as exc:  # noqa: BLE001
            sink.append({"t": time.time(), "error": str(exc)})
        stop.wait(0.25)


def run_bench(root: Path, base: str, concurrency: int, output_len: int, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    samples: list[dict] = []
    thread = threading.Thread(target=sample_loop, args=(base, stop, samples), daemon=True)
    thread.start()
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/bench_openai_stream.py"),
            "--base-url",
            base,
            "--model",
            "awq",
            "--input-len",
            "1024",
            "--output-len",
            str(output_len),
            "--num-prompts",
            str(concurrency),
            "--concurrency",
            str(concurrency),
            "--out",
            str(dest),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    stop.set()
    thread.join(timeout=2)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:] or proc.stdout[-2000:])
    payload = json.loads(dest.read_text())
    payload["gauge_samples"] = samples
    dest.write_text(json.dumps(payload))
    return payload


def position_p95(intervals_ms: list[float]) -> dict[str, float | None]:
    n = len(intervals_ms)
    if n < 8:
        return {"first32": None, "middle": None, "last32": None}
    mid = n // 2
    middle = intervals_ms[max(0, mid - 16) : mid + 16]
    return {
        "first32": percentile(intervals_ms[:32], 0.95),
        "middle": percentile(middle, 0.95),
        "last32": percentile(intervals_ms[-32:], 0.95),
    }


def summarize(payload: dict) -> dict:
    ok = [row for row in payload["rows"] if row.get("ok")]
    ttfts = [row["ttft_s"] * 1000 for row in ok if row.get("ttft_s") is not None]
    itls = [value * 1000 for row in ok for value in row["itl_s"]]
    tpots = []
    per_request = []
    for index, row in enumerate(ok):
        gaps = [value * 1000 for value in row["itl_s"]]
        tokens = row.get("completion_tokens") or 0
        if tokens > 1 and row.get("latency_s") is not None and row.get("ttft_s") is not None:
            tpot = (row["latency_s"] - row["ttft_s"]) / (tokens - 1) * 1000
        else:
            tpot = None
        if tpot is not None:
            tpots.append(tpot)
        per_request.append(
            {
                "request_index": index,
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": tokens,
                "ttft_ms": None if row.get("ttft_s") is None else row["ttft_s"] * 1000,
                "tpot_ms": tpot,
                "itl_p50_ms": percentile(gaps, 0.50),
                "itl_p95_ms": percentile(gaps, 0.95),
                "itl_p99_ms": percentile(gaps, 0.99),
                "itl_max_ms": max(gaps) if gaps else None,
                **{f"itl_{key}_p95_ms": value for key, value in position_p95(gaps).items()},
            }
        )
    running = [sample["running"] for sample in payload.get("gauge_samples", []) if "running" in sample]
    waiting = [sample["waiting"] for sample in payload.get("gauge_samples", []) if "waiting" in sample]
    kv = [sample["kv_perc"] for sample in payload.get("gauge_samples", []) if "kv_perc" in sample]
    return {
        "successful": payload["successful"],
        "failed": payload["failed"],
        "prompt_tokens": ok[0].get("prompt_tokens") if ok else None,
        "completion_tokens": ok[0].get("completion_tokens") if ok else None,
        "output_tok_s": payload["output_throughput"],
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "ttft_p99_ms": percentile(ttfts, 0.99),
        "itl_p50_ms": percentile(itls, 0.50),
        "itl_p95_ms": percentile(itls, 0.95),
        "itl_p99_ms": percentile(itls, 0.99),
        "itl_max_ms": max(itls) if itls else None,
        "tpot_p50_ms": percentile(tpots, 0.50),
        "tpot_p95_ms": percentile(tpots, 0.95),
        "position_p95_ms": position_p95(itls),
        "per_request": per_request,
        "server_running_max": max(running) if running else None,
        "server_waiting_max": max(waiting) if waiting else None,
        "kv_perc_max": max(kv) if kv else None,
    }


def main() -> int:
    profile = sys.argv[1]
    out_root = Path(sys.argv[2])
    base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8000/v1"
    root = Path(__file__).resolve().parents[1]
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"warmup {profile}", flush=True)
    for concurrency in (1, 4, 8):
        run_bench(root, base, concurrency, 32, out_root / f"warmup_c{concurrency}.json")
    cells = []
    for concurrency in (1, 4, 8):
        for rep in (1, 2, 3):
            dest = out_root / f"c{concurrency}_rep{rep}.json"
            print(f"measure {profile} C{concurrency} rep {rep}", flush=True)
            payload = run_bench(root, base, concurrency, 1024, dest)
            summary = summarize(payload)
            summary["profile"] = profile
            summary["concurrency"] = concurrency
            summary["rep"] = rep
            cells.append(summary)
            print(
                f"  tok/s={summary['output_tok_s']:.2f} "
                f"ttft_p95={summary['ttft_p95_ms']:.2f} "
                f"itl_p50={summary['itl_p50_ms']:.2f} "
                f"itl_p95={summary['itl_p95_ms']:.2f} "
                f"itl_p99={summary['itl_p99_ms']:.2f} "
                f"max={summary['itl_max_ms']:.2f} "
                f"wait={summary['server_waiting_max']}",
                flush=True,
            )
    (out_root / "summary.json").write_text(json.dumps(cells, indent=2))
    print("wrote", out_root / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
