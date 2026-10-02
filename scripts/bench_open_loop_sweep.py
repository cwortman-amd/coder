#!/usr/bin/env python3
"""
Open-Loop Offered-Load & SLO Goodput Sweep.

Evaluates Experiment 1 ("Median benchmarks mislead") and Experiment 7 ("SLOs vs raw tokens/s"):
  - Generates open-loop Poisson or paced request arrivals at target rate lambda (req/s).
  - Records scheduled_send_ns vs actual_send_ns to detect client load-generator saturation.
  - Sweeps offered load from light load to near and past the throughput knee.
  - Computes:
    * p50, p95, p99 TTFT and E2E latency vs. offered load
    * Completed requests/s vs. Offered requests/s
    * Completed tokens/s vs. SLO-qualified Goodput (tokens meeting interactive TTFT & ITL gates)
    * Deadline-miss rate (% requests exceeding latency budget)

Usage:
  python3 scripts/bench_open_loop_sweep.py --url http://127.0.0.1:8000/v1 --model Qwen3.8-27B-Quark-AWQ-MXFP4 --qps-list 2 5 10
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from request_event_schema import BenchmarkCorpusSummary, RequestEvent, summarize_events
from streaming_client import EventMode, stream_chat


def execute_single_request(
    request_id: str,
    base_url: str,
    api_key: Optional[str],
    model: str,
    prompt: str,
    max_tokens: int,
    scheduled_send_ns: int,
    timeout: float = 120.0,
) -> RequestEvent:
    # Sleep until scheduled time if in future
    now_ns = time.time_ns()
    if scheduled_send_ns > now_ns:
        time.sleep((scheduled_send_ns - now_ns) / 1e9)

    result = stream_chat(
        base_url=base_url,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        mode=EventMode.CONTENT,
        max_tokens=max_tokens,
        api_key=api_key,
        timeout=timeout,
        request_id=request_id,
        scheduled_send_ns=scheduled_send_ns,
        prompt_length_bucket="medium_mix",
    )
    return result.event


def run_open_loop_cell(
    qps: float,
    duration_s: float,
    base_url: str,
    api_key: Optional[str],
    model: str,
    max_tokens: int,
    poisson: bool = True,
    ttft_slo_ms: float = 1000.0,
    worst_itl_slo_ms: float = 100.0,
    max_concurrency: int = 64,
) -> Tuple[BenchmarkCorpusSummary, List[RequestEvent]]:
    total_expected = int(qps * duration_s)
    # Generate schedule
    intervals = []
    for _ in range(total_expected):
        dt = random.expovariate(qps) if poisson else (1.0 / qps)
        intervals.append(dt)

    t_start_ns = time.time_ns()
    scheduled_times_ns = []
    accum = t_start_ns
    for dt in intervals:
        accum += int(dt * 1e9)
        scheduled_times_ns.append(accum)

    events: List[RequestEvent] = []
    prompts = [
        "Summarize the trade-offs of continuous batching.",
        "List three architectural differences between prefill and decode.",
        "Write a concise Python function to calculate rolling percentiles.",
        "Explain why tail latency compounds non-linearly across tool chains.",
    ]

    t_run_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = [
            executor.submit(
                execute_single_request,
                request_id=f"qps_{qps}_req_{i+1}",
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompts[i % len(prompts)],
                max_tokens=max_tokens,
                scheduled_send_ns=sched_ns,
            )
            for i, sched_ns in enumerate(scheduled_times_ns)
        ]
        for fut in concurrent.futures.as_completed(futures):
            events.append(fut.result())

    t_run_end = time.perf_counter()
    cell_duration = t_run_end - t_run_start
    summary = summarize_events(
        events=events,
        duration_s=cell_duration,
        ttft_slo_ms=ttft_slo_ms,
        worst_itl_slo_ms=worst_itl_slo_ms,
    )
    return summary, events


def run_open_loop_sweep(
    qps_list: List[float],
    duration_per_cell_s: float,
    base_url: str,
    api_key: Optional[str],
    model: str,
    max_tokens: int = 64,
    poisson: bool = True,
    ttft_slo_ms: float = 1000.0,
    worst_itl_slo_ms: float = 100.0,
    output_dir: Path = Path("_results/open_loop_sweep"),
    max_in_flight: int = 64,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "open_loop_manifest.json"

    print("\n=======================================================")
    print("  OPEN-LOOP OFFERED LOAD & SLO GOODPUT SWEEP")
    print(f"  Target: {base_url} (Model: {model})")
    print(f"  Target QPS List: {qps_list} (Duration: {duration_per_cell_s}s per cell)")
    print(f"  Arrival Distribution: {'Poisson' if poisson else 'Paced'}")
    print(f"  Interactive SLOs: TTFT <= {ttft_slo_ms}ms | Worst ITL <= {worst_itl_slo_ms}ms")
    print("=======================================================\n")

    cell_summaries = {}
    all_events_by_qps = {}

    for qps in qps_list:
        print(f"[*] Running Offered Load λ = {qps:.1f} req/s ({int(qps * duration_per_cell_s)} requests)...", flush=True)
        summary, events = run_open_loop_cell(
            qps=qps,
            duration_s=duration_per_cell_s,
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            poisson=poisson,
            ttft_slo_ms=ttft_slo_ms,
            worst_itl_slo_ms=worst_itl_slo_ms,
            max_concurrency=max_in_flight,
        )
        cell_summaries[f"qps_{qps}"] = summary.to_dict()
        all_events_by_qps[f"qps_{qps}"] = [e.request_id for e in events]

        print(f"    -> Offered: {summary.offered_requests_per_s} req/s | Completed: {summary.completed_requests_per_s} req/s")
        print(f"       Raw Tok/s: {summary.completed_tokens_per_s:.1f} | Qualified Goodput: {summary.slo_qualified_goodput_tokens_per_s:.1f} tok/s")
        print(f"       TTFT: p50={summary.ttft_p50_ms or 0:.0f}ms, p95={summary.ttft_p95_ms or 0:.0f}ms, p99={summary.ttft_p99_ms or 0:.0f}ms")
        print(f"       Client Send Lag p95: {summary.client_send_lag_p95_ms:.1f}ms")

    # Master Scorecard Table
    print("\n---------------------------------------------------------------------------------------------")
    print("  OPEN-LOOP LOAD VS. GOODPUT SCORECARD (EXPERIMENTS 1 & 7)")
    print("---------------------------------------------------------------------------------------------")
    print(f"| Offered λ | Completed req/s | Completed tok/s | Qualified Goodput | TTFT p50 | TTFT p95 | TTFT p99 | Pass Rate |")
    print(f"|---:|---:|---:|---:|---:|---:|---:|---:|")
    for qps in qps_list:
        c = cell_summaries[f"qps_{qps}"]
        pass_rate = round(c["slo_passed_requests"] / max(c["completed_requests"], 1) * 100.0, 1)
        print(
            f"| {qps:.1f} req/s | "
            f"{c['completed_requests_per_s']:.1f} req/s | "
            f"{c['completed_tokens_per_s']:.1f} tok/s | "
            f"**{c['slo_qualified_goodput_tokens_per_s']:.1f} tok/s** | "
            f"{c['ttft_p50_ms'] or 0:.0f} ms | "
            f"{c['ttft_p95_ms'] or 0:.0f} ms | "
            f"{c['ttft_p99_ms'] or 0:.0f} ms | "
            f"**{pass_rate}%** |"
        )
    print("---------------------------------------------------------------------------------------------\n")

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "base_url": base_url,
        "qps_list": qps_list,
        "duration_per_cell_s": duration_per_cell_s,
        "ttft_slo_ms": ttft_slo_ms,
        "worst_itl_slo_ms": worst_itl_slo_ms,
        "cells": cell_summaries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[+] Open-loop sweep manifest saved to: {manifest_path}\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", os.environ.get("DO_MODEL_ACCESS_KEY")))
    parser.add_argument("--model", default="Qwen3.8-27B-Quark-AWQ-MXFP4")
    parser.add_argument("--qps-list", nargs="+", type=float, default=[1.0, 3.0, 6.0], help="List of offered request rates (req/s)")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds per QPS cell")
    parser.add_argument("--tokens", type=int, default=32, help="Output tokens per request")
    parser.add_argument("--paced", action="store_true", help="Use constant paced arrival instead of Poisson")
    parser.add_argument("--ttft-slo-ms", type=float, default=1000.0, help="TTFT SLO gate in ms")
    parser.add_argument("--worst-itl-slo-ms", type=float, default=100.0, help="Worst ITL SLO gate in ms")
    parser.add_argument("--out-dir", type=Path, default=Path("_results/open_loop_sweep"))
    parser.add_argument("--max-in-flight", type=int, default=64, help="Client worker cap")
    args = parser.parse_args()

    run_open_loop_sweep(
        qps_list=args.qps_list,
        duration_per_cell_s=args.duration,
        base_url=args.url,
        api_key=args.api_key,
        model=args.model,
        max_tokens=args.tokens,
        poisson=not args.paced,
        ttft_slo_ms=args.ttft_slo_ms,
        worst_itl_slo_ms=args.worst_itl_slo_ms,
        output_dir=args.out_dir,
        max_in_flight=args.max_in_flight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
