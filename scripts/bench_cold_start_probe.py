#!/usr/bin/env python3
"""
Cold-Start & Bimodality Latency Probe.

Measures:
  1. Cold vs. Warm latency divergence (first-request penalty).
  2. Idle interval cool-down sweep (probes endpoint at 0s, 30s, 60s, 120s, 300s gaps).
  3. Bimodal distribution detection (identifies cluster gaps where tail is separated
     from the warm body by a multi-second dead zone, typical of cold starts).

Usage:
  python3 scripts/bench_cold_start_probe.py --url http://127.0.0.1:8000/v1 --model Qwen3.8-27B-Quark-AWQ-MXFP4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streaming_client import EventMode, stream_chat  # noqa: E402


def probe_single_request(
    base_url: str,
    api_key: Optional[str],
    model: str,
    prompt: str = "Ping",
    max_tokens: int = 16,
    timeout: float = 180.0,
) -> Dict[str, Any]:
    """ANY_SSE: the first data line is TTFT, whether or not it parses as JSON."""
    result = stream_chat(
        base_url=base_url,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        mode=EventMode.ANY_SSE,
        max_tokens=max_tokens,
        api_key=api_key,
        timeout=timeout,
    )
    event = result.event
    if event.status != "completed":
        return {
            "success": False,
            "error": event.error_message or "error",
            "ttft_ms": None,
            "total_ms": event.e2e_latency_ms,
        }
    ttft = event.ttft_ms if event.ttft_ms is not None else event.e2e_latency_ms
    return {
        "success": True,
        "ttft_ms": round(ttft, 2),
        "total_ms": round(event.e2e_latency_ms, 2),
        "chunks": len(event.token_timestamps_ns),
    }


def run_cold_probe(
    base_url: str,
    api_key: Optional[str],
    model: str,
    idle_intervals: List[int],
    warm_reps_per_interval: int = 3,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    print("\n=======================================================")
    print("  COLD-START & BIMODALITY LATENCY PROBE")
    print(f"  Target: {base_url} | Model: {model}")
    print(f"  Idle intervals to test: {idle_intervals} seconds")
    print("=======================================================\n")

    results = []
    for interval in idle_intervals:
        if interval > 0:
            print(f"[*] Sleeping for {interval}s to allow idle replica spin-down / memory eviction...", flush=True)
            time.sleep(interval)

        print(f"[*] Sending Probe 1 (First request after {interval}s idle)...", flush=True)
        cold_res = probe_single_request(base_url, api_key, model)
        cold_res["idle_gap_s"] = interval
        cold_res["probe_type"] = "first_after_idle"
        results.append(cold_res)
        print(f"    -> TTFT: {cold_res.get('ttft_ms')} ms | Total: {cold_res.get('total_ms')} ms")

        # Warm follow-ups
        for w in range(1, warm_reps_per_interval + 1):
            time.sleep(0.5)
            warm_res = probe_single_request(base_url, api_key, model)
            warm_res["idle_gap_s"] = 0
            warm_res["probe_type"] = f"warm_followup_{w}"
            results.append(warm_res)
            print(f"    -> Warm {w} TTFT: {warm_res.get('ttft_ms')} ms | Total: {warm_res.get('total_ms')} ms")

    # Analyze bimodality
    all_ttfts = [r["ttft_ms"] for r in results if r["success"] and r.get("ttft_ms")]
    p50 = statistics.median(all_ttfts) if all_ttfts else 0.0
    first_ttfts = [r["ttft_ms"] for r in results if r["success"] and r.get("probe_type") == "first_after_idle"]
    warm_ttfts = [r["ttft_ms"] for r in results if r["success"] and "warm" in r.get("probe_type", "")]

    mean_cold = statistics.mean(first_ttfts) if first_ttfts else 0.0
    mean_warm = statistics.mean(warm_ttfts) if warm_ttfts else 0.0
    cold_penalty_ms = max(mean_cold - mean_warm, 0.0)

    # Check for bimodal separation
    bimodal_detected = False
    if first_ttfts and warm_ttfts and (mean_cold >= mean_warm * 2.5) and (mean_cold - mean_warm > 500.0):
        bimodal_detected = True

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "base_url": base_url,
        "total_probes": len(results),
        "mean_cold_ttft_ms": round(mean_cold, 2),
        "mean_warm_ttft_ms": round(mean_warm, 2),
        "cold_penalty_ms": round(cold_penalty_ms, 2),
        "bimodal_distribution_detected": bimodal_detected,
        "probes": results,
    }

    print("\n-------------------------------------------------------")
    print(f"  SUMMARY DIAGNOSTIC:")
    print(f"  • Warm Mean TTFT: {mean_warm:.1f} ms")
    print(f"  • Cold / Post-Idle Mean TTFT: {mean_cold:.1f} ms")
    print(f"  • Cold-Start Penalty: +{cold_penalty_ms:.1f} ms ({mean_cold/max(mean_warm, 1.0):.1f}x)")
    print(f"  • Bimodal Pattern Detected: {'YES (Cold Start / Memory Eviction confirmed)' if bimodal_detected else 'NO (Consistent warm serving)'}")
    print("-------------------------------------------------------\n")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"[+] Output written to: {output_path}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1", help="Inference API base URL")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", os.environ.get("DO_MODEL_ACCESS_KEY")), help="API Key")
    parser.add_argument("--model", default="Qwen3.8-27B-Quark-AWQ-MXFP4", help="Model name")
    parser.add_argument("--idle-intervals", nargs="+", type=int, default=[0, 5, 15], help="Idle seconds before probing")
    parser.add_argument("--warm-reps", type=int, default=2, help="Warm follow-up requests per interval")
    parser.add_argument("--out", type=Path, default=Path("_results/cold_start_probe.json"), help="Output JSON path")
    args = parser.parse_args()

    run_cold_probe(
        base_url=args.url,
        api_key=args.api_key,
        model=args.model,
        idle_intervals=args.idle_intervals,
        warm_reps_per_interval=args.warm_reps,
        output_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
