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
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def probe_single_request(
    base_url: str,
    api_key: Optional[str],
    model: str,
    prompt: str = "Ping",
    max_tokens: int = 16,
    timeout: float = 180.0,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
        method="POST",
    )

    t_start = time.perf_counter()
    t_first = None
    chunks = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    chunks += 1
    except Exception as exc:
        t_end = time.perf_counter()
        return {
            "success": False,
            "error": str(exc),
            "ttft_ms": None,
            "total_ms": (t_end - t_start) * 1000.0,
        }

    t_end = time.perf_counter()
    return {
        "success": True,
        "ttft_ms": round(((t_first - t_start) if t_first else (t_end - t_start)) * 1000.0, 2),
        "total_ms": round((t_end - t_start) * 1000.0, 2),
        "chunks": chunks,
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
