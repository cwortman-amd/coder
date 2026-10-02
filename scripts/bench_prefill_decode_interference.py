#!/usr/bin/env python3
"""
Prefill/Decode Interference Benchmark (Controlled Server-Side Experiment).

Evaluates the specific hypothesis:
  "Long-prompt prefill bursts interrupt concurrent active decode streams,
   causing conspicuous ITL latency spikes and SLO breaches."

Methodology:
  1. Spawns N concurrent steady decode streams (e.g., short prompt, generating 128+ tokens).
  2. While all decode streams are actively generating, injects a timed burst of
     long-prompt prefill requests (e.g., 4K or 8K input tokens).
  3. Measures the precise ITL timeline of active decode streams to quantify:
     - Peak ITL pause during prefill ingestion
     - Degradation from nominal TPOT (e.g., 25ms -> 450ms stall)
     - Percentage of decode streams experiencing stalls (>50ms, >100ms, >1000ms)
     - Prefill TTFT and total completion time
  4. Supports A/B comparison between:
     - Collocated baseline (default chunked prefill vs. small chunked prefill)
     - Disaggregated P/D (Prefill on GPU 0, Decode on GPU 1)

Usage:
  python3 scripts/bench_prefill_decode_interference.py --url http://127.0.0.1:8000/v1 --model Qwen3.8-27B-Quark-AWQ-MXFP4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from request_event_schema import RequestEvent, linear_percentile
from streaming_client import EventMode, stream_chat


def generate_prompt_by_tokens(target_tokens: int) -> str:
    """Generate deterministic synthetic text with approximately target_tokens."""
    # Approximate 1 token ~ 4 characters
    words_needed = max(target_tokens - 10, 1)
    return "Evaluate system response under load. " + ("alpha beta gamma " * (words_needed // 3 + 1))[:words_needed * 5]


def execute_streaming_request(
    request_id: str,
    base_url: str,
    api_key: Optional[str],
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    request_type: str, # "steady_decode" or "injected_prefill"
    timeout: float = 120.0,
) -> RequestEvent:
    result = stream_chat(
        base_url=base_url,
        model=model,
        messages=messages,
        mode=EventMode.CONTENT,
        max_tokens=max_tokens,
        api_key=api_key,
        timeout=timeout,
        request_id=request_id,
        scheduled_send_ns=time.time_ns(),
        prompt_length_bucket=request_type,
    )
    return result.event


def run_interference_experiment(
    base_url: str,
    api_key: Optional[str],
    model: str,
    num_decode_streams: int = 4,
    decode_tokens: int = 128,
    prefill_prompt_tokens: int = 4096,
    prefill_burst_size: int = 2,
    injection_delay_s: float = 1.0,
    output_dir: Path = Path("_results/prefill_interference"),
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "interference_manifest.json"

    print("\n=======================================================")
    print("  PREFILL / DECODE INTERFERENCE EXPERIMENT")
    print(f"  Target: {base_url} (Model: {model})")
    print(f"  Steady Decode Streams: {num_decode_streams} (Target: {decode_tokens} tokens)")
    print(f"  Injected Burst: {prefill_burst_size}x {prefill_prompt_tokens}-token prompts")
    print(f"  Burst Injection Delay: {injection_delay_s:.1f}s after decode starts")
    print("=======================================================\n")

    short_prompt = [{"role": "user", "content": "Count from 1 to 200 with brief descriptions."}]
    long_prompt_text = generate_prompt_by_tokens(prefill_prompt_tokens)
    long_prompt = [{"role": "user", "content": long_prompt_text}]

    events: List[RequestEvent] = []
    t_experiment_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_decode_streams + prefill_burst_size) as executor:
        futures = []

        print(f"[*] Launching {num_decode_streams} steady decode streams...", flush=True)
        for i in range(num_decode_streams):
            f = executor.submit(
                execute_streaming_request,
                request_id=f"decode_stream_{i+1}",
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=short_prompt,
                max_tokens=decode_tokens,
                request_type="steady_decode",
            )
            futures.append(f)

        # Wait until decode streams are actively generating
        print(f"[*] Waiting {injection_delay_s:.1f}s for steady decoders to enter autoregressive generation phase...", flush=True)
        time.sleep(injection_delay_s)

        # Inject prefill burst
        t_burst_inject_ns = time.time_ns()
        print(f"⚡ INJECTING BURST: {prefill_burst_size}x long-prompt ({prefill_prompt_tokens} tokens) requests simultaneously!", flush=True)
        for j in range(prefill_burst_size):
            f = executor.submit(
                execute_streaming_request,
                request_id=f"prefill_burst_{j+1}",
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=long_prompt,
                max_tokens=16,
                request_type="injected_prefill",
            )
            futures.append(f)

        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            events.append(res)
            print(f"    -> {res.request_id} ({res.prompt_length_bucket}) finished: status={res.status} | E2E={res.e2e_latency_ms:.1f}ms | TTFT={res.ttft_ms or 0:.1f}ms | Worst ITL={res.worst_itl_ms or 0:.1f}ms", flush=True)

    t_experiment_end = time.perf_counter()

    # Disaggregate results by role
    decode_events = [e for e in events if "decode" in e.prompt_length_bucket and e.status == "completed"]
    prefill_events = [e for e in events if "prefill" in e.prompt_length_bucket and e.status == "completed"]

    all_decode_itls: List[float] = []
    worst_itls_per_stream: List[float] = []
    for d in decode_events:
        itls = d.itl_intervals_ms
        all_decode_itls.extend(itls)
        if itls:
            worst_itls_per_stream.append(max(itls))

    nominal_tpot = linear_percentile(sorted(all_decode_itls), 50.0) or 0.0
    p95_decode_itl = linear_percentile(sorted(all_decode_itls), 95.0) or 0.0
    max_decode_stall = max(worst_itls_per_stream) if worst_itls_per_stream else 0.0

    stalls_gt_50ms = sum(1 for itl in all_decode_itls if itl > 50.0)
    stalls_gt_100ms = sum(1 for itl in all_decode_itls if itl > 100.0)
    stalls_gt_500ms = sum(1 for itl in all_decode_itls if itl > 500.0)

    prefill_ttfts = [p.ttft_ms for p in prefill_events if p.ttft_ms is not None]

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "base_url": base_url,
        "num_decode_streams": num_decode_streams,
        "decode_tokens": decode_tokens,
        "prefill_prompt_tokens": prefill_prompt_tokens,
        "prefill_burst_size": prefill_burst_size,
        "burst_injected_at_ns": t_burst_inject_ns,
        "experiment_duration_s": round(t_experiment_end - t_experiment_start, 2),
        "decode_metrics": {
            "streams_completed": len(decode_events),
            "nominal_median_tpot_ms": round(nominal_tpot, 2),
            "p95_token_itl_ms": round(p95_decode_itl, 2),
            "max_observed_stall_ms": round(max_decode_stall, 2),
            "worst_itl_multiplier": round(max_decode_stall / max(nominal_tpot, 1.0), 2),
            "total_token_intervals": len(all_decode_itls),
            "stalls_over_50ms": stalls_gt_50ms,
            "stalls_over_100ms": stalls_gt_100ms,
            "stalls_over_500ms": stalls_gt_500ms,
            "stream_stalls": worst_itls_per_stream,
        },
        "prefill_burst_metrics": {
            "burst_requests_completed": len(prefill_events),
            "prefill_ttft_p50_ms": linear_percentile(sorted(prefill_ttfts), 50.0),
            "prefill_ttft_max_ms": max(prefill_ttfts) if prefill_ttfts else None,
        },
    }

    # Print scorecard
    print("\n-------------------------------------------------------")
    print("  INTERFERENCE SCORECARD:")
    print(f"  • Nominal Median TPOT:           {nominal_tpot:.1f} ms")
    print(f"  • Peak Decode Stall Observed:    {max_decode_stall:.1f} ms ({summary['decode_metrics']['worst_itl_multiplier']}x nominal)")
    print(f"  • Token Intervals > 100ms:       {stalls_gt_100ms} / {len(all_decode_itls)}")
    print(f"  • Token Intervals > 500ms:       {stalls_gt_500ms} / {len(all_decode_itls)}")
    print(f"  • Injected Prefill Mean TTFT:    {statistics.mean(prefill_ttfts) if prefill_ttfts else 0:.1f} ms")
    if max_decode_stall > 300.0:
        print("  ⚠️  HYPOTHESIS CONFIRMED: Long prefill bursts visibly freeze active decoders.")
    else:
        print("  ✓ Smooth execution: Minimal decode stall observed.")
    print("-------------------------------------------------------\n")

    manifest_path.write_text(json.dumps(summary, indent=2))
    print(f"[+] Interference manifest written to: {manifest_path}\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", os.environ.get("DO_MODEL_ACCESS_KEY")))
    parser.add_argument("--model", default="Qwen3.8-27B-Quark-AWQ-MXFP4")
    parser.add_argument("--decode-streams", type=int, default=4, help="Number of concurrent active decoders")
    parser.add_argument("--decode-tokens", type=int, default=128, help="Output tokens per decoder")
    parser.add_argument("--prefill-tokens", type=int, default=4096, help="Input prompt size of injected burst")
    parser.add_argument("--burst-size", type=int, default=2, help="Number of concurrent burst prefill requests")
    parser.add_argument("--delay-s", type=float, default=1.0, help="Delay before injecting burst")
    parser.add_argument("--out-dir", type=Path, default=Path("_results/prefill_interference"))
    args = parser.parse_args()

    run_interference_experiment(
        base_url=args.url,
        api_key=args.api_key,
        model=args.model,
        num_decode_streams=args.decode_streams,
        decode_tokens=args.decode_tokens,
        prefill_prompt_tokens=args.prefill_tokens,
        prefill_burst_size=args.burst_size,
        injection_delay_s=args.delay_s,
        output_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
