#!/usr/bin/env python3
"""
Agent Chain & Latency Distribution Benchmark.

Measures & Disaggregates:
  1. Per-Request Latency Distributions (matched prompt/output length buckets):
     - TTFT (Time to First Token: arrival of first reasoning or content chunk)
     - TTFAT (Time to First Answer Token: arrival of first user-facing answer token)
     - Decode generation rate: (M - 1) / (t_last - t_first) evaluated strictly when M > 1
     - End-to-End Latency (complete wall-clock request duration)
     - Load generator lag: actual_send_ns - scheduled_arrival_ns
     - Deadline-Miss Rate (SLO violation rate against interactive targets)
  2. Complete-Task Agent Trajectory Benchmark (M independent N-call chains):
     - Complete task wall-clock: T_task = sum(T_LLM,i) + sum(T_tool,j) + T_orchestration
     - Context accumulation mode (simulates tool call results and prefix growth)
     - Fraction of tasks containing a slow call (defined by fixed baseline threshold)
     - Compounding Analysis:
       * At-least-one exceedance probability reference: 1 - (1 - q)^N under i.i.d. assumption
       * Sequential additive convolution with correlated queueing
       * Parallel scatter-gather reference: max(t_1, ..., t_k) from The Tail at Scale
     - TWO Resampling Baselines:
       * Baseline 1: Independent random draws from matched single-call distribution
       * Baseline 2: Block/time-preserving resampling retaining temporal correlation
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from okf_docs import ensure_frontmatter
from distribution_stats import distribution_core
from request_event_schema import percentile
from streaming_client import EventMode, stream_chat


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def calculate_distribution_stats(
    samples: List[float],
    deadline_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute distribution statistics and deadline miss rate without assigning speculative causes."""
    core = distribution_core(samples)
    clean = core.pop("clean", [])
    core.pop("_p50", None)
    core.pop("_p99_to_p50", None)
    core.pop("_p95_to_p50", None)
    if core["n"] == 0:
        return core
    n = core["n"]
    deadline_miss_pct = None
    if deadline_threshold is not None and clean:
        misses = sum(1 for value in clean if value > deadline_threshold)
        deadline_miss_pct = round(misses / n * 100.0, 2)
    core["deadline_threshold"] = deadline_threshold
    core["deadline_miss_pct"] = deadline_miss_pct
    core["sample_size_note"] = (
        f"N={n} is insufficient for asymptotic p99 estimation (observed order statistic only)."
        if n < 100 else
        f"N={n} provides empirical tail resolution (>= 1.0%)."
    )
    return core


def streamed_chat_request(
    base_url: str,
    api_key: Optional[str],
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 64,
    temperature: float = 0.0,
    scheduled_arrival_ns: int = 0,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Send an OpenAI-compatible streaming chat completion request.
    Disaggregates:
      - Generator delay: actual send vs. scheduled arrival
      - TTFT: Time to first token (first token arrival timestamp)
      - TTFAT: Time to first answer token (first visible non-reasoning token)
      - Decode rate: (M - 1) / (t_last_token - t_first_token) strictly when M > 1
      - Inter-token intervals: uncompressed token-event timeline
      - End-to-End latency: wall-clock duration to stream completion
    """
    result = stream_chat(
        base_url=base_url,
        model=model,
        messages=messages,
        mode=EventMode.REASONING_AND_ANSWER,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
        timeout=timeout,
        scheduled_send_ns=scheduled_arrival_ns,
    )
    event = result.event
    itls_ms = event.itl_intervals_ms
    ttft_ms = event.ttft_ms if event.ttft_ms is not None else event.e2e_latency_ms
    ttfat_ms = event.ttfat_ms if event.ttfat_ms is not None else ttft_ms
    if event.status != "completed":
        return {
            "success": False,
            "error": event.error_message or "error",
            "finish_reason": "error",
            "generator_lag_ms": round(event.generator_delay_ms, 2),
            "ttft_ms": round(event.ttft_ms, 2) if event.ttft_ms is not None else None,
            "ttfat_ms": round(event.ttfat_ms, 2) if event.ttfat_ms is not None else None,
            "total_time_ms": round(event.e2e_latency_ms, 2),
            "completion_tokens": event.completion_tokens,
            "prompt_tokens": result.prompt_tokens,
            "decode_tps": None,
            "generated_text": result.generated_text,
        }
    return {
        "success": True,
        "finish_reason": event.finish_reason,
        "generator_lag_ms": round(event.generator_delay_ms, 2),
        "ttft_ms": round(ttft_ms, 2),
        "ttfat_ms": round(ttfat_ms, 2),
        "total_time_ms": round(event.e2e_latency_ms, 2),
        "decode_tps": round(event.decode_tokens_per_second, 2) if event.decode_tokens_per_second is not None else None,
        "completion_tokens": event.completion_tokens,
        "prompt_tokens": result.prompt_tokens,
        "mean_itl_ms": round(statistics.mean(itls_ms), 2) if itls_ms else None,
        "p95_itl_ms": round(percentile(itls_ms, 95.0), 2) if itls_ms else None,
        "worst_itl_ms": round(event.worst_itl_ms, 2) if event.worst_itl_ms is not None else None,
        "stalls_over_1s": sum(1 for itl in itls_ms if itl > 1000.0),
        "generated_text": result.generated_text,
    }


def execute_agent_chain(
    chain_id: int,
    base_url: str,
    api_key: Optional[str],
    model: str,
    chain_length: int = 10,
    accumulate_context: bool = True,
    tokens_per_call: int = 64,
    simulated_tool_delay_ms: float = 30.0,
    single_call_slow_threshold_ms: float = 2000.0,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Execute a sequential N-call agent workflow at the complete-task level:
      T_task = sum(T_LLM,i) + sum(T_tool,j) + T_orchestration
    """
    history: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": "You are an autonomous coding assistant inspecting repository state and executing commands.",
        },
        {
            "role": "user",
            "content": f"Task #{chain_id}: Inspect the latency anomalies and trace the execution path.",
        },
    ]

    step_results = []
    total_tool_ms = 0.0
    t_task_wall_start = time.perf_counter()

    for step in range(chain_length):
        t_call_start = time.perf_counter()
        res = streamed_chat_request(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=history,
            max_tokens=tokens_per_call,
            timeout=timeout,
        )
        res["step_index"] = step + 1
        res["call_wall_ms"] = round((time.perf_counter() - t_call_start) * 1000.0, 2)
        step_results.append(res)

        if not res["success"]:
            break

        # Simulate external tool execution (filesystem read / bash execution)
        if step < chain_length - 1:
            time.sleep(simulated_tool_delay_ms / 1000.0)
            total_tool_ms += simulated_tool_delay_ms

        if accumulate_context:
            history.append({"role": "assistant", "content": res.get("generated_text", f"Step {step + 1} completed.")[:200]})
            history.append({
                "role": "user",
                "content": f"Tool output step {step + 1}: Found 42 lines matching signature. Next action?",
            })

    t_task_wall_end = time.perf_counter()
    total_task_wall_ms = (t_task_wall_end - t_task_wall_start) * 1000.0

    llm_times = [s["total_time_ms"] for s in step_results if s["success"]]
    sum_llm_ms = sum(llm_times)
    orchestration_ms = max(total_task_wall_ms - (sum_llm_ms + total_tool_ms), 0.0)

    # Check if task encountered a slow call based on fixed baseline threshold
    contains_slow_call = any(t > single_call_slow_threshold_ms for t in llm_times)

    return {
        "chain_id": chain_id,
        "completed_steps": len(step_results),
        "target_steps": chain_length,
        "all_succeeded": len(step_results) == chain_length and all(s["success"] for s in step_results),
        "total_task_wall_ms": round(total_task_wall_ms, 2),
        "sum_llm_duration_ms": round(sum_llm_ms, 2),
        "sum_tool_duration_ms": round(total_tool_ms, 2),
        "orchestration_duration_ms": round(orchestration_ms, 2),
        "contains_slow_call": contains_slow_call,
        "worst_call_ms": round(max(llm_times), 2) if llm_times else None,
        "worst_ttft_ms": round(max(ttfts), 2) if (ttfts := [s["ttft_ms"] for s in step_results if s.get("ttft_ms") is not None]) else None,
        "steps": step_results,
    }


def compute_compound_tail_risk_reference(per_call_tail_rate: float, chain_lengths: List[int]) -> Dict[int, float]:
    """Calculates independent exceedance reference: 1 - (1 - q)^N."""
    return {
        n: round((1.0 - math.pow(1.0 - per_call_tail_rate, n)) * 100.0, 2)
        for n in chain_lengths
    }


def resample_chains_independent(single_call_latencies: List[float], chain_length: int, iterations: int = 5000) -> List[float]:
    """Resampling Baseline 1: Independent random draws from matched single-call distribution."""
    if not single_call_latencies:
        return []
    simulated_totals = []
    for _ in range(iterations):
        sample_sum = sum(random.choices(single_call_latencies, k=chain_length))
        simulated_totals.append(sample_sum)
    return simulated_totals


def resample_chains_block_preserving(single_call_latencies: List[float], chain_length: int, iterations: int = 5000) -> List[float]:
    """Resampling Baseline 2: Block/time-window preserving resampling to retain queue correlation."""
    n = len(single_call_latencies)
    if n < chain_length:
        return resample_chains_independent(single_call_latencies, chain_length, iterations)
    simulated_totals = []
    max_start = n - chain_length
    for _ in range(iterations):
        start_idx = random.randint(0, max_start)
        block_sum = sum(single_call_latencies[start_idx : start_idx + chain_length])
        simulated_totals.append(block_sum)
    return simulated_totals


def run_benchmark(
    base_url: str,
    api_key: Optional[str],
    model: str,
    concurrency_list: List[int],
    requests_per_cell: int,
    num_chains: int,
    chain_length: int,
    tokens_per_call: int,
    accumulate_context: bool,
    ttft_deadline_ms: float,
    task_deadline_s: float,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "agent_chain_manifest.json"

    print(f"\n=======================================================")
    print(f"  REVISED AGENT CHAIN & LATENCY DISTRIBUTION BENCHMARK")
    print(f"  Target: {base_url} (Model: {model})")
    print(f"  Concurrency Sweep: {concurrency_list} ({requests_per_cell} reqs/cell)")
    print(f"  Complete-Task Agent Benchmark: {num_chains} chains of {chain_length} calls")
    print(f"  Deadlines: TTFT <= {ttft_deadline_ms}ms | Task <= {task_deadline_s}s")
    print(f"=======================================================\n")

    benchmark_data: Dict[str, Any] = {
        "metadata": {
            "timestamp": utc_now(),
            "base_url": base_url,
            "model": model,
            "concurrency_list": concurrency_list,
            "requests_per_cell": requests_per_cell,
            "num_chains": num_chains,
            "chain_length": chain_length,
            "tokens_per_call": tokens_per_call,
            "accumulate_context": accumulate_context,
            "ttft_deadline_ms": ttft_deadline_ms,
            "task_deadline_s": task_deadline_s,
            "workload_mode": "closed_loop_concurrency",
            "prompt_length_bucket": "controlled_~100_tokens",
            "output_length_bucket": f"target_{tokens_per_call}_tokens",
        },
        "concurrency_sweep": {},
        "chained_benchmark": {},
        "compound_risk_reference": {},
    }

    # 1. Concurrency Sweep
    c1_latencies: List[float] = []

    for c in concurrency_list:
        print(f"[*] Running Single-Call Concurrency C={c} ({requests_per_cell} requests)...", flush=True)
        t0 = time.time()
        results: List[Dict[str, Any]] = []

        prompt_messages = [
            {"role": "user", "content": "Explain the trade-off between median and tail latency in distributed LLM inference."}
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
            futures = [
                executor.submit(
                    streamed_chat_request,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=prompt_messages,
                    max_tokens=tokens_per_call,
                )
                for _ in range(requests_per_cell)
            ]
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

        elapsed = time.time() - t0
        successful = [r for r in results if r["success"]]
        ttfts = [r["ttft_ms"] for r in successful]
        ttfats = [r["ttfat_ms"] for r in successful if r.get("ttfat_ms") is not None]
        totals = [r["total_time_ms"] for r in successful]
        dec_tps = [r["decode_tps"] for r in successful if r.get("decode_tps") is not None]

        if c == 1:
            c1_latencies = totals

        ttft_stats = calculate_distribution_stats(ttfts, deadline_threshold=ttft_deadline_ms)
        ttfat_stats = calculate_distribution_stats(ttfats, deadline_threshold=ttft_deadline_ms)
        total_stats = calculate_distribution_stats(totals)
        tps_stats = calculate_distribution_stats(dec_tps)

        print(f"    -> C={c} in {elapsed:.1f}s | Success: {len(successful)}/{requests_per_cell}")
        if not successful and not benchmark_data["concurrency_sweep"]:
            err = next((r.get("error") for r in results if r.get("error")), "request failed")
            print(f"       Error: {err}", flush=True)
            raise SystemExit(f"Every request failed at C={c}: {err}")
        print(f"       TTFT: p50={ttft_stats.get('p50')}ms, p95={ttft_stats.get('p95')}ms, p99={ttft_stats.get('p99')}ms (p99:p50={ttft_stats.get('p99_to_p50_ratio')}x | Miss: {ttft_stats.get('deadline_miss_pct')}%)")
        print(f"       TTFAT (1st Answer): p50={ttfat_stats.get('p50')}ms, p95={ttfat_stats.get('p95')}ms")
        print(f"       Decode Rate (M>1): p50={tps_stats.get('p50')} tok/s")
        print(f"       Total E2E: p50={total_stats.get('p50')}ms, p95={total_stats.get('p95')}ms, p99={total_stats.get('p99')}ms")

        benchmark_data["concurrency_sweep"][f"c{c}"] = {
            "concurrency": c,
            "requests_sent": requests_per_cell,
            "requests_succeeded": len(successful),
            "elapsed_seconds": round(elapsed, 2),
            "ttft_stats": ttft_stats,
            "ttfat_stats": ttfat_stats,
            "total_time_stats": total_stats,
            "decode_tps_stats": tps_stats,
            "raw_records": results,
        }

    # Derive baseline slow-call threshold from C=1 baseline p95
    baseline_slow_threshold = calculate_distribution_stats(c1_latencies).get("p95") or 2000.0

    # 2. Complete-Task Chained Agent Benchmark
    print(f"\n[*] Running Complete-Task Chained Agent Loop ({num_chains} chains, length={chain_length})...", flush=True)
    chain_records = []
    t_chains_start = time.time()

    for ch_idx in range(1, num_chains + 1):
        chain_res = execute_agent_chain(
            chain_id=ch_idx,
            base_url=base_url,
            api_key=api_key,
            model=model,
            chain_length=chain_length,
            accumulate_context=accumulate_context,
            tokens_per_call=tokens_per_call,
            simulated_tool_delay_ms=30.0,
            single_call_slow_threshold_ms=baseline_slow_threshold,
        )
        chain_records.append(chain_res)
        status_sym = "✓" if chain_res["all_succeeded"] else "✗"
        slow_tag = " [SLOW CALL ENCOUNTERED]" if chain_res["contains_slow_call"] else ""
        print(f"    [Chain {ch_idx:02d}/{num_chains:02d}] {status_sym} Wall: {chain_res['total_task_wall_ms']/1000.0:.2f}s | LLM sum: {chain_res['sum_llm_duration_ms']/1000.0:.2f}s{slow_tag}", flush=True)

    elapsed_chains = time.time() - t_chains_start
    chain_wall_totals = [c["total_task_wall_ms"] for c in chain_records if c["all_succeeded"]]
    chain_stats = calculate_distribution_stats(chain_wall_totals, deadline_threshold=task_deadline_s * 1000.0)

    # 3. Two Resampling Baselines
    resample_indep = resample_chains_independent(c1_latencies, chain_length, iterations=5000)
    indep_stats = calculate_distribution_stats(resample_indep, deadline_threshold=task_deadline_s * 1000.0)

    resample_block = resample_chains_block_preserving(c1_latencies, chain_length, iterations=5000)
    block_stats = calculate_distribution_stats(resample_block, deadline_threshold=task_deadline_s * 1000.0)

    # 4. Exceedance Risk Reference Math
    tail_prob_p99 = compute_compound_tail_risk_reference(0.01, [1, 2, 5, 10, 15, 20])
    tail_prob_p95 = compute_compound_tail_risk_reference(0.05, [1, 2, 5, 10, 15, 20])

    tasks_with_slow_call = sum(1 for c in chain_records if c["contains_slow_call"])
    pct_tasks_with_slow_call = round(tasks_with_slow_call / max(len(chain_records), 1) * 100.0, 1)

    benchmark_data["chained_benchmark"] = {
        "num_chains": num_chains,
        "chain_length": chain_length,
        "accumulate_context": accumulate_context,
        "elapsed_seconds": round(elapsed_chains, 2),
        "baseline_single_call_threshold_ms": baseline_slow_threshold,
        "tasks_with_slow_call_pct": pct_tasks_with_slow_call,
        "empirical_task_duration_stats_ms": chain_stats,
        "resampling_baseline_1_independent_ms": indep_stats,
        "resampling_baseline_2_block_correlated_ms": block_stats,
        "chains": chain_records,
    }

    benchmark_data["compound_risk_reference"] = {
        "p99_reference_exceedance_pct": tail_prob_p99,
        "p95_reference_exceedance_pct": tail_prob_p95,
    }

    # Write output JSON
    manifest_path.write_text(json.dumps(benchmark_data, indent=2))
    print(f"\n[+] Raw results saved to: {manifest_path}")

    # Generate Markdown summary report
    report_path = output_dir / "agent_chain_report.md"
    generate_markdown_report(benchmark_data, report_path)
    print(f"[+] Markdown report saved to: {report_path}\n")

    return benchmark_data


def generate_markdown_report(data: Dict[str, Any], report_path: Path) -> None:
    meta = data["metadata"]
    c_sweep = data["concurrency_sweep"]
    ch_bench = data["chained_benchmark"]
    emp_stats = ch_bench["empirical_task_duration_stats_ms"]
    indep_stats = ch_bench["resampling_baseline_1_independent_ms"]
    block_stats = ch_bench["resampling_baseline_2_block_correlated_ms"]
    risks = data["compound_risk_reference"]

    md = []
    md.append(f"# Complete-Task Agent Latency & Compounding Report")
    md.append(f"\n**Model:** `{meta['model']}`  ")
    md.append(f"**Target URL:** `{meta['base_url']}`  ")
    md.append(f"**Workload Mode:** `{meta['workload_mode']}`  ")
    md.append(f"**Length Buckets:** Prompt `{meta['prompt_length_bucket']}` | Output `{meta['output_length_bucket']}`  ")
    md.append(f"**Interactive Deadlines:** TTFT $\\le {meta['ttft_deadline_ms']}$ ms | Task $\\le {meta['task_deadline_s']}$ s  ")
    md.append(f"**Generated:** {meta['timestamp']}  \n")
    md.append(f"---\n")

    # Table 1: Concurrency Sweep
    md.append(f"## 1. Single-Call Latency Distributions (TTFT, TTFAT, and Decode Rates)")
    md.append(f"\n| Concurrency | Requests | TTFT p50/p95/p99 | TTFAT (1st Answer) p50/p95 | Decode Rate (M>1) | Total Latency p50/p99 | TTFT Miss (% >{meta['ttft_deadline_ms']}ms) |")
    md.append(f"|---:|---:|---:|---:|---:|---:|---:|")

    for c_key, c_data in c_sweep.items():
        ttft = c_data["ttft_stats"]
        ttfat = c_data.get("ttfat_stats", {})
        tot = c_data["total_time_stats"]
        tps = c_data.get("decode_tps_stats", {})
        md.append(
            f"| C={c_data['concurrency']} | {c_data['requests_succeeded']}/{c_data['requests_sent']} | "
            f"{ttft.get('p50', 0):.0f}/{ttft.get('p95', 0):.0f}/{ttft.get('p99', 0):.0f} ms | "
            f"{ttfat.get('p50', 0):.0f}/{ttfat.get('p95', 0):.0f} ms | "
            f"{tps.get('p50', 0):.1f} tok/s | "
            f"{tot.get('p50', 0):.0f}/{tot.get('p99', 0):.0f} ms | "
            f"**{ttft.get('deadline_miss_pct', 0.0)}%** |"
        )

    # Table 2: Complete-Task Benchmark
    md.append(f"\n## 2. Complete-Task Agent Trajectory Benchmark")
    md.append(f"\n*Workload: {meta['num_chains']} independent sequential chains of {meta['chain_length']} calls per chain.*\n")
    md.append(f"| Metric | Empirical Measured Tasks | Resampling Baseline 1 (Independent) | Resampling Baseline 2 (Correlated Block) |")
    md.append(f"|:---|---:|---:|---:|")
    md.append(f"| **Sample Count ($N$)** | {emp_stats.get('n', 0)} | {indep_stats.get('n', 0)} | {block_stats.get('n', 0)} |")
    md.append(f"| **Task Duration p50** | **{emp_stats.get('p50', 0)/1000.0:.2f} s** | {indep_stats.get('p50', 0)/1000.0:.2f} s | {block_stats.get('p50', 0)/1000.0:.2f} s |")
    md.append(f"| **Task Duration p95** | **{emp_stats.get('p95', 0)/1000.0:.2f} s** | {indep_stats.get('p95', 0)/1000.0:.2f} s | {block_stats.get('p95', 0)/1000.0:.2f} s |")
    md.append(f"| **Task Duration p99** | **{emp_stats.get('p99', 0)/1000.0:.2f} s** | {indep_stats.get('p99', 0)/1000.0:.2f} s | {block_stats.get('p99', 0)/1000.0:.2f} s |")
    md.append(f"| **Worst-Case Task** | **{emp_stats.get('max', 0)/1000.0:.2f} s** | {indep_stats.get('max', 0)/1000.0:.2f} s | {block_stats.get('max', 0)/1000.0:.2f} s |")
    md.append(f"| **p99:p50 Spread Ratio** | **{emp_stats.get('p99_to_p50_ratio', 1.0):.2f}x** | {indep_stats.get('p99_to_p50_ratio', 1.0):.2f}x | {block_stats.get('p99_to_p50_ratio', 1.0):.2f}x |")
    md.append(f"| **Coefficient of Variation (CV)** | {emp_stats.get('cv_pct', 0):.1f}% | {indep_stats.get('cv_pct', 0):.1f}% | {block_stats.get('cv_pct', 0):.1f}% |")
    md.append(f"| **Task Deadline Miss (% >{meta['task_deadline_s']}s)** | **{emp_stats.get('deadline_miss_pct', 0.0)}%** | {indep_stats.get('deadline_miss_pct', 0.0)}% | {block_stats.get('deadline_miss_pct', 0.0)}% |")
    md.append(f"| **Tasks with >=1 Slow Call** | **{ch_bench['tasks_with_slow_call_pct']}%** | N/A | N/A |")

    # Section 3: Mathematical Systems Foundations & Corrected Crossover
    md.append(f"""
## 3. Mathematical Systems Foundations & Corrected Trade-Offs

### 3.1 Corrected TTFT-vs-Decode Crossover Formula
For a constant-gap model:
$$T_{{\\mathrm{{complete}}}} \\approx T_{{\\mathrm{{TTFT}}}} + (M - 1) \\cdot \\overline{{\\mathrm{{ITL}}}}$$

For Model A ($T_{{\\mathrm{{TTFT}}}}=200\\text{{ ms}}$, rate $= 20\\text{{ tok/s}} \\implies \\overline{{\\mathrm{{ITL}}}} = 0.05\\text{{ s}}$) versus Model B ($T_{{\\mathrm{{TTFT}}}}=1000\\text{{ ms}}$, rate $= 100\\text{{ tok/s}} \\implies \\overline{{\\mathrm{{ITL}}}} = 0.01\\text{{ s}}$):
$$\\Delta T_{{\\mathrm{{TTFT}}}} = 1.0 - 0.2 = 0.8\\text{{ s}}$$
$$\\Delta \\overline{{\\mathrm{{ITL}}}} = 0.05 - 0.01 = 0.04\\text{{ s/interval}}$$
$$\\text{{Crossover at }} M - 1 = \\frac{{0.8}}{{0.04}} = 20 \\implies M \\approx 21\\text{{ output tokens}}$$

*For any task generating more than approximately 21 tokens, Model B completes earlier despite having a $5\\times$ slower initial token.*

### 3.2 Compounding Exceedance Reference vs. Complete Task Duration
Under the independent, identically distributed assumption:
$$P(\\text{{at least one call exceeds threshold }} T) = 1 - (1 - q)^N$$

| Chain Length ($N$) | Independent Exceedance Reference ($q=1%$) | Independent Exceedance Reference ($q=5%$) |
|---:|---:|---:|""")

    p99_risks = risks.get("p99_reference_exceedance_pct", {})
    p95_risks = risks.get("p95_reference_exceedance_pct", {})
    for n in [1, 2, 5, 10, 15, 20]:
        r99 = p99_risks.get(str(n), p99_risks.get(n, 0.0))
        r95 = p95_risks.get(str(n), p95_risks.get(n, 0.0))
        md.append(f"| {n} calls | {r99}% | **{r95}%** |")

    md.append(f"""
*Important distinction:* This predicts the probability of encountering an outlier call; it is **not** the workflow's p95 or p99 completion time. Sequential call durations add ($T_{{\\text{{task}}}} = \\sum T_i$), and shared queue congestion correlates outliers, causing the empirical task tail to diverge from independent convolutions.
""")

    report_path.write_text(
        ensure_frontmatter(
            "\n".join(md) + "\n",
            doc_type="Benchmark Report",
            title="Complete-Task Agent Latency & Compounding Report",
            description=(
                "Measured single-call and complete-task latency distributions "
                "for sequential agent chains."
            ),
            tags=["agent-chain", "tail-latency", "benchmark"],
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", os.environ.get("DO_MODEL_ACCESS_KEY")))
    parser.add_argument("--model", default="Qwen3.8-27B-Quark-AWQ-MXFP4")
    parser.add_argument("--concurrency-list", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--requests-per-cell", type=int, default=75)
    parser.add_argument("--num-chains", type=int, default=30)
    parser.add_argument("--chain-length", type=int, default=10)
    parser.add_argument("--tokens-per-call", type=int, default=64)
    parser.add_argument("--no-context-accumulation", action="store_true")
    parser.add_argument("--ttft-deadline-ms", type=float, default=1000.0)
    parser.add_argument("--task-deadline-s", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("_results/agent_chain_bench"))
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_dir = args.output_dir / stamp

    run_benchmark(
        base_url=args.url,
        api_key=args.api_key,
        model=args.model,
        concurrency_list=args.concurrency_list,
        requests_per_cell=args.requests_per_cell,
        num_chains=args.num_chains,
        chain_length=args.chain_length,
        tokens_per_call=args.tokens_per_call,
        accumulate_context=not args.no_context_accumulation,
        ttft_deadline_ms=args.ttft_deadline_ms,
        task_deadline_s=args.task_deadline_s,
        output_dir=target_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
