#!/usr/bin/env python3
"""
Tail Latency & Agent Distribution Analyzer.

Analyzes raw benchmark JSON manifests to surface:
  1. Statistical Descriptors (p50, p90, p95, p99, CV, p99:p50 spread ratio, standard error).
  2. Distribution Shape Observation (avoids assigning server-side causes without server spans):
     - Moderate spread vs. Heavy tail
     - Apparent multi-mode candidate (flags potential cluster separation for investigation)
  3. TTFT vs. Task Completion Crossover & Ranking Inversion:
     - Calculates the exact crossover point M ≈ Delta_TTFT / Delta_ITL + 1
     - Evaluates whether single-request TTFT ranking reverses at the complete-task level.
  4. Task-Level Compounding vs. Independent Exceedance Reference:
     - Reports observed fraction of tasks containing slow calls vs. independent model reference.

Usage:
  python3 scripts/analyze_tail_metrics.py --input _results/agent_chain_bench/<stamp>/agent_chain_manifest.json
  python3 scripts/analyze_tail_metrics.py --compare run1.json run2.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)


def compute_distribution(samples: List[float]) -> Dict[str, Any]:
    clean = [float(v) for v in samples if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"n": 0}

    n = len(clean)
    mean_v = statistics.mean(clean)
    std_v = statistics.stdev(clean) if n > 1 else 0.0
    stderr = std_v / math.sqrt(n) if n > 1 else 0.0
    p50 = percentile(clean, 50.0) or 0.0
    p90 = percentile(clean, 90.0) or 0.0
    p95 = percentile(clean, 95.0) or 0.0
    p99 = percentile(clean, 99.0) or 0.0
    max_v = max(clean)
    min_v = min(clean)

    cv = (std_v / mean_v * 100.0) if mean_v > 0 else 0.0
    p99_to_p50 = (p99 / p50) if p50 > 0 else 1.0
    p95_to_p50 = (p95 / p50) if p50 > 0 else 1.0

    # Observe shape without asserting server-side cause
    is_multimodal = False
    sorted_clean = sorted(clean)
    if n >= 15:
        high_outliers = [v for v in sorted_clean if v >= p50 * 4.0]
        if len(high_outliers) >= 2 and (len(high_outliers) / n) <= 0.25:
            gap = min(high_outliers) - percentile(sorted_clean, 75.0)
            if gap > p50 * 2.0:
                is_multimodal = True

    if is_multimodal:
        shape_descriptor = "apparent_bimodal_candidate"
    elif p99_to_p50 >= 3.0:
        shape_descriptor = "heavy_tailed_spread"
    else:
        shape_descriptor = "moderate_spread"

    return {
        "n": n,
        "mean": round(mean_v, 2),
        "std": round(std_v, 2),
        "stderr": round(stderr, 2),
        "min": round(min_v, 2),
        "p50": round(p50, 2),
        "p90": round(p90, 2),
        "p95": round(p95, 2),
        "p99": round(p99, 2),
        "max": round(max_v, 2),
        "cv_pct": round(cv, 1),
        "p99_to_p50_ratio": round(p99_to_p50, 2),
        "p95_to_p50_ratio": round(p95_to_p50, 2),
        "shape_descriptor": shape_descriptor,
    }


def analyze_single_manifest(manifest_path: Path) -> Dict[str, Any]:
    with manifest_path.open() as f:
        data = json.load(f)

    report = {"path": str(manifest_path), "sections": {}}

    if "chained_benchmark" in data:
        meta = data.get("metadata", {})
        report["model"] = meta.get("model", "unknown")
        report["target_url"] = meta.get("base_url", "unknown")

        ch = data["chained_benchmark"]
        emp = ch.get("empirical_task_duration_stats_ms", {})
        b1 = ch.get("resampling_baseline_1_independent_ms", {})
        b2 = ch.get("resampling_baseline_2_block_correlated_ms", {})

        report["sections"]["chained_task"] = {
            "num_chains": ch.get("num_chains"),
            "chain_length": ch.get("chain_length"),
            "empirical_task_p50_s": round(emp.get("p50", 0) / 1000.0, 2),
            "empirical_task_p95_s": round(emp.get("p95", 0) / 1000.0, 2),
            "empirical_task_p99_s": round(emp.get("p99", 0) / 1000.0, 2),
            "empirical_task_max_s": round(emp.get("max", 0) / 1000.0, 2),
            "empirical_p99_to_p50_ratio": emp.get("p99_to_p50_ratio"),
            "resample_indep_p50_s": round(b1.get("p50", 0) / 1000.0, 2) if b1 else None,
            "resample_block_p50_s": round(b2.get("p50", 0) / 1000.0, 2) if b2 else None,
            "tasks_with_slow_call_pct": ch.get("tasks_with_slow_call_pct"),
            "shape_descriptor": emp.get("shape_descriptor"),
        }

        c_sweep = data.get("concurrency_sweep", {})
        sweep_report = {}
        for c_key, c_row in c_sweep.items():
            sweep_report[c_key] = {
                "concurrency": c_row.get("concurrency"),
                "ttft": c_row.get("ttft_stats"),
                "ttfat": c_row.get("ttfat_stats"),
                "total": c_row.get("total_time_stats"),
                "decode_tps": c_row.get("decode_tps_stats") or c_row.get("generation_tps_stats"),
            }
        report["sections"]["concurrency_sweep"] = sweep_report

    elif "metrics" in data or "per_concurrency" in data or "runs" in data:
        report["kind"] = "agentx"
        report["raw"] = data

    return report


def compare_manifests(paths: List[Path]) -> None:
    """Print side-by-side comparison with crossover calculation and ranking audit."""
    reports = [analyze_single_manifest(p) for p in paths]

    print("\n" + "=" * 78)
    print("  SIDE-BY-SIDE MODEL / RUN COMPARISON (PERFORMANCE TRADE-OFF AUDIT)")
    print("=" * 78)

    rows = []
    for r in reports:
        model = r.get("model", Path(r["path"]).parent.name)
        c_sweep = r.get("sections", {}).get("concurrency_sweep", {})
        c1 = c_sweep.get("c1", {})
        ttft_c1 = c1.get("ttft", {})
        tot_c1 = c1.get("total", {})
        tps_c1 = c1.get("decode_tps", {})

        ch = r.get("sections", {}).get("chained_task", {})

        rows.append({
            "model": model,
            "path": r["path"],
            "ttft_p50_ms": ttft_c1.get("p50"),
            "ttft_p99_ms": ttft_c1.get("p99"),
            "ttft_ratio": ttft_c1.get("p99_to_p50_ratio"),
            "gen_tps_p50": tps_c1.get("p50"),
            "tot_p50_ms": tot_c1.get("p50"),
            "tot_p99_ms": tot_c1.get("p99"),
            "chain_p50_s": ch.get("empirical_task_p50_s"),
            "chain_p95_s": ch.get("empirical_task_p95_s"),
            "chain_max_s": ch.get("empirical_task_max_s"),
            "slow_call_pct": ch.get("tasks_with_slow_call_pct"),
            "shape": ch.get("shape_descriptor") or tot_c1.get("shape_descriptor"),
        })

    # Summary Table
    print(f"\n| Model / Run | TTFT p50 | TTFT p99 | Decode tok/s | 10-Call Task p50 | 10-Call Task p95 | Slow Call % | Shape Descriptor |")
    print(f"|:---|---:|---:|---:|---:|---:|---:|:---|")
    for row in rows:
        print(
            f"| `{row['model']}` | "
            f"{row['ttft_p50_ms'] or 'N/A'} ms | "
            f"{row['ttft_p99_ms'] or 'N/A'} ms | "
            f"{row['gen_tps_p50'] or 'N/A'} tok/s | "
            f"**{row['chain_p50_s'] or 'N/A'} s** | "
            f"**{row['chain_p95_s'] or 'N/A'} s** | "
            f"{row['slow_call_pct'] or 'N/A'}% | "
            f"`{row['shape'] or 'N/A'}` |"
        )

    # Inversion & Crossover Audit
    if len(rows) >= 2:
        r1, r2 = rows[0], rows[1]
        print("\n" + "-" * 78)
        print("  TRADE-OFF & CROSSOVER ANALYSIS:")

        t1_ttft, t2_ttft = r1.get("ttft_p50_ms"), r2.get("ttft_p50_ms")
        t1_tps, t2_tps = r1.get("gen_tps_p50"), r2.get("gen_tps_p50")

        if t1_ttft and t2_ttft and t1_tps and t2_tps and t1_tps > 0 and t2_tps > 0:
            itl1_s = 1.0 / t1_tps
            itl2_s = 1.0 / t2_tps
            delta_ttft_s = abs(t1_ttft - t2_ttft) / 1000.0
            delta_itl_s = abs(itl1_s - itl2_s)

            if delta_itl_s > 0.0001:
                crossover_m = round(delta_ttft_s / delta_itl_s) + 1
                faster_ttft_model = r1["model"] if t1_ttft < t2_ttft else r2["model"]
                faster_tps_model = r1["model"] if t1_tps > t2_tps else r2["model"]
                print(f"  • Faster TTFT: `{faster_ttft_model}` ({min(t1_ttft, t2_ttft):.0f} ms vs {max(t1_ttft, t2_ttft):.0f} ms)")
                print(f"  • Faster Decode Rate: `{faster_tps_model}` ({max(t1_tps, t2_tps):.1f} tok/s vs {min(t1_tps, t2_tps):.1f} tok/s)")
                print(f"  • Theoretical Completion Crossover: M ≈ {crossover_m} output tokens")
                print(f"    (Calculated via: Delta_TTFT / Delta_ITL + 1 = {delta_ttft_s:.3f}s / {delta_itl_s:.4f}s + 1)")

        if r1["ttft_p50_ms"] and r2["ttft_p50_ms"] and r1["chain_p50_s"] and r2["chain_p50_s"]:
            ttft_winner = r1["model"] if r1["ttft_p50_ms"] < r2["ttft_p50_ms"] else r2["model"]
            task_winner = r1["model"] if r1["chain_p50_s"] < r2["chain_p50_s"] else r2["model"]

            if ttft_winner != task_winner:
                print(f"\n  ⚠️  RANKING REVERSAL OBSERVED:")
                print(f"     • Winner on Single-Call TTFT: `{ttft_winner}`")
                print(f"     • Winner on Complete Agent Task: `{task_winner}`")
                print(f"     Result: A faster initial token did not translate to faster task completion under these conditions.")
            else:
                print(f"\n  ✓ Consistent Ranking: `{task_winner}` completed earlier on both TTFT and the complete task.")
        print("-" * 78 + "\n")


def print_single_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(f"  BENCHMARK DIAGNOSTIC REPORT: {report.get('model', 'Model')}")
    print("=" * 78)

    sections = report.get("sections", {})
    if "concurrency_sweep" in sections:
        print("\n--- Concurrency Sweep Metrics ---")
        print(f"| C | TTFT p50 | TTFT p95 | TTFAT p50 | p99:p50 | Total p50 | Total p99 | Miss (%) | CV (%) |")
        print(f"|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c_key, c_row in sections["concurrency_sweep"].items():
            ttft = c_row.get("ttft") or {}
            ttfat = c_row.get("ttfat") or {}
            tot = c_row.get("total") or {}
            c_val = c_row.get("concurrency")
            print(
                f"| C={c_val} | "
                f"{ttft.get('p50', 0):.0f} ms | {ttft.get('p95', 0):.0f} ms | "
                f"{ttfat.get('p50', 0):.0f} ms | "
                f"{ttft.get('p99_to_p50_ratio', 1.0):.2f}x | "
                f"{tot.get('p50', 0):.0f} ms | {tot.get('p99', 0):.0f} ms | "
                f"{ttft.get('deadline_miss_pct', 0.0)}% | "
                f"{tot.get('cv_pct', 0):.1f}% |"
            )

    if "chained_task" in sections:
        ch = sections["chained_task"]
        print(f"\n--- Complete-Task Agent Trajectory Duration ---")
        print(f"  • Chains Measured: {ch.get('num_chains')} chains of {ch.get('chain_length')} sequential calls")
        print(f"  • Empirical Duration: p50={ch.get('empirical_task_p50_s')}s, p95={ch.get('empirical_task_p95_s')}s, p99={ch.get('empirical_task_p99_s')}s (Max: {ch.get('empirical_task_max_s')}s)")
        print(f"  • Resampling Baseline 1 (Independent): p50={ch.get('resample_indep_p50_s')}s")
        print(f"  • Resampling Baseline 2 (Block-Correlated): p50={ch.get('resample_block_p50_s')}s")
        print(f"  • Tasks Encountering a Slow Call: {ch.get('tasks_with_slow_call_pct')}%")
        print(f"  • Spread Ratio (p99:p50): {ch.get('empirical_p99_to_p50_ratio')}x")

    print("\n" + "=" * 78 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Single benchmark JSON manifest to analyze")
    parser.add_argument("--compare", nargs="+", type=Path, help="Compare two or more JSON manifests")
    args = parser.parse_args()

    if args.compare:
        compare_manifests(args.compare)
    elif args.input:
        report = analyze_single_manifest(args.input)
        print_single_report(report)
    else:
        parser.error("Specify either --input or --compare")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
