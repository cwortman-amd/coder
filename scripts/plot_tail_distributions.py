#!/usr/bin/env python3
"""
Generate publication-quality diagnostic plots for Agent Chain & Tail Latency Benchmark.

Creates a 4-panel master dashboard:
  Panel A: Empirical CDF of TTFT across Concurrencies (Log Scale)
  Panel B: Chained Agent Task Duration (Empirical vs. Bootstrap Resampled)
  Panel C: Compounding Tail Risk as a Function of Chain Length (Theory vs. Observation)
  Panel D: Single-Call vs. 10-Call Task Duration Spread (Box / Violin comparison)

Requires: matplotlib, numpy
Usage:
  .venv/bin/python3 scripts/plot_tail_distributions.py --input docs/results/tail_study/agent_chain_manifest.json --out docs/figures/tail_study_dashboard.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import BLUE, GREEN, GRAY, ORANGE, PURPLE, RED, TEAL  # noqa: E402


PALETTE = {
    "c1": BLUE,
    "c5": TEAL,
    "c20": RED,
    "chain": PURPLE,
    "boot": ORANGE,
    "accent": GREEN,
    "grid": "#E0E0E0",
}


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def plot_dashboard(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = data.get("metadata", {})
    model_name = meta.get("model", "Model")
    c_sweep = data.get("concurrency_sweep", {})
    ch_bench = data.get("chained_benchmark", {})
    risks = data.get("compound_risk_reference") or data.get("compound_risk_analysis") or {}

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig.suptitle(
        f"LLM Latency & Multi-Step Agent Serving Dynamics: {model_name}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # ----------------------------------------------------
    # Panel A: Empirical CDF of TTFT across Concurrencies
    # ----------------------------------------------------
    ax_a = axes[0, 0]
    has_ttft = False
    colors = [PALETTE["c1"], PALETTE["c5"], PALETTE["c20"], "#3949AB", "#D81B60"]

    for idx, (c_key, c_row) in enumerate(c_sweep.items()):
        raws = c_row.get("raw_records", [])
        ttfts = sorted([r["ttft_ms"] for r in raws if r.get("ttft_ms") is not None])
        if not ttfts:
            continue
        has_ttft = True
        c_val = c_row.get("concurrency", c_key)
        y = np.linspace(0, 100, len(ttfts))
        color = colors[idx % len(colors)]
        p50 = np.percentile(ttfts, 50)
        p99 = np.percentile(ttfts, 99)
        ax_a.plot(ttfts, y, label=f"C={c_val} (p50: {p50:.0f}ms, p99: {p99:.0f}ms)", color=color, linewidth=2.2)

    if has_ttft:
        ax_a.set_xscale("log")
        ax_a.axvline(1000, color=GRAY, linestyle="--", linewidth=1.2, label="1.0s Interactive Threshold")
        ax_a.axhline(99, color=RED, linestyle=":", linewidth=1.2, label="p99 Percentile Target")
        ax_a.set_title("A. TTFT Cumulative Distribution (CDF)", fontsize=11, fontweight="bold")
        ax_a.set_xlabel("Time-to-First-Token (TTFT, ms) [Log Scale]", fontsize=10)
        ax_a.set_ylabel("Percentage of Requests (%)", fontsize=10)
        ax_a.grid(True, which="both", linestyle="--", alpha=0.5)
        ax_a.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    else:
        ax_a.text(0.5, 0.5, "No Concurrency TTFT Data", ha="center", va="center")

    # ----------------------------------------------------
    # Panel B: Chained Agent Task Duration (Empirical vs. Dual Resampling)
    # ----------------------------------------------------
    ax_b = axes[0, 1]
    chains = ch_bench.get("chains", [])
    emp_totals = [c.get("total_task_wall_ms", c.get("total_chain_duration_ms", 0.0)) / 1000.0 for c in chains if c.get("all_succeeded")]
    base1_stats = ch_bench.get("resampling_baseline_1_independent_ms", ch_bench.get("bootstrap_simulated_task_stats_ms", {}))
    base2_stats = ch_bench.get("resampling_baseline_2_block_correlated_ms", {})
    chain_len = ch_bench.get("chain_length", 10)

    if emp_totals:
        emp_sorted = np.sort(emp_totals)
        y_emp = np.linspace(0, 100, len(emp_sorted))
        p50_emp = np.percentile(emp_sorted, 50)
        p95_emp = np.percentile(emp_sorted, 95)
        ax_b.plot(emp_sorted, y_emp, label=f"Empirical Tasks (p50: {p50_emp:.2f}s, p95: {p95_emp:.2f}s)", color=PALETTE["chain"], linewidth=2.5)

        # Plot Resampling Baseline 1 (Independent i.i.d.)
        if base1_stats and base1_stats.get("p50"):
            b1_p50 = base1_stats.get("p50", 0) / 1000.0
            b1_p95 = base1_stats.get("p95", 0) / 1000.0
            ax_b.axvline(b1_p50, color=PALETTE["boot"], linestyle="--", linewidth=1.8, label=f"Baseline 1 (Indep) p50 ({b1_p50:.2f}s)")
            ax_b.axvline(b1_p95, color=PALETTE["boot"], linestyle=":", linewidth=1.8, label=f"Baseline 1 (Indep) p95 ({b1_p95:.2f}s)")

        # Plot Resampling Baseline 2 (Block Correlated)
        if base2_stats and base2_stats.get("p95"):
            b2_p95 = base2_stats.get("p95", 0) / 1000.0
            ax_b.axvline(b2_p95, color=PURPLE, linestyle="-.", linewidth=1.8, label=f"Baseline 2 (Block) p95 ({b2_p95:.2f}s)")

        ax_b.set_title(f"B. {chain_len}-Call Chained Task Completion Time", fontsize=11, fontweight="bold")
        ax_b.set_xlabel("Total Task Duration (seconds)", fontsize=10)
        ax_b.set_ylabel("Percentage of Chains (%)", fontsize=10)
        ax_b.grid(True, linestyle="--", alpha=0.5)
        ax_b.legend(loc="lower right", fontsize=8.0, framealpha=0.9)
    else:
        ax_b.text(0.5, 0.5, "No Chained Task Data", ha="center", va="center")

    # ----------------------------------------------------
    # Panel C: Compounding Tail Risk as Function of Chain Length
    # ----------------------------------------------------
    ax_c = axes[1, 0]
    p95_ref = risks.get("p95_reference_exceedance_pct") or {}
    p99_ref = risks.get("p99_reference_exceedance_pct") or {}
    if p95_ref and p99_ref:
        lengths = sorted(int(n) for n in p95_ref)
        risk_p95 = [float(p95_ref[str(n)] if str(n) in p95_ref else p95_ref[n]) for n in lengths]
        risk_p99 = [float(p99_ref[str(n)] if str(n) in p99_ref else p99_ref[n]) for n in lengths]
    else:
        lengths = [1, 2, 3, 5, 8, 10, 12, 15, 20, 25]
        risk_p99 = [(1.0 - math.pow(0.99, n)) * 100.0 for n in lengths]
        risk_p95 = [(1.0 - math.pow(0.95, n)) * 100.0 for n in lengths]

    ax_c.plot(lengths, risk_p95, marker="o", color=RED, linewidth=2.2, label="Single-Call p95 Tail (q=5%)")
    ax_c.plot(lengths, risk_p99, marker="s", color=BLUE, linewidth=2.2, label="Single-Call p99 Tail (q=1%)")
    ax_c.axhline(50, color=GRAY, linestyle="--", linewidth=1.2, label="50% Coin-Flip Threshold")
    ax_c.axvline(10, color=PURPLE, linestyle=":", linewidth=1.5, label="N=10 Typical Agent Chain")

    ax_c.set_title(r"C. Mathematical Tail Compounding: $P(\geq 1\ \mathrm{tail}) = 1-(1-q)^N$", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Sequential Chain Length (Model Calls per Task)", fontsize=10)
    ax_c.set_ylabel("Probability of Hitting At Least 1 Tail Event (%)", fontsize=10)
    ax_c.set_ylim(0, 100)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # ----------------------------------------------------
    # Panel D: Single-Call vs. Chained Call Spread (Box Plot)
    # ----------------------------------------------------
    ax_d = axes[1, 1]
    plot_data = []
    labels = []

    # Add single call C=1
    c1_raws = c_sweep.get("c1", {}).get("raw_records", [])
    c1_totals_s = [r["total_time_ms"] / 1000.0 for r in c1_raws if r.get("total_time_ms") is not None]
    if c1_totals_s:
        plot_data.append(c1_totals_s)
        labels.append("Single Call\n(C=1)")

    # Add highest concurrency single call
    highest_c = list(c_sweep.keys())[-1] if c_sweep else None
    if highest_c and highest_c != "c1":
        high_raws = c_sweep[highest_c].get("raw_records", [])
        high_totals_s = [r["total_time_ms"] / 1000.0 for r in high_raws if r.get("total_time_ms") is not None]
        if high_totals_s:
            plot_data.append(high_totals_s)
            c_num = c_sweep[highest_c].get("concurrency", highest_c)
            labels.append(f"Single Call\n(C={c_num})")

    # Add chain totals scaled to per-call average and total task
    if emp_totals:
        plot_data.append(emp_totals)
        labels.append(f"{chain_len}-Call Agent\nTask Total")

    if plot_data:
        try:
            bp = ax_d.boxplot(plot_data, patch_artist=True, tick_labels=labels)
        except TypeError:
            bp = ax_d.boxplot(plot_data, patch_artist=True)
            ax_d.set_xticks(range(1, len(labels) + 1))
            ax_d.set_xticklabels(labels)
        box_colors = [PALETTE["c1"], PALETTE["c20"], PALETTE["chain"]]
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.8)

        ax_d.set_title("D. Latency Spread Expansion (Single Calls vs. Chained Tasks)", fontsize=11, fontweight="bold")
        ax_d.set_ylabel("Duration (seconds)", fontsize=10)
        ax_d.grid(True, linestyle="--", alpha=0.5)
    else:
        ax_d.text(0.5, 0.5, "Insufficient Data for Boxplot", ha="center", va="center")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Publication dashboard plotted successfully: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input agent_chain_manifest.json")
    parser.add_argument("--out", type=Path, default=Path("docs/figures/agent_chain_dashboard.png"), help="Output image file")
    args = parser.parse_args()

    data = load_manifest(args.input)
    plot_dashboard(data, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
