#!/usr/bin/env python3
"""Generate publication-quality figures for the AgentX concurrency tail sweep.

Visualizes empirical TTFT, ITL, KV cache saturation, and queueing dynamics for
Qwen3.8-27B MXFP4 on AMD Radeon AI PRO R9700 using real Claude Code traces
from semianalysis_cc_traces_weka_062126 across concurrencies C = 1, 2, 4, 8, 16, 32.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import BLUE, GREEN, ORANGE, PURPLE, RED, TEAL  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_DIR / "docs" / "results" / "agentx"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "docs" / "figures" / "agentx"
RESULTS_DIR = DEFAULT_RESULTS_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
USE_R9700_CAPTIONS = True
CAMPAIGN_TITLE = (
    "AMD Radeon AI PRO R9700 (32 GB) — Qwen3.8-27B MXFP4 (max_model_len=65,536)"
)
DASHBOARD_TITLE = (
    "AMD Radeon AI PRO R9700 (32 GB GDDR6) — Qwen3.8-27B-Quark-AWQ-MXFP4"
)
ARTIFACT_DIR_ENV = os.environ.get("ARTIFACT_DIR")
ARTIFACT_DIR = Path(ARTIFACT_DIR_ENV) if ARTIFACT_DIR_ENV else None

# Styling palette
COLORS = {
    1: BLUE,
    2: TEAL,
    4: GREEN,
    8: ORANGE,
    16: PURPLE,
    32: RED,
}


def load_data() -> tuple[dict, list[dict]]:
    analysis_path = RESULTS_DIR / "analysis.json"
    samples_path = RESULTS_DIR / "request_samples.csv"
    if not analysis_path.is_file() or not samples_path.is_file():
        raise SystemExit(
            f"AgentX figures need {analysis_path} and {samples_path}. "
            "Publish the campaign under docs/results/agentx and run analyze.sh."
        )
    with analysis_path.open() as f:
        analysis = json.load(f)
    samples = []
    with samples_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            samples.append(r)
    return analysis, samples


def plot_ttft_histogram(samples: list[dict]) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

    by_c: dict[int, list[float]] = {c: [] for c in [1, 2, 4, 8, 16, 32]}
    for r in samples:
        c = int(r["concurrency"])
        if r["ttft_ms"]:
            by_c[c].append(float(r["ttft_ms"]))

    # Panel A: TTFT Histogram (Log Scale)
    bins = np.logspace(np.log10(100), np.log10(500000), 50)
    for c in [1, 2, 4, 8, 16, 32]:
        vals = by_c[c]
        if not vals:
            continue
        p50 = np.median(vals)
        p95 = np.percentile(vals, 95)
        ax1.hist(
            vals,
            bins=bins,
            histtype="step",
            linewidth=2.2,
            label=f"C={c} (p50: {p50/1000:.1f}s, p95: {p95/1000:.1f}s)",
            color=COLORS[c],
        )

    ax1.set_xscale("log")
    ax1.axvline(3000, color="#D32F2F", linestyle="--", linewidth=1.8, label="3.0s Interactive SLO")
    ax1.axvspan(100, 3000, color="#E8F5E9", alpha=0.3, label="Interactive Zone (≤ 3.0s)")
    ax1.axvspan(3000, 500000, color="#FFEBEE", alpha=0.25, label="Queueing Delay Zone (> 3.0s)")
    ax1.set_title("A. TTFT Empirical Distribution (Log-Histogram)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time-to-First-Token (TTFT, ms) [Log Scale]", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Request Count", fontsize=11, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax1.set_xlim(100, 500000)

    # Panel B: Cumulative Distribution Function (CDF)
    for c in [1, 2, 4, 8, 16, 32]:
        vals = np.sort(by_c[c])
        if len(vals) == 0:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax2.plot(vals, cdf * 100, label=f"C={c}", color=COLORS[c], linewidth=2.4)

    ax2.set_xscale("log")
    ax2.axvline(3000, color="#D32F2F", linestyle="--", linewidth=1.8, label="3.0s Interactive SLO")
    ax2.axhline(95, color="#757575", linestyle=":", linewidth=1.5, label="p95 Target")
    ax2.axhline(50, color="#BDBDBD", linestyle=":", linewidth=1.2, label="p50 Target")
    ax2.set_title("B. Empirical Cumulative Distribution Function (CDF)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time-to-First-Token (TTFT, ms) [Log Scale]", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Requests (%)", fontsize=11, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax2.set_xlim(100, 500000)
    ax2.set_ylim(0, 102)

    fig.suptitle(
        "AgentX Multi-Turn Tail Latency (semianalysis_cc_traces_weka_062126)\n"
        + CAMPAIGN_TITLE,
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    out_file = OUTPUT_DIR / "01_agentx_ttft_histogram.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    if ARTIFACT_DIR:
        shutil.copy(out_file, ARTIFACT_DIR / "01_agentx_ttft_histogram.png")
    return str(out_file)


def plot_master_dashboard(analysis: dict, samples: list[dict]) -> str:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    measured = [r for r in analysis["runs"] if r.get("ttft_ms")]
    concurrencies = [r["concurrency"] for r in measured]
    ttft_p50 = [r["ttft_ms"]["p50"] / 1000 for r in measured]
    ttft_p95 = [r["ttft_ms"]["p95"] / 1000 for r in measured]
    ttft_p99 = [r["ttft_ms"]["p99"] / 1000 for r in measured]

    itl_p50 = [r["request_avg_itl_ms"]["p50"] for r in measured]
    itl_p95 = [r["request_avg_itl_ms"]["p95"] for r in measured]

    out_tps = [r["output_token_throughput_per_second"] for r in measured]

    prefix_hit = [r["prefix_cache"]["hit_rate_percent"] for r in measured]
    kv_usage_avg = [r["server_load"]["kv_cache_usage_fraction"]["avg"] * 100 for r in measured]
    kv_usage_max = [r["server_load"]["kv_cache_usage_fraction"]["max"] * 100 for r in measured]

    running_reqs = [r["server_load"]["running_requests"]["avg"] for r in measured]
    waiting_reqs = [r["server_load"]["waiting_requests"]["avg"] for r in measured]

    x_labels = [str(c) for c in concurrencies]
    x = np.arange(len(concurrencies))

    # Panel 1: TTFT Scaling (p50, p95, p99)
    ax1.plot(x, ttft_p50, marker="o", linewidth=2.5, color="#1E88E5", label="TTFT p50 (Median)")
    ax1.plot(x, ttft_p95, marker="s", linewidth=2.5, color="#FB8C00", label="TTFT p95 (Tail)")
    ax1.plot(x, ttft_p99, marker="^", linewidth=2.5, color="#E53935", label="TTFT p99 (Extreme Tail)")
    ax1.axhline(3.0, color="#D32F2F", linestyle="--", linewidth=1.5, label="3.0s Interactive SLO Limit")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontweight="bold")
    ax1.set_yscale("log")
    ax1.set_title("A. Time-to-First-Token Scaling (TTFT p50, p95, p99)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Concurrency (C)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("TTFT (Seconds) [Log Scale]", fontsize=10, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8.5)

    if USE_R9700_CAPTIONS and len(ttft_p95) > 3:
        ax1.annotate(
            "Knee Point (C=8)\np50=1.7s, p95=51.8s",
            xy=(3, ttft_p95[3]),
            xytext=(2.2, 100),
            arrowprops=dict(facecolor="#D32F2F", shrink=0.08, width=1.5, headwidth=6),
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#FB8C00", lw=1.5),
        )

    # Panel 2: Output Throughput & ITL
    color_tps = "#2E7D32"
    ax2.bar(x - 0.18, out_tps, width=0.36, color=color_tps, alpha=0.85, label="Output Throughput (tok/s)")
    ax2.set_ylabel("Throughput (tok/s)", color=color_tps, fontsize=10, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_tps)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontweight="bold")
    ax2.set_xlabel("Concurrency (C)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)

    ax2_twin = ax2.twinx()
    color_itl = "#6A1B9A"
    ax2_twin.plot(x + 0.18, itl_p50, marker="o", color=color_itl, linewidth=2.2, label="Avg ITL p50 (ms)")
    ax2_twin.plot(x + 0.18, itl_p95, marker="^", color="#D81B60", linewidth=2.0, linestyle="--", label="Avg ITL p95 (ms)")
    ax2_twin.set_ylabel("Inter-Token Latency (ms)", color=color_itl, fontsize=10, fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor=color_itl)
    ax2.set_title("B. Output Throughput & Inter-Token Latency", fontsize=11, fontweight="bold")

    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_2t, labels_2t = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_2 + lines_2t, labels_2 + labels_2t, loc="upper right", fontsize=8.5)

    # Panel 3: Prefix Cache Hit Rate vs KV Saturation
    ax3.plot(x, prefix_hit, marker="o", color="#00897B", linewidth=2.5, label="Prefix Cache Hit Rate (%)")
    ax3.plot(x, kv_usage_avg, marker="s", color="#FFA000", linewidth=2.2, label="KV Cache Usage Avg (%)")
    ax3.plot(x, kv_usage_max, marker="x", color="#E64A19", linewidth=1.8, linestyle="--", label="KV Cache Usage Max (%)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels, fontweight="bold")
    ax3.set_xlabel("Concurrency (C)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Percentage (%)", fontsize=10, fontweight="bold")
    ax3.set_ylim(-2, 105)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.set_title("C. Prefix Caching Efficiency vs KV Memory Pressure", fontsize=11, fontweight="bold")
    ax3.legend(loc="lower left", fontsize=8.5)

    if USE_R9700_CAPTIONS and len(prefix_hit) > 4:
        ax3.annotate(
            "KV Thrashing & Eviction\nHit Rate collapses 65% -> 9% -> 0%",
            xy=(4, prefix_hit[4]),
            xytext=(3.2, 35),
            arrowprops=dict(facecolor="#D32F2F", shrink=0.08, width=1.5, headwidth=6),
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#D32F2F", lw=1.5),
        )

    # Panel 4: Running vs Waiting Requests (Queue Dynamics)
    width = 0.38
    b1 = ax4.bar(x - width/2, running_reqs, width, label="Avg Running Requests (In GPU Engine)", color="#1976D2")
    b2 = ax4.bar(x + width/2, waiting_reqs, width, label="Avg Waiting Requests (Queue Delay)", color="#D32F2F")
    ax4.set_xticks(x)
    ax4.set_xticklabels(x_labels, fontweight="bold")
    ax4.set_xlabel("Concurrency (C)", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Request Count", fontsize=10, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.set_title("D. Engine Concurrency vs Scheduler Waiting Queue", fontsize=11, fontweight="bold")
    ax4.legend(loc="upper left", fontsize=8.5)

    for bar in b1:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, yval + 0.15, f"{yval:.1f}", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        yval = bar.get_height()
        if yval > 0.05:
            ax4.text(bar.get_x() + bar.get_width()/2, yval + 0.15, f"{yval:.1f}", ha="center", va="bottom", fontsize=8, color="#D32F2F", fontweight="bold")

    fig.suptitle(
        "AgentX Multi-Turn Concurrency & Tail Latency Master Dashboard\n"
        + DASHBOARD_TITLE,
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    out_file = OUTPUT_DIR / "02_agentx_master_dashboard.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    if ARTIFACT_DIR:
        shutil.copy(out_file, ARTIFACT_DIR / "02_agentx_master_dashboard.png")
    return str(out_file)


def main() -> int:
    import argparse

    global RESULTS_DIR, OUTPUT_DIR, USE_R9700_CAPTIONS, CAMPAIGN_TITLE, DASHBOARD_TITLE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    RESULTS_DIR = args.results_dir
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    USE_R9700_CAPTIONS = RESULTS_DIR.resolve() == DEFAULT_RESULTS_DIR.resolve()
    if not USE_R9700_CAPTIONS:
        manifest_path = RESULTS_DIR / "manifest.json"
        model = "AgentX"
        if manifest_path.is_file():
            model = json.loads(manifest_path.read_text()).get("model") or model
        CAMPAIGN_TITLE = f"{model} — {RESULTS_DIR.name}"
        DASHBOARD_TITLE = CAMPAIGN_TITLE
    if ARTIFACT_DIR:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    analysis, samples = load_data()
    f1 = plot_ttft_histogram(samples)
    print(f"Generated: {f1}")
    f2 = plot_master_dashboard(analysis, samples)
    print(f"Generated: {f2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
