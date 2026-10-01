#!/usr/bin/env python3
"""Generate publication-quality tail latency histogram and CDF plots.

Visualizes empirical TTFT and ITL distributions for Qwen3.8-27B MXFP4 on AMD Radeon AI PRO R9700
from raw detailed benchmark samples in docs/results/qwen3.8-27b-mxfp4/latency/r9700/.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_DIR / "docs" / "results" / "qwen3.8-27b-mxfp4" / "latency" / "r9700"
PROFILING_JSON = PROJECT_DIR / "docs" / "profiling" / "qwen3.8-27b-mxfp4-latency.json"
OUTPUT_DIR = PROJECT_DIR / "docs" / "figures" / "latency"
ARTIFACT_DIR_ENV = os.environ.get("ARTIFACT_DIR")
ARTIFACT_DIR = Path(ARTIFACT_DIR_ENV) if ARTIFACT_DIR_ENV else None

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if ARTIFACT_DIR:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Styling palette
COLORS = {
    "C1": "#1E88E5",    # Blue
    "C2": "#00897B",    # Teal
    "C4": "#43A047",    # Green
    "C8": "#FB8C00",    # Orange
    "C16": "#E53935",   # Red
}


def load_raw_samples(workload: str) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    workload_dir = RESULTS_ROOT / workload
    if not workload_dir.is_dir():
        return data

    for path in sorted(workload_dir.glob("*.json.gz")):
        stem = path.stem.replace(".json", "")
        # Filename format: qwen_isl{isl}_osl{osl}_c{C}_n{N}_{stamp}
        parts = stem.split("_")
        c_part = next((p for p in parts if p.startswith("c") and p[1:].isdigit()), None)
        if not c_part:
            continue
        c_label = f"C{c_part[1:]}"
        with gzip.open(path, "rt", encoding="utf-8") as f:
            doc = json.load(f)
        requests = doc.get("requests", [])
        ttfts = [r["ttft_ms"] for r in requests if r.get("ttft_ms") is not None]
        tpots = [r["tpot_ms"] for r in requests if r.get("tpot_ms") is not None]
        e2els = [r["e2el_ms"] for r in requests if r.get("e2el_ms") is not None]
        itls = [v for r in requests for v in r.get("itl_ms", [])]
        data[c_label] = {
            "concurrency": int(c_part[1:]),
            "ttfts": ttfts,
            "tpots": tpots,
            "e2els": e2els,
            "itls": itls,
        }
    return dict(sorted(data.items(), key=lambda item: item[1]["concurrency"]))


def plot_ttft_histogram(data_1k: dict[str, Any], data_8k: dict[str, Any]) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

    # Panel A: 1k64 TTFT Distributions
    # Log scale binning for TTFT from 100 ms to 30,000 ms
    bins_1k = np.logspace(np.log10(100), np.log10(30000), 40)
    for c_label, d in data_1k.items():
        if not d["ttfts"]:
            continue
        ax1.hist(
            d["ttfts"],
            bins=bins_1k,
            histtype="step",
            linewidth=2.2,
            label=f"{c_label} (p50: {np.median(d['ttfts']):.0f} ms, p99: {np.percentile(d['ttfts'], 99):.0f} ms)",
            color=COLORS.get(c_label, "#555555"),
        )
    ax1.set_xscale("log")
    ax1.axvline(3000, color="#D32F2F", linestyle="--", linewidth=2.0, label="3.0s Interactive SLO Limit")
    ax1.axvspan(100, 3000, color="#E8F5E9", alpha=0.3, label="SLO Compliant Zone (≤ 3.0s)")
    ax1.axvspan(3000, 30000, color="#FFEBEE", alpha=0.3, label="Cut-Off Zone (> 3.0s)")
    ax1.set_title("A. 1,024 Input Prompt TTFT Distribution (1k:64)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Time-to-First-Token (TTFT, ms) [Log Scale]", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Request Count (N = 100 per concurrency)", fontsize=11, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax1.set_xlim(100, 30000)

    # Panel B: 8k64 TTFT Distributions
    bins_8k = np.logspace(np.log10(1000), np.log10(80000), 40)
    for c_label, d in data_8k.items():
        if not d["ttfts"]:
            continue
        ax2.hist(
            d["ttfts"],
            bins=bins_8k,
            histtype="step",
            linewidth=2.2,
            label=f"{c_label} (p50: {np.median(d['ttfts']):.0f} ms, p99: {np.percentile(d['ttfts'], 99):.0f} ms)",
            color=COLORS.get(c_label, "#555555"),
        )
    ax2.set_xscale("log")
    ax2.axvline(3000, color="#D32F2F", linestyle="--", linewidth=2.0, label="3.0s Interactive SLO Limit")
    ax2.axvspan(1000, 3000, color="#E8F5E9", alpha=0.3, label="SLO Compliant Zone (≤ 3.0s)")
    ax2.axvspan(3000, 80000, color="#FFEBEE", alpha=0.3, label="Cut-Off Zone (> 3.0s)")
    ax2.set_title("B. 8,192 Input Prompt TTFT Distribution (8k:64)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Time-to-First-Token (TTFT, ms) [Log Scale]", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Request Count (N = 100 per concurrency)", fontsize=11, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax2.set_xlim(1000, 80000)

    fig.suptitle(
        "TTFT Tail Latency Histograms: Shift from Interactive Service to Scheduler Queueing\n"
        "AMD Radeon AI PRO R9700 (32 GB GDDR6) — Qwen3.8-27B MXFP4",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    out_file = OUTPUT_DIR / "01_ttft_tail_histogram.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    return str(out_file)


def plot_itl_histogram(data_8k: dict[str, Any]) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

    # Panel A: ITL Histogram showing the 30ms decode mode vs prefill stalls (>1,000 ms)
    # Filter out initial token-0 delta (<1ms)
    bins_itl = np.linspace(15, 1600, 60)
    for c_label in ["C1", "C2", "C4", "C16"]:
        if c_label not in data_8k:
            continue
        d = data_8k[c_label]
        valid_itls = [v for v in d["itls"] if v >= 1.0]
        ax1.hist(
            valid_itls,
            bins=bins_itl,
            histtype="step",
            linewidth=2.2,
            label=f"{c_label} (p50: {np.median(valid_itls):.1f} ms, max: {max(valid_itls):.0f} ms)",
            color=COLORS.get(c_label, "#555555"),
            density=True,
        )

    ax1.set_yscale("log")
    ax1.axvline(20, color="#388E3C", linestyle=":", linewidth=1.8, label="20 ms Streaming SLO Target")
    ax1.axvline(50, color="#FFA000", linestyle=":", linewidth=1.8, label="50 ms Warning Threshold")
    ax1.axvline(100, color="#D32F2F", linestyle="--", linewidth=2.0, label="100 ms Severe Stall Ceiling")

    ax1.annotate(
        "Isolated Nominal Decode\n~30-33 ms cadence",
        xy=(32, 0.05),
        xytext=(150, 0.02),
        arrowprops=dict(arrowstyle="->", color="#1E88E5", lw=1.5),
        fontweight="bold",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD", edgecolor="#1E88E5", alpha=0.9),
    )

    ax1.annotate(
        "Prefill Interference Freezes\n1,180 - 1,550 ms stall spikes\n(under C4-C16 concurrent 8k bursts)",
        xy=(1250, 0.0004),
        xytext=(750, 0.0012),
        arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.5),
        fontweight="bold",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE", edgecolor="#E53935", alpha=0.9),
    )

    ax1.set_title("A. Inter-Token Latency (ITL) Density & Tail Spikes", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Inter-Token Latency (ms)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Probability Density [Log Scale]", fontsize=11, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax1.set_xlim(15, 1600)

    # Panel B: Cumulative Distribution Function (CDF)
    for c_label in ["C1", "C2", "C4", "C8", "C16"]:
        if c_label not in data_8k:
            continue
        d = data_8k[c_label]
        valid_itls = sorted([v for v in d["itls"] if v >= 1.0])
        y_vals = np.linspace(0, 100, len(valid_itls))
        ax2.plot(
            valid_itls,
            y_vals,
            linewidth=2.2,
            label=f"{c_label} (p95: {np.percentile(valid_itls, 95):.0f} ms, p99: {np.percentile(valid_itls, 99):.0f} ms)",
            color=COLORS.get(c_label, "#555555"),
        )

    ax2.set_xscale("log")
    ax2.axvline(20, color="#388E3C", linestyle=":", linewidth=1.8, label="20 ms Streaming SLO")
    ax2.axvline(50, color="#FFA000", linestyle=":", linewidth=1.8, label="50 ms Streaming Limit")
    ax2.axvline(100, color="#D32F2F", linestyle="--", linewidth=2.0, label="100 ms Ceiling")
    ax2.axhline(95, color="#757575", linestyle="--", alpha=0.7)
    ax2.axhline(99, color="#757575", linestyle="--", alpha=0.7)
    ax2.text(22, 95.5, "p95", fontsize=8.5, fontweight="bold", color="#424242")
    ax2.text(22, 99.5, "p99", fontsize=8.5, fontweight="bold", color="#424242")

    ax2.set_title("B. ITL Empirical Cumulative Distribution (CDF)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Inter-Token Latency (ms) [Log Scale]", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Percentage (%)", fontsize=11, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax2.set_xlim(20, 2000)
    ax2.set_ylim(0, 102)

    fig.suptitle(
        "Inter-Token Latency (ITL) Tail Jitter & Token Freezing under 8k Prefill Contention\n"
        "AMD Radeon AI PRO R9700 (32 GB GDDR6) — Qwen3.8-27B MXFP4",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    out_file = OUTPUT_DIR / "02_itl_tail_histogram.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    return str(out_file)


def plot_master_tail_dashboard(data_1k: dict[str, Any], data_8k: dict[str, Any]) -> str:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # 1. 1k TTFT Histogram
    bins_1k = np.logspace(np.log10(100), np.log10(20000), 35)
    for c_label, d in data_1k.items():
        if d["ttfts"]:
            ax1.hist(
                d["ttfts"],
                bins=bins_1k,
                histtype="step",
                linewidth=2.0,
                label=f"{c_label} (p50: {np.median(d['ttfts']):.0f} ms)",
                color=COLORS.get(c_label, "#555"),
            )
    ax1.set_xscale("log")
    ax1.axvline(3000, color="#D32F2F", linestyle="--", linewidth=1.8, label="3.0s Interactive SLO")
    ax1.set_title("1. TTFT Distribution: 1,024 Input Prompt (1k:64)", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("TTFT (ms) [Log Scale]", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Requests (N=100)", fontsize=10, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8)

    # 2. 8k TTFT Histogram
    bins_8k = np.logspace(np.log10(1000), np.log10(70000), 35)
    for c_label, d in data_8k.items():
        if d["ttfts"]:
            ax2.hist(
                d["ttfts"],
                bins=bins_8k,
                histtype="step",
                linewidth=2.0,
                label=f"{c_label} (p50: {np.median(d['ttfts']):.0f} ms)",
                color=COLORS.get(c_label, "#555"),
            )
    ax2.set_xscale("log")
    ax2.axvline(3000, color="#D32F2F", linestyle="--", linewidth=1.8, label="3.0s Interactive SLO")
    ax2.set_title("2. TTFT Distribution: 8,192 Input Prompt (8k:64)", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("TTFT (ms) [Log Scale]", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Requests (N=100)", fontsize=10, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8)

    # 3. ITL Tail Histogram (8k)
    bins_itl = np.linspace(15, 1600, 50)
    for c_label in ["C1", "C2", "C4", "C16"]:
        if c_label in data_8k:
            valid_itls = [v for v in data_8k[c_label]["itls"] if v >= 1.0]
            ax3.hist(
                valid_itls,
                bins=bins_itl,
                histtype="step",
                linewidth=2.0,
                label=f"{c_label} (max: {max(valid_itls):.0f} ms)",
                color=COLORS.get(c_label, "#555"),
                density=True,
            )
    ax3.set_yscale("log")
    ax3.axvline(20, color="#388E3C", linestyle=":", linewidth=1.5, label="20 ms SLO")
    ax3.axvline(100, color="#D32F2F", linestyle="--", linewidth=1.5, label="100 ms Ceiling")
    ax3.set_title("3. ITL Tail Freezes under 8k Bursts", fontsize=11.5, fontweight="bold")
    ax3.set_xlabel("Inter-Token Latency (ms)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Density [Log Scale]", fontsize=10, fontweight="bold")
    ax3.grid(True, which="both", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8)

    # 4. Tail Metrics Summary Table
    ax4.axis("off")
    table_data = [
        ["Workload", "C", "TTFT p50", "TTFT p95", "TTFT p99", "ITL p50", "ITL p95", "ITL p99", "SLO ≤ 3s"],
        ["1k:64", "C1", "481 ms", "485 ms", "487 ms", "30.0 ms", "31.1 ms", "32.0 ms", "100% Pass"],
        ["1k:64", "C2", "864 ms", "919 ms", "921 ms", "31.0 ms", "32.2 ms", "35.4 ms", "100% Pass"],
        ["1k:64", "C4", "1,589 ms", "1,603 ms", "1,604 ms", "32.2 ms", "33.4 ms", "35.9 ms", "100% Pass"],
        ["1k:64", "C8", "5,159 ms", "5,183 ms", "5,185 ms", "32.2 ms", "33.5 ms", "75.0 ms", "Queue Fail"],
        ["1k:64", "C16", "12,383 ms", "12,440 ms", "12,449 ms", "32.2 ms", "33.6 ms", "76.7 ms", "Queue Fail"],
        ["8k:64", "C1", "3,152 ms", "3,169 ms", "3,171 ms", "30.5 ms", "31.7 ms", "32.6 ms", "Marginal"],
        ["8k:64", "C2", "5,611 ms", "6,155 ms", "6,158 ms", "31.9 ms", "33.2 ms", "328 ms", "Fail"],
        ["8k:64", "C4", "8,173 ms", "9,715 ms", "10,501 ms", "33.9 ms", "1,183 ms", "1,501 ms", "Contention"],
        ["8k:64", "C8", "20,585 ms", "21,745 ms", "24,944 ms", "33.9 ms", "1,267 ms", "1,509 ms", "Queue Fail"],
        ["8k:64", "C16", "49,733 ms", "50,913 ms", "54,016 ms", "33.9 ms", "1,266 ms", "1,508 ms", "Queue Fail"],
    ]
    t = ax4.table(cellText=table_data, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1.0, 1.6)

    # Style header row
    for col in range(len(table_data[0])):
        t[(0, col)].set_facecolor("#37474F")
        t[(0, col)].get_text().set_color("white")
        t[(0, col)].get_text().set_weight("bold")

    # Style pass / fail column
    for row in range(1, len(table_data)):
        status = table_data[row][-1]
        cell = t[(row, len(table_data[0]) - 1)]
        if "Pass" in status:
            cell.set_facecolor("#E8F5E9")
            cell.get_text().set_color("#2E7D32")
            cell.get_text().set_weight("bold")
        elif "Marginal" in status:
            cell.set_facecolor("#FFF9C4")
            cell.get_text().set_color("#F57F17")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#FFEBEE")
            cell.get_text().set_color("#C62828")
            cell.get_text().set_weight("bold")

    ax4.set_title("4. Empirical Tail Percentile Summary (N = 100 requests per cell)", fontsize=11.5, fontweight="bold", pad=10)

    fig.suptitle(
        "Tail Latency Master Evaluation Dashboard: TTFT & ITL Distribution Analysis\n"
        "AMD Radeon AI PRO R9700 (32 GB GDDR6) — Qwen3.8-27B MXFP4",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_file = OUTPUT_DIR / "03_tail_latency_master_dashboard.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    return str(out_file)


def main() -> int:
    print("Loading empirical raw latency samples...")
    data_1k = load_raw_samples("1k64")
    data_8k = load_raw_samples("8k64")

    print(f"Loaded 1k64 concurrencies: {list(data_1k.keys())}")
    print(f"Loaded 8k64 concurrencies: {list(data_8k.keys())}")

    p1 = plot_ttft_histogram(data_1k, data_8k)
    print(f"Generated: {p1}")

    p2 = plot_itl_histogram(data_8k)
    print(f"Generated: {p2}")

    p3 = plot_master_tail_dashboard(data_1k, data_8k)
    print(f"Generated: {p3}")

    if ARTIFACT_DIR:
        for p in [p1, p2, p3]:
            dest = ARTIFACT_DIR / Path(p).name
            shutil.copy(p, dest)
            print(f"Copied to artifact dir: {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
