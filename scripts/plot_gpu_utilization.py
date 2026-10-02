#!/usr/bin/env python3
"""Plot power, memory-bandwidth, and PCIe utilization for the latest published run.

Reads docs/profiling/gpu_metrics.json. That file is written when test.sh copies
a throughput run. The figure is the latest run only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import BLUE, ORANGE, TEAL  # noqa: E402
from publish_suite_results import (  # noqa: E402
    GPU_METRICS_JSON,
    latest_gpu_rows,
    write_gpu_metrics_report,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "figures" / "utilization"


def _label(workload: str) -> str:
    if ":" not in workload:
        return workload
    incoming, outgoing = workload.split(":", 1)
    return f"{incoming} in\n{outgoing} out"


def _bars(ax, labels: list[str], values: list[float | None], color: str, ylabel: str, title: str) -> None:
    present = [0.0 if value is None else float(value) for value in values]
    ax.bar(labels, present, color=color)
    ceiling = max(present + [1.0])
    ax.set_ylim(0, max(ceiling * 1.25, 1.0))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    for index, value in enumerate(present):
        ax.text(index, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)


def plot_latest(rows: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [_label(str(row.get("workload") or "")) for row in rows]
    run = rows[0].get("run")
    profile = rows[0].get("gpu_profile") or "gpu"
    link = rows[0].get("pcie_link_gbs")
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8), dpi=300)
    _bars(
        axes[0],
        labels,
        [row.get("power_util_pct") for row in rows],
        BLUE,
        "% of TDP",
        "Socket power",
    )
    _bars(
        axes[1],
        labels,
        [row.get("memory_bw_util_pct") for row in rows],
        TEAL,
        "% of catalog peak",
        "Memory bandwidth",
    )
    _bars(
        axes[2],
        labels,
        [row.get("pcie_util_pct") for row in rows],
        ORANGE,
        f"% of {link} GB/s",
        "PCIe traffic",
    )
    fig.suptitle(
        f"{profile} utilization  {run}",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Memory GB/s is UMC percent times catalog peak, not a calibrated HBM counter. "
        "PCIe percent is amd-smi traffic against the PCIe 5.0 x16 payload ceiling.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    output = OUTPUT_DIR / "01_suite_utilization.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main() -> int:
    if not GPU_METRICS_JSON.is_file():
        print(f"No published GPU metrics at {GPU_METRICS_JSON}", file=sys.stderr)
        return 1
    document = json.loads(GPU_METRICS_JSON.read_text())
    report = write_gpu_metrics_report(document)
    rows = latest_gpu_rows(document)
    if not rows:
        print(f"No utilization rows in {GPU_METRICS_JSON}", file=sys.stderr)
        return 1
    figure = plot_latest(rows)
    print(f"Utilization report: {report.relative_to(ROOT)}")
    print(f"Generated: {figure.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
