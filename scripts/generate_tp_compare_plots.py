#!/usr/bin/env python3
"""Plot the controlled MI350P TP=1 versus TP=2 concurrency sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


CONCURRENCIES = (1, 2, 4, 8)
POSITIONS = tuple(range(len(CONCURRENCIES)))
COLORS = {"TP=1": "#2f78c4", "TP=2": "#d35f3f"}


def load_series(results_dir: Path, tp: int) -> dict[str, list[float]]:
    rows = []
    for concurrency in CONCURRENCIES:
        path = results_dir / f"tp{tp}_1k128_c{concurrency}.json"
        row = json.loads(path.read_text())
        if row["completed"] != row["num_prompts"] or row["failed"]:
            raise ValueError(f"incomplete benchmark: {path}")
        rows.append(row)
    return {
        "throughput": [row["output_throughput"] for row in rows],
        "ttft": [row["mean_ttft_ms"] for row in rows],
        "ttft_p99": [row["p99_ttft_ms"] for row in rows],
    }


def style_axis(ax, ylabel: str) -> None:
    ax.set_xlabel("Maximum request concurrency")
    ax.set_ylabel(ylabel)
    ax.set_xticks(POSITIONS, [str(value) for value in CONCURRENCIES])
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def annotate(ax, xs, ys, suffix: str = "") -> None:
    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y:,.1f}{suffix}",
            (x, y),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )


def draw_metric(ax, data, key: str, ylabel: str, title: str) -> None:
    for label in ("TP=1", "TP=2"):
        values = data[label][key]
        ax.plot(
            POSITIONS,
            values,
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=label,
            color=COLORS[label],
        )
        annotate(ax, POSITIONS, values)
    style_axis(ax, ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "docs" / "results" / "tp_compare" / "mi350p",
        help="Published TP=1 and TP=2 series under docs/results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "figures" / "tp",
    )
    args = parser.parse_args()

    data = {
        "TP=1": load_series(args.results_dir, 1),
        "TP=2": load_series(args.results_dir, 2),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    draw_metric(
        ax,
        data,
        "throughput",
        "Output throughput (tokens/s)",
        "Qwen3.8-27B MXFP4 — TP throughput over PCIe",
    )
    fig.text(
        0.01,
        0.01,
        "2× MI350P PCIe · 1,024 input / 128 output · 8 prompts · prefix cache off · 30 Sep 2026",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.output_dir / "tp-throughput.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    draw_metric(
        ax,
        data,
        "ttft",
        "Mean TTFT (ms)",
        "Qwen3.8-27B MXFP4 — TP time to first token",
    )
    fig.text(
        0.01,
        0.01,
        "2× MI350P PCIe · 1,024 input / 128 output · 8 prompts · prefix cache off · 30 Sep 2026",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.output_dir / "tp-ttft.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    draw_metric(
        axes[0],
        data,
        "throughput",
        "Output throughput (tokens/s)",
        "Output throughput",
    )
    draw_metric(
        axes[1],
        data,
        "ttft",
        "Mean TTFT (ms)",
        "Mean TTFT",
    )
    fig.suptitle(
        "TP=1 versus TP=2 on two cross-socket MI350P PCIe GPUs",
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Qwen3.8-27B Quark AWQ MXFP4 · 1,024 input / 128 output · 8 prompts · prefix cache off",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(args.output_dir / "tp-compare.png")
    plt.close(fig)

    summary = {
        "concurrency": list(CONCURRENCIES),
        "workload": {"input_tokens": 1024, "output_tokens": 128, "prompts": 8},
        "series": data,
    }
    (args.output_dir / "tp-compare.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
