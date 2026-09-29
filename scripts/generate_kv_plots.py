#!/usr/bin/env python3
"""Publish KV measurements to docs/results, then plot that copy."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from publish_results import PUBLISHED_ROOT, publish_and_summarize

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "kv")


def plot_kv_handoff(summary: dict) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    serving = summary["serving_8k1k"]
    waterfall = summary["waterfall_8k_cold"]
    fig, (ax_rate, ax_phase) = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=300)

    labels = [f"R{index}" for index in range(len(serving["requests"]))]
    rates = [row["tok_s"] for row in serving["requests"]]
    ax_rate.bar(labels, rates, color="#1A237E")
    ax_rate.axhline(
        serving["output_throughput_tok_s"],
        color="#C62828",
        linestyle="--",
        linewidth=1.6,
        label=f"Aggregate {serving['output_throughput_tok_s']:.2f} tok/s",
    )
    ax_rate.set_ylabel("Output tok/s")
    ax_rate.set_title("8K in / 1K out through HIP-IPC")
    ax_rate.legend(frameon=False)
    ax_rate.grid(axis="y", linestyle="--", alpha=0.6)

    phases = ["Prefill", "NIXL post", "NIXL transfer", "Client"]
    p50 = [
        waterfall["prefill_ms"]["p50"],
        waterfall["post_ms"]["p50"],
        waterfall["xfer_ms"]["p50"],
        waterfall["client_ms"]["p50"],
    ]
    ax_phase.bar(phases, p50, color=["#546E7A", "#00897B", "#E65100", "#1A237E"])
    ax_phase.set_ylabel("p50 ms")
    ax_phase.set_title("Cold 8K handoff, max_tokens=1")
    ax_phase.grid(axis="y", linestyle="--", alpha=0.6)

    fig.suptitle(
        f"MI350P 1P1D  ITL p50 {serving['itl_p50_ms']:.2f} ms"
        f"  TTFT p50 {serving['ttft_p50_ms']:.0f} ms",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    output = os.path.join(OUTPUT_DIR, "01_hip_ipc_8k1k.png")
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main() -> None:
    summary = publish_and_summarize()
    output = plot_kv_handoff(summary)
    print(f"Published measurements in {PUBLISHED_ROOT}")
    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
