#!/usr/bin/env python3
"""Scorecard for the MI350P expert-parallelism study.

Fills MI350P cells from vLLM ``bench serve`` JSON. The four NVIDIA columns
are the published burst in
https://forums.developer.nvidia.com/t/expert-parallelism-using-6000-pro-pcie-gen5-vs-b200-nvlink/378258
(16 prompts, 1000 in / 1000 out, request rate 10000, ignore_eos).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Published 27 Jul 2026 forum post. Transcribed, not re-measured.
NVIDIA = {
    "rtx6000_tp": {
        "label": "2× RTX PRO 6000, TP=2",
        "output_throughput": 516.44,
        "total_token_throughput": 1032.87,
        "duration": 30.98,
        "mean_ttft_ms": 18354.95,
        "median_ttft_ms": 18326.58,
        "p99_ttft_ms": 18486.20,
        "mean_tpot_ms": 12.63,
        "median_tpot_ms": 12.66,
        "p99_tpot_ms": 12.89,
        "mean_itl_ms": 12.63,
        "max_output_tokens_per_s": 1344.00,
    },
    "b200_tp": {
        "label": "2× B200 NVLink, TP=2",
        "output_throughput": 1885.26,
        "total_token_throughput": 3770.52,
        "duration": 8.49,
        "mean_ttft_ms": 1965.40,
        "median_ttft_ms": 1935.22,
        "p99_ttft_ms": 2069.94,
        "mean_tpot_ms": 6.52,
        "median_tpot_ms": 6.55,
        "p99_tpot_ms": 6.89,
        "mean_itl_ms": 6.53,
        "max_output_tokens_per_s": 2544.00,
    },
    "rtx6000_ep": {
        "label": "2× RTX PRO 6000, TP=2+EP",
        "output_throughput": 1186.12,
        "total_token_throughput": 2372.23,
        "duration": 13.49,
        "mean_ttft_ms": 672.99,
        "median_ttft_ms": 645.70,
        "p99_ttft_ms": 801.48,
        "mean_tpot_ms": 12.82,
        "median_tpot_ms": 12.85,
        "p99_tpot_ms": 13.08,
        "mean_itl_ms": 12.82,
        "max_output_tokens_per_s": 1328.00,
    },
    "b200_ep": {
        "label": "2× B200 NVLink, TP=2+EP",
        "output_throughput": 1965.18,
        "total_token_throughput": 3930.36,
        "duration": 8.14,
        "mean_ttft_ms": 2510.53,
        "median_ttft_ms": 2462.16,
        "p99_ttft_ms": 2661.18,
        "mean_tpot_ms": 5.63,
        "median_tpot_ms": 5.68,
        "p99_tpot_ms": 6.16,
        "mean_itl_ms": 5.63,
        "max_output_tokens_per_s": 2960.00,
    },
}

COLUMNS = (
    ("rtx6000_tp", "6000 TP"),
    ("rtx6000_ep", "6000 EP"),
    ("b200_tp", "B200 TP"),
    ("b200_ep", "B200 EP"),
    ("mi350p_tp", "MI350P TP"),
    ("mi350p_ep", "MI350P EP"),
)

ROWS = (
    ("output_throughput", "Output tok/s", ".2f"),
    ("total_token_throughput", "Total tok/s", ".2f"),
    ("duration", "Duration (s)", ".2f"),
    ("mean_ttft_ms", "Mean TTFT (ms)", ".2f"),
    ("median_ttft_ms", "Median TTFT (ms)", ".2f"),
    ("p99_ttft_ms", "P99 TTFT (ms)", ".2f"),
    ("mean_tpot_ms", "Mean TPOT (ms)", ".2f"),
    ("median_tpot_ms", "Median TPOT (ms)", ".2f"),
    ("p99_tpot_ms", "P99 TPOT (ms)", ".2f"),
    ("mean_itl_ms", "Mean ITL (ms)", ".2f"),
    ("max_output_tokens_per_s", "Peak output tok/s (window)", ".2f"),
)

# Later runs keep tp.json / ep.json as the eager pair that still built the
# vision tower. These filenames are the arms the runner writes now.
EXTRA_PAIRS = (
    ("text_tp", "text_ep", "Eager, language model only"),
    ("graphs_tp", "graphs_ep", "Graphs, vision tower loaded"),
    ("graphs_text_tp", "graphs_text_ep", "Graphs, language model only"),
)


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if data.get("failed", 0) not in (0, None) or data.get("completed") not in (16, None):
        data["_note"] = (
            f"completed={data.get('completed')} failed={data.get('failed')}"
        )
    return data


def _cell(row: dict | None, key: str, fmt: str) -> str:
    if not row or row.get(key) is None:
        return "—"
    return format(float(row[key]), fmt)


def _ratio(num: dict | None, den: dict | None, key: str) -> str:
    if not num or not den or not den.get(key) or num.get(key) is None:
        return "—"
    return f"{float(num[key]) / float(den[key]):.2f}×"


def render_mxfp4(results: Path) -> str:
    """MXFP4 kernel arms. Empty when those JSON files are absent."""
    specs = (
        ("mxfp4_triton_tp", "Triton TP"),
        ("mxfp4_triton_ep", "Triton EP"),
        ("mxfp4_aiter_tp", "AITER TP"),
        ("mxfp4_aiter_ep", "AITER EP"),
        ("mxfp4_triton_text_tp", "Triton text TP"),
        ("mxfp4_triton_text_ep", "Triton text EP"),
        ("mxfp4_aiter_text_tp", "AITER text TP"),
        ("mxfp4_aiter_text_ep", "AITER text EP"),
        ("mxfp4_triton_graphs_text_tp", "Triton graphs TP"),
        ("mxfp4_triton_graphs_text_ep", "Triton graphs EP"),
        ("mxfp4_aiter_graphs_text_tp", "AITER graphs TP"),
        ("mxfp4_aiter_graphs_text_ep", "AITER graphs EP"),
    )
    specs = tuple((name, label) for name, label in specs if (results / f"{name}.json").is_file())
    if not specs:
        return ""
    loaded = {name: _load(results / f"{name}.json") for name, _ in specs}
    lines = [
        "## MXFP4 kernels",
        "",
        "Checkpoint `amd/Qwen3.5-35B-A3B-MXFP4`. Same 16×1000/1000 burst. Triton is `--moe-backend emulation` (OCP MX Triton experts) with `VLLM_ROCM_USE_AITER=0`. `--moe-backend triton` rejects this checkpoint's MXFP4 activations. AITER is `--moe-backend aiter_mxfp4_mxfp4` with `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_MOE=1`, and unified attention off. A `text` column passed `--language-model-only`. A `graphs` column dropped `--enforce-eager`.",
        "",
        "| Metric | " + " | ".join(label for _, label in specs) + " |",
        "|---| " + " | ".join("---:" for _ in specs) + " |",
    ]
    for key, label, fmt in ROWS:
        cells = [_cell(loaded[name], key, fmt) for name, _ in specs]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    bf16_tp = _load(results / "tp.json")
    triton_tp = loaded.get("mxfp4_triton_text_tp") or loaded.get("mxfp4_triton_tp")
    aiter_tp = loaded.get("mxfp4_aiter_text_tp") or loaded.get("mxfp4_aiter_tp")
    lines.append(
        "MXFP4 Triton TP / BF16 eager TP output: "
        + _ratio(triton_tp, bf16_tp, "output_throughput")
        + ". MXFP4 AITER TP / BF16 eager TP output: "
        + _ratio(aiter_tp, bf16_tp, "output_throughput")
        + "."
    )
    lines.append("")
    return "\n".join(lines)


def _gib(nbytes: float | None) -> str:
    if nbytes is None:
        return "—"
    return f"{nbytes / (1024 ** 3):.1f} GiB"


def _gpu_rows(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    samples = [row for row in data.get("samples", []) if row.get("gpus")]
    if not samples:
        return None
    cards: dict[str, dict[str, list[float]]] = {}
    for row in samples:
        for card, info in row["gpus"].items():
            bucket = cards.setdefault(card, {"use": [], "vram": []})
            if info.get("use_pct") is not None:
                bucket["use"].append(float(info["use_pct"]))
            if info.get("vram_used_bytes") is not None:
                bucket["vram"].append(float(info["vram_used_bytes"]))
    lines = []
    for card in sorted(cards):
        use = cards[card]["use"]
        vram = cards[card]["vram"]
        use_cell = "—" if not use else f"{sum(use) / len(use):.0f}% / {max(use):.0f}%"
        vram_cell = "—" if not vram else _gib(sum(vram) / len(vram))
        lines.append(f"| `{path.name}` | {card} | {use_cell} | {vram_cell} | {len(samples)} |")
    return lines


def _extra_section(results: Path) -> list[str]:
    blocks = []
    for tp_name, ep_name, label in EXTRA_PAIRS:
        tp = _load(results / f"{tp_name}.json")
        ep = _load(results / f"{ep_name}.json")
        if tp is None and ep is None:
            continue
        blocks.append(f"| {label} | {tp_name}.json / {ep_name}.json | "
                      f"{_cell(tp, 'output_throughput', '.2f')} / {_cell(ep, 'output_throughput', '.2f')} | "
                      f"{_cell(tp, 'mean_ttft_ms', '.2f')} / {_cell(ep, 'mean_ttft_ms', '.2f')} | "
                      f"{_ratio(ep, tp, 'output_throughput')} | "
                      f"{_ratio(tp, ep, 'mean_ttft_ms')} |")
    if not blocks:
        return []
    return [
        "## Later MI350P arms",
        "",
        "The table above stays the eager pair in `tp.json` and `ep.json`. A graphed pair is the one to set next to the forum columns.",
        "",
        "| Arm | Files | Output tok/s TP / EP | Mean TTFT ms TP / EP | Output (EP/TP) | TTFT (TP/EP) |",
        "|---|---|---:|---:|---:|---:|",
        *blocks,
        "",
    ]


def _gpu_section(results: Path) -> list[str]:
    rows = []
    for path in sorted(results.glob("*.gpus.json")):
        found = _gpu_rows(path)
        if found:
            rows.extend(found)
    lines = [
        "## GPU use during the burst",
        "",
        "`rocm-smi` samples every 2 s while `vllm bench serve` runs. Use is mean / max of `GPU use (%)`. VRAM is the mean of used bytes.",
        "",
    ]
    if not rows:
        lines.append("The eager `tp.json` / `ep.json` pair has no `*.gpus.json`. Per-GPU use during that burst was not recorded.")
        lines.append("")
        return lines
    lines += [
        "| File | GPU | Use mean / max | VRAM mean | Samples |",
        "|---|---|---:|---:|---:|",
        *rows,
        "",
    ]
    return lines


def render(results: Path) -> str:
    arms = {
        "mi350p_tp": _load(results / "tp.json"),
        "mi350p_ep": _load(results / "ep.json"),
    }
    arms.update(NVIDIA)
    topo_path = results / "topology.json"
    topo = json.loads(topo_path.read_text()) if topo_path.is_file() else None

    lines = [
        "# Qwen3.5-35B-A3B expert parallel, same burst",
        "",
        "Protocol match for the [NVIDIA developer forum post](https://forums.developer.nvidia.com/t/expert-parallelism-using-6000-pro-pcie-gen5-vs-b200-nvlink/378258): `Qwen/Qwen3.5-35B-A3B`, TP=2, `--gpu-memory-utilization 0.9`, `--max-model-len 32768`, then the same serve command with `--enable-expert-parallel`. Client: `vllm bench serve --dataset-name random --random-input-len 1000 --random-output-len 1000 --request-rate 10000 --num-prompts 16 --ignore-eos`.",
        "",
        "NVIDIA cells are that post. MI350P cells are `tp.json` and `ep.json`: `--enforce-eager`, and the vision tower still constructed (`--skip-mm-profiling` only). An em dash means that arm has not been saved yet. Peak output tok/s is the bench client's short window. Sustained rate is the Output tok/s row.",
        "",
        "| Metric | " + " | ".join(label for _, label in COLUMNS) + " |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label, fmt in ROWS:
        cells = [_cell(arms[name], key, fmt) for name, _ in COLUMNS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    tp = arms["mi350p_tp"]
    ep = arms["mi350p_ep"]
    lines += [
        "",
        "## EP / TP on the same platform",
        "",
        "| Platform | Output tok/s (EP/TP) | Mean TTFT (TP/EP) | Mean TPOT (EP/TP) |",
        "|---|---:|---:|---:|",
        "| RTX PRO 6000 PCIe | "
        + " | ".join(
            [
                _ratio(NVIDIA["rtx6000_ep"], NVIDIA["rtx6000_tp"], "output_throughput"),
                _ratio(NVIDIA["rtx6000_tp"], NVIDIA["rtx6000_ep"], "mean_ttft_ms"),
                _ratio(NVIDIA["rtx6000_ep"], NVIDIA["rtx6000_tp"], "mean_tpot_ms"),
            ]
        )
        + " |",
        "| B200 NVLink | "
        + " | ".join(
            [
                _ratio(NVIDIA["b200_ep"], NVIDIA["b200_tp"], "output_throughput"),
                _ratio(NVIDIA["b200_tp"], NVIDIA["b200_ep"], "mean_ttft_ms"),
                _ratio(NVIDIA["b200_ep"], NVIDIA["b200_tp"], "mean_tpot_ms"),
            ]
        )
        + " |",
        "| MI350P PCIe | "
        + " | ".join(
            [
                _ratio(ep, tp, "output_throughput"),
                _ratio(tp, ep, "mean_ttft_ms"),
                _ratio(ep, tp, "mean_tpot_ms"),
            ]
        )
        + " |",
        "",
        "A TTFT ratio above 1 means expert parallel shortened time to first token. The forum columns are graphed CUDA runs. Compare those ratios with a `graphs_*.json` pair when that pair exists. The MI350P row above is the eager baseline.",
        "",
    ]
    lines += _extra_section(results)
    lines += _gpu_section(results)
    notes = []
    for name, row in (("tp", tp), ("ep", ep)):
        if row and row.get("_note"):
            notes.append(f"- `{name}.json`: {row['_note']}")
    if notes:
        lines += ["## Arm checks", ""] + notes + [""]
    if topo:
        lines += ["## Link snapshot", "", "```json", json.dumps(topo, indent=2), "```", ""]
    extra = render_mxfp4(results)
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = render(args.results_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(args.out)


if __name__ == "__main__":
    main()
