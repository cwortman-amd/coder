#!/usr/bin/env python3
"""Copy the suite that just finished into docs/results.

test.sh records a start time and calls this when it exits. Accuracy reports,
throughput runs, and experiment directories newer than that time are copied.
Plot scripts read this tree and do not copy scratch themselves.

Qwen latency samples are already written under
docs/results/qwen3.8-27b-mxfp4/latency/ and docs/profiling/ while the TTFT
grid runs. This script publishes the rest of the suite.

Throughput power files carry socket power, the UMC memory-bandwidth estimate,
and amd-smi PCIe bandwidth. Those summaries are reduced into
docs/profiling/gpu_metrics.json. The 0.25 s sample traces stay in _results.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "_results"
PUBLISHED = ROOT / "docs" / "results"
PROFILING = ROOT / "docs" / "profiling"
GPU_METRICS_JSON = PROFILING / "gpu_metrics.json"
GPU_METRICS_MD = PROFILING / "gpu_metrics.md"
AGENTX_KIND = "agentx_concurrency_tail_pilot"
GPU_METRIC_FIELDS = (
    "gpu_profile",
    "gpu_index",
    "duration_s",
    "sample_count",
    "sample_interval_s",
    "device_tdp_w",
    "device_peak_bw_gbs",
    "avg_power_w",
    "max_power_w",
    "power_util_pct",
    "max_power_util_pct",
    "total_energy_joules",
    "gfx_activity_pct_mean",
    "umc_activity_pct_mean",
    "umc_gbs_estimate",
    "pcie_mbps_mean",
    "pcie_mbps_peak",
    "vram_used_mb_mean",
)
GPU_METRICS_NOTE = (
    "Socket power, UMC memory-bandwidth estimate, and amd-smi PCIe bandwidth "
    "from each published throughput run. UMC GB/s is UMC percent times catalog "
    "peak and is not a calibrated HBM counter. PCIe MB/s is measured traffic "
    "(PCIE_BANDWIDTH), not the link peak. Sample traces are not published."
)


def touched_since(path: Path, since: float) -> bool:
    if path.stat().st_mtime >= since:
        return True
    if path.is_file():
        return False
    return any(child.stat().st_mtime >= since for child in path.rglob("*") if child.exists())


def has_files(path: Path) -> bool:
    return path.is_file() or any(child.is_file() for child in path.rglob("*"))


def publish_tree(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("*.pid", "*.begin", "*.samples.jsonl"),
    )
    return destination


def _workload(name: str) -> str:
    parts = name.removesuffix(".json").split("_")
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        return f"{parts[-2]}:{parts[-1]}"
    return name.removesuffix(".json")


def _power_summaries(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in run_dir.glob("power_*.json")
        if path.is_file() and ".samples." not in path.name and not path.name.endswith(".begin")
    )


def reduce_power_file(path: Path, run_name: str) -> dict | None:
    document = json.loads(path.read_text())
    if "pcie_mbps_mean" not in document and "avg_power_w" not in document:
        return None
    row = {field: document.get(field) for field in GPU_METRIC_FIELDS}
    row["run"] = run_name
    row["workload"] = _workload(path.name)
    row["source"] = str(path.relative_to(ROOT))
    return row


def publish_gpu_metrics(run_dirs: list[Path]) -> Path | None:
    """Fold power summaries from published run directories into docs/profiling."""
    incoming = []
    for run_dir in run_dirs:
        for path in _power_summaries(run_dir):
            row = reduce_power_file(path, run_dir.name)
            if row is not None:
                incoming.append(row)
    if not incoming and not GPU_METRICS_JSON.is_file():
        return None
    current = {"note": GPU_METRICS_NOTE, "runs": []}
    if GPU_METRICS_JSON.is_file():
        loaded = json.loads(GPU_METRICS_JSON.read_text())
        if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
            current = loaded
    by_key = {(row.get("run"), row.get("workload")): row for row in current.get("runs") or []}
    for row in incoming:
        by_key[(row["run"], row["workload"])] = row
    document = {
        "note": GPU_METRICS_NOTE,
        "runs": sorted(by_key.values(), key=lambda row: (str(row.get("run")), str(row.get("workload")))),
    }
    GPU_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    GPU_METRICS_JSON.write_text(json.dumps(document, indent=2) + "\n")
    write_gpu_metrics_report(document)
    return GPU_METRICS_JSON


def pcie_gen5_x16_gbs() -> float:
    """PCIe 5.0 x16 payload rate: 32 GT/s, 128b/130b encoding."""
    return 32e9 * (128 / 130) / 8 * 16 / 1e9


def with_utilization(row: dict) -> dict:
    """Add power, memory, and PCIe utilization percents to a measured row."""
    annotated = dict(row)
    annotated["memory_bw_util_pct"] = row.get("umc_activity_pct_mean")
    link_gbs = pcie_gen5_x16_gbs()
    annotated["pcie_link_gbs"] = round(link_gbs, 2)
    mean_mbps = row.get("pcie_mbps_mean")
    if isinstance(mean_mbps, (int, float)) and link_gbs:
        annotated["pcie_util_pct"] = round((float(mean_mbps) / 1000.0) / link_gbs * 100.0, 3)
    else:
        annotated["pcie_util_pct"] = None
    return annotated


def latest_gpu_rows(document: dict) -> list[dict]:
    runs = [with_utilization(row) for row in document.get("runs") or []]
    if not runs:
        return []
    latest = max(str(row.get("run") or "") for row in runs)
    selected = [row for row in runs if str(row.get("run") or "") == latest]
    return sorted(selected, key=lambda row: str(row.get("workload") or ""))


def write_gpu_metrics_report(document: dict | None = None) -> Path:
    if document is None:
        if not GPU_METRICS_JSON.is_file():
            raise FileNotFoundError(f"missing {GPU_METRICS_JSON}")
        document = json.loads(GPU_METRICS_JSON.read_text())
    lines = [
        "# GPU metrics",
        "",
        document.get("note") or GPU_METRICS_NOTE,
        "",
        "| Run | Shape | Power % TDP | Memory % peak | PCIe % of Gen5 x16 | Power (W) | UMC GB/s | PCIe MB/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (with_utilization(item) for item in document.get("runs") or []):
        lines.append(
            "| {run} | {workload} | {power_util_pct} | {memory_bw_util_pct} | {pcie_util_pct} | "
            "{avg_power_w} | {umc_gbs_estimate} | {pcie_mbps_mean} |".format(
                run=row.get("run") or "",
                workload=row.get("workload") or "",
                power_util_pct=row.get("power_util_pct"),
                memory_bw_util_pct=row.get("memory_bw_util_pct"),
                pcie_util_pct=row.get("pcie_util_pct"),
                avg_power_w=row.get("avg_power_w"),
                umc_gbs_estimate=row.get("umc_gbs_estimate"),
                pcie_mbps_mean=row.get("pcie_mbps_mean"),
            )
        )
    lines.append("")
    GPU_METRICS_MD.parent.mkdir(parents=True, exist_ok=True)
    GPU_METRICS_MD.write_text("\n".join(lines))
    return GPU_METRICS_MD


def retarget_agentx_manifests(published: Path) -> None:
    """Point campaign manifests at the exports copied beside them."""
    for manifest_path in published.rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("kind") != AGENTX_KIND:
            continue
        campaign = manifest_path.parent
        changed = False
        for run in manifest.get("runs") or []:
            local = campaign / f"c{run.get('concurrency')}"
            if (local / "profile_export.jsonl").is_file():
                run["artifact_dir"] = str(local.relative_to(ROOT))
                changed = True
        if changed:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def publish_since(since: float) -> list[Path]:
    copied: list[Path] = []
    profiled: list[Path] = []
    accuracy = SOURCE
    if accuracy.is_dir():
        for report in sorted(accuracy.glob("accuracy_summary_*.md")):
            if touched_since(report, since):
                destination = PUBLISHED / "accuracy" / report.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(report, destination)
                copied.append(destination)

    throughput = SOURCE / "throughput"
    if throughput.is_dir():
        for run_dir in sorted(path for path in throughput.iterdir() if path.is_dir()):
            if touched_since(run_dir, since) and has_files(run_dir):
                destination = publish_tree(run_dir, PUBLISHED / "throughput" / run_dir.name)
                copied.append(destination)
                profiled.append(destination)

    experiments = SOURCE / "experiments"
    if experiments.is_dir():
        for run_dir in sorted(path for path in experiments.iterdir() if path.is_dir()):
            if touched_since(run_dir, since) and has_files(run_dir):
                destination = publish_tree(run_dir, PUBLISHED / "experiments" / run_dir.name)
                retarget_agentx_manifests(destination)
                copied.append(destination)
    if profiled:
        metrics = publish_gpu_metrics(profiled)
        if metrics is not None:
            copied.append(metrics)
            copied.append(GPU_METRICS_MD)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=float, help="Unix time the suite started")
    parser.add_argument(
        "--gpu-metrics-report",
        action="store_true",
        help="Rewrite docs/profiling/gpu_metrics.md from the published JSON",
    )
    args = parser.parse_args()
    if args.gpu_metrics_report:
        report = write_gpu_metrics_report()
        print(f"Wrote {report.relative_to(ROOT)}")
        return 0
    if args.since is None:
        parser.error("--since is required unless --gpu-metrics-report is set")
    copied = publish_since(args.since)
    if not copied:
        print("No new suite results to publish.")
        return 0
    for path in copied:
        print(f"Published {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
