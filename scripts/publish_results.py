#!/usr/bin/env python3
"""Copy the measurement records a plot needs into docs/results.

Benchmarks write under _results. That tree is local scratch. The files
below are the records post-processing reads, so they are copied to
docs/results before a plot script runs. A missing source is left alone
when the published copy is already present.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "_results"
PUBLISHED_ROOT = ROOT / "docs" / "results"

# Source path relative to _results, published path relative to docs/results.
MEASUREMENTS = (
    ("kv_xfer_mi350p/gates/hip_ipc_pd_gpu1p_gpu0d_8k1k.json", "kv_xfer/hip_ipc_pd_8k1k.json"),
    ("kv_xfer_mi350p/gates/correctness_pd.json", "kv_xfer/correctness_pd.json"),
    ("kv_xfer_mi350p/gates/correctness_single.json", "kv_xfer/correctness_single.json"),
    ("kv_xfer_mi350p/gates/hip_8k_tokens.json", "kv_xfer/hip_8k_tokens.json"),
    ("kv_xfer_mi350p/gates/single_8k_tokens.json", "kv_xfer/single_8k_tokens.json"),
    ("kv_xfer_mi350p/gates/hip_8k_cold_unique.json", "kv_xfer/hip_8k_cold_unique.json"),
    ("kv_xfer_mi350p/gates/hip_8k_concurrent_cold.json", "kv_xfer/hip_8k_concurrent_cold.json"),
    ("kv_xfer_mi350p/nixl_hip_ipc_regions.jsonl", "kv_xfer/nixl_hip_ipc_regions.jsonl"),
    (
        "phase_isolation_matched/dedicated_prefill/phase_profile_summary_20260929_130722.json",
        "kv_xfer/dedicated_prefill_8k.json",
    ),
    (
        "pd_emulator/pd_emulator_summary_20260929_042526.json",
        "pd/pd_emulator_summary_20260929_042526.json",
    ),
)


def publish_measurements() -> list[Path]:
    copied: list[Path] = []
    for source_rel, published_rel in MEASUREMENTS:
        source = SOURCE_ROOT / source_rel
        published = PUBLISHED_ROOT / published_rel
        if not source.is_file():
            if not published.is_file():
                raise FileNotFoundError(
                    f"missing measurement {source_rel}; no published copy at {published}"
                )
            continue
        published.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, published)
        copied.append(published)
    return copied


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        raise ValueError("empty sample")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _summary_stats(samples: list[float]) -> dict[str, float | int]:
    return {
        "n": len(samples),
        "p50": round(_percentile(samples, 50), 3),
        "p95": round(_percentile(samples, 95), 3),
        "p99": round(_percentile(samples, 99), 3),
    }


def _load(published_rel: str):
    path = PUBLISHED_ROOT / published_rel
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text())


def _first_diff(left: list[int] | None, right: list[int] | None) -> int | None:
    left = left or []
    right = right or []
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        if lhs != rhs:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _record(document: dict) -> dict:
    records = document.get("records") or []
    if not records:
        raise ValueError("measurement has no records")
    return records[0]


def build_kv_summary() -> dict:
    """Reduce the published copies to the fields the KV plots read."""
    serving = _load("kv_xfer/hip_ipc_pd_8k1k.json")
    requests = []
    for row in serving.get("rows") or []:
        if not row.get("ok"):
            continue
        itl_ms = [value * 1000 for value in row.get("itl_s") or []]
        requests.append(
            {
                "tok_s": round(row["completion_tokens"] / row["dt"], 2),
                "ttft_ms": round((row.get("ttft_s") or 0) * 1000, 1),
                "mean_itl_ms": round(sum(itl_ms) / len(itl_ms), 2) if itl_ms else None,
                "max_itl_ms": round(max(itl_ms), 1) if itl_ms else None,
            }
        )

    cold = _load("kv_xfer/hip_8k_cold_unique.json")
    waterfall = {
        "prefill_ms": _summary_stats([row["pd_metrics"]["prefill_ms"] for row in cold]),
        "post_ms": _summary_stats([row["nixl"]["post_ms"] for row in cold]),
        "xfer_ms": _summary_stats([row["nixl"]["xfer_ms"] for row in cold]),
        "client_ms": _summary_stats([row["elapsed_ms"] for row in cold]),
        "bytes": int(cold[0]["nixl"]["bytes_sum"]),
        "failed": int(sum(row["nixl"]["failed"] for row in cold)),
    }
    concurrent = _load("kv_xfer/hip_8k_concurrent_cold.json")
    batch = concurrent["batch"]

    short_pd = _record(_load("kv_xfer/correctness_pd.json"))
    short_single = _record(_load("kv_xfer/correctness_single.json"))
    long_pd = _record(_load("kv_xfer/hip_8k_tokens.json"))
    long_single = _record(_load("kv_xfer/single_8k_tokens.json"))

    prefill = _load("kv_xfer/dedicated_prefill_8k.json")
    prefill_8k = next(
        row for row in prefill["suites"]["prefill"] if row.get("input_tokens") == 8192
    )

    summary = {
        "serving_8k1k": {
            "output_throughput_tok_s": round(serving["output_throughput"], 2),
            "itl_p50_ms": serving["itl_p50_ms"],
            "itl_p95_ms": serving["itl_p95_ms"],
            "ttft_p50_ms": serving["ttft_p50_ms"],
            "ttft_p95_ms": serving["ttft_p95_ms"],
            "duration_s": round(serving["duration"], 2),
            "generated_tokens": serving["total_generated_tokens"],
            "requests": requests,
        },
        "waterfall_8k_cold": waterfall,
        "waterfall_8k_concurrent": {
            "client_ms": _summary_stats([row["client_ms"] for row in concurrent["records"]]),
            "prefill_ms": _summary_stats(
                [row["pd_metrics"]["prefill_ms"] for row in concurrent["records"]]
            ),
            "xfer_ms_sum": round(batch["xfer_ms"], 3),
            "bytes_sum": int(batch["bytes_sum"]),
            "failed": int(batch["failed"]),
        },
        "dedicated_prefill_8k_s": prefill_8k["engine_prefill_s"],
        "token_gate": {
            "short_prompt_equal": short_pd.get("prompt_token_ids") == short_single.get("prompt_token_ids"),
            "short_first_diff": _first_diff(short_pd.get("token_ids"), short_single.get("token_ids")),
            "long_prompt_equal": long_pd.get("prompt_token_ids") == long_single.get("prompt_token_ids"),
            "long_first_diff": _first_diff(long_pd.get("token_ids"), long_single.get("token_ids")),
        },
    }
    destination = PUBLISHED_ROOT / "kv_connector_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


POWER_BANDWIDTH_JSON = ROOT / "docs" / "profiling" / "power_bandwidth.json"
CONCURRENCY_DIR = SOURCE_ROOT / "priority_eval" / "tco_mi350p" / "concurrency_20260929"


def publish_power_bandwidth() -> Path:
    """Keep the socket-power and UMC summaries the utilization plots read.

    The per-sample JSONL traces stay in local scratch. Joules per token is
    the scalar already reduced onto each concurrency result.
    """
    if not CONCURRENCY_DIR.is_dir():
        if not POWER_BANDWIDTH_JSON.is_file():
            raise FileNotFoundError(
                f"missing {CONCURRENCY_DIR}; no published copy at {POWER_BANDWIDTH_JSON}"
            )
        return POWER_BANDWIDTH_JSON

    runs = []
    for power_path in sorted(CONCURRENCY_DIR.glob("*_power.json")):
        stem = power_path.name[: -len("_power.json")]
        workload, separator, concurrency = stem.rpartition("_c")
        if not separator:
            continue
        power = json.loads(power_path.read_text())
        bench_path = CONCURRENCY_DIR / f"{stem}.json"
        joules = None
        if bench_path.is_file():
            bench = json.loads(bench_path.read_text())
            joules = bench.get("joules_per_token")
        runs.append(
            {
                "workload": workload,
                "concurrency": int(concurrency),
                "power_util_pct": power["power_util_pct"],
                "avg_power_w": power["avg_power_w"],
                "umc_activity_pct": power["umc_activity_pct_mean"],
                "umc_gbs_estimate": power.get("umc_gbs_estimate"),
                "device_tdp_w": power["device_tdp_w"],
                "device_peak_bw_gbs": power["device_peak_bw_gbs"],
                "joules_per_token": joules,
            }
        )
    runs.sort(key=lambda row: (row["workload"], row["concurrency"]))
    POWER_BANDWIDTH_JSON.parent.mkdir(parents=True, exist_ok=True)
    POWER_BANDWIDTH_JSON.write_text(
        json.dumps(
            {
                "gpu_profile": "mi350p",
                "note": (
                    "Socket power and UMC activity for the power and bandwidth plots. "
                    "UMC GB/s is UMC percent times the catalog peak and is not a calibrated HBM counter. "
                    "Sample traces are not published."
                ),
                "runs": runs,
            },
            indent=2,
        )
        + "\n"
    )
    return POWER_BANDWIDTH_JSON


def publish_and_summarize() -> dict:
    publish_measurements()
    publish_power_bandwidth()
    return build_kv_summary()
