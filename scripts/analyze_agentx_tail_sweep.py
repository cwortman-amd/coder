#!/usr/bin/env python3
"""Summarize request and token tails from an AgentX concurrency campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PERCENTILES = (50, 90, 95, 99)
OUTPUT_BINS = (
    ("1-64", 1, 64),
    ("65-256", 65, 256),
    ("257-1024", 257, 1024),
    (">1024", 1025, math.inf),
)
SESSION_RE = re.compile(
    r"Phase profiling .*?complete .*?sessions: completed=(\d+), cancelled=(\d+)"
)
SENDING_RE = re.compile(
    r"Phase profiling .*?sending complete .*?sessions: sent=(\d+), completed=(\d+)"
)


def percentile(samples: list[float], percent: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(samples: Iterable[float]) -> dict[str, float | int] | None:
    values = [float(value) for value in samples if math.isfinite(float(value))]
    if not values:
        return None
    result: dict[str, float | int] = {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }
    for percent in PERCENTILES:
        result[f"p{percent}"] = round(percentile(values, percent), 3)
    return result


def metric_value(metrics: dict[str, Any], name: str) -> float | None:
    metric = metrics.get(name)
    if not isinstance(metric, dict):
        return None
    value = metric.get("value")
    return float(value) if isinstance(value, (int, float)) else None


def output_band(output_length: float | None) -> str:
    if output_length is None:
        return "unknown"
    for label, lower, upper in OUTPUT_BINS:
        if lower <= output_length <= upper:
            return label
    return "unknown"


def load_requests(path: Path, concurrency: int) -> list[dict[str, Any]]:
    requests = []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("metadata") or {}
            if metadata.get("benchmark_phase") != "profiling":
                continue
            metrics = record.get("metrics") or {}
            output_length = metric_value(metrics, "output_sequence_length")
            requests.append(
                {
                    "concurrency": concurrency,
                    "conversation_id": metadata.get("conversation_id"),
                    "root_correlation_id": metadata.get("root_correlation_id"),
                    "turn_index": metadata.get("turn_index"),
                    "source_kind": metadata.get("source_kind"),
                    "agent_depth": metadata.get("agent_depth"),
                    "input_length": metric_value(metrics, "input_sequence_length"),
                    "output_length": output_length,
                    "output_band": output_band(output_length),
                    "ttft_ms": metric_value(metrics, "time_to_first_token"),
                    "request_latency_ms": metric_value(metrics, "request_latency"),
                    "decode_duration_ms": metric_value(metrics, "decode_duration"),
                    "request_avg_itl_ms": metric_value(
                        metrics, "inter_token_latency"
                    ),
                }
            )
    return requests


def server_metric(document: dict[str, Any], name: str) -> dict[str, Any] | None:
    metric = (document.get("metrics") or {}).get(name)
    if not isinstance(metric, dict):
        return None
    series = metric.get("series") or []
    return series[0] if series else None


def counter_total(document: dict[str, Any], name: str) -> float | None:
    series = server_metric(document, name)
    if not series:
        return None
    total = (series.get("stats") or {}).get("total")
    return float(total) if isinstance(total, (int, float)) else None


def gauge_stats(document: dict[str, Any], name: str) -> dict[str, Any] | None:
    series = server_metric(document, name)
    return (series or {}).get("stats")


def token_gap_summary(document: dict[str, Any]) -> dict[str, Any] | None:
    series = server_metric(document, "vllm:inter_token_latency_seconds")
    if not series:
        return None
    stats = series.get("stats") or {}
    buckets = series.get("buckets") or {}
    total = int(stats.get("count") or 0)
    at_or_below_one = buckets.get("1.0")
    if isinstance(at_or_below_one, dict):
        at_or_below_one = at_or_below_one.get("count")
    above_one = (
        total - int(at_or_below_one)
        if total and isinstance(at_or_below_one, (int, float))
        else None
    )
    return {
        "intervals": total,
        "gaps_above_1s": above_one,
        "freeze_rate_per_1000": (
            round(above_one / total * 1000, 4)
            if above_one is not None and total
            else None
        ),
        "histogram_seconds_cumulative": buckets,
        "estimated_percentiles_seconds": {
            key: stats.get(f"{key}_estimate") for key in ("p50", "p90", "p95", "p99")
        },
    }


def session_counts(log_path: Path) -> dict[str, int | None]:
    if not log_path.is_file():
        return {
            "sent": None,
            "completed_at_send_stop": None,
            "completed": None,
            "cancelled": None,
        }
    text = log_path.read_text(errors="replace")
    sending = SENDING_RE.findall(text)
    complete = SESSION_RE.findall(text)
    return {
        "sent": int(sending[-1][0]) if sending else None,
        "completed_at_send_stop": int(sending[-1][1]) if sending else None,
        "completed": int(complete[-1][0]) if complete else None,
        "cancelled": int(complete[-1][1]) if complete else None,
    }


def metric_summary(requests: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    return summarize(
        request[key]
        for request in requests
        if isinstance(request.get(key), (int, float))
    )


def newest_campaign(root: Path) -> Path:
    manifests = sorted(
        root.glob("*/manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        raise FileNotFoundError(f"no campaign manifests under {root}")
    return manifests[0].parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", nargs="?", type=Path)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=ROOT / "_results" / "agentx_tail_sweep",
    )
    args = parser.parse_args()
    campaign_dir = args.campaign_dir or newest_campaign(args.campaign_root)
    manifest_path = campaign_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    all_requests: list[dict[str, Any]] = []
    summaries = []
    for run in manifest.get("runs", []):
        if run.get("status") != "completed":
            summaries.append(
                {
                    "concurrency": run.get("concurrency"),
                    "status": run.get("status"),
                }
            )
            continue
        concurrency = int(run["concurrency"])
        run_dir = Path(run["artifact_dir"])
        request_path = run_dir / "profile_export.jsonl"
        aggregate_path = run_dir / "profile_export_aiperf.json"
        server_path = run_dir / "server_metrics_export.json"
        missing = [
            str(path)
            for path in (request_path, aggregate_path, server_path)
            if not path.is_file()
        ]
        if missing:
            summaries.append(
                {
                    "concurrency": concurrency,
                    "status": "missing_artifacts",
                    "missing": missing,
                }
            )
            continue

        requests = load_requests(request_path, concurrency)
        all_requests.extend(requests)
        aggregate = json.loads(aggregate_path.read_text())
        server = json.loads(server_path.read_text())
        hits = counter_total(server, "vllm:prefix_cache_hits")
        queries = counter_total(server, "vllm:prefix_cache_queries")
        by_output = {}
        for label, _, _ in OUTPUT_BINS:
            matching = [
                request
                for request in requests
                if request["output_band"] == label
            ]
            by_output[label] = {
                "requests": len(matching),
                "request_latency_ms": metric_summary(
                    matching, "request_latency_ms"
                ),
                "ttft_ms": metric_summary(matching, "ttft_ms"),
            }

        summary = {
            "concurrency": concurrency,
            "status": "completed",
            "requests": len(requests),
            "distinct_source_conversations": len(
                {
                    request["conversation_id"]
                    for request in requests
                    if request["conversation_id"]
                }
            ),
            "sessions": session_counts(run_dir / "logs" / "aiperf.log"),
            "empirical_tail_resolution_percent": (
                round(100 / len(requests), 3) if requests else None
            ),
            "request_throughput_per_second": (
                aggregate.get("request_throughput") or {}
            ).get("avg"),
            "output_token_throughput_per_second": (
                aggregate.get("output_token_throughput") or {}
            ).get("avg"),
            "errors": len(aggregate.get("error_summary") or []),
            "submission_valid": (
                aggregate.get("metadata") or {}
            ).get("submission_valid"),
            "ttft_ms": metric_summary(requests, "ttft_ms"),
            "request_latency_ms": metric_summary(
                requests, "request_latency_ms"
            ),
            "decode_duration_ms": metric_summary(requests, "decode_duration_ms"),
            "request_avg_itl_ms": metric_summary(
                requests, "request_avg_itl_ms"
            ),
            "input_length_tokens": metric_summary(requests, "input_length"),
            "output_length_tokens": metric_summary(requests, "output_length"),
            "request_latency_by_output_length": by_output,
            "source_kind_counts": {
                kind: sum(request["source_kind"] == kind for request in requests)
                for kind in sorted(
                    {
                        request["source_kind"]
                        for request in requests
                        if request["source_kind"]
                    }
                )
            },
            "token_gaps": token_gap_summary(server),
            "prefix_cache": {
                "hits": hits,
                "queries": queries,
                "hit_rate_percent": (
                    round(hits / queries * 100, 3)
                    if hits is not None and queries
                    else None
                ),
            },
            "server_load": {
                "running_requests": gauge_stats(
                    server, "vllm:num_requests_running"
                ),
                "waiting_requests": gauge_stats(
                    server, "vllm:num_requests_waiting"
                ),
                "kv_cache_usage_fraction": gauge_stats(
                    server, "vllm:kv_cache_usage_perc"
                ),
            },
        }
        summaries.append(summary)

    csv_path = campaign_dir / "request_samples.csv"
    if all_requests:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_requests[0]))
            writer.writeheader()
            writer.writerows(all_requests)

    analysis = {
        "schema_version": 1,
        "campaign_manifest": str(manifest_path),
        "interpretation": {
            "kind": "bounded_concurrency_pilot",
            "campaign_budget_seconds": manifest.get("campaign_budget_seconds"),
            "profiling_duration_seconds": manifest.get(
                "profiling_duration_seconds"
            ),
            "stable_request_p99": False,
            "session_bootstrap_allowed": False,
            "request_p99_note": (
                "p99 is retained as an observed order statistic; use the per-cell "
                "empirical_tail_resolution_percent before interpreting it."
            ),
            "token_gap_note": (
                "vLLM token-gap histograms are population-level and cannot assign "
                "a >1s gap to a request or conversation."
            ),
        },
        "runs": summaries,
    }
    analysis_path = campaign_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")

    report_path = campaign_dir / "report.md"
    budget_minutes = (manifest.get("campaign_budget_seconds") or 0) / 60
    profiling_minutes = (manifest.get("profiling_duration_seconds") or 0) / 60
    lines = [
        "# AgentX concurrency tail pilot",
        "",
        f"This is a bounded load-spread pilot with a {budget_minutes:g}-minute "
        f"campaign budget and {profiling_minutes:g}-minute profiling windows. "
        "Request p99 values are observed order statistics, not stable population "
        "estimates.",
        "",
        "| C | Requests | Sessions completed | Tail resolution | TTFT p50/p95 | "
        "E2E p50/p95 | Avg-ITL p50/p95 | >1s gaps / intervals | Prefix hit |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summaries:
        if run.get("status") != "completed":
            lines.append(
                f"| {run.get('concurrency')} | — | — | — | "
                f"{run.get('status')} | — | — | — | — |"
            )
            continue
        gaps = run.get("token_gaps") or {}
        prefix = run.get("prefix_cache") or {}
        sessions = run.get("sessions") or {}

        def pair(name: str) -> str:
            values = run.get(name) or {}
            return f"{values.get('p50', '—'):.0f}/{values.get('p95', '—'):.0f}"

        lines.append(
            f"| {run['concurrency']} | {run['requests']} | "
            f"{sessions.get('completed', '—')} | "
            f"{run['empirical_tail_resolution_percent']:.2f}% | "
            f"{pair('ttft_ms')} ms | {pair('request_latency_ms')} ms | "
            f"{pair('request_avg_itl_ms')} ms | "
            f"{gaps.get('gaps_above_1s', '—')} / {gaps.get('intervals', '—')} | "
            f"{prefix.get('hit_rate_percent', '—')}% |"
        )
    lines.extend(
        [
            "",
            "Raw request samples: `request_samples.csv`",
            "",
            "Machine-readable analysis: `analysis.json`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Analysis: {analysis_path}")
    print(f"Report:   {report_path}")
    if all_requests:
        print(f"Samples:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
