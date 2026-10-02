#!/usr/bin/env python3
"""Publish durable Qwen latency samples and report-ready summaries.

Benchmark scratch lives under ``_results`` and is not persistent. This module
normalizes either ``vllm bench serve --save-detailed`` output or this repo's
streaming-client output, strips generated text, and writes:

* exact request/token samples to ``docs/results`` as compressed JSON;
* a small report index to ``docs/profiling``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distribution_stats import percentile_summary  # noqa: E402
from request_event_schema import percentile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "docs" / "results" / "qwen3.8-27b-mxfp4" / "latency"
PROFILE_PATH = ROOT / "docs" / "profiling" / "qwen3.8-27b-mxfp4-latency.json"

PERCENTILES = (50, 75, 90, 95, 99)
TTFT_BINS_MS = (0, 50, 100, 250, 500, 1000, 2000, 3000, 5000, 10000, 30000)
ITL_BINS_MS = (0, 5, 10, 15, 20, 30, 50, 100, 250, 500, 1000)


class MissingDetailedLatency(ValueError):
    """Raised when a benchmark result contains summaries but no raw samples."""


def summary(samples: list[float]) -> dict[str, float | int] | None:
    return percentile_summary(samples, PERCENTILES, with_mean=True, with_max=True)


def histogram(samples: list[float], edges: tuple[int, ...]) -> dict[str, Any]:
    labels = [f"{left}-{right}" for left, right in zip(edges, edges[1:])]
    labels.append(f">={edges[-1]}")
    counts = [
        sum(left <= value < right for value in samples)
        for left, right in zip(edges, edges[1:])
    ]
    counts.append(sum(value >= edges[-1] for value in samples))
    return {"unit": "ms", "bins": labels, "counts": counts}


def _milliseconds(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return round(float(value) * 1000, 6)


def _from_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    requests = []
    for index, row in enumerate(document.get("rows") or []):
        itl_ms = [
            converted
            for value in row.get("itl_s") or []
            if (converted := _milliseconds(value)) is not None
        ]
        ttft_ms = _milliseconds(row.get("ttft_s"))
        e2el_ms = _milliseconds(row.get("latency_s", row.get("dt")))
        output_len = row.get("completion_tokens")
        tpot_ms = None
        if (
            ttft_ms is not None
            and e2el_ms is not None
            and isinstance(output_len, int)
            and output_len > 1
        ):
            tpot_ms = round((e2el_ms - ttft_ms) / (output_len - 1), 6)
        requests.append(
            {
                "request_index": index,
                "ok": bool(row.get("ok")),
                "input_len": row.get("prompt_tokens"),
                "output_len": output_len,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "e2el_ms": e2el_ms,
                "itl_ms": itl_ms,
                "error": row.get("error"),
            }
        )
    return requests


def _from_vllm_detailed(document: dict[str, Any]) -> list[dict[str, Any]]:
    ttfts = document.get("ttfts")
    itls = document.get("itls")
    if not isinstance(ttfts, list) or not isinstance(itls, list):
        raise MissingDetailedLatency(
            "result has no raw ttfts/itls; run vllm bench serve with --save-detailed"
        )
    arrays = {
        "input_len": document.get("input_lens") or [],
        "output_len": document.get("output_lens") or [],
        "ttft": ttfts,
        "tpot": document.get("tpots") or [],
        "e2el": document.get("e2els") or [],
        "itl": itls,
        "error": document.get("errors") or [],
    }
    request_count = max(len(values) for values in arrays.values())

    def at(name: str, index: int) -> Any:
        values = arrays[name]
        return values[index] if index < len(values) else None

    requests = []
    for index in range(request_count):
        error = at("error", index)
        raw_itl = at("itl", index) or []
        itl_ms = [
            converted
            for value in raw_itl
            if (converted := _milliseconds(value)) is not None
        ]
        ttft_ms = _milliseconds(at("ttft", index))
        tpot_ms = _milliseconds(at("tpot", index))
        if tpot_ms is None and len(itl_ms) > 1:
            tpot_ms = round(sum(itl_ms[1:]) / (len(itl_ms) - 1), 6)
        e2el_ms = _milliseconds(at("e2el", index))
        if e2el_ms is None and ttft_ms is not None:
            e2el_ms = round(ttft_ms + (sum(itl_ms[1:]) if len(itl_ms) > 1 else sum(itl_ms)), 6)
        requests.append(
            {
                "request_index": index,
                "ok": not bool(error),
                "input_len": at("input_len", index),
                "output_len": at("output_len", index),
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "e2el_ms": e2el_ms,
                "itl_ms": itl_ms,
                "error": error or None,
            }
        )
    return requests


def normalize(document: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(document.get("rows"), list):
        requests = _from_rows(document)
    else:
        requests = _from_vllm_detailed(document)
    if not requests or not any(request["ttft_ms"] is not None for request in requests):
        raise MissingDetailedLatency("result contains no request-level TTFT samples")
    return requests


def _run_summary(run_id: str, metadata: dict[str, Any], requests: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [request for request in requests if request["ok"]]
    ttft = [request["ttft_ms"] for request in successful if request["ttft_ms"] is not None]
    tpot = [request["tpot_ms"] for request in successful if request["tpot_ms"] is not None]
    e2el = [request["e2el_ms"] for request in successful if request["e2el_ms"] is not None]
    itl = [value for request in successful for value in request["itl_ms"]]
    request_slo = sum(value <= 3000 for value in ttft)
    return {
        "run_id": run_id,
        **metadata,
        "requests": {
            "total": len(requests),
            "successful": len(successful),
            "failed": len(requests) - len(successful),
            "tail_resolution_pct": round(100 / len(successful), 3) if successful else None,
            "p99_sample_quality": (
                "stable" if len(successful) >= 1000 else "minimum" if len(successful) >= 100 else "insufficient"
            ),
            "ttft_at_or_below_3000_ms": request_slo,
            "ttft_slo_rate_pct": round(request_slo / len(ttft) * 100, 3) if ttft else None,
        },
        "ttft_ms": summary(ttft),
        "tpot_ms": summary(tpot),
        "e2el_ms": summary(e2el),
        "itl_ms": summary(itl),
        "ttft_histogram": histogram(ttft, TTFT_BINS_MS),
        "itl_histogram": histogram(itl, ITL_BINS_MS),
        "streaming_slo": {
            "limits_ms": {"ttft": 3000, "tpot": 20, "itl_p95": 20, "itl_p99": 50, "itl_max": 100},
            "passes": bool(
                ttft
                and tpot
                and itl
                and max(ttft) <= 3000
                and percentile(tpot, 50) <= 20
                and percentile(itl, 95) <= 20
                and percentile(itl, 99) <= 50
                and max(itl) <= 100
            ),
        },
    }


def _update_profile(run: dict[str, Any], profile_path: Path) -> None:
    if profile_path.is_file():
        profile = json.loads(profile_path.read_text())
    else:
        profile = {
            "schema_version": 1,
            "model": "Qwen3.8-27B MXFP4",
            "note": (
                "Report-ready index. Exact request and token samples are in the "
                "referenced compressed files under docs/results."
            ),
            "runs": [],
        }
    profile["runs"] = [
        existing for existing in profile.get("runs", []) if existing.get("run_id") != run["run_id"]
    ]
    profile["runs"].append(run)
    profile["runs"].sort(
        key=lambda item: (
            item.get("gpu_profile", ""),
            item.get("workload", ""),
            item.get("concurrency", 0),
            item.get("repetition", 0),
            item.get("run_id", ""),
        )
    )
    profile["updated_utc"] = datetime.now(timezone.utc).isoformat()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")


def publish_latency_result(
    source: Path,
    *,
    gpu_profile: str,
    workload: str,
    concurrency: int,
    repetition: int = 1,
    raw_root: Path = RAW_ROOT,
    profile_path: Path = PROFILE_PATH,
) -> tuple[Path, Path]:
    document = json.loads(source.read_text())
    requests = normalize(document)
    run_id = source.stem
    relative = Path(gpu_profile) / workload / f"{run_id}.json.gz"
    destination = raw_root / relative
    measured_input_len = next(
        (request["input_len"] for request in requests if request["input_len"] is not None),
        None,
    )
    measured_output_len = next(
        (request["output_len"] for request in requests if request["output_len"] is not None),
        None,
    )
    metadata = {
        "gpu_profile": gpu_profile,
        "workload": workload,
        "input_len": document.get("input_len", document.get("random_input_len", measured_input_len)),
        "output_len": document.get("output_len", document.get("random_output_len", measured_output_len)),
        "concurrency": concurrency,
        "repetition": repetition,
        "source_format": "streaming_client" if "rows" in document else "vllm_save_detailed",
        "raw_samples": str(Path("docs/results/qwen3.8-27b-mxfp4/latency") / relative),
        "benchmark": {
            key: document.get(key)
            for key in (
                "date",
                "backend",
                "request_rate",
                "duration",
                "duration_s",
                "num_prompts",
                "completed",
                "successful",
                "failed",
                "model_id",
                "model",
                "tokenizer_id",
            )
            if document.get(key) is not None
        },
    }
    archive = {
        "schema_version": 1,
        "model": "Qwen3.8-27B MXFP4",
        "run_id": run_id,
        "metadata": metadata,
        "requests": requests,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt", encoding="utf-8") as handle:
        json.dump(archive, handle, separators=(",", ":"))
        handle.write("\n")
    run = _run_summary(run_id, metadata, requests)
    _update_profile(run, profile_path)
    return destination, profile_path


def _infer_cell(path: Path) -> tuple[str, int] | None:
    stem = path.stem
    if "_power" in stem or stem == "SUMMARY":
        return None
    workload, separator, concurrency = stem.rpartition("_c")
    if not separator or not concurrency.isdigit():
        return None
    return workload, int(concurrency)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--gpu-profile", required=True)
    parser.add_argument("--workload", default="")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--repetition", type=int, default=1)
    args = parser.parse_args()
    if bool(args.source) == bool(args.source_dir):
        parser.error("provide exactly one of --source or --source-dir")

    jobs: list[tuple[Path, str, int]] = []
    if args.source:
        if not args.workload or args.concurrency is None:
            parser.error("--source requires --workload and --concurrency")
        jobs.append((args.source, args.workload, args.concurrency))
    else:
        for path in sorted(args.source_dir.glob("*.json")):
            cell = _infer_cell(path)
            if cell:
                jobs.append((path, *cell))

    for source, workload, concurrency in jobs:
        raw_path, profile_path = publish_latency_result(
            source,
            gpu_profile=args.gpu_profile,
            workload=workload,
            concurrency=concurrency,
            repetition=args.repetition,
        )
        print(f"published {source} -> {raw_path}")
        print(f"updated {profile_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
