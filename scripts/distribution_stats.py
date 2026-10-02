"""Shared linear-percentile distribution summary for the tail benches."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any

from request_event_schema import percentile


def distribution_core(samples: list[float]) -> dict[str, Any]:
    """Finite-sample mean, spread, and p50/p90/p95/p99. Ratios are None at p50 0."""
    clean = [float(value) for value in samples if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"n": 0}
    n = len(clean)
    mean_val = statistics.mean(clean)
    std_val = statistics.stdev(clean) if n > 1 else 0.0
    stderr = std_val / math.sqrt(n) if n > 1 else 0.0
    p50 = percentile(clean, 50.0) or 0.0
    p90 = percentile(clean, 90.0) or 0.0
    p95 = percentile(clean, 95.0) or 0.0
    p99 = percentile(clean, 99.0) or 0.0
    cv = (std_val / mean_val * 100.0) if mean_val > 0 else 0.0
    p99_to_p50 = (p99 / p50) if p50 > 0 else None
    p95_to_p50 = (p95 / p50) if p50 > 0 else None
    return {
        "n": n,
        "clean": clean,
        "min": round(min(clean), 2),
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "stderr": round(stderr, 2),
        "cv_pct": round(cv, 1),
        "p50": round(p50, 2),
        "p90": round(p90, 2),
        "p95": round(p95, 2),
        "p99": round(p99, 2),
        "max": round(max(clean), 2),
        "p99_to_p50_ratio": round(p99_to_p50, 2) if p99_to_p50 is not None else None,
        "p95_to_p50_ratio": round(p95_to_p50, 2) if p95_to_p50 is not None else None,
        "_p50": p50,
        "_p99_to_p50": p99_to_p50,
        "_p95_to_p50": p95_to_p50,
    }


def percentile_summary(
    samples: Iterable[float],
    percents: tuple[int, ...],
    *,
    digits: int = 3,
    with_mean: bool = False,
    with_min: bool = False,
    with_max: bool = False,
    on_empty: str = "none",
) -> dict[str, float | int] | None:
    """Rounded n / optional mean / percentiles. Empty samples return None or raise."""
    values = list(samples)
    if not values:
        if on_empty == "raise":
            raise ValueError("empty sample")
        return None
    result: dict[str, float | int] = {"n": len(values)}
    if with_mean:
        result["mean"] = round(statistics.mean(values), digits)
    if with_min:
        result["min"] = round(min(values), digits)
    if with_max:
        result["max"] = round(max(values), digits)
    for pct in percents:
        value = percentile(values, pct)
        assert value is not None
        result[f"p{pct}"] = round(value, digits)
    return result
