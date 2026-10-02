"""Load prices, SLO lines, and published P/D summary fields for the figure scripts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ASSUMPTIONS_PATH = ROOT / "config" / "plot_assumptions.yaml"
TABLE_PATH = ROOT / "config" / "latency_dashboard_table.yaml"


@lru_cache(maxsize=1)
def assumptions() -> dict[str, Any]:
    with ASSUMPTIONS_PATH.open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"{ASSUMPTIONS_PATH} did not contain a mapping")
    return loaded


def slo() -> dict[str, float]:
    return {key: float(value) for key, value in assumptions()["slo"].items()}


def electricity_usd_per_kwh() -> float:
    return float(assumptions()["prices"]["electricity_usd_per_kwh"])


def r9700_gpu_usd() -> float:
    return float(assumptions()["prices"]["r9700_gpu_usd"])


def capex() -> dict[str, Any]:
    return assumptions()["capex"]


def illustrative_itl() -> dict[str, float]:
    return assumptions()["illustrative_itl"]


def latency_dashboard_rows() -> list[list[str]]:
    with TABLE_PATH.open() as handle:
        loaded = yaml.safe_load(handle)
    return [list(row) for row in loaded["rows"]]


def published_pd_slo(summary_path: Path) -> dict[str, float]:
    """SLO block from the published emulator summary, with the assumption file as fallback."""
    fallback = slo()
    if not summary_path.is_file():
        return {
            "max_ttft_ms": fallback["ttft_ms"],
            "max_tpot_ms": fallback["tpot_ms"],
            "max_p95_itl_ms": fallback["itl_p95_ms"],
            "max_peak_itl_ms": fallback["peak_itl_ms"],
        }
    with summary_path.open() as handle:
        loaded = json.load(handle)
    block = dict(loaded.get("slo_config") or {})
    block.setdefault("max_ttft_ms", fallback["ttft_ms"])
    block.setdefault("max_tpot_ms", fallback["tpot_ms"])
    block.setdefault("max_p95_itl_ms", fallback["itl_p95_ms"])
    block.setdefault("max_peak_itl_ms", fallback["peak_itl_ms"])
    return block
