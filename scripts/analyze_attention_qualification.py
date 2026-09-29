#!/usr/bin/env python3
"""Analyze paired quality, sampling, and latency against predeclared gates."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path("_results/quality/attention_qualification")
PROFILES = {
    "stock": ROOT / "rocm_attn_greedy.json",
    "aiter": ROOT / "aiter_unified_greedy.json",
    "triton": ROOT / "triton_greedy.json",
}
THROUGHPUT = {
    "stock": {"c1": 79.35, "c8": 553.58},
    "aiter": {"c1": 99.58, "c8": 653.96},
    "triton": {"c1": 100.36, "c8": 660.53},
}


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def paired(stock: list[dict], candidate: list[dict], seed: int = 7) -> dict:
    stock_by_id = {row["id"]: row for row in stock}
    candidate_by_id = {row["id"]: row for row in candidate}
    ids = sorted(set(stock_by_id) & set(candidate_by_id))
    pairs = [
        (
            int(bool(stock_by_id[ident]["correct"])),
            int(bool(candidate_by_id[ident]["correct"])),
        )
        for ident in ids
    ]
    wins = sum(left == 0 and right == 1 for left, right in pairs)
    losses = sum(left == 1 and right == 0 for left, right in pairs)
    delta = sum(right - left for left, right in pairs) / len(pairs)
    rng = random.Random(seed)
    boot = []
    for _ in range(10000):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        boot.append(sum(right - left for left, right in sample) / len(sample))
    return {
        "n": len(pairs),
        "stock_correct": sum(left for left, _ in pairs),
        "candidate_correct": sum(right for _, right in pairs),
        "delta": delta,
        "wins": wins,
        "losses": losses,
        "ties": len(pairs) - wins - losses,
        "bootstrap_95_ci": [percentile(boot, 0.025), percentile(boot, 0.975)],
    }


def by_group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value:
            groups[str(value)].append(row)
    return groups


def load_latency(profile: str) -> dict:
    root = ROOT / "latency" / profile
    result = {}
    for path in root.glob("c*.json"):
        if path.name.endswith("_warmup.json"):
            continue
        result[path.stem] = json.loads(path.read_text())
    return result


def latency_gate(name: str, stock: dict, candidate: dict) -> dict:
    failures = []
    for shape, base in stock.items():
        test = candidate[shape]
        if test["failed"]:
            failures.append(f"{shape}:request_failure")
        allowed_ttft = base["ttft_p95_ms"] + max(base["ttft_p95_ms"] * 0.10, 5.0)
        if test["ttft_p95_ms"] > allowed_ttft:
            failures.append(
                f"{shape}:ttft_p95 {test['ttft_p95_ms']:.2f}>{allowed_ttft:.2f}"
            )
        if test["itl_p95_ms"] > base["itl_p95_ms"]:
            failures.append(
                f"{shape}:itl_p95 {test['itl_p95_ms']:.2f}>{base['itl_p95_ms']:.2f}"
            )
        if "_1k_" in shape and test["itl_p95_ms"] > 11.0:
            failures.append(f"{shape}:itl_p95 {test['itl_p95_ms']:.2f}>11.00")
    if THROUGHPUT[name]["c1"] < 98:
        failures.append("c1_1k_1k_throughput")
    if THROUGHPUT[name]["c8"] < 640:
        failures.append("c8_1k_1k_throughput")
    return {"pass": not failures, "failures": failures}


def quality_gate(stock_rows: list[dict], candidate_rows: list[dict]) -> dict:
    overall = paired(stock_rows, candidate_rows)
    stock_groups = by_group(stock_rows, "category")
    candidate_groups = by_group(candidate_rows, "category")
    categories = {
        key: paired(stock_groups[key], candidate_groups[key])
        for key in sorted(stock_groups)
    }
    failures = []
    if overall["delta"] < -0.02:
        failures.append(f"overall_delta={overall['delta']:.3f}")
    for category in ("math", "reasoning", "code"):
        metric = categories[category]
        if metric["delta"] < -0.05:
            failures.append(f"{category}_delta={metric['delta']:.3f}")
        if metric["losses"] - metric["wins"] > 1:
            failures.append(
                f"{category}_net_losses={metric['losses'] - metric['wins']}"
            )
    json_rows = candidate_groups["json_tool"]
    if any(not row["valid"] for row in json_rows):
        failures.append("json_validity")
    if categories["json_tool"]["losses"]:
        failures.append(f"json_new_failures={categories['json_tool']['losses']}")
    stock_bands = by_group(stock_groups["long_context"], "band")
    candidate_bands = by_group(candidate_groups["long_context"], "band")
    bands = {
        band: paired(stock_bands[band], candidate_bands[band])
        for band in sorted(stock_bands)
    }
    for band, metric in bands.items():
        if metric["losses"]:
            failures.append(f"long_{band}_new_failures={metric['losses']}")
    if any(not row["ok"] for row in candidate_rows):
        failures.append("request_failure")
    return {
        "pass": not failures,
        "failures": failures,
        "overall": overall,
        "categories": categories,
        "long_bands": bands,
    }


def sampling_summary() -> dict:
    result = {}
    for profile in ("stock", "aiter"):
        files = sorted((ROOT / "sampling").glob(f"{profile}_*.json"))
        category_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        validity: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for path in files:
            data = json.loads(path.read_text())
            for row in data["rows"]:
                category_counts[row["category"]][0] += int(bool(row["correct"]))
                category_counts[row["category"]][1] += 1
                validity[row["category"]][0] += int(bool(row["valid"]))
                validity[row["category"]][1] += 1
        result[profile] = {
            category: {
                "correct": values[0],
                "n": values[1],
                "accuracy": values[0] / values[1],
                "validity": validity[category][0] / validity[category][1],
            }
            for category, values in sorted(category_counts.items())
        }
    failures = []
    for category, stock in result["stock"].items():
        candidate = result["aiter"][category]
        if candidate["accuracy"] < stock["accuracy"] - 0.05:
            failures.append(
                f"{category}_delta={candidate['accuracy'] - stock['accuracy']:.3f}"
            )
    if result["aiter"]["json_tool"]["validity"] < 1:
        failures.append("json_validity")
    result["aiter_gate"] = {"pass": not failures, "failures": failures}
    return result


def main() -> int:
    data = {
        name: json.loads(path.read_text()) for name, path in PROFILES.items()
    }
    stock_rows = data["stock"]["rows"]
    quality = {
        name: quality_gate(stock_rows, data[name]["rows"])
        for name in ("aiter", "triton")
    }
    stock_latency = load_latency("rocm_attn")
    latency = {
        "aiter": latency_gate(
            "aiter", stock_latency, load_latency("aiter_unified")
        ),
        "triton": latency_gate("triton", stock_latency, load_latency("triton")),
    }
    sampling = sampling_summary()
    decision = {
        name: {
            "quality_pass": quality[name]["pass"],
            "latency_pass": latency[name]["pass"],
            "promote": quality[name]["pass"] and latency[name]["pass"],
        }
        for name in ("aiter", "triton")
    }
    result = {
        "quality": quality,
        "sampling": sampling,
        "latency": latency,
        "throughput_reference": THROUGHPUT,
        "decision": decision,
    }
    path = ROOT / "analysis.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
