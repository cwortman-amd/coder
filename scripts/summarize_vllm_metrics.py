#!/usr/bin/env python3
"""Compute Prometheus counter deltas for a vLLM profiling window."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)\s*$"
)


def parse(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = LINE_RE.match(line)
            if not match:
                continue
            key = match.group("name")
            if match.group("labels"):
                key = f"{key}{{{match.group('labels')}}}"
            out[key] = float(match.group("value"))
    return out


def family_sum(metrics: dict[str, float], prefix: str) -> float:
    total = 0.0
    for name, value in metrics.items():
        bare = name.split("{", 1)[0]
        if bare == prefix or bare.startswith(prefix + "_"):
            if bare.endswith("_created"):
                continue
            if "bucket" in bare:
                continue
            total += value
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    before = parse(args.before)
    after = parse(args.after)
    keys = sorted(set(before) | set(after))
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in keys}

    interesting = [
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:e2e_request_latency_seconds_count",
        "vllm:e2e_request_latency_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:inter_token_latency_seconds_count",
        "vllm:inter_token_latency_seconds_sum",
        "vllm:request_queue_time_seconds_count",
        "vllm:request_queue_time_seconds_sum",
        "vllm:request_prefill_time_seconds_count",
        "vllm:request_prefill_time_seconds_sum",
        "vllm:request_decode_time_seconds_count",
        "vllm:request_decode_time_seconds_sum",
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_emitted_tokens_total",
    ]
    compact = {}
    for name in interesting:
        b = sum(v for k, v in before.items() if k.split("{", 1)[0] == name)
        a = sum(v for k, v in after.items() if k.split("{", 1)[0] == name)
        compact[name] = {"before": b, "after": a, "delta": a - b}

    gauges = {}
    for name in (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:spec_decode_draft_acceptance_rate",
        "vllm:spec_decode_efficiency",
    ):
        gauges[name] = {
            "before": sum(v for k, v in before.items() if k.split("{", 1)[0] == name),
            "after": sum(v for k, v in after.items() if k.split("{", 1)[0] == name),
        }

    draft = compact.get("vllm:spec_decode_num_draft_tokens_total", {}).get("delta", 0.0)
    accepted = compact.get("vllm:spec_decode_num_accepted_tokens_total", {}).get("delta", 0.0)
    drafts = compact.get("vllm:spec_decode_num_drafts_total", {}).get("delta", 0.0)
    generated = compact.get("vllm:generation_tokens_total", {}).get("delta", 0.0)
    ttft_n = compact.get("vllm:time_to_first_token_seconds_count", {}).get("delta", 0.0)
    ttft_s = compact.get("vllm:time_to_first_token_seconds_sum", {}).get("delta", 0.0)
    itl_n = compact.get("vllm:inter_token_latency_seconds_count", {}).get("delta", 0.0)
    itl_s = compact.get("vllm:inter_token_latency_seconds_sum", {}).get("delta", 0.0)

    derived = {
        "draft_acceptance": (accepted / draft) if draft else None,
        "accepted_per_draft_step": (accepted / drafts) if drafts else None,
        "mean_acceptance_length_if_drafts_are_verifies": (
            (generated / drafts) if drafts else None
        ),
        "mean_ttft_s": (ttft_s / ttft_n) if ttft_n else None,
        "mean_itl_s": (itl_s / itl_n) if itl_n else None,
    }

    pos = defaultdict(float)
    for key, value in delta.items():
        if key.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            pos[key] = value

    result = {"counters": compact, "gauges_end": gauges, "derived": derived, "accepted_per_pos_delta": pos}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"derived": derived, "generation_delta": generated, "draft_delta": draft, "accepted_delta": accepted}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
