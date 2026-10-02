#!/usr/bin/env python3
"""Token-ID and serving-waterfall gates for the Qwen3.8 P/D pair.

The router reports prefill HTTP time and the gap before the decode request.
NIXL transfer post/completion come from the decode engine's Prometheus counters,
which advance once per successful transfer when requests are serial.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from request_event_schema import percentile  # noqa: E402

PROMPT = (
    "Reply with exactly these words and no preamble: alpha bravo charlie "
    "delta echo foxtrot golf hotel. Then add one short sentence about GPUs."
)
METRICS = {
    "xfer_sum": "vllm:nixl_xfer_time_seconds_sum",
    "xfer_count": "vllm:nixl_xfer_time_seconds_count",
    "post_sum": "vllm:nixl_post_time_seconds_sum",
    "post_count": "vllm:nixl_post_time_seconds_count",
    "bytes_sum": "vllm:nixl_bytes_transferred_sum",
    "desc_sum": "vllm:nixl_num_descriptors_sum",
    "ext_hits": "vllm:external_prefix_cache_hits_total",
    "ext_queries": "vllm:external_prefix_cache_queries_total",
    "failed": "vllm:nixl_num_failed_transfers_total",
}


def fetch(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def metric_values(port):
    raw = urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=10
    ).read().decode()
    values = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, rest = line.partition("{")
        if key not in METRICS.values():
            continue
        values[key] = float(rest.rsplit(" ", 1)[-1])
    return values


def summarize(samples):
    return {
        "n": len(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
    }


def chat(url, content, max_tokens):
    started = time.perf_counter()
    body = fetch(
        url,
        {
            "model": "awq",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 1,
            "ignore_eos": True,
            "return_token_ids": True,
        },
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    choice = body["choices"][0]
    return {
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": body.get("usage", {}).get("completion_tokens"),
        "prompt_token_ids": body.get("prompt_token_ids"),
        "token_ids": choice.get("token_ids"),
        "pd_metrics": body.get("pd_metrics"),
        "finish_reason": choice.get("finish_reason"),
    }


def delta(before, after):
    out = {}
    for name, key in METRICS.items():
        if key in before and key in after:
            out[name] = after[key] - before[key]
    if out.get("xfer_count"):
        out["xfer_ms"] = out["xfer_sum"] / out["xfer_count"] * 1000
        out["post_ms"] = out["post_sum"] / out["post_count"] * 1000
    out["recomputed_tokens"] = out.get("ext_queries", 0) - out.get("ext_hits", 0)
    return out


def eight_k_prompt(tokenize_url):
    sentence = "Intra-node KV handoff must preserve attention, convolution, and SSM state. "
    low, high = 1, 800
    target = 8192
    best = sentence
    best_delta = abs(fetch(tokenize_url, {"model": "awq", "prompt": sentence})["count"] - target)
    while low <= high:
        count = (low + high) // 2
        text = sentence * count
        token_count = fetch(tokenize_url, {"model": "awq", "prompt": text})["count"]
        delta = abs(token_count - target)
        if delta < best_delta:
            best = text
            best_delta = delta
        if token_count < target:
            low = count + 1
        else:
            high = count - 1
    return best


def first_diff(left, right):
    if left is None or right is None:
        return {"equal": False, "reason": "missing token ids"}
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return {
                "equal": False,
                "first_diff": index,
                "left": left[index],
                "right": right[index],
                "left_len": len(left),
                "right_len": len(right),
            }
    return {
        "equal": len(left) == len(right),
        "first_diff": None if len(left) == len(right) else limit,
        "left_len": len(left),
        "right_len": len(right),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pd", "single", "compare", "prompt"])
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--metrics-port", type=int, default=8200)
    parser.add_argument("--tokenize-url", default="http://127.0.0.1:8100/tokenize")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--eight-k", action="store_true")
    parser.add_argument("--prompt-file")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "prompt":
        text = eight_k_prompt(args.tokenize_url)
        destination.write_text(text)
        print(json.dumps({"chars": len(text), "path": str(destination)}))
        return

    if args.mode == "compare":
        left = json.loads(Path(args.url).read_text()) if False else None
        files = json.loads(Path(args.out).read_text()) if destination.exists() else None
        raise SystemExit("compare is performed by the caller")

    if args.eight_k:
        content = Path(args.prompt_file).read_text()
    else:
        content = PROMPT
    records = []
    if args.concurrency == 1:
        for index in range(args.iters):
            before = metric_values(args.metrics_port) if args.mode == "pd" else {}
            record = chat(args.url, content, args.max_tokens)
            record["iter"] = index
            if args.mode == "pd":
                record["nixl"] = delta(before, metric_values(args.metrics_port))
            records.append(record)
            print(json.dumps({
                "iter": index,
                "elapsed_ms": round(record["elapsed_ms"], 2),
                "prompt_tokens": record["prompt_tokens"],
                "nixl": record.get("nixl"),
                "pd": record.get("pd_metrics"),
            }))
    else:
        import concurrent.futures
        before = metric_values(args.metrics_port) if args.mode == "pd" else {}
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            futures = [
                pool.submit(chat, args.url, content, args.max_tokens)
                for _ in range(args.iters)
            ]
            for index, future in enumerate(futures):
                record = future.result()
                record["iter"] = index
                records.append(record)
        if args.mode == "pd":
            records[0]["nixl_batch"] = delta(before, metric_values(args.metrics_port))

    summary = {
        "mode": args.mode,
        "elapsed_ms": summarize([item["elapsed_ms"] for item in records]),
        "records": records,
    }
    if args.mode == "pd":
        summary["prefill_ms"] = summarize([
            item["pd_metrics"]["prefill_ms"] for item in records if item.get("pd_metrics")
        ])
        summary["xfer_ms"] = summarize([
            item["nixl"]["xfer_ms"] for item in records
            if item.get("nixl", {}).get("xfer_count") == 1
        ])
        summary["post_ms"] = summarize([
            item["nixl"]["post_ms"] for item in records
            if item.get("nixl", {}).get("post_count") == 1
        ])
    destination.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}))


if __name__ == "__main__":
    main()
