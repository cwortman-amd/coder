#!/usr/bin/env python3
"""Measure OpenAI-compatible streaming TTFT and inter-chunk latency."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def chat(base: str, model: str, n_in: int, n_out: int, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say the word ping. " + ("alpha " * max(n_in - 8, 1))}],
        "max_tokens": n_out,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "stream_interval": 1,
        "return_token_ids": True,
        "ignore_eos": True,
        "skip_special_tokens": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    token_times: list[float] = []
    chunk_times: list[float] = []
    usage: dict = {}
    with urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            body = json.loads(line[6:])
            if body.get("usage"):
                usage = body["usage"]
            choices = body.get("choices") or []
            token_ids = choices[0].get("token_ids") if choices else None
            if token_ids:
                received = time.perf_counter()
                chunk_times.append(received)
                token_times.extend([received] * len(token_ids))
    ended = time.perf_counter()
    intervals = [b - a for a, b in zip(token_times, token_times[1:])]
    chunk_intervals = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
    return {
        "ok": True,
        "latency_s": ended - started,
        "ttft_s": token_times[0] - started if token_times else None,
        "itl_s": intervals,
        "chunk_itl_s": chunk_intervals,
        "stream_chunks": len(chunk_times),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(chat, args.base_url, args.model, args.input_len, args.output_len, args.timeout)
            for _ in range(args.num_prompts)
        ]
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001
                rows.append({"ok": False, "error": str(exc)})
    wall = time.perf_counter() - started

    ok = [row for row in rows if row.get("ok")]
    ttfts = [row["ttft_s"] for row in ok if row["ttft_s"] is not None]
    itls = [itl for row in ok for itl in row["itl_s"]]
    chunk_itls = [itl for row in ok for itl in row["chunk_itl_s"]]
    generated = sum(row["completion_tokens"] for row in ok)
    result = {
        "base_url": args.base_url,
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "successful": len(ok),
        "failed": len(rows) - len(ok),
        "duration_s": wall,
        "total_generated_tokens": generated,
        "output_throughput": generated / wall if wall else 0,
        "ttft_p50_ms": percentile(ttfts, 0.50) * 1000 if ttfts else None,
        "ttft_p95_ms": percentile(ttfts, 0.95) * 1000 if ttfts else None,
        "itl_mean_ms": statistics.mean(itls) * 1000 if itls else None,
        "itl_p50_ms": percentile(itls, 0.50) * 1000 if itls else None,
        "itl_p95_ms": percentile(itls, 0.95) * 1000 if itls else None,
        "chunk_itl_p50_ms": percentile(chunk_itls, 0.50) * 1000 if chunk_itls else None,
        "chunk_itl_p95_ms": percentile(chunk_itls, 0.95) * 1000 if chunk_itls else None,
        "mean_latency_s": statistics.mean(row["latency_s"] for row in ok) if ok else None,
        "rows": rows,
        "note": (
            "Token ITL uses token_ids arrival times. Tokens delivered in one speculative SSE "
            "chunk have zero arrival interval; chunk ITL reports user-visible burst cadence."
        ),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
