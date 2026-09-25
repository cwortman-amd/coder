#!/usr/bin/env python3
"""Lightweight OpenAI-chat throughput client (no GPU)."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen


def chat(
    base: str,
    model: str,
    n_in: int,
    n_out: int,
    timeout: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    enable_thinking: bool,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say the word ping. " + ("alpha " * max(n_in - 8, 1))}],
        "max_tokens": n_out,
        "temperature": temperature,
        "stream": False,
        "ignore_eos": True,
        "skip_special_tokens": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    req = Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    dt = time.perf_counter() - t0
    usage = body.get("usage") or {}
    return {
        "ok": True,
        "dt": dt,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t_all = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [
            pool.submit(
                chat,
                args.base_url,
                args.model,
                args.input_len,
                args.output_len,
                args.timeout,
                args.temperature,
                args.top_p,
                args.top_k,
                args.enable_thinking,
            )
            for _ in range(args.num_prompts)
        ]
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                rows.append({"ok": False, "error": str(exc), "dt": 0, "prompt_tokens": 0, "completion_tokens": 0})
    wall = time.perf_counter() - t_all
    ok = [r for r in rows if r.get("ok")]
    gen = sum(r["completion_tokens"] for r in ok)
    prompt = sum(r["prompt_tokens"] for r in ok)
    dts = [r["dt"] for r in ok]
    result = {
        "base_url": args.base_url,
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "enable_thinking": args.enable_thinking,
        "successful": len(ok),
        "failed": len(rows) - len(ok),
        "duration": wall,
        "total_prompt_tokens": prompt,
        "total_generated_tokens": gen,
        "output_throughput": gen / wall if wall else 0,
        "total_token_throughput": (gen + prompt) / wall if wall else 0,
        "mean_latency_s": statistics.mean(dts) if dts else None,
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({k: result[k] for k in result if k != "rows"}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
