#!/usr/bin/env python3
"""Greedy token-ID capture against a localhost OpenAI-compatible server."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


def complete(base: str, model: str, prompt: str, max_tokens: int, timeout: int, seed: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": seed,
        "stream": False,
        "ignore_eos": False,
        "skip_special_tokens": False,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    elapsed = time.perf_counter() - started
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}
    return {
        "ok": True,
        "latency_s": elapsed,
        "content": message.get("content"),
        "token_ids": choice.get("token_ids"),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = []
    with open(args.corpus, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    results = []
    failures = 0
    for index, row in enumerate(rows, 1):
        try:
            out = complete(
                args.base_url,
                args.model,
                row["prompt"],
                int(row.get("max_tokens", 128)),
                args.timeout,
                args.seed,
            )
        except Exception as exc:  # noqa: BLE001
            out = {"ok": False, "error": str(exc)}
            failures += 1
        record = {
            "id": row["id"],
            "category": row["category"],
            "max_tokens": row.get("max_tokens", 128),
            **out,
        }
        results.append(record)
        status = "ok" if record.get("ok") else "FAIL"
        n_tok = len(record.get("token_ids") or [])
        print(f"[{index}/{len(rows)}] {status} {row['id']} tokens={n_tok}", flush=True)

    payload = {
        "profile": args.profile,
        "model": args.model,
        "seed": args.seed,
        "temperature": 0,
        "top_p": 1.0,
        "top_k": -1,
        "ignore_eos": False,
        "n": len(results),
        "failed": failures,
        "rows": results,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(path), "n": len(results), "failed": failures}))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
