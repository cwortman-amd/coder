#!/usr/bin/env python3
"""Teacher-forced top-two logits for attention-backend divergences.

Run once against each already-started server. Thresholds are fixed here,
before either backend's logits are interpreted:

- small margin: top-1 minus top-2 < 1.0 nat on both backends, and each
  argmax appears in the other backend's top-5
- large margin: top-1 minus top-2 >= 2.0 nat on either backend while the
  argmax tokens differ
- page proximity: context length within 2 tokens of a multiple of 784
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

PAGE = 784
SMALL_MARGIN_NATS = 1.0
LARGE_MARGIN_NATS = 2.0
PAGE_PROXIMITY = 2
SPECIAL_MIN_ID = 248000


def post(base: str, path: str, payload: dict, timeout: int) -> dict:
    req = Request(
        f"{base.rstrip('/')}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def chat_prompt_ids(base: str, model: str, prompt: str, timeout: int) -> list[int]:
    body = post(
        base,
        "/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
    )
    return body["tokens"]


def token_id_for_piece(base: str, model: str, piece: str, timeout: int) -> list[int]:
    body = post(base, "/tokenize", {"model": model, "prompt": piece}, timeout)
    return body["tokens"]


def complete_ids(base: str, model: str, prompt_ids: list[int], max_tokens: int, timeout: int, seed: int) -> dict:
    return post(
        base,
        "/v1/completions",
        {
            "model": model,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": seed,
            "logprobs": 5,
            "return_token_ids": True,
        },
        timeout,
    )


def chat_generate(base: str, model: str, prompt: str, max_tokens: int, timeout: int, seed: int) -> dict:
    # Same chat path as eval_greedy_tokens.py. Completions on identical token
    # ids can emit a different greedy token, so it is not a valid substitute.
    return post(
        base,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": seed,
            "ignore_eos": False,
            "skip_special_tokens": False,
            "return_token_ids": True,
            "logprobs": True,
            "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
    )


def top_table(base: str, model: str, entry: dict, timeout: int) -> tuple[list[dict], float | None]:
    ranked = entry.get("top_logprobs") or []
    rows = []
    for item in ranked:
        piece = item.get("token")
        ids = token_id_for_piece(base, model, piece, timeout)
        rows.append(
            {
                "piece": piece,
                "logprob": item.get("logprob"),
                "bytes": item.get("bytes"),
                "token_ids": ids,
                "single_token": len(ids) == 1,
                "token_id": ids[0] if len(ids) == 1 else None,
            }
        )
    if len(rows) >= 2 and rows[0]["logprob"] is not None and rows[1]["logprob"] is not None:
        margin = rows[0]["logprob"] - rows[1]["logprob"]
    else:
        margin = None
    return rows, margin


def page_geometry(context_len: int) -> dict:
    remainder = context_len % PAGE
    distance = min(remainder, PAGE - remainder) if context_len else None
    return {
        "context_tokens_before_prediction": context_len,
        "page_size": PAGE,
        "page_index": context_len // PAGE,
        "offset_in_page": remainder,
        "tokens_from_page_boundary": distance,
        "near_page_boundary": distance is not None and distance <= PAGE_PROXIMITY,
    }


def first_diff(left: list[int], right: list[int]) -> int | None:
    n = min(len(left), len(right))
    for i in range(n):
        if left[i] != right[i]:
            return i
    if len(left) != len(right):
        return n
    return None


def stage(diff: int) -> str:
    if diff <= 0:
        return "prefill_first_token"
    if diff < 16:
        return "early_decode"
    return "late_decode"


def fit_prompt(base: str, model: str, target: int, timeout: int) -> tuple[str, list[int]]:
    lo, hi = 1, max(target, 2)
    best_text, best_ids = "a", chat_prompt_ids(base, model, "a", timeout)
    while lo <= hi:
        mid = (lo + hi) // 2
        text = "alpha " * mid
        ids = chat_prompt_ids(base, model, text, timeout)
        if abs(len(ids) - target) < abs(len(best_ids) - target):
            best_text, best_ids = text, ids
        if len(ids) < target:
            lo = mid + 1
        elif len(ids) > target:
            hi = mid - 1
        else:
            return text, ids
    # Character trim to land on the exact chat-template length when possible.
    text = best_text
    ids = best_ids
    if len(ids) > target:
        while text and len(ids) > target:
            text = text[:-1]
            ids = chat_prompt_ids(base, model, text, timeout)
    elif len(ids) < target:
        while len(ids) < target:
            text += "x"
            ids = chat_prompt_ids(base, model, text, timeout)
            if len(ids) > target + 4:
                break
    return text, ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="awq")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--corpus", default="_results/quality/corpus.jsonl")
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    base = args.base_url[: -len("/v1")] if args.base_url.endswith("/v1") else args.base_url

    corpus = {
        json.loads(line)["id"]: json.loads(line)
        for line in Path(args.corpus).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    control = {row["id"]: row for row in json.loads(Path(args.control).read_text())["rows"]}
    candidate = {row["id"]: row for row in json.loads(Path(args.candidate).read_text())["rows"]}
    mismatches = []
    for ident, left in control.items():
        right = candidate.get(ident)
        if not right or not left.get("ok") or not right.get("ok"):
            continue
        diff = first_diff(left.get("token_ids") or [], right.get("token_ids") or [])
        if diff is None:
            continue
        mismatches.append((ident, diff, left, right))

    rows = []
    for ident, diff, left, right in mismatches:
        prompt = corpus[ident]["prompt"]
        started = time.perf_counter()
        body = chat_generate(base, args.model, prompt, diff + 1, args.timeout, args.seed)
        choice = body["choices"][0]
        live_ids = choice.get("token_ids") or []
        content = (choice.get("logprobs") or {}).get("content") or []
        entry = content[diff] if diff < len(content) else {}
        top, margin = top_table(base, args.model, entry, args.timeout)
        chosen = live_ids[diff] if diff < len(live_ids) else None
        prompt_len = len(body.get("prompt_token_ids") or [])
        control_ids = left.get("token_ids") or []
        candidate_ids = right.get("token_ids") or []
        control_next = control_ids[diff] if diff < len(control_ids) else None
        candidate_next = candidate_ids[diff] if diff < len(candidate_ids) else None
        prefix_ok = live_ids[:diff] == control_ids[:diff]
        record = {
            "id": ident,
            "category": left.get("category"),
            "first_diff_index": diff,
            "stage": stage(diff),
            "stored_control_token": control_next,
            "stored_candidate_token": candidate_next,
            "live_token": chosen,
            "live_prefix_matches_stored_control": prefix_ok,
            "matches_stored_control": chosen == control_next,
            "matches_stored_candidate": chosen == candidate_next,
            "special_or_format_token": bool(
                chosen is not None and (chosen >= SPECIAL_MIN_ID or (entry.get("token") or "").startswith("<|"))
            ),
            "top": top,
            "top1_top2_margin_nats": margin,
            "prompt_tokens": prompt_len,
            "batch_shape": "C1",
            "serving_path": "chat",
            "latency_s": time.perf_counter() - started,
            **page_geometry(prompt_len + diff),
        }
        rows.append(record)
        print(
            f"{ident} diff={diff} ctx={prompt_len + diff} chosen={chosen} margin={None if margin is None else round(margin, 4)} prefix_ok={prefix_ok}",
            flush=True,
        )

    page_rows = []
    targets = []
    for page in (1, 2, 3):
        boundary = PAGE * page
        targets.extend([boundary - 1, boundary, boundary + 1])
    for target in targets:
        text, ids = fit_prompt(base, args.model, target, args.timeout)
        body = chat_generate(base, args.model, text, 8, args.timeout, args.seed)
        choice = body["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        positions = []
        for index, entry in enumerate(content):
            top, margin = top_table(base, args.model, entry, args.timeout)
            positions.append({"index": index, "top": top[:2], "top1_top2_margin_nats": margin})
        top = positions[0]["top"] if positions else []
        margin = positions[0]["top1_top2_margin_nats"] if positions else None
        actual = len(body.get("prompt_token_ids") or ids)
        page_rows.append(
            {
                "requested_context": target,
                "actual_context": actual,
                "exact": actual == target,
                "generated_token_ids": choice.get("token_ids"),
                "top": top,
                "top1_top2_margin_nats": margin,
                "positions": positions,
                **page_geometry(actual),
            }
        )
        print(f"page target={target} actual={actual} ids={choice.get('token_ids')}", flush=True)

    nested_rows = []
    _, nested_ids = fit_prompt(base, args.model, PAGE * 3 + 1, args.timeout)
    for target in targets:
        prefix = nested_ids[:target]
        body = complete_ids(base, args.model, prefix, 1, args.timeout, args.seed)
        choice = body["choices"][0]
        logprobs = (choice.get("logprobs") or {}).get("top_logprobs") or [{}]
        ranked = sorted(logprobs[0].items(), key=lambda item: item[1], reverse=True)
        top = []
        for piece, logprob in ranked[:2]:
            ids = token_id_for_piece(base, args.model, piece, args.timeout)
            top.append(
                {
                    "piece": piece,
                    "logprob": logprob,
                    "token_id": ids[0] if len(ids) == 1 else None,
                    "single_token": len(ids) == 1,
                }
            )
        margin = top[0]["logprob"] - top[1]["logprob"] if len(top) == 2 else None
        nested_rows.append(
            {
                "requested_context": target,
                "actual_context": len(prefix),
                "nested_prefix": True,
                "serving_path": "completions_token_ids",
                "generated_token_ids": choice.get("token_ids"),
                "top": top,
                "top1_top2_margin_nats": margin,
                **page_geometry(len(prefix)),
            }
        )
        print(
            f"nested target={target} chosen={choice.get('token_ids')} margin={None if margin is None else round(margin, 4)}",
            flush=True,
        )

    payload = {
        "profile": args.profile,
        "model": args.model,
        "seed": args.seed,
        "temperature": 0,
        "thresholds": {
            "small_margin_nats": SMALL_MARGIN_NATS,
            "large_margin_nats": LARGE_MARGIN_NATS,
            "page_proximity_tokens": PAGE_PROXIMITY,
            "page_size": PAGE,
        },
        "mismatches": rows,
        "page_probes": page_rows,
        "nested_page_probes": nested_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "mismatches": len(rows), "pages": len(page_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
