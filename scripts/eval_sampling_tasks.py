#!/usr/bin/env python3
"""Small production-sampling task suite for an OpenAI-compatible chat server."""
from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen


MATH_TASKS = [
    ("math_gsm_00", "Maya has 18 marbles, buys 7, then gives away 9. How many remain? End with only the integer.", "16"),
    ("math_gsm_01", "A bus holds 48 people. Three buses are full and 17 people take a fourth. How many people total? End with only the integer.", "161"),
    ("math_gsm_02", "A $36 book is discounted by 25%. What is its price in whole dollars? End with only the integer.", "27"),
    ("math_gsm_03", "Five boxes each contain 24 pencils. Twelve pencils are lost. How many remain? End with only the integer.", "108"),
    ("math_gsm_04", "Lina reads 15 pages daily for 6 days, then 10 pages on day 7. How many pages total? End with only the integer.", "100"),
    ("math_gsm_05", "A recipe needs 3 eggs per cake. How many eggs are needed for 14 cakes? End with only the integer.", "42"),
    ("math_gsm_06", "Sam had $100 and bought 3 shirts at $18 each. How many whole dollars remain? End with only the integer.", "46"),
    ("math_gsm_07", "A 96-meter rope is cut into 8 equal pieces. How many meters is each piece? End with only the integer.", "12"),
]

JSON_TASKS = [
    ("json_00", 'Return only JSON equal to {"status":"ok","count":3}.', {"status": "ok", "count": 3}),
    ("json_01", 'Return only JSON equal to {"ok":true}.', {"ok": True}),
    ("json_02", "Return only a JSON array containing the integers 1, 2, and 3.", [1, 2, 3]),
    ("json_03", 'Return only JSON with user.id equal to 7: {"user":{"id":7}}.', {"user": {"id": 7}}),
    ("json_04", 'Return only JSON equal to {"enabled":false,"extra":null}.', {"enabled": False, "extra": None}),
    ("json_05", 'Return only JSON equal to {"items":[]}.', {"items": []}),
    ("json_06", 'Return only JSON equal to {"tool":"search","args":{"q":"rocm"}}.', {"tool": "search", "args": {"q": "rocm"}}),
    ("json_07", 'Return only JSON equal to {"a":1,"b":2,"c":3}.', {"a": 1, "b": 2, "c": 3}),
]

CODE_TASKS = [
    ("code_00", "Write a Python function square(x) that returns x squared. Return only code.", "square", ["Pow", "Mult"]),
    ("code_01", "Write a Python function is_even(n) returning whether n is even. Return only code.", "is_even", ["Mod"]),
    ("code_02", "Write a Python function add(a, b) returning their sum. Return only code.", "add", ["Add"]),
    ("code_03", "Write a Python function first(xs) returning the first item. Return only code.", "first", ["Subscript"]),
    ("code_04", "Write a Python function greet(name) returning 'Hello, ' plus name. Return only code.", "greet", ["Add", "JoinedStr"]),
    ("code_05", "Write a Python function maximum(a, b) returning the larger value. Return only code.", "maximum", ["Compare", "Call", "IfExp"]),
    ("code_06", "Write a Python function length(xs) returning its length. Return only code.", "length", ["Call"]),
    ("code_07", "Write a Python function negate(flag) returning logical not of flag. Return only code.", "negate", ["Not"]),
]

CORPUS_IDS = [
    "multilingual_01", "multilingual_04", "multilingual_07", "multilingual_09",
    "reasoning_00", "reasoning_02", "reasoning_03", "reasoning_07",
]
CORPUS_ANSWERS = {
    "multilingual_01": ["buenos dias"],
    "multilingual_04": ["grazie"],
    "multilingual_07": ["привет"],
    "multilingual_09": ["inferenz", "schlussfolgerung"],
    "reasoning_00": ["5"],
    "reasoning_02": ["yes"],
    "reasoning_03": ["9"],
    "reasoning_07": ["29"],
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode() or text
    return re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE).strip()


def clean_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.fullmatch(r"\s*```(?:python|json)?\s*(.*?)\s*```\s*", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def stream_complete(base: str, payload: dict, timeout: int) -> dict:
    req = Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at = None
    chunks: list[str] = []
    finish_reason = None
    usage: dict = {}
    token_ids: list[int] = []
    with urlopen(req, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunks.append(piece)
                token_ids.extend(choice.get("token_ids") or delta.get("token_ids") or [])
                finish_reason = choice.get("finish_reason") or finish_reason
    ended = time.perf_counter()
    return {
        "content": "".join(chunks),
        "ttft_s": None if first_token_at is None else first_token_at - started,
        "latency_s": ended - started,
        "finish_reason": finish_reason,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "token_ids": token_ids or None,
    }


def parse_json(text: str) -> tuple[bool, object | None]:
    try:
        return True, json.loads(clean_output(text))
    except (json.JSONDecodeError, TypeError):
        return False, None


def score_record(task: dict, content: str) -> tuple[bool, bool, str]:
    cleaned = clean_output(content)
    if task["category"] == "math":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        correct = bool(numbers) and numbers[-1] == task["answer"]
        return correct, bool(numbers), numbers[-1] if numbers else ""
    if task["category"] == "json":
        valid, parsed = parse_json(cleaned)
        return valid and parsed == task["answer"], valid, repr(parsed)
    if task["category"] == "code":
        try:
            tree = ast.parse(cleaned)
        except SyntaxError as exc:
            return False, False, str(exc)
        names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        op_present = any(type(node).__name__ in task["operators"] for node in ast.walk(tree))
        return task["function"] in names and op_present, True, f"functions={sorted(names)}"
    answer = normalize(cleaned)
    expected = [normalize(value) for value in task["answers"]]
    # Reasoning prompts request a leading short answer; tolerate an explanation
    # after that answer while preserving exact matching for multilingual prompts.
    correct = any(
        value == answer or (task["category"] == "reasoning" and answer.startswith(value + " "))
        for value in expected
    )
    return correct, bool(cleaned), answer


def load_tasks(corpus_path: Path) -> list[dict]:
    tasks = [
        {"id": ident, "category": "math", "prompt": prompt, "answer": answer}
        for ident, prompt, answer in MATH_TASKS
    ]
    tasks += [
        {"id": ident, "category": "json", "prompt": prompt, "answer": answer}
        for ident, prompt, answer in JSON_TASKS
    ]
    tasks += [
        {"id": ident, "category": "code", "prompt": prompt, "function": function, "operators": operators}
        for ident, prompt, function, operators in CODE_TASKS
    ]
    corpus = {
        row["id"]: row
        for row in (json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines())
    }
    for ident in CORPUS_IDS:
        row = corpus[ident]
        tasks.append({
            "id": ident,
            "category": row["category"],
            "prompt": row["prompt"],
            "answers": CORPUS_ANSWERS[ident],
        })
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="awq")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--corpus", default="_results/quality/corpus.jsonl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tasks = load_tasks(Path(args.corpus))
    if args.limit:
        tasks = tasks[:args.limit]
    rows = []
    failures = 0
    for index, task in enumerate(tasks, 1):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": task["prompt"]}],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            result = stream_complete(args.base_url, payload, args.timeout)
            correct, valid, scored_value = score_record(task, result["content"])
            record = {
                "id": task["id"], "category": task["category"], "ok": True,
                "correct": correct, "valid": valid, "scored_value": scored_value, **result,
            }
        except Exception as exc:  # noqa: BLE001
            failures += 1
            record = {
                "id": task["id"], "category": task["category"], "ok": False,
                "correct": False, "valid": False, "error": str(exc),
            }
        rows.append(record)
        print(
            f"[{index:02d}/{len(tasks)}] {task['id']}: "
            f"{'PASS' if record['correct'] else 'FAIL'} ttft={record.get('ttft_s')}",
            flush=True,
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    def metrics(items: list[dict]) -> dict:
        ttfts = [row["ttft_s"] for row in items if row.get("ttft_s") is not None]
        return {
            "n": len(items),
            "correct": sum(bool(row["correct"]) for row in items),
            "accuracy": sum(bool(row["correct"]) for row in items) / len(items),
            "valid": sum(bool(row["valid"]) for row in items),
            "validity": sum(bool(row["valid"]) for row in items) / len(items),
            "ttft_mean_s": statistics.fmean(ttfts) if ttfts else None,
            "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
        }

    output = {
        "profile": args.profile,
        "model": args.model,
        "endpoint": args.base_url,
        "sampling": {
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "seed": args.seed, "max_tokens": args.max_tokens, "enable_thinking": False,
            "stream": True,
        },
        "suite": "32 small single-sample tasks; code score is static syntax/shape, not execution",
        "n": len(rows),
        "failed_requests": failures,
        "summary": metrics(rows),
        "by_category": {category: metrics(items) for category, items in grouped.items()},
        "rows": rows,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(path), "failed_requests": failures, **output["summary"]}))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
