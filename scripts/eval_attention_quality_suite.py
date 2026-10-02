#!/usr/bin/env python3
"""Run and score the predeclared paired attention-backend quality suite."""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_http import post_chat, strip_think  # noqa: E402


def clean(text: str) -> str:
    text = strip_think(text)
    match = re.fullmatch(
        r"\s*```(?:python|json|regex)?\s*(.*?)\s*```\s*",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return (match.group(1) if match else text).strip()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", clean(text)).casefold().strip()
    return re.sub(r"\s+", " ", text)


def complete(base: str, model: str, task: dict, args) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "max_tokens": int(task.get("max_tokens", args.max_tokens)),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "stream": False,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body, elapsed = post_chat(base, payload, args.timeout)
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    return {
        "content": (choice.get("message") or {}).get("content") or "",
        "token_ids": choice.get("token_ids"),
        "finish_reason": choice.get("finish_reason"),
        "latency_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def score_answer(task: dict, content: str) -> tuple[bool, bool, str]:
    value = normalize(content)
    expected = [normalize(item) for item in task["answers"]]
    if task["category"] == "math":
        candidates = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", value)
        scored = candidates[-1] if candidates else ""
        return scored in expected, bool(candidates), scored
    return value in expected, bool(value), value


def validate_python(tree: ast.AST) -> str | None:
    forbidden_names = {"__import__", "eval", "exec", "compile", "open", "input"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            return type(node).__name__
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            return node.id
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return node.attr
    return None


def score_python(task: dict, content: str) -> tuple[bool, bool, str]:
    code = clean(content)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, False, f"syntax:{exc.msg}"
    reason = validate_python(tree)
    if reason:
        return False, False, f"forbidden:{reason}"
    functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    if task["function"] not in functions:
        return False, True, f"missing:{task['function']}"
    # Python literals are required here: JSON true/null are not valid Python.
    tests = repr(task["tests"])
    harness = (
        code
        + "\n"
        + f"_tests = {tests}\n"
        + f"for _args, _expected in _tests:\n"
        + f"    _actual = {task['function']}(*_args)\n"
        + "    assert _actual == _expected, repr((_args, _actual, _expected))\n"
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(harness, encoding="utf-8")
        try:
            run = subprocess.run(
                ["python3", "-I", "-S", str(path)],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
                env={"PATH": "/usr/bin:/bin"},
            )
        except subprocess.TimeoutExpired:
            return False, True, "timeout"
    return run.returncode == 0, True, (run.stderr or "tests_passed")[-500:]


def score_regex(task: dict, content: str) -> tuple[bool, bool, str]:
    pattern = clean(content).strip()
    if pattern.startswith("/") and pattern.endswith("/") and len(pattern) > 2:
        pattern = pattern[1:-1]
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return False, False, str(exc)
    positive = all(compiled.fullmatch(value) for value in task["positive"])
    negative = all(not compiled.fullmatch(value) for value in task["negative"])
    return positive and negative, True, pattern


def score(task: dict, content: str) -> tuple[bool, bool, str]:
    kind = task["kind"]
    if kind == "answer":
        return score_answer(task, content)
    if kind == "json":
        try:
            parsed = json.loads(clean(content))
        except json.JSONDecodeError as exc:
            return False, False, str(exc)
        return parsed == task["expected"], True, repr(parsed)
    if kind == "python":
        return score_python(task, content)
    if kind == "regex":
        return score_regex(task, content)
    raise ValueError(kind)


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "valid": sum(bool(row["valid"]) for row in rows),
        "validity": sum(bool(row["valid"]) for row in rows) / len(rows),
        "failed_requests": sum(not bool(row["ok"]) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="awq")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--categories",
        default="",
        help="Comma-separated category filter; empty runs the full suite.",
    )
    args = parser.parse_args()

    tasks = [
        json.loads(line)
        for line in Path(args.suite).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.categories:
        selected = {value.strip() for value in args.categories.split(",") if value.strip()}
        tasks = [task for task in tasks if task["category"] in selected]
    rows = []
    for index, task in enumerate(tasks, 1):
        try:
            response = complete(args.base_url, args.model, task, args)
            correct, valid, scored = score(task, response["content"])
            row = {
                "id": task["id"],
                "category": task["category"],
                "kind": task["kind"],
                "band": task.get("band"),
                "position": task.get("position"),
                "critical": bool(task.get("critical")),
                "ok": True,
                "correct": correct,
                "valid": valid,
                "scored": scored,
                **response,
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                "id": task["id"],
                "category": task["category"],
                "kind": task["kind"],
                "band": task.get("band"),
                "position": task.get("position"),
                "critical": bool(task.get("critical")),
                "ok": False,
                "correct": False,
                "valid": False,
                "error": str(exc),
            }
        rows.append(row)
        print(
            f"[{index:03d}/{len(tasks)}] {task['id']}: "
            f"{'PASS' if row['correct'] else 'FAIL'} "
            f"prompt={row.get('prompt_tokens')} completion={row.get('completion_tokens')}",
            flush=True,
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    result = {
        "profile": args.profile,
        "model": args.model,
        "suite": args.suite,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "enable_thinking": False,
        },
        "summary": summarize(rows),
        "by_category": {
            category: summarize(items) for category, items in sorted(grouped.items())
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(out), **result["summary"]}))
    return 0 if result["summary"]["failed_requests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
