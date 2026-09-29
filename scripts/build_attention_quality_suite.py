#!/usr/bin/env python3
"""Build the predeclared deterministic attention-backend quality suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MATH = [
    ("math_00", "Maya has 18 marbles, buys 7, then gives away 9. Integer only.", "16"),
    ("math_01", "Three full buses hold 48 people each and a fourth holds 17. Total people? Integer only.", "161"),
    ("math_02", "A $36 book is discounted by 25%. Whole-dollar price? Integer only.", "27"),
    ("math_03", "Five boxes contain 24 pencils each; 12 are lost. Remaining? Integer only.", "108"),
    ("math_04", "Compute 37 * 41. Integer only.", "1517"),
    ("math_05", "Compute 2^10. Integer only.", "1024"),
    ("math_06", "Greatest common divisor of 48 and 18. Integer only.", "6"),
    ("math_07", "Least common multiple of 4 and 6. Integer only.", "12"),
    ("math_08", "What is 7 factorial? Integer only.", "5040"),
    ("math_09", "Remainder of 100 divided by 9. Integer only.", "1"),
    ("math_10", "Solve 3x + 7 = 31. Integer only.", "8"),
    ("math_11", "What is 15% of 240? Integer only.", "36"),
    ("math_12", "Average of 8, 12, 16, and 20. Integer only.", "14"),
    ("math_13", "A rectangle is 9 by 7. Area? Integer only.", "63"),
    ("math_14", "A cube has side length 4. Volume? Integer only.", "64"),
    ("math_15", "Convert 3.5 hours to minutes. Integer only.", "210"),
    ("math_16", "Simplify 18/24 to lowest terms as a/b.", "3/4"),
    ("math_17", "Compute 19*21. Integer only.", "399"),
    ("math_18", "What is 0.5 as a fraction in lowest terms?", "1/2"),
    ("math_19", "A dozen boxes each hold a dozen items. Total? Integer only.", "144"),
    ("math_20", "How many degrees are in the interior angles of a triangle? Integer only.", "180"),
    ("math_21", "If 5 notebooks cost $30, cost of 8 at the same rate? Whole dollars, integer only.", "48"),
    ("math_22", "The sequence is 2, 6, 18, 54. Next term? Integer only.", "162"),
    ("math_23", "Square root of 2025. Integer only.", "45"),
]

REASONING = [
    ("reasoning_00", "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball in cents? Integer only.", ["5"]),
    ("reasoning_01", "A farmer has 17 sheep. All but 9 die. How many remain? Integer only.", ["9"]),
    ("reasoning_02", "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Yes or no.", ["yes"]),
    ("reasoning_03", "A lily pad doubles daily and covers a lake on day 30. On which day is it half covered? Integer only.", ["29"]),
    ("reasoning_04", "Which is heavier: one pound of feathers or one pound of steel? Answer: same or different.", ["same"]),
    ("reasoning_05", "Five machines make five widgets in five minutes. Minutes for 100 machines to make 100 widgets? Integer only.", ["5"]),
    ("reasoning_06", "Alice is older than Bob; Bob is older than Cara. Is Alice older than Cara? Yes or no.", ["yes"]),
    ("reasoning_07", "Every red object is round. This object is red. Is it round? Yes or no.", ["yes"]),
    ("reasoning_08", "Some cats are black. Must every cat be black? Yes or no.", ["no"]),
    ("reasoning_09", "A train leaves at 14:35 and travels 95 minutes. Arrival in 24-hour HH:MM only.", ["16:10"]),
    ("reasoning_10", "You face north, turn right, then turn around. Final direction only.", ["west"]),
    ("reasoning_11", "Tom is Sam's brother and Sam is Ana's mother. Tom is Ana's what? One word.", ["uncle"]),
    ("reasoning_12", "If yesterday was Monday, what day is tomorrow? One word.", ["wednesday"]),
    ("reasoning_13", "A drawer has only red and blue socks. Minimum socks drawn blind to guarantee a matching color pair? Integer only.", ["3"]),
    ("reasoning_14", "Three switches are all off. Toggle switch 1, then 2, then 1. Which switch is on? Integer only.", ["2"]),
    ("reasoning_15", "A book and pen cost $12 total. The book costs $10 more. Pen cost in whole dollars? Integer only.", ["1"]),
    ("reasoning_16", "If no poets are robots and Ada is a poet, can Ada be a robot? Yes or no.", ["no"]),
    ("reasoning_17", "Sequence A, C, F, J increases gaps by one letter. Next letter only.", ["o"]),
    ("reasoning_18", "There are four corners in a room and one cat in each corner. Each cat sees the other three. Total cats? Integer only.", ["4"]),
    ("reasoning_19", "A clock shows 3:00. Smaller angle between hands in degrees? Integer only.", ["90"]),
]

JSON_TASKS = [
    ("json_00", 'Return only JSON equal to {"status":"ok","count":3}.', {"status": "ok", "count": 3}),
    ("json_01", 'Return only JSON equal to {"ok":true}.', {"ok": True}),
    ("json_02", "Return only a JSON array containing integers 1, 2, 3.", [1, 2, 3]),
    ("json_03", 'Return only JSON equal to {"user":{"id":7}}.', {"user": {"id": 7}}),
    ("json_04", 'Return only JSON equal to {"enabled":false,"extra":null}.', {"enabled": False, "extra": None}),
    ("json_05", 'Return only JSON equal to {"items":[]}.', {"items": []}),
    ("json_06", 'Return only JSON equal to {"tool":"search","args":{"q":"rocm"}}.', {"tool": "search", "args": {"q": "rocm"}}),
    ("json_07", 'Return only JSON equal to {"a":1,"b":2,"c":3}.', {"a": 1, "b": 2, "c": 3}),
    ("json_08", 'Return only JSON equal to {"name":"echo","arguments":{"text":"ping"}}.', {"name": "echo", "arguments": {"text": "ping"}}),
    ("json_09", 'Return only JSON equal to {"temperature":0,"top_p":1}.', {"temperature": 0, "top_p": 1}),
    ("json_10", 'Return only JSON equal to {"ids":[2,4,6]}.', {"ids": [2, 4, 6]}),
    ("json_11", 'Return only JSON equal to {"error":{"code":404,"message":"not found"}}.', {"error": {"code": 404, "message": "not found"}}),
    ("json_12", 'Return only JSON equal to {"role":"admin","active":true}.', {"role": "admin", "active": True}),
    ("json_13", 'Return only JSON equal to {"x":-3,"y":2.5}.', {"x": -3, "y": 2.5}),
    ("json_14", 'Return only JSON equal to {"tags":["amd","rocm"]}.', {"tags": ["amd", "rocm"]}),
    ("json_15", 'Return only JSON equal to {"nested":{"empty":{}}}.', {"nested": {"empty": {}}}),
    ("json_16", 'Return only JSON equal to {"schema":{"type":"object","required":["name"]}}.', {"schema": {"type": "object", "required": ["name"]}}),
    ("json_17", 'Return only JSON equal to {"tool":"sum","args":{"a":2,"b":3}}.', {"tool": "sum", "args": {"a": 2, "b": 3}}),
    ("json_18", 'Return only JSON equal to {"success":true,"data":null}.', {"success": True, "data": None}),
    ("json_19", 'Return only JSON equal to {"messages":[{"role":"user","content":"hi"}]}.', {"messages": [{"role": "user", "content": "hi"}]}),
]

CODE = [
    ("code_00", "square", "Write Python function square(x). Return code only.", [[[3], 9], [[-4], 16], [[0], 0]]),
    ("code_01", "is_even", "Write Python function is_even(n). Return code only.", [[[2], True], [[7], False]]),
    ("code_02", "add", "Write Python function add(a,b). Return code only.", [[[2, 3], 5], [[-2, 5], 3]]),
    ("code_03", "first", "Write Python function first(xs). Return code only.", [[[[4, 5]], 4], [[["a", "b"]], "a"]]),
    ("code_04", "greet", "Write Python function greet(name) returning 'Hello, ' plus name. Return code only.", [[["Ada"], "Hello, Ada"]]),
    ("code_05", "maximum", "Write Python function maximum(a,b). Return code only.", [[[2, 8], 8], [[7, -1], 7]]),
    ("code_06", "length", "Write Python function length(xs). Return code only.", [[[[1, 2, 3]], 3], [[[]], 0]]),
    ("code_07", "negate", "Write Python function negate(flag). Return code only.", [[[True], False], [[False], True]]),
    ("code_09", "clamp", "Write Python function clamp(x,lo,hi). Return code only.", [[[5, 0, 3], 3], [[-1, 0, 3], 0], [[2, 0, 3], 2]]),
    ("code_10", "factorial", "Write iterative Python function factorial(n). Return code only.", [[[0], 1], [[5], 120]]),
    ("code_11", "reverse", "Write Python function reverse(s). Return code only.", [[["abc"], "cba"], [[""], ""]]),
    ("code_12", "count_vowels", "Write Python function count_vowels(s). Return code only.", [[["hello"], 2], [["rhythm"], 0]]),
    ("code_13", "unique", "Write Python function unique(xs) preserving first-seen order. Return code only.", [[[[1, 2, 1, 3]], [1, 2, 3]]]),
    ("code_14", "sum_positive", "Write Python function sum_positive(xs). Return code only.", [[[[-2, 3, 4]], 7], [[[-1]], 0]]),
    ("code_15", "is_palindrome", "Write Python function is_palindrome(s). Return code only.", [[["level"], True], [["hello"], False]]),
    ("code_16", "safe_div", "Write Python function safe_div(a,b) returning None when b is zero. Return code only.", [[[6, 2], 3], [[1, 0], None]]),
    ("code_17", "flatten_one", "Write Python function flatten_one(xss) flattening one list level. Return code only.", [[[[[1, 2], [3]]], [1, 2, 3]]]),
    ("code_18", "word_count", "Write Python function word_count(s) counting whitespace-separated words. Return code only.", [[["a b c"], 3], [["  hi  "], 1]]),
    ("code_19", "sign", "Write Python function sign(x) returning -1, 0, or 1. Return code only.", [[[-4], -1], [[0], 0], [[7], 1]]),
    ("code_20", "merge_dicts", "Write Python function merge_dicts(a,b), with b overriding a. Return code only.", [[[{"x": 1}, {"x": 2, "y": 3}], {"x": 2, "y": 3}]]),
]

REGEX_TASK = {
    "id": "code_08",
    "category": "code",
    "kind": "regex",
    "prompt": "Write a regex that matches an integer. Return only the regex.",
    "positive": ["0", "-1", "42", "+7"],
    "negative": ["1.2", "x3", "", "--2"],
    "max_tokens": 32,
}


def long_prompt(repeats: int, needle: str, position: str) -> str:
    sentence = "Operational notes discuss cooling, PCIe topology, NUMA placement, graph capture, and queue policy. "
    filler = sentence * repeats
    fact = f"RETRIEVAL KEY: {needle}. "
    if position == "start":
        body = fact + filler
    elif position == "middle":
        half = len(filler) // 2
        body = filler[:half] + fact + filler[half:]
    else:
        body = filler + fact
    return f"{body}\nQuestion: What is the retrieval key? Return only the key."


def build() -> list[dict]:
    tasks: list[dict] = []
    tasks.extend(
        {"id": ident, "category": "math", "kind": "answer", "prompt": prompt, "answers": [answer], "max_tokens": 64}
        for ident, prompt, answer in MATH
    )
    tasks.extend(
        {"id": ident, "category": "reasoning", "kind": "answer", "prompt": prompt, "answers": answers, "max_tokens": 128}
        for ident, prompt, answers in REASONING
    )
    tasks.extend(
        {"id": ident, "category": "json_tool", "kind": "json", "prompt": prompt, "expected": expected, "max_tokens": 128, "critical": True}
        for ident, prompt, expected in JSON_TASKS
    )
    tasks.extend(
        {"id": ident, "category": "code", "kind": "python", "prompt": prompt, "function": function, "tests": tests, "max_tokens": 192}
        for ident, function, prompt, tests in CODE
    )
    tasks.append(REGEX_TASK)
    for band, repeats in (("2k", 100), ("8k", 420), ("16k", 820)):
        for index, position in enumerate(("start", "middle", "end", "middle", "end")):
            needle = f"{band}-{position}-{index}-q950"
            tasks.append(
                {
                    "id": f"long_{band}_{index:02d}",
                    "category": "long_context",
                    "kind": "answer",
                    "prompt": long_prompt(repeats, needle, position),
                    "answers": [needle],
                    "band": band,
                    "position": position,
                    "max_tokens": 32,
                }
            )
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tasks = build()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["category"]] = counts.get(task["category"], 0) + 1
    print(json.dumps({"out": str(path), "n": len(tasks), "categories": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
