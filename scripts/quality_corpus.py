#!/usr/bin/env python3
"""Build a text-only greedy-equality corpus for MXFP4 vs DFlash/MTP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SHORT_CHAT = [
    "Reply with exactly: MI350P READY",
    "Say yes or no: is 2 an even number?",
    "Translate 'The system is ready' into French. Return only the translation.",
    "What is the capital of Japan? Answer with only the city name.",
    "List three primary colors, comma-separated, nothing else.",
    "Rewrite this sentence in past tense: The server starts quickly.",
    "Give a one-sentence definition of latency.",
    "Is water H2O? Answer yes or no.",
    "Name the planet closest to the Sun.",
    "Convert 3 hours to minutes. Integer only.",
    "What day follows Tuesday? One word.",
    "Spell the word 'queue' backwards.",
    "Return the ISO 639-1 code for English.",
    "Who wrote Hamlet? Last name only.",
    "Give the chemical symbol for gold.",
    "What is 9 + 10? Integer only.",
    "Is Python a compiled-only language? Yes or no.",
    "Name one HTTP method used to retrieve a resource.",
    "What timezone abbreviation is Coordinated Universal Time?",
    "Complete the proverb: Look before you ____.",
]

CODE = [
    "Write a Python function named square that returns the square of its argument. Return only code.",
    "Write a Python one-liner that reverses the string s. Return only the expression.",
    "Write a bash command that prints the current working directory. Command only.",
    "Write a Python function is_even(n) that returns True for even integers. Code only.",
    "Fix this Python: def add(a,b) return a+b. Return the corrected function only.",
    "Write a SQL query that selects name from users where id = 1. SQL only.",
    "Write a JavaScript function that returns the length of a string x. Code only.",
    "Write a Python list comprehension of squares of 0..4. Expression only.",
    "Write a regex that matches an integer. Return only the regex.",
    "Write a Python dataclass named Point with x: int and y: int. Code only.",
    "Write a Go function Sum(a, b int) int that returns a+b. Code only.",
    "Write a Python type hint for a function that maps str to int. Signature only.",
    "Write a C function int min2(int a, int b) that returns the smaller value. Code only.",
    "Write a Python generator that yields 1, 2, 3. Code only.",
    "Write a Rust function add(a: i32, b: i32) -> i32. Code only.",
    "Write a Python context manager using contextlib.contextmanager that yields None. Code only.",
    "Write an HTML paragraph containing Hello. Markup only.",
    "Write a Dockerfile line that sets WORKDIR to /app. Line only.",
    "Write a Python unittest that asserts 1+1==2. Code only.",
    "Write a recursive Python factorial function named fact. Code only.",
]

JSON_TOOL = [
    "Return a compact JSON object with keys status and count, values ok and 3. JSON only.",
    'Return JSON {"ok": true}. JSON only.',
    "Return a JSON array of the first three positive integers. JSON only.",
    'Return JSON with keys name and role, values "ada" and "admin". JSON only.',
    "Return a JSON object with a nested key user.id equal to 7. JSON only.",
    "Return JSON {\"tool\": \"search\", \"args\": {\"q\": \"rocm\"}}. JSON only.",
    "Return a JSON list of objects each with id 1 then 2. JSON only.",
    "Return JSON with boolean false under key enabled. JSON only.",
    "Return JSON with a null value under key extra. JSON only.",
    "Return JSON {\"items\": []}. JSON only.",
    "Return a JSON object representing an error with code 404 and message not found. JSON only.",
    "Return JSON with keys a, b, c equal to 1, 2, 3. JSON only.",
    "Return a JSON schema-like object with type object and required [name]. JSON only.",
    "Return JSON for a tool call: name echo, argument text ping. JSON only.",
    "Return JSON {\"temperature\": 0, \"top_p\": 1}. JSON only.",
]

MATH = [
    "Compute 37 * 41. Give only the integer.",
    "Compute 128 / 4. Integer only.",
    "What is 2^10? Integer only.",
    "What is 15% of 200? Integer only.",
    "Solve for x: 2x + 3 = 11. Integer only.",
    "Greatest common divisor of 48 and 18. Integer only.",
    "Least common multiple of 4 and 6. Integer only.",
    "What is 7 factorial? Integer only.",
    "Remainder of 100 divided by 9. Integer only.",
    "Absolute value of -42. Integer only.",
    "Round 3.7 to the nearest integer.",
    "How many degrees in a right angle? Integer only.",
    "What is the 6th Fibonacci number if F1=1, F2=1? Integer only.",
    "Convert 5 km to meters. Integer only.",
    "If a=3 and b=4, what is a^2+b^2? Integer only.",
    "What is 111 + 222? Integer only.",
    "Simplify 18/24 to lowest terms as a/b.",
    "How many sides does a hexagon have? Integer only.",
    "What is 0.5 as a fraction in lowest terms?",
    "Compute 19*21. Integer only.",
]

RAG = [
    "Context: The gateway binds to 127.0.0.1:8000. Question: Which host does the gateway bind? Answer with the host only.",
    "Context: Quark AWQ MXFP4 is the serving checkpoint. Question: What quantization is used? Short phrase.",
    "Context: DFlash2 drafts seven tokens per block. Question: How many draft tokens per block? Integer only.",
    "Context: MI350P has 144 GB HBM. Question: How much HBM? Include the unit.",
    "Context: Tensor parallel size is 1. Question: What is TP? Integer only.",
    "Context: Prefix caching is enabled by default in this engine. Question: Is prefix caching enabled? Yes or no.",
    "Context: The vision encoder remains BF16. Question: What dtype is vision? Token only.",
    "Context: Native MTP failed a shape assertion in qwen3_5_mtp.py. Question: Which file asserted? Filename only.",
    "Context: Saturated serving uses no speculation. Question: Does the saturated profile use DFlash? Yes or no.",
    "Context: max-model-len is 16384. Question: What is the max model length? Integer only.",
    "Context: The draft checkpoint is 12 shards totaling 3.58 GiB. Question: How many shards? Integer only.",
    "Context: Attention backend auto-selected ROCM_ATTN. Question: Which attention backend? Token only.",
    "Context: ignore_eos is for throughput tests only. Question: Should correctness tests set ignore_eos? Yes or no.",
    "Context: The serving name is awq. Question: What is the served-model-name? Token only.",
    "Context: Graph mode FULL_AND_PIECEWISE is the default O2 path. Question: Which graph mode? Token only.",
]

MULTILINGUAL = [
    "Responde con una sola palabra: hola.",
    "Traduce 'good morning' al español. Solo la traducción.",
    "用一个汉字回答：水。",
    "Antworten Sie mit einem Wort: ja.",
    "Traduci 'thank you' in italiano. Solo la traduzione.",
    "日本語で「はい」とだけ答えてください。",
    "Réponds par un seul mot : d'accord.",
    "Переведи 'hello' на русский. Только перевод.",
    "Responda em português com uma palavra: sim.",
    "Translate 'inference' into German. One word.",
]

REASONING = [
    "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball in cents? Integer only.",
    "There are 3 boxes. One is labeled apples, one oranges, one apples-and-oranges, all labels wrong. You draw an apple from the box labeled apples-and-oranges. What is actually in the box labeled apples? One word: apples, oranges, or mixed.",
    "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Yes or no.",
    "A farmer has 17 sheep. All but 9 die. How many are left? Integer only.",
    "You have a 3-liter and 5-liter jug. How can you measure 4 liters? Give the shortest sequence of fills/pours in one sentence.",
    "Which is heavier: a pound of feathers or a pound of steel? One short sentence.",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets? Minutes, integer only.",
    "A lily pad doubles every day and covers the lake on day 30. On which day is it half covered? Integer only.",
    "True or false: a valid JSON object may have duplicate keys according to RFC 8259. One word.",
    "If n is even, n^2 is even. Contrapositive in one sentence.",
]


def long_context(n_words: int, needle: str, question: str) -> str:
    filler = ("The cluster description repeats operational notes about cooling, PCIe, and NUMA locality. ") * n_words
    return (
        f"You will be asked about a hidden fact.\n\n{filler}\n\n"
        f"HIDDEN FACT: {needle}\n\n{filler}\n\nQuestion: {question} Answer with only the fact value."
    )


def build() -> list[dict]:
    rows: list[dict] = []
    for category, prompts in (
        ("short_chat", SHORT_CHAT),
        ("code", CODE),
        ("json_tool", JSON_TOOL),
        ("math", MATH),
        ("rag", RAG),
        ("multilingual", MULTILINGUAL),
        ("reasoning", REASONING),
    ):
        for i, prompt in enumerate(prompts):
            rows.append(
                {
                    "id": f"{category}_{i:02d}",
                    "category": category,
                    "max_tokens": 128,
                    "prompt": prompt,
                }
            )
    long_specs = [
        ("long_2k_00", 80, "reservation-alpha", "What is the hidden fact?"),
        ("long_2k_01", 80, "ticket-1842", "What is the hidden fact?"),
        ("long_8k_00", 420, "needle-8k-blue", "What is the hidden fact?"),
        ("long_8k_01", 420, "needle-8k-green", "What is the hidden fact?"),
        ("long_8k_02", 420, "needle-8k-red", "What is the hidden fact?"),
        ("long_8k_03", 420, "needle-8k-gold", "What is the hidden fact?"),
    ]
    for ident, n_words, needle, question in long_specs:
        rows.append(
            {
                "id": ident,
                "category": "long_context",
                "max_tokens": 32,
                "prompt": long_context(n_words, needle, question),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = build()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    print(json.dumps({"n": len(rows), "by_category": counts, "out": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
