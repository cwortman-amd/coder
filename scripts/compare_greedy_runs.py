#!/usr/bin/env python3
"""Compare greedy token-ID captures from two serving profiles."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_rows(path: str) -> dict[str, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["id"]: row for row in payload["rows"]}


def first_diff(a: list | None, b: list | None) -> int | None:
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = load_rows(args.baseline)
    cand = load_rows(args.candidate)
    ids = sorted(set(base) & set(cand))
    by_cat = Counter()
    mismatches = []
    exact_tokens = 0
    exact_text = 0
    both_ok = 0
    for ident in ids:
        left, right = base[ident], cand[ident]
        category = left.get("category") or "unknown"
        if not left.get("ok") or not right.get("ok"):
            mismatches.append({"id": ident, "category": category, "reason": "request_failed"})
            continue
        both_ok += 1
        token_equal = left.get("token_ids") == right.get("token_ids")
        text_equal = left.get("content") == right.get("content")
        if token_equal:
            exact_tokens += 1
        if text_equal:
            exact_text += 1
        if token_equal:
            by_cat[f"{category}:match"] += 1
        else:
            by_cat[f"{category}:mismatch"] += 1
            mismatches.append(
                {
                    "id": ident,
                    "category": category,
                    "reason": "token_mismatch" if not token_equal else "text_only_mismatch",
                    "first_diff_index": first_diff(left.get("token_ids"), right.get("token_ids")),
                    "baseline_tokens": (left.get("token_ids") or [])[:32],
                    "candidate_tokens": (right.get("token_ids") or [])[:32],
                    "baseline_text": (left.get("content") or "")[:240],
                    "candidate_text": (right.get("content") or "")[:240],
                    "baseline_finish": left.get("finish_reason"),
                    "candidate_finish": right.get("finish_reason"),
                }
            )

    summary = {
        "n_compared": len(ids),
        "both_ok": both_ok,
        "exact_token_id_matches": exact_tokens,
        "exact_text_matches": exact_text,
        "token_match_rate": (exact_tokens / both_ok) if both_ok else None,
        "text_match_rate": (exact_text / both_ok) if both_ok else None,
        "by_category": dict(sorted(by_cat.items())),
        "mismatches": mismatches,
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "mismatches"}, indent=2))
    if mismatches:
        print("mismatches", len(mismatches))
        for row in mismatches[:12]:
            print(row["id"], row["reason"], "diff@", row.get("first_diff_index"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
