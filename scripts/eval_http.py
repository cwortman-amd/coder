"""Non-streaming chat POST and thinking-tag strip for the quality evals."""

from __future__ import annotations

import json
import re
import time
from urllib.request import Request, urlopen


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def post_chat(base: str, payload: dict, timeout: int) -> tuple[dict, float]:
    request = Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    return body, time.perf_counter() - started
