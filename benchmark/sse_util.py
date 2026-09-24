"""Parse OpenAI-compatible SSE lines into content-bearing token events."""

from __future__ import annotations

import json
from typing import Any, Optional


def sse_has_content(line: str) -> bool:
    payload = sse_json_payload(line)
    if payload is None:
        return False
    choices = payload.get("choices") or []
    if not choices:
        return False
    choice = choices[0] if isinstance(choices[0], dict) else {}
    text = choice.get("text")
    if isinstance(text, str) and text:
        return True
    delta = choice.get("delta") or {}
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str) and content:
            return True
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            return True
    message = choice.get("message") or {}
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content:
            return True
    return False


def sse_json_payload(line: str) -> Optional[dict[str, Any]]:
    if not line:
        return None
    raw = line.strip()
    if raw.startswith("data:"):
        raw = raw[5:].strip()
    if not raw or raw == "[DONE]":
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
