#!/usr/bin/env python3
"""One OpenAI-compatible streaming client for the latency benches.

Callers choose an event mode. The mode is the measurement definition, so a
content-only sweep does not silently start counting reasoning tokens.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from request_event_schema import RequestEvent


class EventMode(str, Enum):
    """Which stream events become token timestamps."""

    CONTENT = "content"
    REASONING_AND_ANSWER = "reasoning_and_answer"
    ANY_SSE = "any_sse"
    CHUNK = "chunk"
    TOKEN_IDS = "token_ids"


@dataclass
class StreamResult:
    event: RequestEvent
    generated_text: str = ""
    reasoning_text: str = ""
    prompt_tokens: int = 0
    chunk_timestamps_ns: List[int] = field(default_factory=list)


def stream_chat(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    mode: EventMode,
    max_tokens: int = 64,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
    request_id: str = "",
    scheduled_send_ns: int = 0,
    prompt_length_bucket: str = "",
    output_length_bucket: str = "",
    extra_payload: Optional[Dict[str, Any]] = None,
) -> StreamResult:
    """Send one streaming chat completion and return a RequestEvent."""
    if scheduled_send_ns > time.time_ns():
        time.sleep((scheduled_send_ns - time.time_ns()) / 1e9)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra_payload:
        payload.update(extra_payload)

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    actual_send_ns = time.time_ns()
    event = RequestEvent(
        request_id=request_id,
        model=model,
        prompt_length_bucket=prompt_length_bucket,
        output_length_bucket=output_length_bucket or f"max_{max_tokens}",
        scheduled_send_ns=scheduled_send_ns,
        actual_send_ns=actual_send_ns,
    )
    result = StreamResult(event=event)
    token_timestamps: List[int] = []
    in_think_block = False

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                now_ns = time.time_ns()
                if mode is EventMode.ANY_SSE:
                    token_timestamps.append(now_ns)
                    continue
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if mode is EventMode.CHUNK:
                    result.chunk_timestamps_ns.append(now_ns)
                _consume_usage(chunk, event, result)
                in_think_block = _record_chunk(
                    mode, chunk, now_ns, token_timestamps, result, event, in_think_block
                )
    except Exception as exc:
        event.status = "error"
        event.error_message = str(exc)
        event.finish_reason = "error"
        event.stream_end_ns = time.time_ns()
        event.completion_or_timeout_ns = event.stream_end_ns
        event.token_timestamps_ns = token_timestamps
        if token_timestamps and event.first_output_token_ns is None:
            event.first_output_token_ns = token_timestamps[0]
            event.first_streamed_output_ns = token_timestamps[0]
        return result

    event.status = "completed"
    event.token_timestamps_ns = token_timestamps
    if event.first_output_token_ns is None and token_timestamps:
        event.first_output_token_ns = token_timestamps[0]
    event.first_streamed_output_ns = event.first_output_token_ns
    if event.usage_source != "server_usage_chunk":
        counted = len(token_timestamps)
        event.completion_tokens = counted
        event.actual_output_tokens = counted
    event.stream_end_ns = time.time_ns()
    event.completion_or_timeout_ns = event.stream_end_ns
    return result


def _consume_usage(chunk: Dict[str, Any], event: RequestEvent, result: StreamResult) -> None:
    usage = chunk.get("usage") or {}
    if not usage:
        return
    if usage.get("completion_tokens") is not None:
        event.completion_tokens = int(usage["completion_tokens"])
        event.actual_output_tokens = event.completion_tokens
    if usage.get("prompt_tokens") is not None:
        result.prompt_tokens = int(usage["prompt_tokens"])
    event.usage_source = "server_usage_chunk"


def _record_chunk(
    mode: EventMode,
    chunk: Dict[str, Any],
    now_ns: int,
    token_timestamps: List[int],
    result: StreamResult,
    event: RequestEvent,
    in_think_block: bool,
) -> bool:
    choices = chunk.get("choices") or []
    if not choices:
        return in_think_block
    choice = choices[0]
    if choice.get("finish_reason"):
        event.finish_reason = choice["finish_reason"]
    delta = choice.get("delta") or {}

    if mode is EventMode.CHUNK:
        token_ids = choice.get("token_ids")
        if token_ids:
            token_timestamps.extend([now_ns] * len(token_ids))
        elif delta.get("content") or delta.get("text"):
            token_timestamps.append(now_ns)
        return in_think_block

    if mode is EventMode.TOKEN_IDS:
        token_ids = choice.get("token_ids")
        if token_ids:
            result.chunk_timestamps_ns.append(now_ns)
            token_timestamps.extend([now_ns] * len(token_ids))
        return in_think_block

    if mode is EventMode.CONTENT:
        content = delta.get("content") or ""
        if content:
            token_timestamps.append(now_ns)
            result.generated_text += content
        elif choice.get("finish_reason") and event.first_output_token_ns is None and not token_timestamps:
            event.first_output_token_ns = now_ns
        return in_think_block

    reasoning = delta.get("reasoning_content") or ""
    content = delta.get("content") or ""
    if reasoning:
        if event.first_output_token_ns is None:
            event.first_output_token_ns = now_ns
        result.reasoning_text += reasoning
    if content:
        if event.first_output_token_ns is None:
            event.first_output_token_ns = now_ns
        if "<think>" in content:
            in_think_block = True
        if "</think>" in content:
            in_think_block = False
            if event.first_answer_token_ns is None:
                event.first_answer_token_ns = now_ns
        if (
            not in_think_block
            and event.first_answer_token_ns is None
            and content.strip()
            and not content.startswith("<think>")
        ):
            event.first_answer_token_ns = now_ns
        if in_think_block:
            result.reasoning_text += content
        else:
            result.generated_text += content
            token_timestamps.append(now_ns)
    elif choice.get("finish_reason") and event.first_output_token_ns is None:
        event.first_output_token_ns = now_ns
    return in_think_block
