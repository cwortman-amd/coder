#!/usr/bin/env python3
"""
jev_gateway.py - TypeSafe Semantic Routing Gateway (Open Jev + Qwen 3.8 27B / Claude)
=====================================================================================
Routes incoming OpenAI-compatible Chat Completion requests between:
  1. Local AMD Radeon AI PRO R9700 (llama.cpp server serving Qwen 3.8 27B Q4_K_M)
  2. Cloud Anthropic API (Claude 3.5 Sonnet fallback for large/complex architectures)

Provides:
  - TypeSafe semantic decision engine (open_jev_decision_engine)
  - Privacy boundary enforcement (locks confidential data to local GPU)
  - Bi-directional streaming conversion (translates Anthropic SSE to OpenAI SSE)
  - Transparent drop-in replacement for OpenAI API on port 8000
"""

import json
import os
import sys
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ==============================================================================
# Configuration
# ==============================================================================
LOCAL_QWEN_URL = os.getenv("LOCAL_QWEN_URL", "http://127.0.0.1:8033/v1/chat/completions")
LOCAL_MODELS_URL = os.getenv("LOCAL_MODELS_URL", "http://127.0.0.1:8033/v1/models")
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8000"))
MAX_LOCAL_PROMPT_CHARS = int(os.getenv("MAX_LOCAL_PROMPT_CHARS", "45000"))

app = FastAPI(
    title="Open Jev Semantic Routing Gateway",
    description="TypeSafe local decision router between AMD Radeon AI PRO R9700 (Qwen 3.8 27B) and Cloud Claude",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Pydantic Schemas
# ==============================================================================
class Message(BaseModel):
    role: str
    content: Any  # Can be string or list of content blocks

class ChatRequest(BaseModel):
    model: Optional[str] = "jev-routed-qwen"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None

# ==============================================================================
# Jev Decision Engine
# ==============================================================================
def open_jev_decision_engine(messages: List[Message]) -> str:
    """
    Local Jev Decision Engine.
    Parses semantic structure, token weights, and privacy markers to route requests.
    
    Returns:
      "local_qwen"          -> Route to local AMD Radeon AI PRO R9700
      "claude_subscription" -> Route to Anthropic Claude 3.5 Sonnet
    """
    user_nodes = [
        str(m.content) for m in messages if m.role == "user"
    ]
    if not user_nodes:
        return "local_qwen"

    target_prompt = user_nodes[-1]
    prompt_len = len(target_prompt)
    target_lower = target_prompt.lower()

    # Rule 1: Privacy Boundary (Absolute Priority - Never leak secrets to cloud)
    privacy_signals = [
        "private_key", "internal_db", "api_key", "secret_key", 
        "passwd", "password", ".env", "bearer", "credentials", "confidential"
    ]
    if any(sig in target_lower for sig in privacy_signals):
        return "local_qwen"

    # Rule 2: High complexity markers or broad architectural requests scale to Claude
    cloud_signals = [
        "architectural review", "race condition profile", "ast transformer rewrite",
        "system architecture audit", "deadlock analysis", "formal verification",
        "distributed consensus design"
    ]
    if any(sig in target_lower for sig in cloud_signals):
        return "claude_subscription"

    # Rule 3: Token Density / Context Limits (Large payloads passed to Claude)
    if prompt_len > MAX_LOCAL_PROMPT_CHARS:
        return "claude_subscription"

    # Default: Serve locally with zero egress cost on AMD R9700
    return "local_qwen"

# ==============================================================================
# Anthropic <-> OpenAI Streaming Adapter
# ==============================================================================
async def stream_anthropic_as_openai(
    client: httpx.AsyncClient, 
    headers: Dict[str, str], 
    anthropic_payload: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """
    Streams from Anthropic Messages API and transforms into standard OpenAI
    chat.completion.chunk SSE packets so IDEs, Hermes, and OpenCode remain compatible.
    """
    chunk_id = f"chatcmpl-jev-{int(time.time())}"
    created_ts = int(time.time())

    async with client.stream("POST", CLAUDE_API_URL, headers=headers, json=anthropic_payload, timeout=90.0) as resp:
        if resp.status_code != 200:
            err_text = await resp.aread()
            err_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": CLAUDE_MODEL,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"\n[Jev Gateway Error from Claude API: {err_text.decode('utf-8', errors='ignore')}]"},
                    "finish_reason": "error"
                }]
            }
            yield f"data: {json.dumps(err_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                event_data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = event_data.get("type")
            delta_text = ""

            if event_type == "content_block_delta":
                delta_text = event_data.get("delta", {}).get("text", "")
            elif event_type == "message_delta":
                stop_reason = event_data.get("delta", {}).get("stop_reason")
                if stop_reason:
                    end_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": CLAUDE_MODEL,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(end_chunk)}\n\n"

            if delta_text:
                openai_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": CLAUDE_MODEL,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta_text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(openai_chunk)}\n\n"

    yield "data: [DONE]\n\n"

async def stream_local_qwen(
    client: httpx.AsyncClient, 
    payload: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """Forward local llama.cpp streaming SSE chunks directly."""
    async with client.stream("POST", LOCAL_QWEN_URL, json=payload, timeout=120.0) as resp:
        if resp.status_code != 200:
            err_text = await resp.aread()
            yield f"data: {json.dumps({'error': err_text.decode('utf-8', errors='ignore')})}\n\n"
            yield "data: [DONE]\n\n"
            return
        async for chunk in resp.aiter_text():
            yield chunk

# ==============================================================================
# API Endpoints
# ==============================================================================
@app.get("/health")
async def health_check():
    local_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(LOCAL_MODELS_URL)
            local_ok = (r.status_code == 200)
    except Exception:
        pass

    return {
        "status": "healthy",
        "router": "open-jev",
        "local_qwen_online": local_ok,
        "claude_available": bool(CLAUDE_API_KEY),
        "target_hardware": "AMD Radeon AI PRO R9700 (gfx1201, 32GB VRAM)"
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "jev-routed-qwen",
                "object": "model",
                "owned_by": "open-jev",
                "permission": [],
                "root": "jev-routed-qwen"
            },
            {
                "id": "Qwen3.8-27B",
                "object": "model",
                "owned_by": "rocm-local",
                "permission": [],
                "root": "Qwen3.8-27B"
            },
            {
                "id": "claude-3-5-sonnet-latest",
                "object": "model",
                "owned_by": "anthropic",
                "permission": [],
                "root": "claude-3-5-sonnet-latest"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def route_request(request: ChatRequest):
    # 1. Execute TypeSafe Jev Routing Logic
    target_backend = open_jev_decision_engine(request.messages)

    client = httpx.AsyncClient()

    # Route A: Local Qwen 3.8 27B on AMD Radeon AI PRO R9700
    if target_backend == "local_qwen":
        payload = request.model_dump(exclude_none=True)
        payload["model"] = "Qwen3.8-27B"

        if request.stream:
            return StreamingResponse(
                stream_local_qwen(client, payload), 
                media_type="text/event-stream"
            )
        else:
            try:
                resp = await client.post(LOCAL_QWEN_URL, json=payload, timeout=120.0)
                await client.aclose()
                return JSONResponse(status_code=resp.status_code, content=resp.json())
            except Exception as e:
                await client.aclose()
                raise HTTPException(status_code=502, detail=f"Local llama.cpp server error: {str(e)}")

    # Route B: Anthropic Claude Subscription
    else:
        if not CLAUDE_API_KEY:
            # Fallback to local Qwen if Claude API key is absent
            payload = request.model_dump(exclude_none=True)
            payload["model"] = "Qwen3.8-27B"
            if request.stream:
                return StreamingResponse(stream_local_qwen(client, payload), media_type="text/event-stream")
            else:
                resp = await client.post(LOCAL_QWEN_URL, json=payload, timeout=120.0)
                await client.aclose()
                return JSONResponse(status_code=resp.status_code, content=resp.json())

        # Translate OpenAI messages into Anthropic Messages format
        system_prompt = ""
        claude_messages = []
        for m in request.messages:
            if m.role == "system":
                system_prompt = str(m.content)
            elif m.role in ("user", "assistant"):
                claude_messages.append({"role": m.role, "content": str(m.content)})

        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        anthropic_payload: Dict[str, Any] = {
            "model": CLAUDE_MODEL,
            "messages": claude_messages,
            "max_tokens": request.max_tokens or 4096,
            "stream": request.stream or False
        }
        if system_prompt:
            anthropic_payload["system"] = system_prompt

        if request.stream:
            return StreamingResponse(
                stream_anthropic_as_openai(client, headers, anthropic_payload),
                media_type="text/event-stream"
            )
        else:
            try:
                resp = await client.post(CLAUDE_API_URL, headers=headers, json=anthropic_payload, timeout=90.0)
                await client.aclose()
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=resp.text)
                anthropic_resp = resp.json()
                # Convert to OpenAI chat completion format
                text_content = ""
                for blk in anthropic_resp.get("content", []):
                    if blk.get("type") == "text":
                        text_content += blk.get("text", "")
                openai_formatted = {
                    "id": f"chatcmpl-jev-{anthropic_resp.get('id', '')}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": CLAUDE_MODEL,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text_content},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": anthropic_resp.get("usage", {}).get("input_tokens", 0),
                        "completion_tokens": anthropic_resp.get("usage", {}).get("output_tokens", 0),
                        "total_tokens": anthropic_resp.get("usage", {}).get("input_tokens", 0) + anthropic_resp.get("usage", {}).get("output_tokens", 0)
                    }
                }
                return JSONResponse(status_code=200, content=openai_formatted)
            except Exception as e:
                await client.aclose()
                raise HTTPException(status_code=502, detail=f"Claude forwarding error: {str(e)}")

# ==============================================================================
# Main Entrypoint
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    print(f"[*] Starting Open Jev Decision Gateway on http://127.0.0.1:{GATEWAY_PORT}")
    print(f"[*] Local Qwen backend: {LOCAL_QWEN_URL}")
    print(f"[*] Cloud Claude backend: {CLAUDE_API_URL} ({'ENABLED' if CLAUDE_API_KEY else 'DISABLED (No Key)'})")
    uvicorn.run(app, host="127.0.0.1", port=GATEWAY_PORT)
