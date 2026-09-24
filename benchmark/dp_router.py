#!/usr/bin/env python3
"""
Data Parallelism (DP=2) Load Balancing Proxy.

Supports two routing modes:
1. Round-Robin (--routing-policy round-robin): Alternates requests sequentially.
2. Cache-Affine (--routing-policy cache-affine): Hashes session ID / prompt prefix
   to pin multi-turn agent sessions to a consistent GPU replica, maximizing
   vLLM prefix-cache hits while eliminating cross-device KV thrashing.

Target Architecture: Dual homogeneous AMD Radeon AI PRO R9700 (gfx1201)
Endpoints:
  - Replica 0: http://127.0.0.1:8001/v1 (GPU 0)
  - Replica 1: http://127.0.0.1:8002/v1 (GPU 1)
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional
import zlib

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [DP=2 Proxy] %(message)s")
logger = logging.getLogger("dp_router")

CONFIG = {
    "endpoints": ["http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1"],
    "routing_policy": "round-robin",
    "counter": 0,
    "lock": asyncio.Lock(),
    "timeout_s": 120.0
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages persistent HTTP connection pool across all proxy requests."""
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
    timeout = httpx.Timeout(CONFIG["timeout_s"], connect=10.0)
    app.state.client = httpx.AsyncClient(limits=limits, timeout=timeout)
    logger.info(f"Initialized persistent HTTP client pool (Policy: {CONFIG['routing_policy']}).")
    yield
    await app.state.client.aclose()
    logger.info("Closed HTTP client pool.")


app = FastAPI(title="vLLM DP=2 Load Balancing Proxy", version="1.1.0", lifespan=lifespan)


def get_http_client(request: Optional[Request] = None) -> httpx.AsyncClient:
    """Retrieves the pooled AsyncClient instance from app state."""
    if request and hasattr(request.app.state, "client"):
        return request.app.state.client
    if hasattr(app.state, "client"):
        return app.state.client
    return httpx.AsyncClient(timeout=CONFIG["timeout_s"])


def extract_session_key(request: Request, body: Dict[str, Any]) -> str:
    """Extracts or derives a stable session identifier for cache-affine pinning."""
    # 1. Explicit headers
    if "x-session-id" in request.headers:
        return request.headers["x-session-id"]
    if "session-id" in request.headers:
        return request.headers["session-id"]

    # 2. Body metadata
    if body.get("session_id"):
        return str(body["session_id"])
    if body.get("user"):
        return str(body["user"])

    # 3. Derive prefix hash from prompt or messages
    messages = body.get("messages", [])
    if messages and isinstance(messages, list):
        # Hash first system/user prompt (typically invariant across multi-turn session)
        first_content = str(messages[0].get("content", ""))[:256]
        return hashlib.sha256(first_content.encode("utf-8", "ignore")).hexdigest()[:16]

    prompt = body.get("prompt", "")
    if prompt:
        prompt_str = str(prompt)[:256] if not isinstance(prompt, list) else str(prompt[:32])
        return hashlib.sha256(prompt_str.encode("utf-8", "ignore")).hexdigest()[:16]

    return "default_session"


async def get_target_endpoint(request: Request, body: Dict[str, Any]) -> str:
    """Selects the target replica based on the active routing policy."""
    num_replicas = len(CONFIG["endpoints"])
    if num_replicas == 1:
        return CONFIG["endpoints"][0]

    if CONFIG["routing_policy"] == "cache-affine":
        session_key = extract_session_key(request, body)
        idx = zlib.crc32(session_key.encode("utf-8")) % num_replicas
        target = CONFIG["endpoints"][idx]
        logger.debug(f"[Cache-Affine] Session '{session_key}' pinned to Replica {idx} ({target})")
        return target
    else:
        # Default: Round-Robin
        async with CONFIG["lock"]:
            idx = CONFIG["counter"] % num_replicas
            CONFIG["counter"] += 1
            target = CONFIG["endpoints"][idx]
            logger.debug(f"[Round-Robin] Dispatched to Replica {idx} ({target})")
            return target


@app.get("/health")
async def health_check(request: Request):
    status = {"status": "healthy", "routing_policy": CONFIG["routing_policy"], "replicas": {}}
    client = get_http_client(request)
    for idx, ep in enumerate(CONFIG["endpoints"]):
        try:
            r = await client.get(f"{ep}/models", timeout=3.0)
            status["replicas"][f"gpu_{idx}"] = "online" if r.status_code == 200 else f"error_{r.status_code}"
        except Exception as e:
            status["replicas"][f"gpu_{idx}"] = f"unreachable ({e.__class__.__name__})"
    return status


@app.get("/v1/models")
async def list_models(request: Request):
    target = CONFIG["endpoints"][0]
    client = get_http_client(request)
    try:
        r = await client.get(f"{target}/models", timeout=5.0)
        return r.json()
    except Exception as e:
        return {"object": "list", "data": [{"id": "Qwen3.8-27B-Quark-AWQ-MXFP4"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    target_endpoint = await get_target_endpoint(request, body)
    stream = body.get("stream", False)
    client = get_http_client(request)

    if stream:
        async def forward_stream():
            try:
                async with client.stream("POST", f"{target_endpoint}/chat/completions", json=body) as resp:
                    if resp.status_code != 200:
                        err_body = await resp.aread()
                        err_text = err_body.decode(errors="ignore")
                        yield f"data: {json.dumps({'error': f'Replica returned {resp.status_code}: {err_text}'})}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if line:
                            yield f"{line}\n\n"
            except Exception as e:
                logger.error(f"Stream forwarding to {target_endpoint} failed: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(forward_stream(), media_type="text/event-stream")
    else:
        try:
            resp = await client.post(f"{target_endpoint}/chat/completions", json=body)
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception as e:
            logger.error(f"Post request to {target_endpoint} failed: {e}")
            raise HTTPException(status_code=502, detail=f"Replica {target_endpoint} error: {e}")


def main():
    parser = argparse.ArgumentParser(description="DP=2 Load Balancing Proxy")
    parser.add_argument("--host", default="0.0.0.0", help="Proxy listen host")
    parser.add_argument("--port", type=int, default=8000, help="Proxy listen port")
    parser.add_argument("--replica-0", default="http://127.0.0.1:8001/v1", help="Replica 0 URL")
    parser.add_argument("--replica-1", default="http://127.0.0.1:8002/v1", help="Replica 1 URL")
    parser.add_argument(
        "--routing-policy",
        choices=["round-robin", "cache-affine"],
        default="round-robin",
        help="Routing policy: round-robin or cache-affine (pins sessions for prefix caching)"
    )
    args = parser.parse_args()

    CONFIG["endpoints"] = [args.replica_0, args.replica_1]
    CONFIG["routing_policy"] = args.routing_policy
    logger.info(f"Starting DP=2 Proxy on {args.host}:{args.port}")
    logger.info(f"  • Replica 0 (GPU 0): {args.replica_0}")
    logger.info(f"  • Replica 1 (GPU 1): {args.replica_1}")
    logger.info(f"  • Routing Policy   : {args.routing_policy.upper()}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
