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
from typing import Any, Dict, List, Optional, Tuple
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

ROUTER_TELEMETRY = {
    "requests_inflight": 0,
    "streams_inflight": 0,
    "pool_timeouts_total": 0,
    "connect_errors_total": 0,
    "read_errors_total": 0,
    "client_disconnects_total": 0,
    "dp_router_requests_total": 0,
    "affinity_hits": 0,
    "affinity_misses": 0,
    "replica_requests_total": {"replica_0": 0, "replica_1": 0},
}

IDLE_STREAM_TIMEOUT_S = 60.0
SESSION_AFFINITY_MAP: Dict[str, int] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages persistent HTTP connection pool across all proxy requests.
    Configured with connect=5s, pool=5s, write=30s, and read=None, paired with an application-level idle watchdog.
    """
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
    timeout = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)
    app.state.client = httpx.AsyncClient(limits=limits, timeout=timeout)
    logger.info(f"Initialized persistent HTTP client pool (Policy: {CONFIG['routing_policy']}, connect=5s, pool=5s, write=30s, read=None).")
    yield
    await app.state.client.aclose()
    logger.info("Closed HTTP client pool.")


app = FastAPI(title="vLLM DP=2 Load Balancing Proxy", version="1.2.0", lifespan=lifespan)


def get_http_client(request: Optional[Request] = None) -> httpx.AsyncClient:
    """Retrieves the pooled AsyncClient instance from app state."""
    if request and hasattr(request.app.state, "client"):
        return request.app.state.client
    if hasattr(app.state, "client"):
        return app.state.client
    return httpx.AsyncClient(timeout=CONFIG["timeout_s"])


def extract_session_key(request: Request, body: Dict[str, Any]) -> str:
    """
    Extracts or derives a namespaced session identifier: tenant:user:session.
    Precedence:
      1. Explicit headers: x-tenant-id, x-user-id, x-session-id / session-id / x-conversation-id
      2. Body metadata: tenant_id/tenant, user_id/user, session_id/session/conversation_id
      3. Fallback: prompt-derived sha256 hash formatted as default:default:{hash}
    """
    tenant = (
        request.headers.get("x-tenant-id")
        or body.get("tenant_id")
        or body.get("tenant")
        or "default"
    )
    user = (
        request.headers.get("x-user-id")
        or body.get("user_id")
        or body.get("user")
        or "default"
    )
    session = (
        request.headers.get("x-session-id")
        or request.headers.get("session-id")
        or request.headers.get("x-conversation-id")
        or body.get("session_id")
        or body.get("session")
        or body.get("conversation_id")
    )
    if session:
        return f"{tenant}:{user}:{session}"

    # Prompt-derived fallback
    messages = body.get("messages", [])
    if messages and isinstance(messages, list) and len(messages) > 0:
        first_content = str(messages[0].get("content", ""))[:256]
        prompt_hash = hashlib.sha256(first_content.encode("utf-8", "ignore")).hexdigest()[:16]
        return f"{tenant}:{user}:{prompt_hash}"

    prompt = body.get("prompt", "")
    if prompt:
        prompt_str = str(prompt)[:256] if not isinstance(prompt, list) else str(prompt[:32])
        prompt_hash = hashlib.sha256(prompt_str.encode("utf-8", "ignore")).hexdigest()[:16]
        return f"{tenant}:{user}:{prompt_hash}"

    return f"{tenant}:{user}:default_session"


async def get_target_endpoint(request: Request, body: Dict[str, Any]) -> Tuple[str, int, str]:
    """
    Selects target replica based on active routing policy.
    Returns (endpoint_url, replica_idx, session_key).
    """
    num_replicas = len(CONFIG["endpoints"])
    session_key = extract_session_key(request, body)

    if num_replicas == 1:
        return CONFIG["endpoints"][0], 0, session_key

    if CONFIG["routing_policy"] == "cache-affine":
        idx = zlib.crc32(session_key.encode("utf-8")) % num_replicas
        target = CONFIG["endpoints"][idx]
        if session_key in SESSION_AFFINITY_MAP:
            ROUTER_TELEMETRY["affinity_hits"] += 1
        else:
            ROUTER_TELEMETRY["affinity_misses"] += 1
            SESSION_AFFINITY_MAP[session_key] = idx
        logger.debug(f"[Cache-Affine] Key '{session_key}' pinned to Replica {idx} ({target})")
        return target, idx, session_key
    else:
        # Default: Round-Robin
        async with CONFIG["lock"]:
            idx = CONFIG["counter"] % num_replicas
            CONFIG["counter"] += 1
            target = CONFIG["endpoints"][idx]
            logger.debug(f"[Round-Robin] Dispatched to Replica {idx} ({target})")
            return target, idx, session_key


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


@app.get("/metrics/dp")
async def get_dp_metrics():
    """Return DP=2 load balancer telemetry and replica metrics."""
    return {
        "routing_policy": CONFIG["routing_policy"],
        "pool_telemetry": dict(ROUTER_TELEMETRY),
        "endpoints": CONFIG["endpoints"],
        "active_sessions_tracked": len(SESSION_AFFINITY_MAP)
    }


@app.get("/v1/models")
async def list_models(request: Request):
    target = CONFIG["endpoints"][0]
    client = get_http_client(request)
    try:
        r = await client.get(f"{target}/models", timeout=5.0)
        return r.json()
    except Exception:
        return {"object": "list", "data": [{"id": "Qwen3.8-27B-Quark-AWQ-MXFP4"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    target_endpoint, replica_idx, session_key = await get_target_endpoint(request, body)
    stream = body.get("stream", False)
    client = get_http_client(request)

    ROUTER_TELEMETRY["dp_router_requests_total"] += 1
    ROUTER_TELEMETRY["requests_inflight"] += 1
    replica_key = f"replica_{replica_idx}"
    if replica_key in ROUTER_TELEMETRY["replica_requests_total"]:
        ROUTER_TELEMETRY["replica_requests_total"][replica_key] += 1
    else:
        ROUTER_TELEMETRY["replica_requests_total"][replica_key] = 1

    try:
        if stream:
            async def forward_stream():
                ROUTER_TELEMETRY["streams_inflight"] += 1
                try:
                    async with client.stream("POST", f"{target_endpoint}/chat/completions", json=body) as resp:
                        if resp.status_code != 200:
                            err_body = await resp.aread()
                            err_text = err_body.decode(errors="ignore")
                            yield f"data: {json.dumps({'error': f'Replica {replica_idx} error ({resp.status_code}): {err_text}'})}\n\n"
                            return

                        iterator = resp.aiter_lines()
                        while True:
                            try:
                                line = await asyncio.wait_for(iterator.__anext__(), timeout=IDLE_STREAM_TIMEOUT_S)
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError:
                                ROUTER_TELEMETRY["read_errors_total"] += 1
                                logger.warning(f"Idle stream timeout ({IDLE_STREAM_TIMEOUT_S}s) on replica {replica_idx}.")
                                yield f"data: {json.dumps({'error': f'Replica idle timeout after {IDLE_STREAM_TIMEOUT_S}s'})}\n\n"
                                break

                            if line:
                                yield f"{line}\n\n"
                except (asyncio.CancelledError, GeneratorExit):
                    ROUTER_TELEMETRY["client_disconnects_total"] += 1
                    logger.info(f"Client disconnected during stream to replica {replica_idx}.")
                    raise
                except httpx.PoolTimeout:
                    ROUTER_TELEMETRY["pool_timeouts_total"] += 1
                    yield f"data: {json.dumps({'error': 'Connection pool timeout reaching replica'})}\n\n"
                except httpx.ConnectError:
                    ROUTER_TELEMETRY["connect_errors_total"] += 1
                    yield f"data: {json.dumps({'error': f'Connection failed to replica {replica_idx}'})}\n\n"
                except Exception as e:
                    ROUTER_TELEMETRY["read_errors_total"] += 1
                    logger.error(f"Stream forwarding to {target_endpoint} failed: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                finally:
                    ROUTER_TELEMETRY["streams_inflight"] = max(0, ROUTER_TELEMETRY["streams_inflight"] - 1)

            return StreamingResponse(forward_stream(), media_type="text/event-stream")
        else:
            try:
                resp = await client.post(f"{target_endpoint}/chat/completions", json=body)
                return JSONResponse(status_code=resp.status_code, content=resp.json())
            except httpx.PoolTimeout:
                ROUTER_TELEMETRY["pool_timeouts_total"] += 1
                raise HTTPException(status_code=504, detail=f"Connection pool timeout reaching replica {replica_idx}")
            except httpx.ConnectError:
                ROUTER_TELEMETRY["connect_errors_total"] += 1
                raise HTTPException(status_code=502, detail=f"Replica {replica_idx} connection failed")
            except Exception as e:
                logger.error(f"Post request to {target_endpoint} failed: {e}")
                raise HTTPException(status_code=502, detail=f"Replica {target_endpoint} error: {e}")
    finally:
        ROUTER_TELEMETRY["requests_inflight"] = max(0, ROUTER_TELEMETRY["requests_inflight"] - 1)


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
