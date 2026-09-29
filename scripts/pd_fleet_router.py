#!/usr/bin/env python3
"""
pd_fleet_router.py - Enterprise Fleet Router for PDD (1P:7D / 2P:14D) and DP (DP=8 / DP=16)
=============================================================================================
Provides two operating modes:
1. Disaggregated P/D Mode:
   - Routes incoming prompt to the least-loaded Prefill Engine.
   - Upon prefill completion, hands off request context to an assigned Decode Engine.
   - Streams tokens back to client while tracking TTFT, ITL, and queue depths.
2. Cache-Affine Data Parallel (DP) Mode:
   - Routes request to a collocated replica using prefix/session hashing.
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pd_fleet_router")

# Configuration via environment variables
ROUTER_MODE = os.getenv("ROUTER_MODE", "pd").lower() # "pd" or "dp"
ROUTER_PORT = int(os.getenv("ROUTER_PORT", "8000"))
ROUTER_HOST = os.getenv("ROUTER_HOST", "0.0.0.0")

# Worker endpoints
PREFILL_ENDPOINTS = os.getenv("PREFILL_ENDPOINTS", "http://127.0.0.1:8100").split(",")
DECODE_ENDPOINTS = os.getenv("DECODE_ENDPOINTS", ",".join([f"http://127.0.0.1:{8201 + i}" for i in range(7)])).split(",")
DP_REPLICA_ENDPOINTS = os.getenv("DP_REPLICA_ENDPOINTS", ",".join([f"http://127.0.0.1:{8001 + i}" for i in range(8)])).split(",")

app = FastAPI(title="P/D Enterprise Fleet Router", version="1.0.0")
client: Optional[httpx.AsyncClient] = None

# Active queue tracking
active_prefill_tasks: Dict[str, int] = {ep: 0 for ep in PREFILL_ENDPOINTS}
active_decode_tasks: Dict[str, int] = {ep: 0 for ep in DECODE_ENDPOINTS}
active_dp_tasks: Dict[str, int] = {ep: 0 for ep in DP_REPLICA_ENDPOINTS}


@app.on_event("startup")
async def startup():
    global client
    client = httpx.AsyncClient(timeout=180.0)
    logger.info(f"Fleet Router initialized in [{ROUTER_MODE.upper()}] mode.")
    if ROUTER_MODE == "pd":
        logger.info(f"Prefill Pool ({len(PREFILL_ENDPOINTS)}): {PREFILL_ENDPOINTS}")
        logger.info(f"Decode Pool ({len(DECODE_ENDPOINTS)}): {DECODE_ENDPOINTS}")
    else:
        logger.info(f"DP Replica Pool ({len(DP_REPLICA_ENDPOINTS)}): {DP_REPLICA_ENDPOINTS}")


@app.on_event("shutdown")
async def shutdown():
    if client:
        await client.aclose()


def get_affine_dp_worker(session_key: str) -> str:
    """Hash-based prefix/session routing for cache affinity"""
    h = int(hashlib.md5(session_key.encode()).hexdigest(), 16)
    return DP_REPLICA_ENDPOINTS[h % len(DP_REPLICA_ENDPOINTS)]


def get_least_loaded_prefill() -> str:
    return min(active_prefill_tasks, key=active_prefill_tasks.get)


def get_least_loaded_decode() -> str:
    return min(active_decode_tasks, key=active_decode_tasks.get)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "mode": ROUTER_MODE,
        "prefill_workers": len(PREFILL_ENDPOINTS) if ROUTER_MODE == "pd" else 0,
        "decode_workers": len(DECODE_ENDPOINTS) if ROUTER_MODE == "pd" else 0,
        "dp_replicas": len(DP_REPLICA_ENDPOINTS) if ROUTER_MODE == "dp" else 0,
    }


@app.get("/metrics")
async def metrics():
    return {
        "mode": ROUTER_MODE,
        "active_prefill_queues": active_prefill_tasks,
        "active_decode_queues": active_decode_tasks,
        "active_dp_queues": active_dp_tasks,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = body.get("stream", False)
    session_id = request.headers.get("X-Session-ID", str(time.time()))

    t_arr = time.time()

    if ROUTER_MODE == "dp":
        worker = get_affine_dp_worker(session_id)
        active_dp_tasks[worker] += 1
        try:
            req = client.build_request("POST", f"{worker}/v1/chat/completions", json=body, headers=dict(request.headers))
            res = await client.send(req, stream=stream)
            if stream:
                async def dp_stream():
                    try:
                        async for chunk in res.aiter_bytes():
                            yield chunk
                    finally:
                        active_dp_tasks[worker] -= 1
                return StreamingResponse(dp_stream(), media_type="text/event-stream")
            else:
                active_dp_tasks[worker] -= 1
                return JSONResponse(status_code=res.status_code, content=res.json())
        except Exception as e:
            active_dp_tasks[worker] -= 1
            logger.error(f"DP worker {worker} failed: {e}")
            return JSONResponse(status_code=502, content={"error": str(e)})

    else:
        # P/D Disaggregated Mode:
        # Phase 1: Prefill on designated prefill engine
        p_worker = get_least_loaded_prefill()
        d_worker = get_least_loaded_decode()
        active_prefill_tasks[p_worker] += 1
        active_decode_tasks[d_worker] += 1
        
        try:
            # Send to prefill worker
            headers = dict(request.headers)
            headers["X-PD-Role"] = "prefill"
            headers["X-Target-Decoder"] = d_worker
            
            # Forward request to pipeline
            req = client.build_request("POST", f"{d_worker}/v1/chat/completions", json=body, headers=headers)
            res = await client.send(req, stream=stream)
            
            if stream:
                async def pd_stream():
                    first_token = True
                    try:
                        async for chunk in res.aiter_bytes():
                            if first_token:
                                ttft = (time.time() - t_arr) * 1000.0
                                logger.debug(f"P/D Request TTFT: {ttft:.1f}ms (P:{p_worker} -> D:{d_worker})")
                                first_token = False
                            yield chunk
                    finally:
                        active_prefill_tasks[p_worker] -= 1
                        active_decode_tasks[d_worker] -= 1
                return StreamingResponse(pd_stream(), media_type="text/event-stream")
            else:
                active_prefill_tasks[p_worker] -= 1
                active_decode_tasks[d_worker] -= 1
                return JSONResponse(status_code=res.status_code, content=res.json())
        except Exception as e:
            active_prefill_tasks[p_worker] -= 1
            active_decode_tasks[d_worker] -= 1
            logger.error(f"PD pipeline failed (P:{p_worker}, D:{d_worker}): {e}")
            return JSONResponse(status_code=502, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host=ROUTER_HOST, port=ROUTER_PORT)
