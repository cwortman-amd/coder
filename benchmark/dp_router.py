#!/usr/bin/env python3
"""
Data Parallelism (DP=2) Round-Robin Load Balancing Proxy.

Balances incoming OpenAI-compatible chat completion requests across two
independent vLLM model replicas running on Dual Radeon AI PRO R9700 GPUs:
  - Replica 0: http://127.0.0.1:8001/v1 (GPU 0)
  - Replica 1: http://127.0.0.1:8002/v1 (GPU 1)
"""

import argparse
import asyncio
import itertools
import logging
from typing import List

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [DP=2 Proxy] %(message)s")
logger = logging.getLogger("dp_router")

app = FastAPI(title="vLLM DP=2 Round-Robin Proxy", version="1.0.0")

CONFIG = {
    "endpoints": ["http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1"],
    "counter": 0,
    "lock": asyncio.Lock()
}


async def get_next_endpoint() -> str:
    async with CONFIG["lock"]:
        idx = CONFIG["counter"] % len(CONFIG["endpoints"])
        CONFIG["counter"] += 1
        return CONFIG["endpoints"][idx]


@app.get("/health")
async def health_check():
    status = {"status": "healthy", "replicas": {}}
    async with httpx.AsyncClient(timeout=3.0) as client:
        for idx, ep in enumerate(CONFIG["endpoints"]):
            try:
                r = await client.get(f"{ep}/models")
                status["replicas"][f"gpu_{idx}"] = "online" if r.status_code == 200 else f"error_{r.status_code}"
            except Exception as e:
                status["replicas"][f"gpu_{idx}"] = f"unreachable ({e.__class__.__name__})"
    return status


@app.get("/v1/models")
async def list_models():
    target = CONFIG["endpoints"][0]
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{target}/models")
            return r.json()
        except Exception as e:
            return {"object": "list", "data": [{"id": "Qwen3.8-27B-Quark-AWQ-MXFP4"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    target_endpoint = await get_next_endpoint()
    body = await request.json()
    stream = body.get("stream", False)

    client = httpx.AsyncClient(timeout=120.0)
    if stream:
        async def forward_stream():
            try:
                async with client.stream("POST", f"{target_endpoint}/chat/completions", json=body) as resp:
                    if resp.status_code != 200:
                        yield f"data: {{\"error\": \"Replica returned {resp.status_code}\"}}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if line:
                            yield f"{line}\n\n"
            finally:
                await client.aclose()

        return StreamingResponse(forward_stream(), media_type="text/event-stream")
    else:
        try:
            resp = await client.post(f"{target_endpoint}/chat/completions", json=body)
            await client.aclose()
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception as e:
            await client.aclose()
            raise HTTPException(status_code=502, detail=f"Replica {target_endpoint} error: {e}")


def main():
    parser = argparse.ArgumentParser(description="DP=2 Round-Robin Proxy")
    parser.add_argument("--host", default="0.0.0.0", help="Proxy listen host")
    parser.add_argument("--port", type=int, default=8000, help="Proxy listen port")
    parser.add_argument("--replica-0", default="http://127.0.0.1:8001/v1", help="Replica 0 URL")
    parser.add_argument("--replica-1", default="http://127.0.0.1:8002/v1", help="Replica 1 URL")
    args = parser.parse_args()

    CONFIG["endpoints"] = [args.replica_0, args.replica_1]
    logger.info(f"Starting DP=2 Proxy on {args.host}:{args.port}")
    logger.info(f"  • Replica 0 (GPU 0): {args.replica_0}")
    logger.info(f"  • Replica 1 (GPU 1): {args.replica_1}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
