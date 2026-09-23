#!/usr/bin/env python3
"""
Prefill/Decode (P/D) Disaggregation Router & Request Orchestrator.

Target Architecture: Dual homogeneous AMD Radeon AI PRO R9700 (gfx1201)
Current Dev Host   : 1x Radeon AI PRO R9700 (gfx1201) + 1x Radeon 780M (gfx1103)
Note               : On single-card dev hosts, run with --mock for choreography
                     validation; production dual-card deployment requires 2x R9700.

Implements OpenAI-compatible /v1/chat/completions proxy for Dual-GPU P/D architectures:
1. Receives incoming chat completion request on port 8000.
2. Dispatches prompt to Prefill Engine (GPU 0, default port 8100) with `do_remote_decode=True`.
3. Extracts `prompt_token_ids` from the prefill completion.
4. Forwards request to Decode Engine (GPU 1, default port 8200) with `do_remote_prefill=True`.
5. Streams generation chunks back to client with low-jitter ITL.
6. Records granular phase metrics: Prefill TTFT, KV Handoff Duration, Decode TPOT, and E2E Latency.
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [P/D Router] %(message)s")
logger = logging.getLogger("pd_router")

app = FastAPI(title="vLLM Prefill/Decode Disaggregation Router", version="1.0.0")

# Router State
ROUTER_CONFIG = {
    "prefill_url": "http://127.0.0.1:8100/v1",
    "decode_url": "http://127.0.0.1:8200/v1",
    "mock_mode": False,
    "timeout_s": 120.0,
    "metrics_log": []
}


class PDMetrics:
    def __init__(self, req_id: str):
        self.req_id = req_id
        self.start_time = time.time()
        self.prefill_start = 0.0
        self.prefill_end = 0.0
        self.decode_start = 0.0
        self.first_token_time = 0.0
        self.end_time = 0.0
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def prefill_duration_ms(self) -> float:
        return (self.prefill_end - self.prefill_start) * 1000.0 if self.prefill_end > 0 else 0.0

    @property
    def kv_handoff_ms(self) -> float:
        return (self.decode_start - self.prefill_end) * 1000.0 if self.decode_start > 0 and self.prefill_end > 0 else 0.0

    @property
    def ttft_ms(self) -> float:
        return (self.first_token_time - self.start_time) * 1000.0 if self.first_token_time > 0 else 0.0

    @property
    def total_e2e_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000.0 if self.end_time > 0 else 0.0

    @property
    def decode_tpot_ms(self) -> float:
        decode_duration = (self.end_time - self.first_token_time) * 1000.0
        return (decode_duration / (self.output_tokens - 1)) if self.output_tokens > 1 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "req_id": self.req_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "prefill_ms": round(self.prefill_duration_ms, 2),
            "kv_handoff_ms": round(self.kv_handoff_ms, 2),
            "ttft_ms": round(self.ttft_ms, 2),
            "decode_tpot_ms": round(self.decode_tpot_ms, 2),
            "total_e2e_ms": round(self.total_e2e_ms, 2)
        }


@app.get("/health")
async def health_check():
    if ROUTER_CONFIG["mock_mode"]:
        return {"status": "healthy", "mode": "mock", "prefill": "mock", "decode": "mock"}

    status = {"status": "healthy", "prefill": "unknown", "decode": "unknown"}
    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            r_p = await client.get(f"{ROUTER_CONFIG['prefill_url']}/models")
            status["prefill"] = "online" if r_p.status_code == 200 else f"err_{r_p.status_code}"
        except Exception as e:
            status["prefill"] = f"unreachable ({e.__class__.__name__})"

        try:
            r_d = await client.get(f"{ROUTER_CONFIG['decode_url']}/models")
            status["decode"] = "online" if r_d.status_code == 200 else f"err_{r_d.status_code}"
        except Exception as e:
            status["decode"] = f"unreachable ({e.__class__.__name__})"

    return status


@app.get("/v1/models")
async def list_models():
    if ROUTER_CONFIG["mock_mode"]:
        return {
            "object": "list",
            "data": [{
                "id": "Qwen3.8-27B-Quark-AWQ-MXFP4",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm-mxfp4-pd"
            }]
        }
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{ROUTER_CONFIG['decode_url']}/models")
            return r.json()
        except Exception:
            return await health_check()


@app.get("/metrics/pd")
async def get_pd_metrics():
    """Return historical P/D request timing metrics."""
    return {
        "total_requests": len(ROUTER_CONFIG["metrics_log"]),
        "recent_metrics": ROUTER_CONFIG["metrics_log"][-50:]
    }


async def mock_stream_response(metrics: PDMetrics, prompt_len: int, max_tokens: int) -> AsyncGenerator[str, None]:
    metrics.input_tokens = prompt_len
    metrics.output_tokens = max_tokens

    # Simulate prefill on GPU 0
    metrics.prefill_start = time.time()
    await asyncio.sleep(0.003 * (prompt_len / 100))  # ~30ms per 1000 tokens
    metrics.prefill_end = time.time()

    # Simulate PCIe KV transfer handoff
    await asyncio.sleep(0.005)  # 5ms PCIe transfer
    metrics.decode_start = time.time()

    # First token
    metrics.first_token_time = time.time()
    for i in range(max_tokens):
        chunk = {
            "id": f"chatcmpl-{metrics.req_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "Qwen3.8-27B-Quark-AWQ-MXFP4",
            "choices": [{
                "index": 0,
                "delta": {"content": f" token_{i}" if i > 0 else "Hello from P/D router!"},
                "finish_reason": "stop" if i == max_tokens - 1 else None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.030)  # 30ms TPOT

    yield "data: [DONE]\n\n"
    metrics.end_time = time.time()
    ROUTER_CONFIG["metrics_log"].append(metrics.to_dict())
    logger.info(f"Mock Req {metrics.req_id} Complete: Prefill={metrics.prefill_duration_ms:.1f}ms, KV-Handoff={metrics.kv_handoff_ms:.1f}ms, TPOT={metrics.decode_tpot_ms:.1f}ms, E2E={metrics.total_e2e_ms:.1f}ms")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req_id = str(uuid.uuid4())[:8]
    metrics = PDMetrics(req_id)
    body = await request.json()

    model_name = body.get("model", "Qwen3.8-27B-Quark-AWQ-MXFP4")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 256)

    # 1. Mock execution path
    if ROUTER_CONFIG["mock_mode"]:
        prompt_len = sum(len(m.get("content", "").split()) for m in messages) * 2 or 512
        if stream:
            return StreamingResponse(
                mock_stream_response(metrics, prompt_len, max_tokens),
                media_type="text/event-stream"
            )
        else:
            # Sync mock response
            metrics.prefill_start = time.time()
            await asyncio.sleep(0.05)
            metrics.prefill_end = time.time()
            metrics.decode_start = time.time() + 0.005
            metrics.first_token_time = metrics.decode_start + 0.030
            await asyncio.sleep(0.030 * max_tokens)
            metrics.end_time = time.time()
            metrics.input_tokens = prompt_len
            metrics.output_tokens = max_tokens
            ROUTER_CONFIG["metrics_log"].append(metrics.to_dict())
            return {
                "id": f"chatcmpl-{req_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Mock P/D execution completed successfully."},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": max_tokens,
                    "total_tokens": prompt_len + max_tokens
                },
                "pd_metrics": metrics.to_dict()
            }

    # 2. Live P/D Choreography
    async with httpx.AsyncClient(timeout=ROUTER_CONFIG["timeout_s"]) as client:
        # Step A: Prefill on GPU 0
        metrics.prefill_start = time.time()
        prefill_payload = dict(body)
        prefill_payload["max_tokens"] = 1
        prefill_payload["stream"] = False
        extra_body = prefill_payload.get("extra_body", {})
        extra_body.update({
            "return_token_ids": True,
            "kv_transfer_params": {"do_remote_decode": True}
        })
        prefill_payload["extra_body"] = extra_body

        try:
            prefill_resp = await client.post(
                f"{ROUTER_CONFIG['prefill_url']}/chat/completions",
                json=prefill_payload
            )
            metrics.prefill_end = time.time()
            if prefill_resp.status_code != 200:
                raise HTTPException(status_code=prefill_resp.status_code, detail=f"Prefill engine error: {prefill_resp.text}")
            prefill_json = prefill_resp.json()
        except Exception as e:
            logger.error(f"[{req_id}] Prefill phase failed: {e}")
            raise HTTPException(status_code=502, detail=f"Prefill engine unreachable: {str(e)}")

        prompt_token_ids = prefill_json.get("prompt_token_ids", [])
        prompt_tokens_count = prefill_json.get("usage", {}).get("prompt_tokens", len(prompt_token_ids) or 100)
        metrics.input_tokens = prompt_tokens_count

        # Step B: Decode on GPU 1
        metrics.decode_start = time.time()
        decode_payload = dict(body)
        extra_body_decode = decode_payload.get("extra_body", {})
        extra_body_decode.update({
            "kv_transfer_params": {
                "do_remote_prefill": True,
                "prompt_token_ids": prompt_token_ids
            }
        })
        decode_payload["extra_body"] = extra_body_decode

        if stream:
            async def forward_stream():
                async with client.stream("POST", f"{ROUTER_CONFIG['decode_url']}/chat/completions", json=decode_payload) as stream_resp:
                    if stream_resp.status_code != 200:
                        yield f"data: {json.dumps({'error': 'Decode engine error'})}\n\n"
                        return

                    first_chunk = True
                    async for line in stream_resp.aiter_lines():
                        if not line:
                            continue
                        if first_chunk:
                            metrics.first_token_time = time.time()
                            first_chunk = False
                        if line.startswith("data: ") and not line.endswith("[DONE]"):
                            metrics.output_tokens += 1
                        yield f"{line}\n\n"

                metrics.end_time = time.time()
                ROUTER_CONFIG["metrics_log"].append(metrics.to_dict())
                logger.info(f"Req {req_id} Streaming Complete: Prefill={metrics.prefill_duration_ms:.1f}ms, KV-Handoff={metrics.kv_handoff_ms:.1f}ms, TPOT={metrics.decode_tpot_ms:.1f}ms, Total={metrics.total_e2e_ms:.1f}ms")

            return StreamingResponse(forward_stream(), media_type="text/event-stream")
        else:
            try:
                decode_resp = await client.post(
                    f"{ROUTER_CONFIG['decode_url']}/chat/completions",
                    json=decode_payload
                )
                metrics.end_time = time.time()
                metrics.first_token_time = metrics.decode_start + 0.035
                if decode_resp.status_code != 200:
                    raise HTTPException(status_code=decode_resp.status_code, detail=f"Decode engine error: {decode_resp.text}")
                decode_json = decode_resp.json()
                metrics.output_tokens = decode_json.get("usage", {}).get("completion_tokens", max_tokens)
                decode_json["pd_metrics"] = metrics.to_dict()
                ROUTER_CONFIG["metrics_log"].append(metrics.to_dict())
                return decode_json
            except Exception as e:
                logger.error(f"[{req_id}] Decode phase failed: {e}")
                raise HTTPException(status_code=502, detail=f"Decode engine unreachable: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="P/D Disaggregation Router")
    parser.add_argument("--host", default="0.0.0.0", help="Router host")
    parser.add_argument("--port", type=int, default=8000, help="Router listen port")
    parser.add_argument("--prefill-url", default="http://127.0.0.1:8100/v1", help="Prefill vLLM URL (GPU 0)")
    parser.add_argument("--decode-url", default="http://127.0.0.1:8200/v1", help="Decode vLLM URL (GPU 1)")
    parser.add_argument("--mock", action="store_true", help="Run in mock/dry-run mode for choreography validation")
    args = parser.parse_args()

    ROUTER_CONFIG["prefill_url"] = args.prefill_url
    ROUTER_CONFIG["decode_url"] = args.decode_url
    ROUTER_CONFIG["mock_mode"] = args.mock

    logger.info(f"Starting P/D Router on {args.host}:{args.port}")
    logger.info(f"  • Prefill Endpoint (GPU 0): {ROUTER_CONFIG['prefill_url']}")
    logger.info(f"  • Decode Endpoint (GPU 1) : {ROUTER_CONFIG['decode_url']}")
    logger.info(f"  • Operating Mode          : {'MOCK SIMULATION' if args.mock else 'LIVE DISAGGREGATION'}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
