#!/usr/bin/env python3
"""
R9700 Prefill Chunk Size Sweep & Mixed Workload SLO Benchmark (Tracks 2 & 3).

Evaluates:
1. Prefill Chunk Size Sweep: --max-num-batched-tokens in [4096, 2048, 1024, 512]
   - Isolated 8K prompt ingestion throughput (tok/s) & TTFT (penalty of smaller chunks)
   - Contention J1 (burst) and J3 (saturation) peak decode ITL stall (ms) & p95 ITL
2. Mixed Workload & Interactive SLO Benchmark:
   - Concurrent active decode streams with Poisson arrivals of 8K prompts
   - Exact SLO violation tracking (% ITL > 50ms, % ITL > 100ms)
"""

import argparse
import asyncio
import json
import logging
import math
import os
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chunk_slo_bench")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
from benchmark.sse_util import sse_has_content  # noqa: E402
RESULTS_DIR = os.path.join(PROJECT_DIR, "_results", "chunk_sweep")
COMPOSE_FILE = os.path.join(PROJECT_DIR, "docker", "docker-compose.mxfp4.yml")


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    def get_pct(pct):
        k = (n - 1) * (pct / 100.0)
        f = int(k)
        c = f + 1
        if c < n:
            return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])
        return sorted_vals[f]
    return {
        "p50": get_pct(50),
        "p90": get_pct(90),
        "p95": get_pct(95),
        "p99": get_pct(99),
        "mean": statistics.mean(sorted_vals),
        "min": sorted_vals[0],
        "max": sorted_vals[-1]
    }


def generate_token_prompt(num_tokens: int, start_token: int = 100) -> List[int]:
    return [(start_token + i) % 32000 for i in range(num_tokens)]


async def wait_for_server_healthy(base_url: str, timeout: int = 240) -> str:
    t0 = time.time()
    logger.info(f"Waiting for server at {base_url} to become healthy (timeout: {timeout}s)...")
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.time() - t0 < timeout:
            try:
                r = await client.get(f"{base_url}/health")
                if r.status_code == 200:
                    # Fetch model name
                    m_resp = await client.get(f"{base_url}/v1/models")
                    if m_resp.status_code == 200:
                        data = m_resp.json().get("data", [])
                        if data:
                            model_id = data[0]["id"]
                            logger.info(f"Server is HEALTHY! Active model: {model_id}")
                            return model_id
            except Exception:
                pass
            await asyncio.sleep(2.0)
    raise RuntimeError(f"Server failed to become healthy within {timeout}s")


def restart_server_with_chunk_size(chunk_size: int):
    logger.info(f"\n=======================================================")
    logger.info(f"  RECONFIGURING SERVER: --max-num-batched-tokens {chunk_size}")
    logger.info(f"=======================================================")
    subprocess.run(
        ["docker", "stop", "rocm-mxfp4-server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run(
        ["docker", "rm", "rocm-mxfp4-server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    env = os.environ.copy()
    env["MAX_NUM_BATCHED_TOKENS"] = str(chunk_size)
    res = subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "up", "-d", "inference"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        logger.error(f"Failed to start container: {res.stderr}")
        raise RuntimeError("Failed to restart container")
    logger.info("Container re-launched. Awaiting initialization...")


async def run_streaming_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: List[int],
    max_tokens: int,
    request_id: str = "stream"
) -> Dict[str, Any]:
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True
    }
    chunk_times = []
    t_submit = time.perf_counter()
    try:
        async with client.stream("POST", url, json=payload, timeout=180.0) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return {"success": False, "error": f"HTTP {resp.status_code}: {body.decode('utf-8', errors='ignore')}"}
            async for line in resp.aiter_lines():
                if not line:
                    continue
                t_arr = time.perf_counter()
                if sse_has_content(line):
                    chunk_times.append(t_arr)

        if not chunk_times:
            return {"success": False, "error": "No SSE tokens received"}

        ttft_ms = (chunk_times[0] - t_submit) * 1000.0
        itls_ms = []
        for i in range(1, len(chunk_times)):
            itls_ms.append((chunk_times[i] - chunk_times[i - 1]) * 1000.0)

        n_gen = len(chunk_times)
        gen_time = chunk_times[-1] - chunk_times[0]
        tpot_ms = (gen_time / (n_gen - 1) * 1000.0) if n_gen > 1 else 0.0
        decode_tok_s = ((n_gen - 1) / gen_time) if gen_time > 0 else 0.0

        return {
            "success": True,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "decode_tok_s": decode_tok_s,
            "itls_ms": itls_ms,
            "num_generated": n_gen,
            "total_time_s": chunk_times[-1] - t_submit
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def measure_isolated_prefill(base_url: str, model: str, num_tokens: int = 8192, trials: int = 3) -> Dict[str, Any]:
    logger.info(f"Measuring isolated {num_tokens}-token prefill ({trials} trials)...")
    durations = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for t in range(trials):
            # Unique prompt to avoid cache hit
            prompt = generate_token_prompt(num_tokens, start_token=10000 + t * 9000)
            t0 = time.perf_counter()
            r = await client.post(
                f"{base_url}/v1/completions",
                json={"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0}
            )
            t1 = time.perf_counter()
            if r.status_code == 200:
                durations.append(t1 - t0)

    median_s = statistics.median(durations) if durations else 0.0
    tok_s = (num_tokens / median_s) if median_s > 0 else 0.0
    logger.info(f"  Isolated {num_tokens} prefill: {median_s:.3f}s ({tok_s:.1f} prompt tok/s)")
    return {
        "duration_s": round(median_s, 3),
        "prompt_tok_s": round(tok_s, 1),
        "ttft_ms": round(median_s * 1000.0, 1)
    }


async def measure_contention_scenario(
    base_url: str,
    model: str,
    burst_interval: Optional[float],
    attacker_tokens: int = 8192,
    victim_prompt_len: int = 1024,
    victim_output_len: int = 256
) -> Dict[str, Any]:
    victim_prompt = generate_token_prompt(victim_prompt_len, start_token=3000)
    attacker_prompt = generate_token_prompt(attacker_tokens, start_token=70000)
    stop_event = asyncio.Event()

    async def attacker_loop():
        if burst_interval is None:
            return
        async with httpx.AsyncClient(timeout=120.0) as client:
            cnt = 0
            while not stop_event.is_set():
                cnt += 1
                try:
                    await client.post(
                        f"{base_url}/v1/completions",
                        json={"model": model, "prompt": attacker_prompt, "max_tokens": 1, "temperature": 0.0}
                    )
                except Exception:
                    pass
                if burst_interval > 0:
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=burst_interval)
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(0.01)

    attacker_task = asyncio.create_task(attacker_loop())
    if burst_interval == 0:
        await asyncio.sleep(0.5)

    async with httpx.AsyncClient(timeout=180.0) as client:
        res = await run_single_streaming_request(client, base_url, model, victim_prompt, victim_output_len)

    stop_event.set()
    await attacker_task

    if not res["success"]:
        return {"error": res.get("error")}

    itls = res.get("itls_ms", [])
    pct = calculate_percentiles(itls)
    return {
        "decode_tok_s": round(res.get("decode_tok_s", 0.0), 2),
        "tpot_ms": round(res.get("tpot_ms", 0.0), 2),
        "itl_p50_ms": round(pct["p50"], 2),
        "itl_p95_ms": round(pct["p95"], 2),
        "itl_p99_ms": round(pct["p99"], 2),
        "peak_itl_ms": round(pct["max"], 2)
    }


async def run_single_streaming_request(client, base_url, model, prompt, max_tokens):
    return await run_streaming_request(client, base_url, model, prompt, max_tokens)


async def run_slo_benchmark(
    base_url: str,
    model: str,
    concurrency: int = 2,
    decode_tokens: int = 256,
    arrival_rate_lambda: float = 0.25,
    test_duration_s: float = 25.0
) -> Dict[str, Any]:
    logger.info(f"Running Mixed SLO Benchmark (concurrency={concurrency}, lambda={arrival_rate_lambda} req/s)...")
    stop_event = asyncio.Event()
    prefill_prompt = generate_token_prompt(8192, start_token=50000)

    # Background Poisson prefill injector
    async def poisson_injector():
        async with httpx.AsyncClient(timeout=120.0) as client:
            while not stop_event.is_set():
                inter_arrival = random.expovariate(arrival_rate_lambda)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=inter_arrival)
                    break
                except asyncio.TimeoutError:
                    pass
                try:
                    await client.post(
                        f"{base_url}/v1/completions",
                        json={"model": model, "prompt": prefill_prompt, "max_tokens": 1, "temperature": 0.0}
                    )
                except Exception:
                    pass

    injector_task = asyncio.create_task(poisson_injector())

    # Concurrent decode streams
    all_itls = []
    stream_results = []

    async def decode_worker(worker_id: int):
        prompt = generate_token_prompt(1024, start_token=1000 + worker_id * 2000)
        async with httpx.AsyncClient(timeout=180.0) as client:
            res = await run_streaming_request(client, base_url, model, prompt, decode_tokens, request_id=f"w{worker_id}")
            if res["success"]:
                stream_results.append(res)
                all_itls.extend(res["itls_ms"])

    workers = [decode_worker(i) for i in range(concurrency)]
    await asyncio.gather(*workers)

    stop_event.set()
    await injector_task

    if not all_itls:
        return {"error": "No ITL data collected"}

    pct = calculate_percentiles(all_itls)
    total_tokens = len(all_itls)
    viol_50 = sum(1 for itl in all_itls if itl > 50.0)
    viol_100 = sum(1 for itl in all_itls if itl > 100.0)
    viol_300 = sum(1 for itl in all_itls if itl > 300.0)

    p95_slo_pct = (viol_50 / total_tokens * 100.0) if total_tokens > 0 else 0.0
    p99_slo_pct = (viol_100 / total_tokens * 100.0) if total_tokens > 0 else 0.0
    stall_slo_pct = (viol_300 / total_tokens * 100.0) if total_tokens > 0 else 0.0

    avg_decode_tok_s = sum(r["decode_tok_s"] for r in stream_results) if stream_results else 0.0

    return {
        "total_decode_tokens": total_tokens,
        "aggregate_decode_tok_s": round(avg_decode_tok_s, 2),
        "itl_p50_ms": round(pct["p50"], 2),
        "itl_p95_ms": round(pct["p95"], 2),
        "itl_p99_ms": round(pct["p99"], 2),
        "peak_itl_ms": round(pct["max"], 2),
        "slo_violation_50ms_pct": round(p95_slo_pct, 2),
        "slo_violation_100ms_pct": round(p99_slo_pct, 2),
        "slo_violation_300ms_pct": round(stall_slo_pct, 2)
    }


async def main():
    parser = argparse.ArgumentParser(description="Prefill Chunk Size Sweep & SLO Benchmark")
    parser.add_argument("--chunks", nargs="+", type=int, default=[4096, 2048, 1024, 512], help="Chunk sizes to sweep")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="vLLM base URL")
    parser.add_argument("--skip-restart-first", action="store_true", help="Skip restarting server for the first chunk size if already running")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    master_results = {"timestamp": timestamp, "chunks": {}}

    print("=" * 95)
    print("  R9700 PREFILL CHUNK SIZE SWEEP & MIXED INTERACTIVE SLO BENCHMARK")
    print(f"  Sweeping Chunk Sizes: {args.chunks}")
    print("=" * 95)

    for idx, chunk in enumerate(args.chunks):
        if idx == 0 and args.skip_restart_first:
            logger.info(f"Using currently running server for chunk {chunk}...")
        else:
            restart_server_with_chunk_size(chunk)

        model_id = await wait_for_server_healthy(args.base_url)
        chunk_data = {"chunk_size": chunk, "model": model_id}

        # 1. Measure Isolated 8K Prefill
        chunk_data["isolated_prefill_8k"] = await measure_isolated_prefill(args.base_url, model_id, num_tokens=8192, trials=2)

        # 2. Measure Contention J1 (burst every 5s)
        logger.info(f"Measuring J1 (5s prefill bursts) under Chunk {chunk}...")
        chunk_data["contention_j1"] = await measure_contention_scenario(args.base_url, model_id, burst_interval=5.0)

        # 3. Measure Contention J3 (continuous prefill)
        logger.info(f"Measuring J3 (continuous prefill) under Chunk {chunk}...")
        chunk_data["contention_j3"] = await measure_contention_scenario(args.base_url, model_id, burst_interval=0.0)

        # 4. Measure Mixed Traffic Interactive SLO
        logger.info(f"Measuring Mixed Interactive SLO under Chunk {chunk}...")
        chunk_data["mixed_slo"] = await run_slo_benchmark(args.base_url, model_id, concurrency=2, decode_tokens=200, arrival_rate_lambda=0.2)

        master_results["chunks"][str(chunk)] = chunk_data

    # Print Master Summary Table
    print("\n" + "=" * 125)
    print("  PREFILL CHUNK SIZE SWEEP & SLO MASTER COMPARISON TABLE")
    print("=" * 125)
    print(f"{'Chunk Size':<12} | {'Prompt tok/s':<14} | {'8K TTFT (s)':<12} | {'J1 Peak Stall':<16} | {'J1 p95 ITL':<14} | {'J3 Peak Stall':<16} | {'SLO Viol >50ms':<16} | {'SLO Viol >100ms':<16}")
    print("-" * 125)
    for chunk_str, data in master_results["chunks"].items():
        pf = data.get("isolated_prefill_8k", {})
        j1 = data.get("contention_j1", {})
        j3 = data.get("contention_j3", {})
        slo = data.get("mixed_slo", {})
        print(f"{chunk_str:<12} | {pf.get('prompt_tok_s', 0.0):<14.1f} | {pf.get('duration_s', 0.0):<12.3f} | {j1.get('peak_itl_ms', 0.0):<16.1f} | {j1.get('itl_p95_ms', 0.0):<14.1f} | {j3.get('peak_itl_ms', 0.0):<16.1f} | {slo.get('slo_violation_50ms_pct', 0.0):<15.1f}% | {slo.get('slo_violation_100ms_pct', 0.0):<15.1f}%")
    print("=" * 125)

    summary_file = os.path.join(RESULTS_DIR, f"chunk_sweep_summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(master_results, f, indent=2)
    logger.info(f"Chunk sweep and SLO summary saved to: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
