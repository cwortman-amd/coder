#!/usr/bin/env python3
"""OpenAI chat client for throughput and streaming latency.

bench_openai_chat.py and bench_openai_stream.py are compatibility entry points.
Token-id streams and chunk streams stay separate measurements.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catalog  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed
from request_event_schema import percentile  # noqa: E402
from streaming_client import EventMode, stream_chat  # noqa: E402
from urllib.request import Request, urlopen


def chat(
    base: str,
    model: str,
    n_in: int,
    n_out: int,
    timeout: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    enable_thinking: bool,
    stream: bool = False,
) -> dict:
    prompt = "Say the word ping. " + ("alpha " * max(n_in - 8, 1))
    if not stream:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_out,
            "temperature": temperature,
            "stream": False,
            "ignore_eos": True,
            "skip_special_tokens": True,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        req = Request(
            f"{base.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        dt = time.perf_counter() - t0
        usage = body.get("usage") or {}
        return {
            "ok": True,
            "dt": dt,
            "ttft_s": None,
            "itl_s": [],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    extra = {
        "ignore_eos": True,
        "skip_special_tokens": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "stream_options": {"include_usage": True},
        "stream_interval": 1,
        "return_token_ids": True,
    }
    if top_p is not None:
        extra["top_p"] = top_p
    if top_k is not None:
        extra["top_k"] = top_k
    result = stream_chat(
        base_url=base,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        mode=EventMode.CHUNK,
        max_tokens=n_out,
        temperature=temperature,
        timeout=timeout,
        extra_payload=extra,
    )
    event = result.event
    if event.status != "completed":
        raise RuntimeError(event.error_message or "stream failed")
    chunks = result.chunk_timestamps_ns
    ttft_s = (chunks[0] - event.actual_send_ns) / 1e9 if chunks else None
    intervals = [(b - a) / 1e9 for a, b in zip(chunks, chunks[1:])]
    gen_toks = event.completion_tokens or (len(event.token_timestamps_ns) if event.token_timestamps_ns else n_out)
    return {
        "ok": True,
        "dt": event.e2e_latency_ms / 1000.0,
        "ttft_s": ttft_s,
        "itl_s": intervals,
        "prompt_tokens": result.prompt_tokens or n_in,
        "completion_tokens": gen_toks,
    }


def main_chat() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--stream", action="store_true", help="Enable SSE streaming to measure TTFT and ITL latencies")
    p.add_argument("--monitor-power", action="store_true", help="Record the telemetry window during the benchmark")
    p.add_argument("--gpu", type=int, default=0, help="Target GPU index for power monitoring")
    p.add_argument("--gpu-profile", default=os.environ.get("GPU_PROFILE", ""), help="Target GPU profile (mi350p, r9700, r9600)")
    p.add_argument("--device-tdp", type=float, default=None, help="Device TDP in Watts (default: 600W for mi350p, 300W for r9700)")
    p.add_argument("--peak-bw-gbs", type=float, default=None, help="Device peak memory bandwidth in GB/s (default: 4096 for mi350p, 960 for r9700)")
    p.add_argument("--model-weight-gib", type=float, default=17.91, help="Model weights size in GiB for memory bandwidth estimation (default: 17.91)")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--publish-latency",
        action="store_true",
        help="Publish Qwen request/token samples to docs/results and docs/profiling",
    )
    p.add_argument("--repetition", type=int, default=1)
    args = p.parse_args()
    if args.publish_latency and not args.stream:
        p.error("--publish-latency requires --stream so request-level TTFT/ITL is collected")

    # Setup optional power monitoring
    telemetry_json = None
    if args.monitor_power:
        telemetry_json = f"{os.path.splitext(args.out)[0]}_power.json"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        begin = [
            sys.executable, os.path.join(script_dir, "telemetry.py"),
            "begin", "--output", telemetry_json, "--gpu", str(args.gpu), "--interval", "0.25",
        ]
        if args.gpu_profile:
            begin.extend(["--profile", args.gpu_profile])
        subprocess.run(begin, check=False)

    t_all = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [
            pool.submit(
                chat,
                args.base_url,
                args.model,
                args.input_len,
                args.output_len,
                args.timeout,
                args.temperature,
                args.top_p,
                args.top_k,
                args.enable_thinking,
                args.stream,
            )
            for _ in range(args.num_prompts)
        ]
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                rows.append({"ok": False, "error": str(exc), "dt": 0, "ttft_s": None, "itl_s": [], "prompt_tokens": 0, "completion_tokens": 0})
    wall = time.perf_counter() - t_all

    # Stop power monitoring if active
    pwr_summary = {}
    if telemetry_json:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            [sys.executable, os.path.join(script_dir, "telemetry.py"), "end", "--output", telemetry_json],
            check=False,
        )
        if os.path.exists(telemetry_json):
            try:
                with open(telemetry_json, "r", encoding="utf-8") as pf:
                    pwr_summary = json.load(pf)
            except Exception as exc:
                print(f"Warning: could not parse power JSON: {exc}", file=sys.stderr)

    ok = [r for r in rows if r.get("ok")]
    gen = sum(r["completion_tokens"] for r in ok)
    prompt = sum(r["prompt_tokens"] for r in ok)
    dts = [r["dt"] for r in ok]

    # TTFT & ITL statistics
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    itls = [itl for r in ok for itl in r.get("itl_s", [])]
    ttft_p50_ms = round(percentile(ttfts, 50) * 1000.0, 1) if ttfts else None
    ttft_p90_ms = round(percentile(ttfts, 90) * 1000.0, 1) if ttfts else None
    ttft_p95_ms = round(percentile(ttfts, 95) * 1000.0, 1) if ttfts else None
    ttft_p99_ms = round(percentile(ttfts, 99) * 1000.0, 1) if ttfts else None
    ttft_max_ms = round(max(ttfts) * 1000.0, 1) if ttfts else None
    mean_ttft_ms = round(statistics.mean(ttfts) * 1000.0, 1) if ttfts else None
    itl_p50_ms = round(percentile(itls, 50) * 1000.0, 2) if itls else None
    itl_p90_ms = round(percentile(itls, 90) * 1000.0, 2) if itls else None
    itl_p95_ms = round(percentile(itls, 95) * 1000.0, 2) if itls else None
    itl_p99_ms = round(percentile(itls, 99) * 1000.0, 2) if itls else None
    itl_max_ms = round(max(itls) * 1000.0, 2) if itls else None

    # Memory bandwidth estimation
    prof = (args.gpu_profile or os.environ.get("GPU_PROFILE", "")).lower()
    spec = catalog.gpu(prof or "r9700")
    peak_bw = args.peak_bw_gbs if args.peak_bw_gbs is not None else float(spec["peak_bw_gbs"])
    catalog_weight = catalog.model(args.model).get("weight_gib")
    weight_gib = args.model_weight_gib if args.model_weight_gib != 17.91 or catalog_weight is None else float(catalog_weight)
    if args.model_weight_gib == 17.91 and catalog_weight is None:
        weight_gib = None

    out_tok_s = gen / wall if wall else 0.0
    replay = catalog.weight_replay_gbs(out_tok_s, args.concurrency, weight_gib)
    mem_bw_gb_s = round(replay, 2) if replay is not None else None
    mem_bw_util_pct = round((mem_bw_gb_s / peak_bw) * 100.0, 2) if mem_bw_gb_s is not None and peak_bw > 0 else None

    # Power metrics integration
    avg_power_w = pwr_summary.get("avg_power_w")
    max_power_w = pwr_summary.get("max_power_w")
    total_energy_j = pwr_summary.get("total_energy_joules")
    power_util_pct = pwr_summary.get("power_util_pct")
    j_per_tok = round(total_energy_j / gen, 3) if (total_energy_j and gen > 0) else None
    tok_per_j = round(gen / total_energy_j, 4) if (total_energy_j and total_energy_j > 0) else None

    result = {
        "base_url": args.base_url,
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "stream": args.stream,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "enable_thinking": args.enable_thinking,
        "successful": len(ok),
        "failed": len(rows) - len(ok),
        "duration": wall,
        "total_prompt_tokens": prompt,
        "total_generated_tokens": gen,
        "output_throughput": out_tok_s,
        "total_token_throughput": (gen + prompt) / wall if wall else 0.0,
        "mean_latency_s": statistics.mean(dts) if dts else None,
        "ttft_p50_ms": ttft_p50_ms,
        "ttft_p90_ms": ttft_p90_ms,
        "ttft_p95_ms": ttft_p95_ms,
        "ttft_p99_ms": ttft_p99_ms,
        "ttft_max_ms": ttft_max_ms,
        "mean_ttft_ms": mean_ttft_ms,
        "itl_p50_ms": itl_p50_ms,
        "itl_p90_ms": itl_p90_ms,
        "itl_p95_ms": itl_p95_ms,
        "itl_p99_ms": itl_p99_ms,
        "itl_max_ms": itl_max_ms,
        "avg_power_w": avg_power_w,
        "max_power_w": max_power_w,
        "power_util_pct": power_util_pct,
        "total_energy_joules": total_energy_j,
        "joules_per_token": j_per_tok,
        "tokens_per_joule": tok_per_j,
        "mem_bw_gb_s": mem_bw_gb_s,
        "mem_bw_util_pct": mem_bw_util_pct,
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    if args.publish_latency:
        from pathlib import Path

        from publish_latency_results import publish_latency_result

        def short(value: int) -> str:
            return f"{value // 1024}k" if value % 1024 == 0 else str(value)

        raw_path, profile_path = publish_latency_result(
            Path(args.out),
            gpu_profile=prof or "unknown",
            workload=f"{short(args.input_len)}{short(args.output_len)}",
            concurrency=args.concurrency,
            repetition=args.repetition,
        )
        print(f"published raw latency: {raw_path}")
        print(f"updated latency profile: {profile_path}")
    print(json.dumps({k: result[k] for k in result if k != "rows"}, indent=2))
    return 0 if ok else 1


def chat_token_ids(base: str, model: str, n_in: int, n_out: int, timeout: int) -> dict:
    result = stream_chat(
        base_url=base,
        model=model,
        messages=[{"role": "user", "content": "Say the word ping. " + ("alpha " * max(n_in - 8, 1))}],
        mode=EventMode.TOKEN_IDS,
        max_tokens=n_out,
        timeout=timeout,
        extra_payload={
            "stream_options": {"include_usage": True},
            "stream_interval": 1,
            "return_token_ids": True,
            "ignore_eos": True,
            "skip_special_tokens": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    event = result.event
    if event.status != "completed":
        raise RuntimeError(event.error_message or "stream failed")
    tokens = event.token_timestamps_ns
    chunks = result.chunk_timestamps_ns
    send = event.actual_send_ns
    return {
        "ok": True,
        "latency_s": event.e2e_latency_ms / 1000.0,
        "ttft_s": (tokens[0] - send) / 1e9 if tokens else None,
        "itl_s": [(b - a) / 1e9 for a, b in zip(tokens, tokens[1:])],
        "chunk_itl_s": [(b - a) / 1e9 for a, b in zip(chunks, chunks[1:])],
        "stream_chunks": len(chunks),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": event.completion_tokens,
    }


def main_stream() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--publish-gpu-profile",
        default="",
        help="Publish Qwen raw samples under docs/results for this GPU profile",
    )
    parser.add_argument("--repetition", type=int, default=1)
    args = parser.parse_args()

    started = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(chat_token_ids, args.base_url, args.model, args.input_len, args.output_len, args.timeout)
            for _ in range(args.num_prompts)
        ]
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001
                rows.append({"ok": False, "error": str(exc)})
    wall = time.perf_counter() - started

    ok = [row for row in rows if row.get("ok")]
    ttfts = [row["ttft_s"] for row in ok if row["ttft_s"] is not None]
    itls = [itl for row in ok for itl in row["itl_s"]]
    chunk_itls = [itl for row in ok for itl in row["chunk_itl_s"]]
    e2els = [row["latency_s"] for row in ok]
    generated = sum(row["completion_tokens"] for row in ok)
    result = {
        "base_url": args.base_url,
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "successful": len(ok),
        "failed": len(rows) - len(ok),
        "duration_s": wall,
        "total_generated_tokens": generated,
        "output_throughput": generated / wall if wall else 0,
        "ttft_p50_ms": percentile(ttfts, 50) * 1000 if ttfts else None,
        "ttft_p90_ms": percentile(ttfts, 90) * 1000 if ttfts else None,
        "ttft_p95_ms": percentile(ttfts, 95) * 1000 if ttfts else None,
        "ttft_p99_ms": percentile(ttfts, 99) * 1000 if ttfts else None,
        "ttft_max_ms": max(ttfts) * 1000 if ttfts else None,
        "itl_mean_ms": statistics.mean(itls) * 1000 if itls else None,
        "itl_p50_ms": percentile(itls, 50) * 1000 if itls else None,
        "itl_p90_ms": percentile(itls, 90) * 1000 if itls else None,
        "itl_p95_ms": percentile(itls, 95) * 1000 if itls else None,
        "itl_p99_ms": percentile(itls, 99) * 1000 if itls else None,
        "itl_max_ms": max(itls) * 1000 if itls else None,
        "chunk_itl_p50_ms": percentile(chunk_itls, 50) * 1000 if chunk_itls else None,
        "chunk_itl_p95_ms": percentile(chunk_itls, 95) * 1000 if chunk_itls else None,
        "e2el_p50_ms": percentile(e2els, 50) * 1000 if e2els else None,
        "e2el_p90_ms": percentile(e2els, 90) * 1000 if e2els else None,
        "e2el_p95_ms": percentile(e2els, 95) * 1000 if e2els else None,
        "e2el_p99_ms": percentile(e2els, 99) * 1000 if e2els else None,
        "e2el_max_ms": max(e2els) * 1000 if e2els else None,
        "mean_latency_s": statistics.mean(row["latency_s"] for row in ok) if ok else None,
        "rows": rows,
        "note": (
            "Token ITL uses token_ids arrival times. Tokens delivered in one speculative SSE "
            "chunk have zero arrival interval; chunk ITL reports user-visible burst cadence."
        ),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if args.publish_gpu_profile:
        from pathlib import Path

        from publish_latency_results import publish_latency_result

        def short(value: int) -> str:
            return f"{value // 1024}k" if value % 1024 == 0 else str(value)

        raw_path, profile_path = publish_latency_result(
            Path(args.out),
            gpu_profile=args.publish_gpu_profile,
            workload=f"{short(args.input_len)}{short(args.output_len)}",
            concurrency=args.concurrency,
            repetition=args.repetition,
        )
        print(f"published raw latency: {raw_path}")
        print(f"updated latency profile: {profile_path}")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0 if ok else 1
