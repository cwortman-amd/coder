#!/usr/bin/env python3
"""Lightweight OpenAI-chat throughput and latency client (with TTFT, Power & Bandwidth tracking)."""
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
from urllib.request import Request, urlopen


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


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
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say the word ping. " + ("alpha " * max(n_in - 8, 1))}],
        "max_tokens": n_out,
        "temperature": temperature,
        "stream": stream,
        "ignore_eos": True,
        "skip_special_tokens": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
        payload["stream_interval"] = 1
        payload["return_token_ids"] = True

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
    if not stream:
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
    else:
        chunk_times: list[float] = []
        token_times: list[float] = []
        usage: dict = {}
        with urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    body = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if body.get("usage"):
                    usage = body["usage"]
                choices = body.get("choices") or []
                received = time.perf_counter()
                chunk_times.append(received)
                token_ids = choices[0].get("token_ids") if choices else None
                if token_ids:
                    token_times.extend([received] * len(token_ids))
                elif choices and (choices[0].get("delta", {}).get("content") or choices[0].get("delta", {}).get("text")):
                    token_times.append(received)
        dt = time.perf_counter() - t0
        ttft_s = (chunk_times[0] - t0) if chunk_times else None
        intervals = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
        gen_toks = usage.get("completion_tokens") or (len(token_times) if token_times else n_out)
        return {
            "ok": True,
            "dt": dt,
            "ttft_s": ttft_s,
            "itl_s": intervals,
            "prompt_tokens": usage.get("prompt_tokens", n_in),
            "completion_tokens": gen_toks,
        }


def main() -> int:
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
    args = p.parse_args()

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
    ttft_p50_ms = round(percentile(ttfts, 0.50) * 1000.0, 1) if ttfts else None
    ttft_p95_ms = round(percentile(ttfts, 0.95) * 1000.0, 1) if ttfts else None
    ttft_p99_ms = round(percentile(ttfts, 0.99) * 1000.0, 1) if ttfts else None
    mean_ttft_ms = round(statistics.mean(ttfts) * 1000.0, 1) if ttfts else None
    itl_p50_ms = round(percentile(itls, 0.50) * 1000.0, 2) if itls else None
    itl_p95_ms = round(percentile(itls, 0.95) * 1000.0, 2) if itls else None

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
        "ttft_p95_ms": ttft_p95_ms,
        "ttft_p99_ms": ttft_p99_ms,
        "mean_ttft_ms": mean_ttft_ms,
        "itl_p50_ms": itl_p50_ms,
        "itl_p95_ms": itl_p95_ms,
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
    print(json.dumps({k: result[k] for k in result if k != "rows"}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
