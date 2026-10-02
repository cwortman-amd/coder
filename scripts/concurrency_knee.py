#!/usr/bin/env python3
"""Burst and paced concurrency curve for the qualified MXFP4 profile.

Invoked by scripts/concurrency.sh. Does not change server flags.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from request_event_schema import percentile  # noqa: E402
from streaming_client import EventMode, stream_chat as shared_stream_chat  # noqa: E402
from vllm_gauges import sample_loop  # noqa: E402
FILL_BETWEEN = {(16, 32): 24, (32, 64): 48, (64, 128): 96}
FROZEN_C1 = 79.35
FROZEN_C8 = 553.58


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def pstdev(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)


def prompt_text(n_in: int) -> str:
    return "Say the word ping. " + ("alpha " * max(n_in - 8, 1))


def start_power(out: Path, gpu: int) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = out.with_suffix(".json")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/telemetry.py"), "begin",
         "--output", str(summary), "--gpu", str(gpu), "--interval", "0.5"],
        check=False,
    )
    return summary


def stop_power(summary: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/telemetry.py"), "end", "--output", str(summary)],
        check=False,
    )


def power_summary(path: Path) -> dict:
    summary = path if path.suffix == ".json" else path.with_suffix(".json")
    if not summary.exists():
        return {"samples": 0, "power_mean_w": None, "power_max_w": None, "gfx_clock_mean_mhz": None, "mem_clock_mean_mhz": None}
    row = json.loads(summary.read_text(encoding="utf-8"))
    return {
        "samples": row.get("sample_count", 0),
        "power_mean_w": row.get("power_mean_w"),
        "power_max_w": row.get("power_max_w"),
        "gfx_clock_mean_mhz": row.get("gfx_clock_mean_mhz"),
        "mem_clock_mean_mhz": row.get("mem_clock_mean_mhz"),
    }


def vram_mib(gpu: int) -> dict[str, float | None]:
    try:
        result = subprocess.run(
            ["amd-smi", "metric", "-g", str(gpu), "-m", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        mem = json.loads(result.stdout or "{}")["gpu_data"][0]["mem_usage"]
        used = float(mem["used_vram"]["value"])
        total = float(mem["total_vram"]["value"])
    except (OSError, subprocess.TimeoutExpired, KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError):
        return {"vram_used_mib": None, "vram_total_mib": None}
    # amd-smi reports MB. Store MiB so older summaries stay on one scale.
    scale = (1000 * 1000) / (1024 * 1024)
    return {"vram_used_mib": used * scale, "vram_total_mib": total * scale}


def summarize_stream(payload: dict, samples: list[dict]) -> dict:
    ok = [row for row in payload.get("rows", []) if row.get("ok")]
    ttfts = [row["ttft_s"] * 1000 for row in ok if row.get("ttft_s") is not None]
    itls = [value * 1000 for row in ok for value in row.get("itl_s") or []]
    tpots = []
    for row in ok:
        tokens = row.get("completion_tokens") or 0
        if tokens > 1 and row.get("latency_s") is not None and row.get("ttft_s") is not None:
            tpots.append((row["latency_s"] - row["ttft_s"]) / (tokens - 1) * 1000)
    running = [sample["running"] for sample in samples if "running" in sample]
    waiting = [sample["waiting"] for sample in samples if "waiting" in sample]
    kv = [sample["kv_perc"] for sample in samples if "kv_perc" in sample]
    generated = payload.get("total_generated_tokens") or sum(row.get("completion_tokens") or 0 for row in ok)
    wall = payload.get("duration_s") or 0
    return {
        "successful": payload.get("successful", len(ok)),
        "failed": payload.get("failed", 0),
        "errors": [row.get("error") for row in payload.get("rows", []) if row.get("error")][:8],
        "prompt_tokens": ok[0].get("prompt_tokens") if ok else None,
        "completion_tokens": ok[0].get("completion_tokens") if ok else None,
        "output_tok_s": payload.get("output_throughput") if payload.get("output_throughput") is not None else (generated / wall if wall else 0),
        "requests_per_s": (len(ok) / wall) if wall else 0,
        "ttft_p50_ms": percentile(ttfts, 50),
        "ttft_p95_ms": percentile(ttfts, 95),
        "ttft_p99_ms": percentile(ttfts, 99),
        "ttft_origin": "http_send",
        "itl_p50_ms": percentile(itls, 50),
        "itl_p95_ms": percentile(itls, 95),
        "itl_p99_ms": percentile(itls, 99),
        "tpot_mean_ms": mean(tpots),
        "server_running_mean": mean(running),
        "server_running_max": max(running) if running else None,
        "server_waiting_max": max(waiting) if waiting else None,
        "kv_perc_max": max(kv) if kv else None,
        "client_queue": "none; num_prompts equals concurrency, so the client does not hold requests back",
    }


def stream_chat(base: str, model: str, n_in: int, n_out: int, timeout: int) -> dict:
    """Token-id stream. TTFT and ITL use the shared client clock."""
    http_send = time.perf_counter()
    result = shared_stream_chat(
        base_url=base,
        model=model,
        messages=[{"role": "user", "content": prompt_text(n_in)}],
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
    ended = time.perf_counter()
    event = result.event
    if event.status != "completed":
        raise RuntimeError(event.error_message or "stream failed")
    tokens = event.token_timestamps_ns
    send = event.actual_send_ns
    return {
        "ok": True,
        "http_send": http_send,
        "latency_s": event.e2e_latency_ms / 1000.0,
        "ttft_s": (tokens[0] - send) / 1e9 if tokens else None,
        "itl_s": [(b - a) / 1e9 for a, b in zip(tokens, tokens[1:])],
        "ended": ended,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": event.completion_tokens,
    }


def run_burst_rep(args, dest: Path, concurrency: int, output_len: int) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    samples: list[dict] = []
    thread = threading.Thread(target=sample_loop, args=(args.base_url, stop, samples), daemon=True)
    power_path = dest.with_suffix(".power.jsonl")
    power_proc = start_power(power_path, args.gpu)
    thread.start()
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/bench_openai_stream.py"),
                "--base-url",
                args.base_url,
                "--model",
                args.model,
                "--input-len",
                str(args.input_len),
                "--output-len",
                str(output_len),
                "--num-prompts",
                str(concurrency),
                "--concurrency",
                str(concurrency),
                "--timeout",
                str(args.timeout),
                "--out",
                str(dest),
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=args.timeout + 60,
        )
    except subprocess.TimeoutExpired:
        return {
            "successful": 0,
            "failed": concurrency,
            "errors": ["bench timed out"],
            "output_tok_s": 0,
            "requests_per_s": 0,
            "ttft_origin": "http_send",
            "client_queue": "none; num_prompts equals concurrency, so the client does not hold requests back",
        }
    finally:
        stop.set()
        thread.join(timeout=2)
        stop_power(power_proc)
    if not dest.exists():
        return {
            "successful": 0,
            "failed": concurrency,
            "errors": [(proc.stderr or proc.stdout or "bench produced no file")[-500:]],
            "output_tok_s": 0,
            "requests_per_s": 0,
            "ttft_origin": "http_send",
            "client_queue": "none; num_prompts equals concurrency, so the client does not hold requests back",
        }
    payload = json.loads(dest.read_text())
    summary = summarize_stream(payload, samples)
    summary.update(power_summary(power_path))
    summary.update(vram_mib(args.gpu))
    if proc.returncode != 0 and not summary["errors"]:
        summary["errors"] = [(proc.stderr or "")[-500:]]
    side = dest.with_suffix(".side.json")
    side.write_text(json.dumps({"summary": summary, "gauge_samples": samples}, indent=2))
    return summary


def run_burst_point(args, out: Path, concurrency: int) -> list[dict]:
    point = out / "burst"
    print(f"warmup C{concurrency}", flush=True)
    run_burst_rep(args, point / f"c{concurrency}_warmup.json", concurrency, args.warmup_output_len)
    reps = []
    for rep in range(1, args.reps + 1):
        print(f"burst C{concurrency} rep {rep}/{args.reps}", flush=True)
        summary = run_burst_rep(args, point / f"c{concurrency}_rep{rep}.json", concurrency, args.output_len)
        summary["concurrency"] = concurrency
        summary["rep"] = rep
        summary["policy"] = "burst"
        reps.append(summary)
        print(
            f"  tok/s={summary.get('output_tok_s')} ttft_p95={summary.get('ttft_p95_ms')} "
            f"itl_p95={summary.get('itl_p95_ms')} wait={summary.get('server_waiting_max')}",
            flush=True,
        )
    return reps


def slo_failure(rep: dict, args) -> str | None:
    if rep.get("failed"):
        return "request_failure"
    if any(rep.get("errors") or []):
        return "request_failure"
    if args.itl_p95_max_ms is not None:
        itl = rep.get("itl_p95_ms")
        if itl is None or itl > args.itl_p95_max_ms:
            return "itl_p95"
    if args.ttft_p95_max_ms is not None:
        ttft = rep.get("ttft_p95_ms")
        if ttft is None or ttft > args.ttft_p95_max_ms:
            return "ttft_p95"
    if args.per_stream_tok_s_min is not None:
        tok_s = rep.get("output_tok_s") or 0
        concurrency = rep.get("concurrency") or 1
        if tok_s / concurrency < args.per_stream_tok_s_min:
            return "per_stream_tok_s"
    return None


def point_mean(reps: list[dict]) -> dict:
    def col(name: str) -> list[float]:
        return [rep[name] for rep in reps if isinstance(rep.get(name), (int, float))]

    tok = col("output_tok_s")
    return {
        "reps": len(reps),
        "output_tok_s": mean(tok),
        "output_tok_s_pstdev": pstdev(tok),
        "requests_per_s": mean(col("requests_per_s")),
        "ttft_p50_ms": mean(col("ttft_p50_ms")),
        "ttft_p95_ms": mean(col("ttft_p95_ms")),
        "ttft_p99_ms": mean(col("ttft_p99_ms")),
        "itl_p50_ms": mean(col("itl_p50_ms")),
        "itl_p95_ms": mean(col("itl_p95_ms")),
        "itl_p99_ms": mean(col("itl_p99_ms")),
        "tpot_mean_ms": mean(col("tpot_mean_ms")),
        "server_running_mean": mean(col("server_running_mean")),
        "server_running_max": max(col("server_running_max")) if col("server_running_max") else None,
        "server_waiting_max": max(col("server_waiting_max")) if col("server_waiting_max") else None,
        "kv_perc_max": max(col("kv_perc_max")) if col("kv_perc_max") else None,
        "power_mean_w": mean(col("power_mean_w")),
        "power_max_w": max(col("power_max_w")) if col("power_max_w") else None,
        "gfx_clock_mean_mhz": mean(col("gfx_clock_mean_mhz")),
        "mem_clock_mean_mhz": mean(col("mem_clock_mean_mhz")),
        "vram_used_mib": mean(col("vram_used_mib")),
        "failed_reps": sum(1 for rep in reps if rep.get("failed") or rep.get("errors")),
    }


def gain(before: float, after: float) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return (after - before) / before


def first_flat_doubling(measured: dict[int, list[dict]], threshold: float) -> tuple[int, int, float] | None:
    present = set(measured)
    for left in sorted(present):
        right = left * 2
        if right not in present:
            continue
        delta = gain(point_mean(measured[left])["output_tok_s"], point_mean(measured[right])["output_tok_s"])
        if delta is not None and delta < threshold:
            return left, right, delta
    return None


def refine_knee(measured: dict[int, list[dict]], threshold: float, flat: tuple[int, int, float] | None) -> dict:
    if flat is None:
        last = max(measured) if measured else None
        last_gain = None
        if last is not None and last // 2 in measured:
            last_gain = gain(
                point_mean(measured[last // 2])["output_tok_s"],
                point_mean(measured[last])["output_tok_s"],
            )
        return {
            "status": "not_located",
            "knee": None,
            "detail": (
                f"No doubling through C{last} gained under {threshold:.0%}. "
                "The throughput knee is at that concurrency or higher."
                if last is not None
                else "No burst points."
            ),
            "last_doubling_gain": last_gain,
        }
    left, right, doubling_gain = flat
    mid = FILL_BETWEEN.get((left, right))
    if mid not in measured:
        return {
            "status": "candidate",
            "knee": left,
            "pair": [left, right],
            "doubling_gain": doubling_gain,
            "detail": (
                f"C{left} to C{right} gained {doubling_gain:.1%}, under {threshold:.0%}. "
                f"Candidate throughput knee is C{left}. No interior fill point was measured."
            ),
        }
    left_mean = point_mean(measured[left])["output_tok_s"]
    mid_mean = point_mean(measured[mid])["output_tok_s"]
    right_mean = point_mean(measured[right])["output_tok_s"]
    to_mid = gain(left_mean, mid_mean)
    to_right = gain(mid_mean, right_mean)
    if to_mid is not None and to_mid < threshold:
        knee = left
        status = "confirmed"
        detail = f"C{left} to C{mid} already gained {to_mid:.1%}. Throughput knee is C{left}."
    elif to_right is not None and to_right < threshold and to_mid is not None and to_mid >= threshold:
        knee = mid
        status = "confirmed"
        detail = (
            f"C{left} to C{mid} gained {to_mid:.1%}, then C{mid} to C{right} gained {to_right:.1%}. "
            f"Throughput knee is C{mid}."
        )
    else:
        knee = None
        status = "not_confirmed"
        detail = (
            f"C{left} to C{right} gained {doubling_gain:.1%}, but the interior point did not confirm a plateau "
            f"(C{left} to C{mid} {None if to_mid is None else f'{to_mid:.1%}'}, "
            f"C{mid} to C{right} {None if to_right is None else f'{to_right:.1%}'})."
        )
    return {
        "status": status,
        "knee": knee,
        "pair": [left, right],
        "fill": mid,
        "doubling_gain": doubling_gain,
        "fill_gains": [to_mid, to_right],
        "detail": detail,
    }


def slo_knee(measured: dict[int, list[dict]], args) -> dict:
    knee = None
    failed_at = None
    reason = None
    for concurrency in sorted(measured):
        failures = [slo_failure(rep, args) for rep in measured[concurrency]]
        if any(failures):
            failed_at = concurrency
            reason = next(item for item in failures if item)
            break
        knee = concurrency
    if knee is None:
        detail = (
            f"No measured concurrency passed. C{failed_at} failed {reason}."
            if failed_at is not None
            else "No burst points."
        )
    elif failed_at is None:
        detail = f"Every measured concurrency passed. SLO knee is at least C{knee}."
    else:
        detail = f"SLO knee is C{knee}. C{failed_at} failed {reason}."
    return {"knee": knee, "failed_at": failed_at, "reason": reason, "detail": detail}


def paced_once(args, concurrency: int, rate: float, num_prompts: int) -> dict:
    slot = threading.Semaphore(concurrency)
    rows: list[dict] = []
    lock = threading.Lock()
    stop = threading.Event()
    samples: list[dict] = []
    thread = threading.Thread(target=sample_loop, args=(args.base_url, stop, samples), daemon=True)
    started = time.perf_counter()

    def worker(index: int) -> None:
        scheduled = started + index / rate
        delay = scheduled - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        queued_at = time.perf_counter()
        slot.acquire()
        acquired = time.perf_counter()
        try:
            row = stream_chat(args.base_url, args.model, args.input_len, args.output_len, args.timeout)
            row["client_queue_s"] = acquired - queued_at
            row["ttft_from_scheduled_s"] = None if row.get("ttft_s") is None else row["ttft_s"] + (acquired - scheduled)
        except Exception as exc:  # noqa: BLE001
            row = {"ok": False, "error": str(exc), "client_queue_s": acquired - queued_at}
        finally:
            slot.release()
        with lock:
            rows.append(row)

    thread.start()
    workers = [threading.Thread(target=worker, args=(index,)) for index in range(num_prompts)]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    stop.set()
    thread.join(timeout=2)
    wall = time.perf_counter() - started
    ok = [row for row in rows if row.get("ok")]
    generated = sum(row.get("completion_tokens") or 0 for row in ok)
    http_ttft = [row["ttft_s"] * 1000 for row in ok if row.get("ttft_s") is not None]
    sched_ttft = [row["ttft_from_scheduled_s"] * 1000 for row in ok if row.get("ttft_from_scheduled_s") is not None]
    queues = [row["client_queue_s"] * 1000 for row in rows if row.get("client_queue_s") is not None]
    itls = [value * 1000 for row in ok for value in row.get("itl_s") or []]
    running = [sample["running"] for sample in samples if "running" in sample]
    waiting = [sample["waiting"] for sample in samples if "waiting" in sample]
    kv = [sample["kv_perc"] for sample in samples if "kv_perc" in sample]
    return {
        "policy": "paced",
        "concurrency": concurrency,
        "offered_rps": rate,
        "num_prompts": num_prompts,
        "successful": len(ok),
        "failed": len(rows) - len(ok),
        "errors": [row.get("error") for row in rows if row.get("error")][:8],
        "duration_s": wall,
        "output_tok_s": generated / wall if wall else 0,
        "requests_per_s": len(ok) / wall if wall else 0,
        "ttft_p50_ms": percentile(http_ttft, 50),
        "ttft_p95_ms": percentile(http_ttft, 95),
        "ttft_p99_ms": percentile(http_ttft, 99),
        "ttft_origin": "http_send",
        "ttft_scheduled_p95_ms": percentile(sched_ttft, 95),
        "client_queue_p95_ms": percentile(queues, 95),
        "itl_p50_ms": percentile(itls, 50),
        "itl_p95_ms": percentile(itls, 95),
        "itl_p99_ms": percentile(itls, 99),
        "server_running_mean": mean(running),
        "server_running_max": max(running) if running else None,
        "server_waiting_max": max(waiting) if waiting else None,
        "kv_perc_max": max(kv) if kv else None,
        "client_queue": "semaphore wait after the scheduled arrival; TTFT above is from HTTP send",
    }


def run_paced(args, out: Path, concurrency: int, rates: list[float]) -> dict:
    num_prompts = max(concurrency, 8)
    scout = []
    last_pass = None
    first_fail = None
    for rate in rates:
        print(f"paced C{concurrency} scout {rate:.3f} req/s", flush=True)
        row = paced_once(args, concurrency, rate, num_prompts)
        row["role"] = "scout"
        dest = out / "paced" / f"c{concurrency}_r{rate:.3f}_scout.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(row, indent=2))
        scout.append(row)
        if slo_failure(row, args):
            first_fail = rate
            break
        last_pass = rate
    confirmed = []
    for label, rate in (("pass", last_pass), ("fail", first_fail)):
        if rate is None:
            continue
        for rep in range(1, args.reps + 1):
            print(f"paced C{concurrency} confirm {label} {rate:.3f} rep {rep}", flush=True)
            row = paced_once(args, concurrency, rate, num_prompts)
            row["role"] = f"confirm_{label}"
            row["rep"] = rep
            dest = out / "paced" / f"c{concurrency}_r{rate:.3f}_rep{rep}.json"
            dest.write_text(json.dumps(row, indent=2))
            confirmed.append(row)
    pass_reps = [row for row in confirmed if row.get("role") == "confirm_pass"]
    goodput = last_pass if pass_reps and all(slo_failure(row, args) is None for row in pass_reps) else None
    return {
        "concurrency": concurrency,
        "rates_scouted": rates,
        "goodput_rps": goodput,
        "first_fail_rps": first_fail,
        "scout": scout,
        "confirmed": confirmed,
        "detail": (
            f"Highest paced rate at C{concurrency} that passed on the HTTP-send clock: {goodput} req/s."
            if goodput is not None
            else f"No paced rate at C{concurrency} passed."
        ),
    }


def fmt(value, digits=2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, args, measured: dict[int, list[dict]], throughput: dict, slo: dict, paced: dict | None) -> None:
    lines = [
        "# Concurrency knees",
        "",
        "Qualified non-speculative MI350P profile: Qwen3.8-27B Quark MXFP4, `ROCM_ATTN`, `VLLM_ROCM_USE_AITER` unset. "
        "This run did not change `--max-num-seqs`, graph mode, or KV dtype.",
        "",
        f"Workload: input {args.input_len} / output {args.output_len}, `ignore_eos`, fixed synthetic prompt, "
        f"{args.reps} measured reps after one warmup. Burst TTFT starts at HTTP send. "
        f"Client concurrency equals the burst size, so the client does not queue.",
        "",
        f"Throughput rule: a doubling that adds under {args.gain_threshold:.0%} aggregate output tok/s is a candidate knee. "
        "The knee is the concurrency beyond which that extra load buys little throughput. "
        "SLO rule: the highest concurrency at which every replicate passes, with no failure at a lower measured concurrency.",
        "",
        "## Result",
        "",
        throughput["detail"],
        "",
        slo["detail"],
        "",
    ]
    if paced:
        lines.append(paced["detail"])
        lines.append("")
        if paced.get("goodput_rps") is not None:
            lines.append(
                "Paced TTFT in the pass/fail decision is from HTTP send. "
                "`ttft_scheduled_p95_ms` includes client semaphore wait. If those two diverge, the offered-load knee is the generator, not vLLM admission."
            )
            lines.append("")
    lines.extend(["## Burst curve", "", "| C | Agg tok/s | Per stream | Req/s | TTFT p95 ms | ITL p95 ms | TPOT mean ms | Running max | Waiting max | KV max | Power mean W | SLO |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
    for concurrency in sorted(measured):
        reps = measured[concurrency]
        row = point_mean(reps)
        per_stream = None if row["output_tok_s"] is None else row["output_tok_s"] / concurrency
        failures = [slo_failure(rep, args) for rep in reps]
        status = "pass" if not any(failures) else "fail " + next(item for item in failures if item)
        lines.append(
            f"| {concurrency} | {fmt(row['output_tok_s'])} | {fmt(per_stream)} | {fmt(row['requests_per_s'], 3)} | "
            f"{fmt(row['ttft_p95_ms'])} | {fmt(row['itl_p95_ms'])} | {fmt(row['tpot_mean_ms'])} | "
            f"{fmt(row['server_running_max'], 1)} | {fmt(row['server_waiting_max'], 1)} | {fmt(row['kv_perc_max'], 3)} | "
            f"{fmt(row['power_mean_w'], 1)} | {status} |"
        )
    lines.extend(["", "Per stream is aggregate tok/s divided by configured concurrency. It is not p95 ITL and not per-request TPOT.", "", "## Doubling gains", "", "| From | To | Gain | Rule |", "|---:|---:|---:|---|"])
    present = set(measured)
    for left in sorted(present):
        right = left * 2
        if right not in present:
            continue
        delta = gain(point_mean(measured[left])["output_tok_s"], point_mean(measured[right])["output_tok_s"])
        flag = "under threshold" if delta is not None and delta < args.gain_threshold else "still rising"
        lines.append(
            f"| {left} | {right} | {fmt(None if delta is None else delta * 100, 1)}% | {flag} |"
        )
    if args.input_len == 1024 and args.output_len == 1024 and 1 in measured:
        c1 = point_mean(measured[1])["output_tok_s"]
        lines.extend(["", f"Frozen campaign C1 is {FROZEN_C1} tok/s. This run's C1 mean is {fmt(c1)}."])
    if args.input_len == 1024 and args.output_len == 1024 and 8 in measured:
        c8 = point_mean(measured[8])["output_tok_s"]
        lines.append(f"Frozen campaign C8 is {FROZEN_C8} tok/s. This run's C8 mean is {fmt(c8)}.")
    lines.extend(
        [
            "",
            f"ITL p95 gate: {args.itl_p95_max_ms} ms. TTFT p95 gate: {args.ttft_p95_max_ms if args.ttft_p95_max_ms is not None else 'not set'}. "
            "The 11 ms default is the predeclared token-arrival ceiling in `_results/quality/attention_qualification/GATES.md`. "
            "The 27 Sep stock `ROCM_ATTN` C1 p95 was 13.36 ms, so this ceiling is stricter than that published stock point.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate MXFP4 concurrency knees")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="awq")
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--warmup-output-len", type=int, default=32)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--concurrencies", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--gain-threshold", type=float, default=0.10)
    parser.add_argument("--itl-p95-max-ms", type=float, default=11.0)
    parser.add_argument("--ttft-p95-max-ms", type=float, default=None)
    parser.add_argument("--per-stream-tok-s-min", type=float, default=None)
    parser.add_argument("--policy", choices=("burst", "paced", "both"), default="both")
    parser.add_argument("--paced-concurrency", type=int, default=None)
    parser.add_argument("--no-fill", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.quick:
        args.reps = 1
        args.input_len = 128
        args.output_len = 16
        args.warmup_output_len = 8
        args.concurrencies = "1"
        args.policy = "burst"
        args.no_fill = True
    args.concurrencies = [int(item) for item in str(args.concurrencies).split(",") if item]
    if not args.out:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out = str(ROOT / "_results" / "priority_eval" / "concurrency_knee" / stamp)
    return args


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    measured: dict[int, list[dict]] = {}
    if args.policy in ("burst", "both"):
        for concurrency in args.concurrencies:
            measured[concurrency] = run_burst_point(args, out, concurrency)
        flat = first_flat_doubling(measured, args.gain_threshold)
        if flat and not args.no_fill:
            mid = FILL_BETWEEN.get((flat[0], flat[1]))
            if mid and mid not in measured:
                print(f"fill C{mid} around C{flat[0]}→C{flat[1]}", flush=True)
                measured[mid] = run_burst_point(args, out, mid)
        throughput = refine_knee(measured, args.gain_threshold, first_flat_doubling(measured, args.gain_threshold))
        slo = slo_knee(measured, args)
    else:
        throughput = {"status": "not_run", "knee": None, "detail": "Burst policy was not run."}
        slo = {"knee": None, "failed_at": None, "reason": None, "detail": "Burst policy was not run."}
    paced = None
    if args.policy in ("paced", "both"):
        concurrency = args.paced_concurrency or slo.get("knee")
        if concurrency is None:
            paced = {
                "concurrency": None,
                "goodput_rps": None,
                "first_fail_rps": None,
                "detail": "Paced policy skipped: no burst concurrency passed the SLO gate. Pass --paced-concurrency to force a rate ramp.",
            }
        else:
            anchor = None
            if concurrency in measured and point_mean(measured[concurrency])["requests_per_s"]:
                anchor = point_mean(measured[concurrency])["requests_per_s"]
            anchor = anchor or 0.1
            rates = []
            rate = anchor * 0.25
            while rate <= anchor * 1.5 + 1e-9 and len(rates) < 8:
                rates.append(round(rate, 4))
                rate *= 1.5
            paced = run_paced(args, out, concurrency, rates)
    report = {
        "throughput_knee": throughput,
        "slo_knee": slo,
        "paced": {key: paced[key] for key in ("concurrency", "goodput_rps", "first_fail_rps", "detail") } if paced else None,
        "burst_means": {str(concurrency): point_mean(reps) for concurrency, reps in sorted(measured.items())},
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_report(out / "KNEE.md", args, measured, throughput, slo, paced)
    print(throughput["detail"], flush=True)
    print(slo["detail"], flush=True)
    if paced:
        print(paced["detail"], flush=True)
    print(f"wrote {out / 'KNEE.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
