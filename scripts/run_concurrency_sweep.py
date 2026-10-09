#!/usr/bin/env python3
"""
Power-of-Two Concurrency Sweep (C = 1, 2, 4, 8, 16) for Qwen3.8-27B Quark AWQ MXFP4 on AMD Radeon AI PRO R9700.
Automates:
- Warm-up phase (10 discarded requests)
- Power-of-two concurrency sweep: C = 1, 2, 4, 8, 16
- GPU telemetry window via scripts/telemetry.py
- Multi-repetition averaging and plateau analysis (Throughput, Efficiency, Latency, Memory)
- Automated report generation in GitHub Markdown
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_serve  # noqa: E402
import catalog  # noqa: E402
from okf_docs import ensure_frontmatter  # noqa: E402
from publish_latency_results import MissingDetailedLatency, publish_latency_result  # noqa: E402
from request_event_schema import percentile  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "_results", "concurrency_sweep")
TELEMETRY_DIR = os.path.join(PROJECT_DIR, "_results", "telemetry")
DOCS_DIR = os.path.join(PROJECT_DIR, "docs")
DOCKER_RESULTS_DIR = "/results/concurrency_sweep"
PUBLISHED_R9700_RESULTS = Path(PROJECT_DIR) / "reports" / "results" / "r9700" / "concurrency.json"
PUBLISHED_POWER_BANDWIDTH = Path(PROJECT_DIR) / "reports" / "profiling" / "power_bandwidth.json"

def _power_sweep_defaults() -> tuple[list[int], dict[int, int], int]:
    """Concurrency and prompt counts live in config/campaigns.yaml."""
    import yaml

    path = Path(PROJECT_DIR) / "config" / "campaigns.yaml"
    loaded = yaml.safe_load(path.read_text())
    row = loaded["campaigns"]["power-sweep"]["power_sweep"]
    counts = {int(key): int(value) for key, value in row["prompt_counts"].items()}
    return (
        [int(value) for value in row["concurrency"]],
        counts,
        int(row["prompt_fallback_per_concurrency"]),
    )


POWER_CONCURRENCY, PROMPT_COUNTS, PROMPT_FALLBACK = _power_sweep_defaults()


def workload_slug(input_len, output_len):
    def short(value):
        return f"{value // 1024}k" if value % 1024 == 0 else str(value)

    return f"{short(input_len)}{short(output_len)}"


def _avg_optional(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def publish_r9700_plot_inputs(
    workload: str,
    sweep_records: list[dict],
    *,
    device_tdp: float,
    peak_bw: float,
) -> None:
    """Upsert one measured workload into the canonical TCO plot inputs."""
    measured = [row for row in sweep_records if row.get("status") == "PASSED"]
    if not measured:
        raise RuntimeError(f"{workload} produced no passed rows to publish")

    results_document = {
        "gpu_profile": "r9700",
        "note": (
            "Measured R9700 serving results used by the TCO plots. "
            "Detailed request samples are indexed under reports/profiling."
        ),
        "runs": [],
    }
    if PUBLISHED_R9700_RESULTS.is_file():
        loaded = json.loads(PUBLISHED_R9700_RESULTS.read_text())
        if isinstance(loaded, dict):
            results_document.update(loaded)
    retained_results = [
        row
        for row in results_document.get("runs") or []
        if row.get("workload") != workload
    ]
    for row in measured:
        retained_results.append(
            {
                "workload": workload,
                "concurrency": row["concurrency"],
                "output_throughput_tok_s": row["agg_tok_s"],
                "server_output_throughput_tok_s": round(row["agg_tok_s"] * 8, 3),
                "ttft_p50_ms": row["ttft_p50_ms"],
                "ttft_p90_ms": row["ttft_p90_ms"],
                "ttft_p95_ms": row["ttft_p95_ms"],
                "ttft_p99_ms": row["ttft_p99_ms"],
                "tpot_p50_ms": row["tpot_p50_ms"],
                "tpot_p90_ms": row["tpot_p90_ms"],
                "tpot_p95_ms": row["tpot_p95_ms"],
                "tpot_p99_ms": row["tpot_p99_ms"],
            }
        )
    results_document["runs"] = sorted(
        retained_results,
        key=lambda row: (str(row.get("workload")), int(row.get("concurrency", 0))),
    )
    PUBLISHED_R9700_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    PUBLISHED_R9700_RESULTS.write_text(json.dumps(results_document, indent=2) + "\n")

    profile_document = {"note": "Measured power and bandwidth series used by report plots.", "runs": []}
    if PUBLISHED_POWER_BANDWIDTH.is_file():
        loaded = json.loads(PUBLISHED_POWER_BANDWIDTH.read_text())
        if isinstance(loaded, dict):
            profile_document.update(loaded)
    retained_profiles = [
        row
        for row in profile_document.get("runs") or []
        if not (row.get("gpu_profile") == "r9700" and row.get("workload") == workload)
    ]
    for row in measured:
        retained_profiles.append(
            {
                "gpu_profile": "r9700",
                "workload": workload,
                "concurrency": row["concurrency"],
                "power_util_pct": row["power_util_pct"],
                "avg_power_w": row["avg_power_w"],
                "bandwidth_util_pct": row["mem_bw_util_pct"],
                "bandwidth_gbs_estimate": row["mem_bw_gb_s"],
                "device_tdp_w": device_tdp,
                "device_peak_bw_gbs": peak_bw,
                "joules_per_token": row["j_per_tok"],
            }
        )
    profile_document["runs"] = sorted(
        retained_profiles,
        key=lambda row: (
            str(row.get("gpu_profile")),
            str(row.get("workload")),
            int(row.get("concurrency", 0)),
        ),
    )
    PUBLISHED_POWER_BANDWIDTH.parent.mkdir(parents=True, exist_ok=True)
    PUBLISHED_POWER_BANDWIDTH.write_text(json.dumps(profile_document, indent=2) + "\n")


def get_vram_peak_mb():
    try:
        res = subprocess.run(["amd-smi", "metric", "-g", "0", "--usage", "--mem-usage", "--json"], capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        gpu_data = data.get("gpu_data", [{}])[0]
        used = (gpu_data.get("mem_usage") or {}).get("used_vram") or {}
        if isinstance(used, dict) and used.get("value") is not None:
            return float(used["value"])
        vram = gpu_data.get("usage", {}).get("vram", {}).get("value", None)
        if vram is not None:
            return float(vram)
    except Exception:
        pass
    return "N/A"

def run_warmup(container_name, count=2, input_len=1024):
    print(f"\n[Warmup] Executing {count} warmup requests ({input_len}:64) to discard...")
    cmd = [
        "docker", "exec", container_name,
        "/opt/vllm/bin/vllm", "bench", "serve",
        "--backend", "openai-chat",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--endpoint", "/v1/chat/completions",
        "--model", "Qwen3.8-27B-Quark-AWQ-MXFP4",
        "--tokenizer", "Qwen/Qwen3.8-27B-FP8",
        "--dataset-name", "random",
        "--random-input-len", str(min(input_len, 8192)),
        "--random-output-len", "64",
        "--num-prompts", str(count),
        "--max-concurrency", "2"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[Warmup] Completed.")

def main():
    parser = argparse.ArgumentParser(description="Run Power-of-Two Concurrency Sweep")
    parser.add_argument("--container", default="rocm-mxfp4-server", help="Docker container name")
    parser.add_argument("--model", default="Qwen3.8-27B-Quark-AWQ-MXFP4", help="Model name")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B-FP8", help="Tokenizer name")
    parser.add_argument("--gpu-profile", default=os.environ.get("GPU_PROFILE", "r9700"), help="GPU profile (r9700, mi350p, r9600)")
    parser.add_argument("--gpu", type=int, default=0, help="Target GPU index for power monitoring")
    parser.add_argument("--device-tdp", type=float, default=None, help="Device TDP in Watts (default: 600W for mi350p, 300W for r9700)")
    parser.add_argument("--peak-bw-gbs", type=float, default=None, help="Device peak memory bandwidth in GB/s (default: 4096 for mi350p, 960 for r9700)")
    parser.add_argument("--model-weight-gib", type=float, default=17.91, help="Model weights size in GiB for memory bandwidth estimation (default: 17.91)")
    parser.add_argument("--concurrency-list", nargs="+", type=int, default=POWER_CONCURRENCY, help="Concurrency list")
    parser.add_argument("--max-concurrency", type=int, default=None, help="Maximum concurrency ceiling")
    parser.add_argument("--repetitions", type=int, default=1, help="Repetitions per point")
    parser.add_argument("--prompt-scale", type=float, default=None, help="Scale factor for prompt counts")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip warmup")
    parser.add_argument("--input-len", type=int, default=8192, help="Input prompt length")
    parser.add_argument("--output-len", type=int, default=1024, help="Output completion length")
    parser.add_argument(
        "--no-publish-latency",
        action="store_true",
        help="Do not copy detailed latency samples to reports/results and reports/profiling",
    )
    args = parser.parse_args()

    prof = args.gpu_profile.lower()
    spec = catalog.gpu(prof)
    device_tdp = args.device_tdp if args.device_tdp is not None else float(spec["tdp_w"])
    peak_bw = args.peak_bw_gbs if args.peak_bw_gbs is not None else float(spec["peak_bw_gbs"])
    catalog_weight = catalog.model(args.model).get("weight_gib")
    weight_gib = float(catalog_weight) if args.model_weight_gib == 17.91 and catalog_weight is not None else (
        None if args.model_weight_gib == 17.91 and catalog_weight is None else args.model_weight_gib
    )

    if args.max_concurrency is not None:
        args.concurrency_list = [c for c in args.concurrency_list if c <= args.max_concurrency]

    if args.prompt_scale is None:
        args.prompt_scale = 1.0 if args.output_len >= 4096 else 0.05

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TELEMETRY_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    if not args.skip_warmup:
        run_warmup(args.container, count=2, input_len=args.input_len)

    sweep_records = []

    for C in args.concurrency_list:
        if args.output_len >= 4096:
            base_prompts = C
        else:
            base_prompts = PROMPT_COUNTS.get(C, C * PROMPT_FALLBACK)
        prompts = max(1, max(C if C > 1 else 1, int(base_prompts * args.prompt_scale)))

        print(f"\n" + "=" * 80)
        print(f"=== TESTING CONCURRENCY C={C} (Prompts per run: {prompts}, Repetitions: {args.repetitions}) ===")
        print("=" * 80)

        c_runs = []

        for rep in range(1, args.repetitions + 1):
            run_id = f"c{C}_r{rep}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            telemetry_json = f"{TELEMETRY_DIR}/{run_id}.json"
            bench_json_host = f"{RESULTS_DIR}/{run_id}.json"
            bench_json_docker_dir = f"{DOCKER_RESULTS_DIR}"
            bench_filename = f"{run_id}.json"

            print(f"\n--- [C={C} | Rep {rep}/{args.repetitions}] Run ID: {run_id} ---")

            subprocess.run(
                [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "telemetry.py"),
                 "begin", "--output", telemetry_json, "--gpu", str(args.gpu),
                 "--profile", prof, "--interval", "0.25"],
                check=False,
            )

            serve_args = bench_serve.bench_serve_args(
                model=args.model,
                tokenizer=args.tokenizer,
                input_len=args.input_len,
                output_len=args.output_len,
                num_prompts=prompts,
                max_concurrency=C,
                request_rate=None,
                result_dir=bench_json_docker_dir,
                result_filename=bench_filename,
                percentile_metrics="tpot,ttft,itl,e2el",
                metric_percentiles="50,90,95,99",
                save_detailed=True,
            )
            bench_cmd = ["docker", "exec", args.container, "/opt/vllm/bin/vllm", *serve_args]

            start_t = time.time()
            bench_res = subprocess.run(bench_cmd, capture_output=True, text=True)
            elapsed_t = time.time() - start_t

            subprocess.run(
                [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "telemetry.py"),
                 "end", "--output", telemetry_json],
                check=False,
            )

            vram_peak = get_vram_peak_mb()

            # Parse results
            bench_data = {}
            if os.path.exists(bench_json_host):
                try:
                    with open(bench_json_host, "r") as bf:
                        bench_data = json.load(bf)
                except Exception as e:
                    print(f"Warning: Could not read benchmark JSON: {e}")

            pwr_data = {}
            if os.path.exists(telemetry_json):
                try:
                    with open(telemetry_json, "r") as pf:
                        pwr_data = json.load(pf)
                except Exception as e:
                    print(f"Warning: Could not read telemetry JSON: {e}")

            success = (bench_res.returncode == 0) and (bench_data.get("completed", 0) > 0)
            completed_prompts = bench_data.get("completed", 0)
            failed_prompts = bench_data.get("failed", 0)
            agg_tok_s = bench_data.get("output_throughput", 0.0)
            total_output_tokens = bench_data.get("total_output_tokens", completed_prompts * args.output_len)
            ttft_p50 = bench_data.get("p50_ttft_ms", bench_data.get("median_ttft_ms", 0.0))
            ttft_p90 = bench_data.get("p90_ttft_ms", 0.0)
            ttft_p95 = bench_data.get("p95_ttft_ms", bench_data.get("p99_ttft_ms", 0.0))
            ttft_p99 = bench_data.get("p99_ttft_ms", 0.0)
            tpot_p50 = bench_data.get("p50_tpot_ms", bench_data.get("median_tpot_ms", 0.0))
            tpot_p90 = bench_data.get("p90_tpot_ms", 0.0)
            tpot_p95 = bench_data.get("p95_tpot_ms", bench_data.get("p99_tpot_ms", 0.0))
            tpot_p99 = bench_data.get("p99_tpot_ms", 0.0)

            raw_ttfts = bench_data.get("ttfts") or []
            if raw_ttfts and (ttft_p90 == 0.0 or ttft_p95 == 0.0 or ttft_p99 == 0.0):
                ttft_samples_ms = [val * 1000.0 for val in raw_ttfts if val is not None]
                if ttft_samples_ms:
                    ttft_p50 = percentile(ttft_samples_ms, 50.0) or ttft_p50
                    ttft_p90 = percentile(ttft_samples_ms, 90.0) or ttft_p90
                    ttft_p95 = percentile(ttft_samples_ms, 95.0) or ttft_p95
                    ttft_p99 = percentile(ttft_samples_ms, 99.0) or ttft_p99

            raw_tpots = bench_data.get("tpots") or []
            if raw_tpots and (tpot_p90 == 0.0 or tpot_p95 == 0.0 or tpot_p99 == 0.0):
                tpot_samples_ms = [val * 1000.0 for val in raw_tpots if val is not None]
                if tpot_samples_ms:
                    tpot_p50 = percentile(tpot_samples_ms, 50.0) or tpot_p50
                    tpot_p90 = percentile(tpot_samples_ms, 90.0) or tpot_p90
                    tpot_p95 = percentile(tpot_samples_ms, 95.0) or tpot_p95
                    tpot_p99 = percentile(tpot_samples_ms, 99.0) or tpot_p99

            avg_power = pwr_data.get("avg_power_w", 0.0)
            max_power = pwr_data.get("max_power_w", 0.0)
            total_energy = pwr_data.get("total_energy_joules", 0.0)
            hotspot_max = pwr_data.get("max_hotspot_c", 0.0)
            power_util_pct = pwr_data.get("power_util_pct", round((avg_power / device_tdp) * 100.0, 1) if device_tdp > 0 else 0.0)

            j_per_tok = (total_energy / total_output_tokens) if total_output_tokens > 0 else 0.0
            tok_per_j = (total_output_tokens / total_energy) if total_energy > 0 else 0.0
            per_stream_tok_s = (agg_tok_s / C) if C > 0 else 0.0
            replay = catalog.weight_replay_gbs(agg_tok_s, C, weight_gib)
            mem_bw_gb_s = round(replay, 2) if replay is not None else None
            mem_bw_util_pct = round((mem_bw_gb_s / peak_bw) * 100.0, 2) if mem_bw_gb_s is not None and peak_bw > 0 else None
            published_latency = None
            if success and not args.no_publish_latency:
                try:
                    raw_path, profile_path = publish_latency_result(
                        Path(bench_json_host),
                        gpu_profile=prof,
                        workload=workload_slug(args.input_len, args.output_len),
                        concurrency=C,
                        repetition=rep,
                    )
                    published_latency = {
                        "raw_samples": str(raw_path.relative_to(PROJECT_DIR)),
                        "profile": str(profile_path.relative_to(PROJECT_DIR)),
                    }
                    print(f"Published detailed latency: {raw_path}")
                except MissingDetailedLatency as exc:
                    raise RuntimeError(
                        f"{exc}. Detailed samples are required because _results is not persistent."
                    ) from exc

            run_record = {
                "concurrency": C,
                "repetition": rep,
                "success": success,
                "completed": completed_prompts,
                "failed": failed_prompts,
                "duration_s": round(elapsed_t, 2),
                "agg_tok_s": round(agg_tok_s, 2),
                "per_stream_tok_s": round(per_stream_tok_s, 2),
                "ttft_p50_ms": round(ttft_p50, 1),
                "ttft_p90_ms": round(ttft_p90, 1),
                "ttft_p95_ms": round(ttft_p95, 1),
                "ttft_p99_ms": round(ttft_p99, 1),
                "tpot_p50_ms": round(tpot_p50, 2),
                "tpot_p90_ms": round(tpot_p90, 2),
                "tpot_p95_ms": round(tpot_p95, 2),
                "tpot_p99_ms": round(tpot_p99, 2),
                "avg_power_w": round(avg_power, 1),
                "max_power_w": round(max_power, 1),
                "power_util_pct": power_util_pct,
                "mem_bw_gb_s": mem_bw_gb_s,
                "mem_bw_util_pct": mem_bw_util_pct,
                "total_energy_j": round(total_energy, 1),
                "j_per_tok": round(j_per_tok, 3),
                "tok_per_j": round(tok_per_j, 4),
                "vram_peak": vram_peak,
                "hotspot_max_c": hotspot_max,
                "published_latency": published_latency,
            }

            c_runs.append(run_record)
            bw_txt = "N/A" if mem_bw_gb_s is None else f"{mem_bw_gb_s:.1f} GB/s ({mem_bw_util_pct:.1f}% Peak)"
            print(f"Result: {agg_tok_s:.2f} tok/s | TTFT p50/p90/p95/p99: {ttft_p50:.1f}/{ttft_p90:.1f}/{ttft_p95:.1f}/{ttft_p99:.1f} ms | TPOT p50/p90/p95/p99: {tpot_p50:.2f}/{tpot_p90:.2f}/{tpot_p95:.2f}/{tpot_p99:.2f} ms | Power: {avg_power:.1f}W ({power_util_pct:.1f}% TDP) | BW: {bw_txt} | {j_per_tok:.3f} J/tok | Hotspot: {hotspot_max}°C")

            time.sleep(5.0)

        # Average across successful repetitions for this C
        valid_runs = [r for r in c_runs if r["success"]]
        if valid_runs:
            avg_record = {
                "concurrency": C,
                "reps": len(valid_runs),
                "agg_tok_s": round(sum(r["agg_tok_s"] for r in valid_runs) / len(valid_runs), 2),
                "per_stream_tok_s": round(sum(r["per_stream_tok_s"] for r in valid_runs) / len(valid_runs), 2),
                "ttft_p50_ms": round(sum(r["ttft_p50_ms"] for r in valid_runs) / len(valid_runs), 1),
                "ttft_p90_ms": round(sum(r["ttft_p90_ms"] for r in valid_runs) / len(valid_runs), 1),
                "ttft_p95_ms": round(sum(r["ttft_p95_ms"] for r in valid_runs) / len(valid_runs), 1),
                "ttft_p99_ms": round(sum(r["ttft_p99_ms"] for r in valid_runs) / len(valid_runs), 1),
                "tpot_p50_ms": round(sum(r["tpot_p50_ms"] for r in valid_runs) / len(valid_runs), 2),
                "tpot_p90_ms": round(sum(r["tpot_p90_ms"] for r in valid_runs) / len(valid_runs), 2),
                "tpot_p95_ms": round(sum(r["tpot_p95_ms"] for r in valid_runs) / len(valid_runs), 2),
                "tpot_p99_ms": round(sum(r["tpot_p99_ms"] for r in valid_runs) / len(valid_runs), 2),
                "avg_power_w": round(sum(r["avg_power_w"] for r in valid_runs) / len(valid_runs), 1),
                "max_power_w": max(r["max_power_w"] for r in valid_runs),
                "power_util_pct": round(sum(r["power_util_pct"] for r in valid_runs) / len(valid_runs), 1),
                "mem_bw_gb_s": _avg_optional(valid_runs, "mem_bw_gb_s"),
                "mem_bw_util_pct": _avg_optional(valid_runs, "mem_bw_util_pct"),
                "total_energy_j": round(sum(r["total_energy_j"] for r in valid_runs) / len(valid_runs), 1),
                "j_per_tok": round(sum(r["j_per_tok"] for r in valid_runs) / len(valid_runs), 3),
                "tok_per_j": round(sum(r["tok_per_j"] for r in valid_runs) / len(valid_runs), 4),
                "vram_peak": valid_runs[-1]["vram_peak"],
                "hotspot_max_c": max(r["hotspot_max_c"] for r in valid_runs),
                "status": "PASSED"
            }
        else:
            avg_record = {
                "concurrency": C,
                "reps": 0,
                "agg_tok_s": 0.0,
                "per_stream_tok_s": 0.0,
                "ttft_p50_ms": 0.0,
                "ttft_p90_ms": 0.0,
                "ttft_p95_ms": 0.0,
                "ttft_p99_ms": 0.0,
                "tpot_p50_ms": 0.0,
                "tpot_p90_ms": 0.0,
                "tpot_p95_ms": 0.0,
                "tpot_p99_ms": 0.0,
                "avg_power_w": 0.0,
                "max_power_w": 0.0,
                "power_util_pct": 0.0,
                "mem_bw_gb_s": 0.0,
                "mem_bw_util_pct": 0.0,
                "total_energy_j": 0.0,
                "j_per_tok": 0.0,
                "tok_per_j": 0.0,
                "vram_peak": "N/A",
                "hotspot_max_c": 0.0,
                "status": "FAILED / OOM"
            }

        sweep_records.append(avg_record)

    # Plateau Analysis & Recommendations
    print("\n" + "=" * 160)
    print("                                      CONCURRENCY SWEEP SUMMARY TABLE")
    print("=" * 160)
    print(f"{'C':<4} | {'Agg tok/s':<10} | {'Stream tok/s':<12} | {'TTFT p50/p90/p95/p99 (ms)':<32} | {'TPOT p50/p90/p95/p99 (ms)':<32} | {'Power (W / %TDP)':<20} | {'Mem BW (%Peak)':<18} | {'J/tok':<8} | {'Status'}")
    print("-" * 160)

    for rec in sweep_records:
        ttft_str = f"{rec['ttft_p50_ms']}/{rec['ttft_p90_ms']}/{rec['ttft_p95_ms']}/{rec['ttft_p99_ms']}"
        tpot_str = f"{rec['tpot_p50_ms']}/{rec['tpot_p90_ms']}/{rec['tpot_p95_ms']}/{rec['tpot_p99_ms']}"
        pwr_str = f"{rec['avg_power_w']}W ({rec['power_util_pct']}%)"
        bw_str = f"{rec['mem_bw_gb_s']}G ({rec['mem_bw_util_pct']}%)"
        print(f"{rec['concurrency']:<4} | {rec['agg_tok_s']:<10.2f} | {rec['per_stream_tok_s']:<12.2f} | {ttft_str:<32} | {tpot_str:<32} | {pwr_str:<20} | {bw_str:<18} | {rec['j_per_tok']:<8.3f} | {rec['status']}")

    # Compute marginal gains
    for i in range(1, len(sweep_records)):
        prev = sweep_records[i-1]
        curr = sweep_records[i]
        if prev["agg_tok_s"] > 0:
            tp_gain = (curr["agg_tok_s"] - prev["agg_tok_s"]) / prev["agg_tok_s"]
            eff_gain = (curr["tok_per_j"] - prev["tok_per_j"]) / prev["tok_per_j"] if prev["tok_per_j"] > 0 else 0.0
            print(f"\n[Marginal Transition C={prev['concurrency']} -> C={curr['concurrency']}]")
            print(f"  • Throughput Gain: +{tp_gain*100:.1f}% {'(Plateau: <10%)' if tp_gain < 0.10 else '(Scaling)'}")
            print(f"  • Efficiency Gain: +{eff_gain*100:.1f}% {'(Plateau: <5%)' if eff_gain < 0.05 else '(Scaling)'}")
            print(f"  • Latency TPOT p95: {curr['tpot_p95_ms']} ms {'(Interactive SLO exceeded >20ms)' if curr['tpot_p95_ms'] > 20 else '(Interactive OK)'}")

    # Save summary report markdown
    report_prefix = "MI350P-SWEEP" if "mi350" in prof else "R9700-SWEEP"
    report_filename = f"{report_prefix}-{args.input_len}-{args.output_len}.md" if (args.input_len != 8192 or args.output_len != 1024) else f"{report_prefix}.md"
    report_file = os.path.join(DOCS_DIR, report_filename)
    hw_label = f"AMD Instinct™ MI350P (`gfx950`, 144 GB HBM3E, {device_tdp:.0f}W TDP, {peak_bw:.0f} GB/s Peak)" if "mi350" in prof else f"AMD Radeon™ AI PRO R9700 (`gfx1201`, 32 GB GDDR6, {device_tdp:.0f}W TDP, {peak_bw:.0f} GB/s Peak)"
    with open(report_file, "w", encoding="utf-8") as rf:
        report_title = f"{report_prefix.replace('-', ' ')} ({args.input_len}:{args.output_len})"
        rf.write(
            ensure_frontmatter(
                "",
                doc_type="Benchmark Report",
                title=report_title,
                description=(
                    f"Concurrency, latency, power, and bandwidth sweep for "
                    f"{args.input_len:,} input and {args.output_len:,} output tokens."
                ),
                tags=["concurrency", "latency", prof],
            )
        )
        rf.write(f"# {report_title}\n\n")
        if "r9700" in prof:
            rf.write("Narrative and earlier sweeps: [R9700.md](R9700.md).\n\n")
        rf.write(f"**Target Hardware**: {hw_label}\n")
        rf.write(f"**Model**: `{args.model}`\n")
        rf.write(f"**Workload**: {args.input_len:,} Input Tokens / {args.output_len:,} Output Tokens\n")
        rf.write(f"**Execution Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EDT')}\n\n")
        rf.write("## Performance, Latency, Power & Bandwidth Ledger\n\n")
        rf.write("| C | Aggregate tok/s | Per-stream tok/s | TTFT p50/p90/p95/p99 (ms) | TPOT p50/p90/p95/p99 (ms) | Power (W / % TDP) | Mem Bandwidth (GB/s / % Peak) | Total Energy (J) | J/token | tokens/Joule | Hotspot Max (°C) | Status |\n")
        rf.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for rec in sweep_records:
            rf.write(f"| **{rec['concurrency']}** | **{rec['agg_tok_s']:.2f}** | {rec['per_stream_tok_s']:.2f} | {rec['ttft_p50_ms']} / {rec['ttft_p90_ms']} / {rec['ttft_p95_ms']} / {rec['ttft_p99_ms']} | {rec['tpot_p50_ms']} / {rec['tpot_p90_ms']} / {rec['tpot_p95_ms']} / {rec['tpot_p99_ms']} | {rec['avg_power_w']} W ({rec['power_util_pct']}%) | {rec['mem_bw_gb_s']} GB/s ({rec['mem_bw_util_pct']}%) | {rec['total_energy_j']:,} | **{rec['j_per_tok']:.3f}** | **{rec['tok_per_j']:.4f}** | {rec['hotspot_max_c']} | `{rec['status']}` |\n")
        rf.write("\n---\n")

    summary_file = os.path.join(RESULTS_DIR, f"sweep_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, "w", encoding="utf-8") as sf:
        json.dump({
            "gpu_profile": prof,
            "model": args.model,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "records": sweep_records,
            "completed_utc": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        }, sf, indent=2)
    if "r9700" in prof or "gfx1201" in prof or "radeon" in prof:
        failed_concurrency = [
            row["concurrency"] for row in sweep_records if row.get("status") != "PASSED"
        ]
        if failed_concurrency:
            raise RuntimeError(
                "Refusing to replace the canonical R9700 plot series because "
                f"these cells failed: {failed_concurrency}"
            )
        publish_r9700_plot_inputs(
            workload_slug(args.input_len, args.output_len),
            sweep_records,
            device_tdp=device_tdp,
            peak_bw=peak_bw,
        )
        print(
            f"[Published] Refreshed {PUBLISHED_R9700_RESULTS} and "
            f"{PUBLISHED_POWER_BANDWIDTH}"
        )
    print(f"\n[Results] JSON summary saved to: {summary_file}")
    print(f"[Report] Markdown summary saved to: {report_file}")

if __name__ == "__main__":
    main()
