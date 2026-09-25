#!/usr/bin/env python3
"""
Power-of-Two Concurrency Sweep (C = 1, 2, 4, 8, 16) for Qwen3.8-27B Quark AWQ MXFP4 on AMD Radeon AI PRO R9700.
Automates:
- Warm-up phase (10 discarded requests)
- Power-of-two concurrency sweep: C = 1, 2, 4, 8, 16
- 250ms high-frequency GPU telemetry & power monitoring via collect_amd_power.py
- Multi-repetition averaging and plateau analysis (Throughput, Efficiency, Latency, Memory)
- Automated report generation in GitHub Markdown
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results", "concurrency_sweep")
TELEMETRY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results", "telemetry")
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
DOCKER_RESULTS_DIR = "/results/concurrency_sweep"

PROMPT_COUNTS = {
    1: 50,
    2: 50,
    4: 80,
    8: 160,
    16: 320
}

def get_vram_peak_mb():
    try:
        res = subprocess.run(["amd-smi", "metric", "-g", "0", "--usage", "--json"], capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        gpu_data = data.get("gpu_data", [{}])[0]
        vram = gpu_data.get("usage", {}).get("vram", {}).get("value", None)
        if vram is not None:
            return float(vram)
    except Exception:
        pass
    try:
        res = subprocess.run(["rocm-smi", "--showmemuse"], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if "GPU[0]" in line and "%" in line:
                return line.strip()
    except Exception:
        pass
    return "N/A"

def run_warmup(container_name, count=10):
    print(f"\n[Warmup] Executing {count} warmup requests (8192:64) to discard...")
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
        "--random-input-len", "8192",
        "--random-output-len", "64",
        "--num-prompts", str(count),
        "--max-concurrency", "2"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[Warmup] Completed.")

def main():
    parser = argparse.ArgumentParser(description="Run Power-of-Two Concurrency Sweep")
    parser.add_argument("--container", default="rocm-mxfp4-server", help="Docker container name")
    parser.add_argument("--concurrency-list", nargs="+", type=int, default=[1, 2, 4, 8, 16], help="Concurrency list")
    parser.add_argument("--repetitions", type=int, default=3, help="Repetitions per point")
    parser.add_argument("--prompt-scale", type=float, default=1.0, help="Scale factor for prompt counts (e.g. 0.1 for rapid calibration)")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip 10-prompt warmup")
    parser.add_argument("--input-len", type=int, default=8192, help="Input prompt length")
    parser.add_argument("--output-len", type=int, default=1024, help="Output completion length")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TELEMETRY_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    if not args.skip_warmup:
        run_warmup(args.container, 10)

    sweep_records = []

    for C in args.concurrency_list:
        base_prompts = PROMPT_COUNTS.get(C, C * 20)
        prompts = max(C, int(base_prompts * args.prompt_scale))

        print(f"\n" + "=" * 80)
        print(f"=== TESTING CONCURRENCY C={C} (Prompts per run: {prompts}, Repetitions: {args.repetitions}) ===")
        print("=" * 80)

        c_runs = []

        for rep in range(1, args.repetitions + 1):
            run_id = f"c{C}_r{rep}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            telemetry_csv = f"{TELEMETRY_DIR}/{run_id}.csv"
            telemetry_json = f"{TELEMETRY_DIR}/{run_id}.json"
            bench_json_host = f"{RESULTS_DIR}/{run_id}.json"
            bench_json_docker_dir = f"{DOCKER_RESULTS_DIR}"
            bench_filename = f"{run_id}.json"

            print(f"\n--- [C={C} | Rep {rep}/{args.repetitions}] Run ID: {run_id} ---")

            # Launch power monitor at 250 ms interval
            power_proc = subprocess.Popen([
                "python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "collect_amd_power.py"),
                "--gpu", "0",
                "--interval", "0.25",
                "--output", telemetry_csv
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            time.sleep(2.0)

            bench_cmd = [
                "docker", "exec", args.container,
                "/opt/vllm/bin/vllm", "bench", "serve",
                "--backend", "openai-chat",
                "--host", "127.0.0.1",
                "--port", "8000",
                "--endpoint", "/v1/chat/completions",
                "--model", "Qwen3.8-27B-Quark-AWQ-MXFP4",
                "--tokenizer", "Qwen/Qwen3.8-27B-FP8",
                "--dataset-name", "random",
                "--random-input-len", str(args.input_len),
                "--random-output-len", str(args.output_len),
                "--num-prompts", str(prompts),
                "--max-concurrency", str(C),
                "--save-result",
                "--result-dir", bench_json_docker_dir,
                "--result-filename", bench_filename
            ]

            start_t = time.time()
            bench_res = subprocess.run(bench_cmd, capture_output=True, text=True)
            elapsed_t = time.time() - start_t

            # Stop power monitor
            try:
                power_proc.send_signal(signal.SIGINT)
                power_proc.wait(timeout=4)
            except Exception:
                power_proc.kill()

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
            ttft_p50 = bench_data.get("median_ttft_ms", 0.0)
            ttft_p95 = bench_data.get("p95_ttft_ms", bench_data.get("p99_ttft_ms", 0.0))
            tpot_p50 = bench_data.get("median_tpot_ms", 0.0)
            tpot_p95 = bench_data.get("p95_tpot_ms", bench_data.get("p99_tpot_ms", 0.0))
            avg_power = pwr_data.get("avg_power_w", 0.0)
            max_power = pwr_data.get("max_power_w", 0.0)
            total_energy = pwr_data.get("total_energy_joules", 0.0)
            hotspot_max = pwr_data.get("max_hotspot_c", 0.0)

            j_per_tok = (total_energy / total_output_tokens) if total_output_tokens > 0 else 0.0
            tok_per_j = (total_output_tokens / total_energy) if total_energy > 0 else 0.0
            per_stream_tok_s = (agg_tok_s / C) if C > 0 else 0.0

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
                "ttft_p95_ms": round(ttft_p95, 1),
                "tpot_p50_ms": round(tpot_p50, 2),
                "tpot_p95_ms": round(tpot_p95, 2),
                "avg_power_w": round(avg_power, 1),
                "max_power_w": round(max_power, 1),
                "total_energy_j": round(total_energy, 1),
                "j_per_tok": round(j_per_tok, 3),
                "tok_per_j": round(tok_per_j, 4),
                "vram_peak": vram_peak,
                "hotspot_max_c": hotspot_max
            }

            c_runs.append(run_record)
            print(f"Result: {agg_tok_s:.2f} tok/s | TPOT p50: {tpot_p50:.2f} ms | TTFT p50: {ttft_p50:.1f} ms | Power: {avg_power:.1f}W | {j_per_tok:.3f} J/tok | Hotspot: {hotspot_max}°C")

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
                "ttft_p95_ms": round(sum(r["ttft_p95_ms"] for r in valid_runs) / len(valid_runs), 1),
                "tpot_p50_ms": round(sum(r["tpot_p50_ms"] for r in valid_runs) / len(valid_runs), 2),
                "tpot_p95_ms": round(sum(r["tpot_p95_ms"] for r in valid_runs) / len(valid_runs), 2),
                "avg_power_w": round(sum(r["avg_power_w"] for r in valid_runs) / len(valid_runs), 1),
                "max_power_w": max(r["max_power_w"] for r in valid_runs),
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
                "ttft_p95_ms": 0.0,
                "tpot_p50_ms": 0.0,
                "tpot_p95_ms": 0.0,
                "avg_power_w": 0.0,
                "max_power_w": 0.0,
                "total_energy_j": 0.0,
                "j_per_tok": 0.0,
                "tok_per_j": 0.0,
                "vram_peak": "N/A",
                "hotspot_max_c": 0.0,
                "status": "FAILED / OOM"
            }

        sweep_records.append(avg_record)

    # Plateau Analysis & Recommendations
    print("\n" + "=" * 80)
    print("                      CONCURRENCY SWEEP SUMMARY TABLE")
    print("=" * 80)
    print(f"{'C':<4} | {'Agg tok/s':<10} | {'Stream tok/s':<12} | {'TTFT p50/p95':<16} | {'TPOT p50/p95':<16} | {'Avg/Max W':<12} | {'J/tok':<8} | {'tok/J':<8} | {'Status'}")
    print("-" * 105)

    for rec in sweep_records:
        ttft_str = f"{rec['ttft_p50_ms']}/{rec['ttft_p95_ms']}"
        tpot_str = f"{rec['tpot_p50_ms']}/{rec['tpot_p95_ms']}"
        pwr_str = f"{rec['avg_power_w']}/{rec['max_power_w']}"
        print(f"{rec['concurrency']:<4} | {rec['agg_tok_s']:<10.2f} | {rec['per_stream_tok_s']:<12.2f} | {ttft_str:<16} | {tpot_str:<16} | {pwr_str:<12} | {rec['j_per_tok']:<8.3f} | {rec['tok_per_j']:<8.4f} | {rec['status']}")

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
            print(f"  • Latency TPOT p95: {curr['tpot_p95_ms']} ms {'(Interactive SLO exceeded >50ms)' if curr['tpot_p95_ms'] > 50 else '(Interactive OK)'}")

    # Save summary report markdown
    report_file = os.path.join(DOCS_DIR, "CONCURRENCY_SWEEP_REPORT.md")
    with open(report_file, "w", encoding="utf-8") as rf:
        rf.write("# Power-of-Two Concurrency Sweep Report (C = 1, 2, 4, 8, 16)\n\n")
        rf.write("**Target Hardware**: AMD Radeon™ AI PRO R9700 (`gfx1201`, 32 GB GDDR6)\n")
        rf.write("**Model**: `Qwen3.8-27B-Quark-AWQ-MXFP4` (Hybrid Attention: 48 GDN + 16 Full Softmax)\n")
        rf.write("**Workload**: 8,192 Input Tokens / 1,024 Output Tokens\n")
        rf.write(f"**Execution Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EDT')}\n\n")
        rf.write("## Performance, Latency & Energy Ledger\n\n")
        rf.write("| C | Aggregate tok/s | Per-stream tok/s | TTFT p50/p95 (ms) | TPOT p50/p95 (ms) | Avg / Max Power (W) | Total Energy (J) | J/token | tokens/Joule | Hotspot Max (°C) | Status |\n")
        rf.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for rec in sweep_records:
            rf.write(f"| **{rec['concurrency']}** | **{rec['agg_tok_s']:.2f}** | {rec['per_stream_tok_s']:.2f} | {rec['ttft_p50_ms']} / {rec['ttft_p95_ms']} | {rec['tpot_p50_ms']} / {rec['tpot_p95_ms']} | {rec['avg_power_w']} / {rec['max_power_w']} | {rec['total_energy_j']:,} | **{rec['j_per_tok']:.3f}** | **{rec['tok_per_j']:.4f}** | {rec['hotspot_max_c']} | `{rec['status']}` |\n")
        rf.write("\n---\n")

    print(f"\n[Report] Markdown summary saved to: {report_file}")

if __name__ == "__main__":
    main()
