#!/usr/bin/env python3
"""
GPU Power & Energy Measurement Wrapper for ROCm Benchmarking on AMD Radeon™ AI PRO R9700.
Monitors GPU power (W), temperature (°C), clocks, and calculates:
- Average Active Power (W)
- Peak Power (W)
- Total Energy (Joules and Wh)
- Energy per Token (J/tok) and Efficiency (Tokens/Joule)
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Measure GPU power consumption during a benchmark.")
    parser.add_argument("--output-csv", default="/tmp/power_monitor.csv", help="Path to save raw power samples CSV")
    parser.add_argument("--output-json", default="/tmp/power_summary.json", help="Path to save power summary JSON")
    parser.add_argument("--sample-interval", type=int, default=1, help="Sampling interval in seconds (default: 1)")
    parser.add_argument("--total-tokens", type=int, default=None, help="Total generated tokens (for J/token calculation)")
    parser.add_argument("--benchmark-json", default=None, help="Path to vLLM bench result JSON to automatically extract tokens and duration")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Benchmark command to execute")

    args = parser.parse_args()

    if not args.cmd:
        print("Error: No command provided to benchmark.")
        sys.exit(1)

    cmd = args.cmd
    if cmd[0] == "--":
        cmd = cmd[1:]

    # Remove existing CSV
    if os.path.exists(args.output_csv):
        try:
            os.remove(args.output_csv)
        except OSError:
            pass

    print(f"[Power Monitor] Starting amd-smi power monitor (sample interval: {args.sample_interval}s)...")
    monitor_cmd = [
        "amd-smi", "monitor",
        "-g", "0",
        "-p", "-t", "-u", "-m",
        "--csv",
        "-w", str(args.sample_interval),
        "--file", args.output_csv,
        "--overwrite"
    ]

    monitor_proc = subprocess.Popen(
        monitor_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )

    # Let monitor initialize
    time.sleep(1.0)
    start_time = time.time()
    print(f"[Power Monitor] Executing command: {' '.join(cmd)}")

    try:
        bench_res = subprocess.run(cmd, check=True)
        ret_code = bench_res.returncode
    except subprocess.CalledProcessError as e:
        print(f"[Power Monitor] Benchmark failed with return code {e.returncode}")
        ret_code = e.returncode
    finally:
        end_time = time.time()
        # Terminate monitor process group
        try:
            os.killpg(os.getpgid(monitor_proc.pid), signal.SIGTERM)
            monitor_proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(monitor_proc.pid), signal.SIGKILL)
            except Exception:
                pass

    total_duration = end_time - start_time
    print(f"[Power Monitor] Benchmark completed in {total_duration:.2f}s. Parsing power samples...")

    # Parse CSV samples
    samples = []
    if os.path.exists(args.output_csv):
        with open(args.output_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Find header line starting with timestamp
            header_idx = -1
            for idx, line in enumerate(lines):
                if line.startswith("timestamp"):
                    header_idx = idx
                    break
            
            if header_idx != -1:
                reader = csv.DictReader(lines[header_idx:])
                for row in reader:
                    try:
                        pwr = float(row.get("power_usage", 0.0))
                        temp_hot = float(row.get("hotspot_temperature", 0.0))
                        temp_mem = float(row.get("memory_temperature", 0.0))
                        gfx_clk = float(row.get("gfx_clk", 0.0))
                        gfx_util = float(row.get("gfx", 0.0))
                        samples.append({
                            "timestamp": float(row.get("timestamp", 0)),
                            "power_w": pwr,
                            "hotspot_temp_c": temp_hot,
                            "mem_temp_c": temp_mem,
                            "gfx_clk_mhz": gfx_clk,
                            "gfx_util_pct": gfx_util
                        })
                    except (ValueError, TypeError):
                        continue

    if not samples:
        print("[Power Monitor] Warning: No valid power samples captured.")
        sys.exit(ret_code)

    powers = [s["power_w"] for s in samples]
    hot_temps = [s["hotspot_temp_c"] for s in samples]
    mem_temps = [s["mem_temp_c"] for s in samples]
    gfx_utils = [s["gfx_util_pct"] for s in samples]

    avg_power = sum(powers) / len(powers)
    peak_power = max(powers)
    min_power = min(powers)
    
    # Calculate energy: sum(P * dt)
    energy_joules = sum(p * args.sample_interval for p in powers)
    energy_wh = energy_joules / 3600.0

    # Auto-extract token metrics from vLLM benchmark JSON if provided
    total_tokens = args.total_tokens
    output_throughput = None
    if args.benchmark_json and os.path.exists(args.benchmark_json):
        try:
            with open(args.benchmark_json, "r") as f:
                bdata = json.load(f)
                total_tokens = bdata.get("total_output_tokens", total_tokens)
                output_throughput = bdata.get("output_throughput", None)
        except Exception as e:
            print(f"[Power Monitor] Note: Could not parse benchmark JSON: {e}")

    joules_per_token = (energy_joules / total_tokens) if (total_tokens and total_tokens > 0) else None
    tokens_per_joule = (total_tokens / energy_joules) if (total_tokens and energy_joules > 0) else None

    summary = {
        "benchmark_duration_s": round(total_duration, 2),
        "sample_count": len(samples),
        "sample_interval_s": args.sample_interval,
        "avg_power_w": round(avg_power, 2),
        "peak_power_w": round(peak_power, 2),
        "min_power_w": round(min_power, 2),
        "total_energy_joules": round(energy_joules, 2),
        "total_energy_wh": round(energy_wh, 4),
        "avg_hotspot_temp_c": round(sum(hot_temps) / len(hot_temps), 1),
        "peak_hotspot_temp_c": round(max(hot_temps), 1),
        "avg_mem_temp_c": round(sum(mem_temps) / len(mem_temps), 1),
        "peak_mem_temp_c": round(max(mem_temps), 1),
        "avg_gfx_util_pct": round(sum(gfx_utils) / len(gfx_utils), 1),
        "total_output_tokens": total_tokens,
        "output_throughput_tok_per_s": round(output_throughput, 2) if output_throughput else None,
        "joules_per_token": round(joules_per_token, 3) if joules_per_token else None,
        "tokens_per_joule": round(tokens_per_joule, 3) if tokens_per_joule else None
    }

    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("           GPU POWER & ENERGY CONSUMPTION REPORT")
    print("=" * 65)
    print(f" Target Device:           AMD Radeon™ AI PRO R9700 (GPU 0)")
    print(f" Benchmark Duration:      {summary['benchmark_duration_s']} s")
    print(f" Samples Captured:        {summary['sample_count']} (interval: {summary['sample_interval_s']}s)")
    print("-" * 65)
    print(f" Average Power:           {summary['avg_power_w']} W")
    print(f" Peak Power:              {summary['peak_power_w']} W (Power Cap: 300 W)")
    print(f" Min / Idle Power:        {summary['min_power_w']} W")
    print(f" Total Energy Consumed:   {summary['total_energy_joules']:,} Joules ({summary['total_energy_wh']:.4f} Wh)")
    print(f" Average Hotspot Temp:    {summary['avg_hotspot_temp_c']} °C (Peak: {summary['peak_hotspot_temp_c']} °C)")
    print(f" Average Memory Temp:     {summary['avg_mem_temp_c']} °C (Peak: {summary['peak_mem_temp_c']} °C)")
    print("-" * 65)
    if total_tokens:
        print(f" Total Generated Tokens:  {summary['total_output_tokens']:,} tokens")
        if output_throughput:
            print(f" Output Throughput:       {summary['output_throughput_tok_per_s']} tok/s")
        if joules_per_token:
            print(f" Energy per Token:        {summary['joules_per_token']} J/token")
        if tokens_per_joule:
            print(f" Energy Efficiency:       {summary['tokens_per_joule']} tokens/Joule")
    print("=" * 65)
    print(f"Raw samples saved to: {args.output_csv}")
    print(f"Summary JSON saved to: {args.output_json}\n")

    sys.exit(ret_code)

if __name__ == "__main__":
    main()
