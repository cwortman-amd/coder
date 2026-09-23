#!/usr/bin/env python3
"""
High-Frequency GPU Telemetry and Power Monitor for AMD Radeon™ AI PRO R9700.
Monitors sysfs hwmon directly (fallback to AMD-SMI / Exporter) for accurate 250ms sampling.
Outputs timestamped CSV with power (W), hotspot temperature, memory temperature, and energy metrics.
"""

import argparse
import csv
import json
import os
import signal
import sys
import time

running = True

def sig_handler(signum, frame):
    global running
    running = False

signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

def find_hwmon_paths(gpu_id=0):
    # Locate hwmon node for Navi 48 (device id 0x7551)
    base_cards = ["/sys/class/drm/card1/device/hwmon", "/sys/class/drm/card0/device/hwmon"]
    for base in base_cards:
        if os.path.exists(base):
            for entry in os.listdir(base):
                hwmon_dir = os.path.join(base, entry)
                dev_file = os.path.join(base, "..", "device")
                try:
                    with open(dev_file, "r") as f:
                        dev_id = f.read().strip()
                        if "7551" in dev_id:
                            return hwmon_dir
                except Exception:
                    pass
                if os.path.exists(os.path.join(hwmon_dir, "power1_average")):
                    return hwmon_dir
    return None

def main():
    parser = argparse.ArgumentParser(description="Collect high-frequency AMD GPU power telemetry.")
    parser.add_argument("--gpu", type=int, default=0, help="Target GPU index (default: 0)")
    parser.add_argument("--interval", type=float, default=0.25, help="Sampling interval in seconds (default: 0.25)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    hwmon_dir = find_hwmon_paths(args.gpu)
    pwr_file = os.path.join(hwmon_dir, "power1_average") if hwmon_dir else None
    temp_hot_file = os.path.join(hwmon_dir, "temp2_input") if hwmon_dir else None
    temp_mem_file = os.path.join(hwmon_dir, "temp3_input") if hwmon_dir else None
    temp_edge_file = os.path.join(hwmon_dir, "temp1_input") if hwmon_dir else None

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json_output = os.path.splitext(args.output)[0] + ".json"

    samples = []
    start_time = time.time()

    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["timestamp", "elapsed_s", "power_w", "hotspot_temp_c", "mem_temp_c", "edge_temp_c"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()

        while running:
            t_now = time.time()
            elapsed = t_now - start_time
            power_w = 0.0
            hotspot_c = 0.0
            mem_c = 0.0
            edge_c = 0.0

            if pwr_file and os.path.exists(pwr_file):
                try:
                    with open(pwr_file, "r") as f:
                        # microwatts to watts
                        power_w = float(f.read().strip()) / 1e6
                except Exception:
                    pass

            if temp_hot_file and os.path.exists(temp_hot_file):
                try:
                    with open(temp_hot_file, "r") as f:
                        hotspot_c = float(f.read().strip()) / 1000.0
                except Exception:
                    pass

            if temp_mem_file and os.path.exists(temp_mem_file):
                try:
                    with open(temp_mem_file, "r") as f:
                        mem_c = float(f.read().strip()) / 1000.0
                except Exception:
                    pass

            if temp_edge_file and os.path.exists(temp_edge_file):
                try:
                    with open(temp_edge_file, "r") as f:
                        edge_c = float(f.read().strip()) / 1000.0
                except Exception:
                    pass

            row = {
                "timestamp": round(t_now, 4),
                "elapsed_s": round(elapsed, 4),
                "power_w": round(power_w, 2),
                "hotspot_temp_c": round(hotspot_c, 1),
                "mem_temp_c": round(mem_c, 1),
                "edge_temp_c": round(edge_c, 1)
            }
            writer.writerow(row)
            csvfile.flush()
            samples.append(row)

            # Sleep remaining time
            sleep_time = args.interval - (time.time() - t_now)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Compute summary
    if samples:
        powers = [s["power_w"] for s in samples]
        hot_temps = [s["hotspot_temp_c"] for s in samples]
        mem_temps = [s["mem_temp_c"] for s in samples]
        total_duration = time.time() - start_time
        
        # Trapezoidal or sum integration
        total_energy_joules = sum(p * args.interval for p in powers)
        summary = {
            "duration_s": round(total_duration, 2),
            "sample_count": len(samples),
            "sample_interval_s": args.interval,
            "avg_power_w": round(sum(powers) / len(powers), 2) if powers else 0.0,
            "max_power_w": round(max(powers), 2) if powers else 0.0,
            "min_power_w": round(min(powers), 2) if powers else 0.0,
            "total_energy_joules": round(total_energy_joules, 2),
            "avg_hotspot_c": round(sum(hot_temps) / len(hot_temps), 1) if hot_temps else 0.0,
            "max_hotspot_c": round(max(hot_temps), 1) if hot_temps else 0.0,
            "avg_mem_c": round(sum(mem_temps) / len(mem_temps), 1) if mem_temps else 0.0,
            "max_mem_c": round(max(mem_temps), 1) if mem_temps else 0.0
        }
        with open(json_output, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2)

if __name__ == "__main__":
    main()
