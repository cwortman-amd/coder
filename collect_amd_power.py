#!/usr/bin/env python3
"""
High-frequency GPU telemetry for AMD Radeon AI PRO R9700 (gfx1201) and
Instinct MI350P (gfx950). Samples sysfs hwmon (power1_average) with trapezoidal
energy integration.
"""

import argparse
import csv
import json
import os
import signal
import sys
import time

running = True

# PCI device id hints (lowercase hex without 0x)
PCI_HINTS = {
    "r9700": ("7551",),
    "mi350p": tuple(f"75a{x}" for x in "0123456789abcdef"),
}


def sig_handler(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read().strip()


def _drm_cards():
    drm = "/sys/class/drm"
    if not os.path.isdir(drm):
        return []
    cards = []
    for name in sorted(os.listdir(drm)):
        if not name.startswith("card") or "-" in name:
            continue
        suffix = name[4:]
        if not suffix.isdigit():
            continue
        cards.append((int(suffix), os.path.join(drm, name, "device")))
    return cards


def find_hwmon_paths(gpu_id=0, profile=None):
    """
    Pick the hwmon directory for a compute GPU.
    gpu_id indexes among amdgpu cards that expose power1_average (skips display-only).
    """
    profile = (profile or os.environ.get("GPU_PROFILE") or "").lower()
    pci_hints = PCI_HINTS.get(profile, ())
    candidates = []
    for card_idx, dev_dir in _drm_cards():
        hwmon_root = os.path.join(dev_dir, "hwmon")
        if not os.path.isdir(hwmon_root):
            continue
        pci_id = ""
        for pci_file in (os.path.join(dev_dir, "device"), os.path.join(dev_dir, "uevent")):
            try:
                pci_id += " " + _read_text(pci_file).lower()
            except OSError:
                pass
        for entry in os.listdir(hwmon_root):
            hwmon_dir = os.path.join(hwmon_root, entry)
            pwr = os.path.join(hwmon_dir, "power1_average")
            if not os.path.exists(pwr):
                continue
            hinted = any(h in pci_id for h in pci_hints) if pci_hints else False
            candidates.append((hinted, card_idx, hwmon_dir))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (not x[0], x[1]))
    hinted = [c for c in candidates if c[0]] or candidates
    if gpu_id < 0 or gpu_id >= len(hinted):
        gpu_id = 0
    return hinted[gpu_id][2]


def main():
    parser = argparse.ArgumentParser(description="Collect high-frequency AMD GPU power telemetry.")
    parser.add_argument("--gpu", type=int, default=0, help="Target GPU index among power-capable amdgpu devices")
    parser.add_argument("--profile", default=os.environ.get("GPU_PROFILE", ""), help="r9700 or mi350p (PCI hint)")
    parser.add_argument("--interval", type=float, default=0.25, help="Sampling interval in seconds (default: 0.25)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    hwmon_dir = find_hwmon_paths(args.gpu, args.profile)
    if not hwmon_dir:
        print("Warning: no amdgpu hwmon with power1_average found; recording 0 W", file=sys.stderr)
    pwr_file = os.path.join(hwmon_dir, "power1_average") if hwmon_dir else None
    temp_hot_file = os.path.join(hwmon_dir, "temp2_input") if hwmon_dir else None
    temp_mem_file = os.path.join(hwmon_dir, "temp3_input") if hwmon_dir else None
    temp_edge_file = os.path.join(hwmon_dir, "temp1_input") if hwmon_dir else None

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json_output = os.path.splitext(args.output)[0] + ".json"

    samples = []
    start_time = time.time()
    prev_t = start_time
    prev_p = 0.0
    total_energy_joules = 0.0

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
                    power_w = float(_read_text(pwr_file)) / 1e6
                except Exception:
                    pass

            if temp_hot_file and os.path.exists(temp_hot_file):
                try:
                    hotspot_c = float(_read_text(temp_hot_file)) / 1000.0
                except Exception:
                    pass

            if temp_mem_file and os.path.exists(temp_mem_file):
                try:
                    mem_c = float(_read_text(temp_mem_file)) / 1000.0
                except Exception:
                    pass

            if temp_edge_file and os.path.exists(temp_edge_file):
                try:
                    edge_c = float(_read_text(temp_edge_file)) / 1000.0
                except Exception:
                    pass

            dt = t_now - prev_t
            if dt > 0:
                total_energy_joules += 0.5 * (prev_p + power_w) * dt
            prev_t = t_now
            prev_p = power_w

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

            sleep_time = args.interval - (time.time() - t_now)
            if sleep_time > 0:
                time.sleep(sleep_time)

    if samples:
        powers = [s["power_w"] for s in samples]
        hot_temps = [s["hotspot_temp_c"] for s in samples]
        mem_temps = [s["mem_temp_c"] for s in samples]
        total_duration = time.time() - start_time
        summary = {
            "duration_s": round(total_duration, 2),
            "sample_count": len(samples),
            "sample_interval_s": args.interval,
            "gpu_index": args.gpu,
            "gpu_profile": args.profile or os.environ.get("GPU_PROFILE", ""),
            "hwmon_dir": hwmon_dir,
            "avg_power_w": round(sum(powers) / len(powers), 2) if powers else 0.0,
            "max_power_w": round(max(powers), 2) if powers else 0.0,
            "min_power_w": round(min(powers), 2) if powers else 0.0,
            "total_energy_joules": round(total_energy_joules, 2),
            "energy_joules": round(total_energy_joules, 2),
            "avg_hotspot_c": round(sum(hot_temps) / len(hot_temps), 1) if hot_temps else 0.0,
            "max_hotspot_c": round(max(hot_temps), 1) if hot_temps else 0.0,
            "avg_mem_c": round(sum(mem_temps) / len(mem_temps), 1) if mem_temps else 0.0,
            "max_mem_c": round(max(mem_temps), 1) if mem_temps else 0.0
        }
        with open(json_output, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2)


if __name__ == "__main__":
    main()
