#!/usr/bin/env python3
"""One begin/end GPU window for benches and profiling captures.

Samples amd-smi for the window. If the Device Metrics Exporter or Prometheus
is already listening, the same window also records those series. This process
does not start or remove those containers.

The summary JSON keeps the keys the benches already read: avg_power_w,
power_util_pct, energy_joules, total_energy_joules.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalog  # noqa: E402

EXPORTER_URL = os.environ.get("DME_METRICS_URL", "http://127.0.0.1:5000/metrics")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://127.0.0.1:9090")
NODE_URL = os.environ.get("NODE_EXPORTER_URL", "http://127.0.0.1:9100/metrics")


def _num(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        value = value.get("value")
    if value is None or value == "N/A":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def read_proc_stat() -> tuple[int, int]:
    idle = total = 0
    for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
        if not line.startswith("cpu "):
            continue
        parts = [int(part) for part in line.split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        total = sum(parts)
        break
    return idle, total


def cpu_util(prev: tuple[int, int], cur: tuple[int, int]) -> Optional[float]:
    idle_d = cur[0] - prev[0]
    total_d = cur[1] - prev[1]
    if total_d <= 0:
        return None
    return 100.0 * (1.0 - idle_d / total_d)


def fetch_text(url: str, timeout: float = 0.4) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def parse_exposition(text: str) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(" ")
        name = name.split("{", 1)[0]
        try:
            found.setdefault(name, []).append(float(rest.split()[0]))
        except (ValueError, IndexError):
            continue
    return found


def interested(name: str) -> bool:
    upper = name.upper()
    tokens = (
        "POWER",
        "GFX_ACTIVITY",
        "UMC_ACTIVITY",
        "PCIE_BANDWIDTH",
        "PCIE_BIDIRECTIONAL",
        "USED_VRAM",
        "NODE_CPU_SECONDS",
    )
    return any(token in upper for token in tokens)


def filter_series(parsed: dict[str, list[float]]) -> dict[str, float]:
    kept: dict[str, float] = {}
    for name, values in parsed.items():
        if interested(name) and values:
            kept[name] = sum(values) / len(values)
    return kept


def sample_amd_smi(gpu: Optional[int]) -> list[dict[str, Any]]:
    cmd = ["amd-smi", "metric", "-u", "-m", "-p", "-P", "-c", "--json"]
    if gpu is not None:
        cmd[2:2] = ["-g", str(gpu)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
        payload = json.loads(proc.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []
    rows = []
    for info in payload.get("gpu_data") or []:
        if not isinstance(info, dict) or info.get("gpu") is None:
            continue
        usage = info.get("usage") or {}
        mem = info.get("mem_usage") or {}
        power = info.get("power") or {}
        pcie = info.get("pcie") or {}
        clock = info.get("clock") or {}
        gfx_clk = None
        mem_clk = None
        if isinstance(clock, dict):
            for key, node in clock.items():
                if gfx_clk is None and "GFX" in key.upper():
                    gfx_clk = _clk(node)
                if mem_clk is None and "MEM" in key.upper():
                    mem_clk = _clk(node)
        used_mb = _num((mem.get("used_vram") or {}))
        total_mb = _num((mem.get("total_vram") or {}))
        rows.append({
            "gpu": info.get("gpu"),
            "socket_power_w": _num(power.get("socket_power")),
            "gfx_activity_pct": _num(usage.get("gfx_activity")),
            "umc_activity_pct": _num(usage.get("umc_activity")),
            "vram_used_mb": used_mb,
            "vram_total_mb": total_mb,
            "vram_used_bytes": None if used_mb is None else used_mb * 1024 * 1024,
            "vram_total_bytes": None if total_mb is None else total_mb * 1024 * 1024,
            "pcie_mbps": _num((pcie.get("bandwidth") or {})),
            "pcie_gt_s": _num((pcie.get("speed") or {})),
            "gfx_clock_mhz": gfx_clk,
            "mem_clock_mhz": mem_clk,
        })
    return rows


def _clk(node: Any) -> Optional[float]:
    if isinstance(node, dict):
        for key in ("clk", "current", "value"):
            if key in node:
                found = _num(node[key])
                if found is not None:
                    return found
        for child in node.values():
            found = _clk(child)
            if found is not None:
                return found
    return _num(node)


def _paths(output: Path) -> dict[str, Path]:
    return {
        "begin": Path(str(output) + ".begin"),
        "pid": Path(str(output) + ".pid"),
        "samples": Path(str(output) + ".samples.jsonl"),
    }


def cmd_begin(output: Path, gpu: Optional[int], profile: str, interval: float, gpus_json: str) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = _paths(output)
    meta = {
        "start_unix": time.time(),
        "gpu": gpu,
        "profile": profile,
        "interval": interval,
        "samples": str(paths["samples"]),
        "gpus_json": gpus_json,
        "exporter": bool(fetch_text(EXPORTER_URL)),
        "prometheus": bool(fetch_text(PROMETHEUS_URL + "/-/ready")),
        "node_exporter": bool(fetch_text(NODE_URL)),
    }
    paths["begin"].write_text(json.dumps(meta), encoding="utf-8")
    paths["samples"].write_text("", encoding="utf-8")
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "_sample",
        "--output", str(output),
        "--interval", str(interval),
        "--profile", profile,
    ]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    proc = subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    paths["pid"].write_text(str(proc.pid), encoding="utf-8")
    return 0


def cmd_sample(output: Path, gpu: Optional[int], interval: float, scrape_exporter: bool) -> int:
    paths = _paths(output)
    running = True

    def _stop(signum, frame):  # noqa: ARG001
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    prev_cpu = read_proc_stat()
    with paths["samples"].open("a", encoding="utf-8") as handle:
        while running:
            started = time.time()
            cpu_now = read_proc_stat()
            row = {
                "unix_s": started,
                "gpus": sample_amd_smi(gpu),
                "cpu_util_pct": cpu_util(prev_cpu, cpu_now),
            }
            prev_cpu = cpu_now
            if scrape_exporter:
                scraped = filter_series(parse_exposition(fetch_text(EXPORTER_URL, timeout=0.8)))
                if scraped:
                    row["exporter"] = scraped
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            time.sleep(max(0.0, interval - (time.time() - started)))
    return 0


def _column(samples: list[dict[str, Any]], gpu_filter: Optional[int], key: str) -> list[float]:
    values = []
    for row in samples:
        for gpu in row.get("gpus") or []:
            if gpu_filter is not None and int(gpu.get("gpu", -1)) != gpu_filter:
                continue
            value = gpu.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
    return values


def summarize(samples: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    spec = catalog.gpu(meta.get("profile") or os.environ.get("GPU_PROFILE") or "r9700")
    gpu_filter = meta.get("gpu")
    if gpu_filter is not None:
        gpu_filter = int(gpu_filter)
    power = _column(samples, gpu_filter, "socket_power_w")
    duration = 0.0
    if samples:
        duration = max(0.0, time.time() - float(meta.get("start_unix") or samples[0]["unix_s"]))
    energy = 0.0
    interval = float(meta.get("interval") or 0.25)
    previous = power[0] if power else 0.0
    for watts in power:
        energy += 0.5 * (previous + watts) * interval
        previous = watts
    avg = _mean(power) or 0.0
    peak = max(power) if power else 0.0
    tdp = float(spec["tdp_w"])
    peak_bw = float(spec["peak_bw_gbs"])
    umc = _column(samples, gpu_filter, "umc_activity_pct")
    umc_mean = _mean(umc)
    summary = {
        "source": "amd-smi",
        "duration_s": round(duration, 2),
        "sample_count": len(samples),
        "sample_interval_s": interval,
        "gpu_index": gpu_filter if gpu_filter is not None else "all",
        "gpu_profile": spec["profile"],
        "device_tdp_w": tdp,
        "device_peak_bw_gbs": peak_bw,
        "avg_power_w": round(avg, 2),
        "max_power_w": round(peak, 2),
        "peak_power_w": round(peak, 2),
        "min_power_w": round(min(power), 2) if power else 0.0,
        "power_util_pct": round((avg / tdp) * 100.0, 2) if tdp else 0.0,
        "max_power_util_pct": round((peak / tdp) * 100.0, 2) if tdp else 0.0,
        "total_energy_joules": round(energy, 2),
        "energy_joules": round(energy, 2),
        "gfx_activity_pct_mean": _round(_mean(_column(samples, gpu_filter, "gfx_activity_pct"))),
        "umc_activity_pct_mean": _round(umc_mean),
        "umc_gbs_estimate": _round(None if umc_mean is None else umc_mean / 100.0 * peak_bw),
        "umc_gbs_estimate_note": "UMC percent times catalog peak GB/s. An estimate, not a calibrated HBM counter.",
        "vram_used_mb_mean": _round(_mean(_column(samples, gpu_filter, "vram_used_mb"))),
        "pcie_mbps_mean": _round(_mean(_column(samples, gpu_filter, "pcie_mbps"))),
        "pcie_mbps_peak": _round(max(_column(samples, gpu_filter, "pcie_mbps")) if _column(samples, gpu_filter, "pcie_mbps") else None),
        "cpu_util_pct_mean": _round(_mean([row["cpu_util_pct"] for row in samples if isinstance(row.get("cpu_util_pct"), (int, float))])),
        "gfx_clock_mean_mhz": _round(_mean(_column(samples, gpu_filter, "gfx_clock_mhz"))),
        "mem_clock_mean_mhz": _round(_mean(_column(samples, gpu_filter, "mem_clock_mhz"))),
        "power_mean_w": round(avg, 2),
        "power_max_w": round(peak, 2),
        "exporter_scraped": bool(meta.get("exporter")),
    }
    exporter_rows = [row.get("exporter") for row in samples if isinstance(row.get("exporter"), dict)]
    if exporter_rows:
        summary["source"] = "amd-smi+device-metrics-exporter"
        summary["exporter_series"] = _merge_exporter(exporter_rows)
    if meta.get("prometheus"):
        prom = prometheus_window(float(meta.get("start_unix") or time.time()), time.time())
        if prom:
            summary["prometheus"] = prom
            summary["source"] = summary["source"] + "+prometheus"
    return summary


def _round(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value, 2)


def _merge_exporter(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = set()
    for row in rows:
        keys.update(row)
    merged = {}
    for key in sorted(keys):
        values = [row[key] for row in rows if key in row]
        if values:
            merged[key] = round(sum(values) / len(values), 3)
    return merged


def prometheus_window(start: float, end: float) -> dict[str, Any]:
    """Query the standing Prometheus only. Empty when it is not up or the names differ."""
    queries = {
        "gpu_power": "avg(GPU_POWER_USAGE)",
        "gpu_gfx": "avg(GPU_GFX_ACTIVITY)",
        "gpu_umc": "avg(GPU_UMC_ACTIVITY)",
        "pcie_mbps": "avg(PCIE_BANDWIDTH)",
    }
    found: dict[str, Any] = {}
    for label, query in queries.items():
        url = (
            f"{PROMETHEUS_URL}/api/v1/query_range?query={urllib.parse.quote(query)}"
            f"&start={start:.3f}&end={end:.3f}&step=1"
        )
        body = fetch_text(url, timeout=1.0)
        if not body:
            return {}
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return {}
        series = ((payload.get("data") or {}).get("result")) or []
        if not series:
            continue
        values = []
        for point in series[0].get("values") or []:
            try:
                values.append(float(point[1]))
            except (TypeError, ValueError, IndexError):
                continue
        if values:
            found[label] = {
                "mean": round(sum(values) / len(values), 3),
                "max": round(max(values), 3),
                "p95": round(_percentile(values, 0.95) or 0.0, 3),
            }
    cpu = meta_node_query(start, end)
    if cpu:
        found["cpu_util_pct"] = cpu
    return found


def meta_node_query(start: float, end: float) -> Optional[dict[str, float]]:
    query = '100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[30s])))'
    url = (
        f"{PROMETHEUS_URL}/api/v1/query_range?query={urllib.parse.quote(query)}"
        f"&start={start:.3f}&end={end:.3f}&step=15"
    )
    body = fetch_text(url, timeout=1.0)
    if not body:
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    series = ((payload.get("data") or {}).get("result")) or []
    if not series:
        return None
    values = []
    for point in series[0].get("values") or []:
        try:
            values.append(float(point[1]))
        except (TypeError, ValueError, IndexError):
            continue
    if not values:
        return None
    return {"mean": round(sum(values) / len(values), 2), "max": round(max(values), 2)}


def write_gpus_json(path: Path, samples: list[dict[str, Any]], interval: float) -> None:
    shaped = []
    for row in samples:
        gpus = {}
        for gpu in row.get("gpus") or []:
            gpus[f"gpu{gpu.get('gpu')}"] = {
                "use_pct": gpu.get("gfx_activity_pct"),
                "vram_used_bytes": gpu.get("vram_used_bytes"),
                "vram_total_bytes": gpu.get("vram_total_bytes"),
            }
        shaped.append({"t": row.get("unix_s"), "gpus": gpus})
    path.write_text(json.dumps({"interval_s": interval, "samples": shaped}) + "\n", encoding="utf-8")


def cmd_end(output: Path) -> int:
    paths = _paths(output)
    meta = {}
    if paths["begin"].is_file():
        meta = json.loads(paths["begin"].read_text(encoding="utf-8"))
    if paths["pid"].is_file():
        pid = int(paths["pid"].read_text(encoding="utf-8").strip() or "0")
        if pid:
            try:
                os.kill(pid, signal.SIGINT)
            except OSError:
                pass
            for _ in range(20):
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
                time.sleep(0.1)
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
    samples = []
    if paths["samples"].is_file():
        for line in paths["samples"].read_text(encoding="utf-8").splitlines():
            if line.strip():
                samples.append(json.loads(line))
    summary = summarize(samples, meta)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    gpus_json = meta.get("gpus_json") or ""
    if gpus_json:
        write_gpus_json(Path(gpus_json), samples, float(meta.get("interval") or 2.0))
    paths["pid"].unlink(missing_ok=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Begin or end a telemetry window")
    sub = parser.add_subparsers(dest="cmd", required=True)

    begin = sub.add_parser("begin")
    begin.add_argument("--output", type=Path, required=True)
    begin.add_argument("--gpu", type=int, default=None)
    begin.add_argument("--all-gpus", action="store_true")
    begin.add_argument("--profile", default=os.environ.get("GPU_PROFILE", ""))
    begin.add_argument("--interval", type=float, default=0.25)
    begin.add_argument("--gpus-json", default="")

    end = sub.add_parser("end")
    end.add_argument("--output", type=Path, required=True)

    sample = sub.add_parser("_sample")
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--gpu", type=int, default=None)
    sample.add_argument("--interval", type=float, default=0.25)
    sample.add_argument("--profile", default="")

    args = parser.parse_args()
    if args.cmd == "begin":
        gpu = None if args.all_gpus else (0 if args.gpu is None else args.gpu)
        return cmd_begin(args.output, gpu, args.profile, args.interval, args.gpus_json)
    if args.cmd == "end":
        return cmd_end(args.output)
    meta = {}
    begin_path = Path(str(args.output) + ".begin")
    if begin_path.is_file():
        meta = json.loads(begin_path.read_text(encoding="utf-8"))
    return cmd_sample(args.output, args.gpu, args.interval, bool(meta.get("exporter")))


if __name__ == "__main__":
    raise SystemExit(main())
