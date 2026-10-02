"""Poll vLLM /metrics for running, waiting, and KV-cache usage."""

from __future__ import annotations

import threading
import time
import urllib.request


def gauges(base: str) -> dict[str, float]:
    url = base.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    wanted = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:kv_cache_usage_perc": "kv_perc",
    }
    found: dict[str, float] = {}
    with urllib.request.urlopen(f"{url}/metrics", timeout=2) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line or line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name in wanted:
                found[wanted[name]] = float(line.rsplit(" ", 1)[-1])
    return found


def sample_loop(base: str, stop: threading.Event, sink: list[dict]) -> None:
    while not stop.is_set():
        try:
            row = gauges(base)
            row["t"] = time.time()
            sink.append(row)
        except Exception as exc:  # noqa: BLE001
            sink.append({"t": time.time(), "error": str(exc)})
        stop.wait(0.25)
