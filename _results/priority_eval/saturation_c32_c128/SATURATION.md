# Instrumented C32–C128 (HF MXFP4 control)

Harness: `scripts/bench_saturation_instrumented.sh`  
Date: 2026-09-25  
Server: frozen HF control on `127.0.0.1:8000`, `VLLM_ROCM_USE_AITER` unset, stock `M_LEQ_8`, graphs O2.

Protocol: 1K input / **256** output, `ignore_eos`, one concurrent batch of size C (not the earlier 1K/1K campaign cells). Prometheus ITL/TTFT are request-window means. `amd-smi metric` was sampled **after** each cell, so SOCKET_POWER (~156–169 W) and 17% GFX are **cooldown**, not decode power. `vllm:kv_cache_usage_perc` is **0.0 after** the batch completes (KV released); it is not peak occupancy.

| C | output tok/s | mean e2e s | prom mean ITL ms | prom mean TTFT ms |
|---:|---:|---:|---:|---:|
| 32 | 1276.7 | 6.40 | 19.8 | 821 |
| 64 | 1620.2 | 10.05 | 32.5 | 737 |
| 96 | 1807.4 | 13.46 | 43.1 | 937 |
| 128 | 1954.5 | 16.50 | 52.9 | 1039 |

Throughput is still rising at C128 on this 1K/256 mix; there is no plateau yet. These cells are **not** interchangeable with the 1K/1K campaign (C32 1504 / C64 2017) because output length is 256 vs 1024.

Post-sweep C1 stream sample (1K/64, n=8): **81.51 tok/s**, TTFT p95 **56.8 ms**, token ITL p95 **11.80 ms** — control still healthy after the saturation window.

Next instrumentation pass should scrape `amd-smi` and KV **during** the in-flight batch, not after it.
