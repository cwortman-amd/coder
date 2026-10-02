# AgentX concurrency tail pilot

This is a bounded load-spread pilot with a 40-minute campaign budget and 15-minute profiling windows. Request p99 values are observed order statistics, not stable population estimates.

| C | Requests | Sessions completed | Tail resolution | TTFT p50/p95 | E2E p50/p95 | Avg-ITL p50/p95 | >1s gaps / intervals | Prefix hit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 21 | 1 | 4.76% | 835/6765 ms | 31249/90597 ms | 86/115 ms | 7 / 12330 | 85.296% |
| 4 | 46 | 2 | 2.17% | 864/4524 ms | 15557/87602 ms | 95/109 ms | 15 / 12321 | 88.835% |

Raw request samples: `request_samples.csv`

Machine-readable analysis: `analysis.json`
