# AgentX concurrency tail pilot

This is a bounded load-spread pilot with a 30-minute campaign budget and 15-minute profiling windows. Request p99 values are observed order statistics, not stable population estimates.

| C | Requests | Sessions completed | Tail resolution | TTFT p50/p95 | E2E p50/p95 | Avg-ITL p50/p95 | >1s gaps / intervals | Prefix hit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 48 | None | 2.08% | 129407/232155 ms | 207664/442445 ms | 604/789 ms | 690 / 12497 | 12.536% |

Raw request samples: `request_samples.csv`

Machine-readable analysis: `analysis.json`
