# AgentX concurrency tail pilot

This is a bounded load-spread pilot with a 180-minute campaign budget and 15-minute profiling windows. Request p99 values are observed order statistics, not stable population estimates.

| C | Requests | Sessions completed | Tail resolution | TTFT p50/p95 | E2E p50/p95 | Avg-ITL p50/p95 | >1s gaps / intervals | Prefix hit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 1 | 3.12% | 1234/4876 ms | 23267/57708 ms | 53/55 ms | 1 / 14341 | 93.169% |
| 2 | 39 | 2 | 2.56% | 1614/15071 ms | 17751/56017 ms | 53/77 ms | 32 / 15352 | 85.607% |
| 4 | 50 | 2 | 2.00% | 1154/23223 ms | 11232/58072 ms | 57/63 ms | 24 / 14925 | 83.083% |
| 8 | 53 | 1 | 1.89% | 1736/51832 ms | 26743/108964 ms | 59/391 ms | 363 / 15869 | 65.319% |
| 16 | 32 | 0 | 3.12% | 125267/213886 ms | 178267/278934 ms | 200/439 ms | 795 / 9145 | 9.054% |
| 32 | 32 | 0 | 3.12% | 331942/436376 ms | 373454/576018 ms | 213/731 ms | 731 / 6812 | 0.0% |

Raw request samples: `request_samples.csv`

Machine-readable analysis: `analysis.json`
