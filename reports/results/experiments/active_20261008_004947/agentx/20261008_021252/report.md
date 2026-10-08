# AgentX concurrency tail pilot

This is a bounded load-spread pilot with a 120-minute campaign budget and 15-minute profiling windows. Request p99 values are observed order statistics, not stable population estimates.

| C | Requests | Sessions completed | Tail resolution | TTFT p50/p95 | E2E p50/p95 | Avg-ITL p50/p95 | >1s gaps / intervals | Prefix hit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41 | None | 2.44% | 1171/7670 ms | 13819/36081 ms | 32/34 ms | 1 / 19779 | 91.966% |
| 2 | 52 | None | 1.92% | 1270/15562 ms | 10745/59175 ms | 33/63 ms | 43 / 20061 | 86.646% |
| 4 | 57 | None | 1.75% | 1094/25029 ms | 7727/46421 ms | 35/76 ms | 98 / 18139 | 79.704% |
| 8 | 61 | None | 1.64% | 1664/33570 ms | 16005/89861 ms | 36/157 ms | 227 / 19124 | 68.668% |
| 16 | 35 | None | 2.86% | 119472/196461 ms | 155529/286285 ms | 186/506 ms | 840 / 10384 | 8.289% |
| 32 | — | — | — | campaign_timeout | — | — | — | — |

Raw request samples: `request_samples.csv`

Machine-readable analysis: `analysis.json`
