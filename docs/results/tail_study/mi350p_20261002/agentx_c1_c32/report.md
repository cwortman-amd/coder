# AgentX concurrency tail pilot

This is a bounded load-spread pilot with a 95-minute campaign budget and 15-minute profiling windows. Request p99 values are observed order statistics, not stable population estimates.

| C | Requests | Sessions completed | Tail resolution | TTFT p50/p95 | E2E p50/p95 | Avg-ITL p50/p95 | >1s gaps / intervals | Prefix hit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | None | 5.00% | 731/4008 ms | 43515/101838 ms | 130/145 ms | 0 / 8089 | 92.313% |
| 2 | 18 | None | 5.56% | 910/6764 ms | 34755/108021 ms | 105/131 ms | 0 / 9611 | 86.835% |
| 4 | 44 | None | 2.27% | 824/7425 ms | 19044/108814 ms | 115/129 ms | 0 / 10191 | 88.346% |
| 8 | 52 | None | 1.92% | 876/7988 ms | 21299/107865 ms | 122/152 ms | 3 / 15590 | 90.583% |
| 16 | 107 | None | 0.94% | 1362/21186 ms | 25836/156535 ms | 138/248 ms | 96 / 25729 | 82.014% |
| 32 | 48 | None | 2.08% | 129407/232155 ms | 207664/442445 ms | 604/789 ms | 690 / 12497 | 12.536% |

Raw request samples: `request_samples.csv`

Machine-readable analysis: `analysis.json`
