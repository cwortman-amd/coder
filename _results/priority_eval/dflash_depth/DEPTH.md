# DFlash depth 1–5 (matched protocol, 25 Sep 2026)

Harness: `scripts/dflash_depth_sweep.sh`. Control rows are the prior crossover process (`crossover/baseline_matched_c*.json`, **78.80 / 318.04 / 426.95 / 557.14**). DFLASH-7 rows are the prior crossover, not re-run in this process.

1K/1K `ignore_eos`; C1 = 4 sequential prompts; Cn = n concurrent prompts. Stream: 1K/256, 8 sequential, token-arrival ITL.

| Depth | C1 tok/s | vs ctrl C1 | C5 | C6 | C8 | vs ctrl C8 | C1 TTFT p95 ms | C1 ITL p95 ms | window accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 78.80 | — | 318.04 | **426.95** | **557.14** | — | 75.7 | 12.05 | — |
| 1 | 85.96 | +9.1% | 285.40 | 337.16 | 420.24 | −24.6% | 137.3 | 18.43 | 64.8% (pos 0 only) |
| 2 | 96.71 | +22.7% | 343.23 | 399.01 | 495.54 | −11.1% | 136.4 | 18.54 | 49.5% (pos 0–1) |
| **3** | **113.75** | **+44.3%** | **353.60** | 416.52 | 510.81 | −8.3% | **136.4** | 18.82 | 39.0% (pos 0–2) |
| 4 | 84.29 | +7.0% | 346.94 | 329.79 | 428.50 | −23.1% | 144.5 | 22.69 | 26.2% (pos 0–3) |
| 5 | 112.58 | +42.9% | 344.63 | 406.33 | 516.41 | −7.3% | **134.5** | 21.21 | 29.1% (pos 0–4) |
| 7 (prior) | 102.68 | +30.3% | 336.65 | 336.09 | 422.30 | −24.2% | 244.4 | 20.57 | ~17% (pos 0–1 only) |

Crossover remains **between C5 and C6** for depths 2, 3, and 5. Depth 3 almost ties C6 (−2.4%). Depth 1 already loses at C5. Depth 4 C1 is a one-run dip (also weak in the earlier DFLASH-4 81.48 C1); do not promote 4 without a repeat.

**If an optional DFlash profile is shipped, use depth 3, not 7**, on this stack: higher C1, lower TTFT than DFLASH-7, and less C8 tax. Completed n=24 streaming confirms it is still not an interactive improvement: TTFT p95 **58.6 → 127.5 ms** and ITL p95 **12.07 → 18.84 ms** versus control. Greedy equality is **101/116 (87.1%)**, so it is not bit-identical. Use only as an opt-in long-output profile.

Accepted-position deltas mix C1–C8 in one Prometheus window. Shorter depths accept on all requested positions; DFLASH-7 previously accepted only 0–1 of 0–6.
