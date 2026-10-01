---
type: Technical Report
title: R9700 concurrency sweep
description: 'Narrative and earlier sweeps: R9700.md.'
tags:
- technical-report
- r9700
- sweep
status: stable
---

# R9700 concurrency sweep

Narrative and earlier sweeps: [R9700.md](R9700.md).

**Target Hardware**: AMD Radeon™ AI PRO R9700 (`gfx1201`, 32 GB GDDR6)
**Model**: `Qwen3.8-27B-Quark-AWQ-MXFP4` (Hybrid Attention: 48 GDN + 16 Full Softmax)
**Workload**: 8,192 Input Tokens / 1,024 Output Tokens
**Execution Date**: 2026-09-28 14:59:21 EDT

## Performance, Latency & Energy Ledger

| C | Aggregate tok/s | Per-stream tok/s | TTFT p50/p95 (ms) | TPOT p50/p95 (ms) | Avg / Max Power (W) | Total Energy (J) | J/token | tokens/Joule | Hotspot Max (°C) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **32.81** | 32.81 | 304.8 / 308.8 | 30.21 / 30.24 | 226.5 / 327.0 | 19,013.1 | **9.284** | **0.1077** | 89.0 | `PASSED` |
| **2** | **62.26** | 31.13 | 443.9 / 577.8 | 31.71 / 31.82 | 186.1 / 307.0 | 10,114.6 | **4.939** | **0.2025** | 87.0 | `PASSED` |
| **4** | **101.80** | 25.45 | 3319.7 / 5914.3 | 35.98 / 38.81 | 199.4 / 322.0 | 12,339.1 | **3.012** | **0.3320** | 89.0 | `PASSED` |
| **8** | **101.92** | 12.74 | 20445.2 / 45939.0 | 33.9 / 39.33 | 238.5 / 311.0 | 24,376.5 | **2.976** | **0.3361** | 91.0 | `PASSED` |
| **16** | **100.20** | 6.26 | 58410.1 / 128953.9 | 36.44 / 39.52 | 265.1 / 325.0 | 49,295.1 | **3.009** | **0.3324** | 94.0 | `PASSED` |

---
