---
type: Benchmark Report
title: Multi-Engine Throughput Benchmark Performance Report
description: 'Target Hardware: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6) GPU
  Profile: r9700 ROCm ISA: gfx1201.'
tags:
- benchmark-report
- throughput
- 20260928-144716
- benchmark
status: stable
---

# Multi-Engine Throughput Benchmark Performance Report

- **Target Hardware**: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6)
- **GPU Profile**: `r9700`
- **ROCm ISA**: `gfx1201`
- **Engines Tested**: `mxfp4`
- **Model**: `Qwen3.8-27B-Quark-AWQ-MXFP4`
- **Server**: `http://127.0.0.1:8000`
- **Benchmark Suite**: vLLM Throughput Matrix (`vllm bench serve`)
- **Reference**: [vLLM Throughput CLI Docs](https://docs.vllm.ai/en/latest/cli/bench/throughput/)
- **Timestamp**: Mon Sep 28 02:48:43 PM EDT 2026
- **Concurrency (CONC)**: 1
- **Prompt Sizing Strategy**: Dynamic by OSL (OSL=8192 -> 20, OSL!=8192 -> 50)

## Comparative Throughput & Latency Matrix

| Engine | Input Tokens (ISL) | Output Tokens (OSL) | Prompts | Total Tokens | Output Throughput | Total Throughput | Mean TTFT | Mean TPOT | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **mxfp4** | **8192** | **1024** | 2 | 18432 | **30.69 tok/s** | **277.78 tok/s** | 2463.63 ms | 30.21 ms | 66.74 s |

## Metric Definitions
- **Input Tokens (ISL)**: Number of prompt context tokens fed into the model.
- **Output Tokens (OSL)**: Number of generative completion tokens sampled.
- **Prompts**: Number of requests executed in this test slice.
- **Output Throughput**: Speed of generated tokens (`completion_tokens / duration`).
- **Total Throughput**: Combined prefill and decode token processing speed (`(input_tokens + output_tokens) / duration`).
- **Mean TTFT (Time to First Token)**: Prefill latency before the first token is emitted.
- **Mean TPOT (Time per Output Token)**: Average decode step time per subsequent token.
