# Multi-Engine Throughput Benchmark Performance Report

- **Target Hardware**: AMD Radeon™ AI PRO R9700 (`gfx1201`, 32 GB GDDR6 VRAM)
- **Engines Tested**: `vllm llama.cpp sglang`
- **Benchmark Suite**: vLLM Throughput Matrix (`vllm bench serve`)
- **Reference**: [vLLM Throughput CLI Docs](https://docs.vllm.ai/en/latest/cli/bench/throughput/)
- **Timestamp**: Wed Sep 23 07:06:57 AM EDT 2026
- **Concurrency (CONC)**: 1
- **Prompt Sizing Strategy**: Dynamic by OSL (OSL=8192 -> 20, OSL!=8192 -> 50)

## Comparative Throughput & Latency Matrix

| Engine | Input Tokens (ISL) | Output Tokens (OSL) | Prompts | Total Tokens | Output Throughput | Total Throughput | Mean TTFT | Mean TPOT | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **vllm** | **128** | **64** | 1 | 192 | **10.94 tok/s** | **42.05 tok/s** | 132.09 ms | 90.76 ms | 5.85 s |
| **llama.cpp** | **128** | **64** | 1 | 192 | **2.20 tok/s** | **8.21 tok/s** | 9798.35 ms | 306.38 ms | 29.10 s |
