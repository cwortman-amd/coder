# Multi-Engine Throughput Benchmark Performance Report

- **Target Hardware**: AMD Radeon™ AI PRO R9700 (`gfx1201`, 32 GB GDDR6 VRAM)
- **Engines Tested**: `vllm llama.cpp sglang`
- **Benchmark Suite**: vLLM Throughput Matrix (`vllm bench serve`)
- **Reference**: [vLLM Throughput CLI Docs](https://docs.vllm.ai/en/latest/cli/bench/throughput/)
- **Timestamp**: Wed Sep 23 07:11:27 AM EDT 2026
- **Concurrency (CONC)**: 1
- **Prompt Sizing Strategy**: Dynamic by OSL (OSL=8192 -> 20, OSL!=8192 -> 50)

## Comparative Throughput & Latency Matrix

| Engine | Input Tokens (ISL) | Output Tokens (OSL) | Prompts | Total Tokens | Output Throughput | Total Throughput | Mean TTFT | Mean TPOT | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **vllm** | **128** | **64** | 1 | 192 | **11.03 tok/s** | **42.39 tok/s** | 132.65 ms | 90.01 ms | 5.80 s |
| **llama.cpp** | **128** | **64** | 1 | 192 | **2.21 tok/s** | **8.25 tok/s** | 9858.27 ms | 303.45 ms | 28.98 s |
| **sglang** | **128** | **64** | N/A | N/A | **UNAVAILABLE tok/s** | **UNAVAILABLE tok/s** | UNAVAILABLE ms | UNAVAILABLE ms | UNAVAILABLE s |

## Metric Definitions
- **Input Tokens (ISL)**: Number of prompt context tokens fed into the model.
- **Output Tokens (OSL)**: Number of generative completion tokens sampled.
- **Prompts**: Number of requests executed in this test slice.
- **Output Throughput**: Speed of generated tokens (`completion_tokens / duration`).
- **Total Throughput**: Combined prefill and decode token processing speed (`(input_tokens + output_tokens) / duration`).
- **Mean TTFT (Time to First Token)**: Prefill latency before the first token is emitted.
- **Mean TPOT (Time per Output Token)**: Average decode step time per subsequent token.
