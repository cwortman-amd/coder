# Multi-Engine Throughput Benchmark Performance Report

- **Target Hardware**: AMD Radeon™ AI PRO R9700 (`gfx1201`, 32 GB GDDR6 VRAM)
- **Engines Tested**: `llama.cpp`
- **Benchmark Suite**: vLLM Throughput Matrix (`vllm bench serve`)
- **Reference**: [vLLM Throughput CLI Docs](https://docs.vllm.ai/en/latest/cli/bench/throughput/)
- **Timestamp**: Wed Sep 23 07:35:09 AM EDT 2026
- **Concurrency (CONC)**: 1
- **Prompt Sizing Strategy**: Dynamic by OSL (OSL=8192 -> 20, OSL!=8192 -> 50)

## Comparative Throughput & Latency Matrix

| Engine | Input Tokens (ISL) | Output Tokens (OSL) | Prompts | Total Tokens | Output Throughput | Total Throughput | Mean TTFT | Mean TPOT | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **llama.cpp** | **8192** | **1024** | 1 | 9216 | **ERROR tok/s** | **ERROR tok/s** | ERROR ms | ERROR ms | ERROR s |

## Metric Definitions
- **Input Tokens (ISL)**: Number of prompt context tokens fed into the model.
- **Output Tokens (OSL)**: Number of generative completion tokens sampled.
- **Prompts**: Number of requests executed in this test slice.
- **Output Throughput**: Speed of generated tokens (`completion_tokens / duration`).
- **Total Throughput**: Combined prefill and decode token processing speed (`(input_tokens + output_tokens) / duration`).
- **Mean TTFT (Time to First Token)**: Prefill latency before the first token is emitted.
- **Mean TPOT (Time per Output Token)**: Average decode step time per subsequent token.
