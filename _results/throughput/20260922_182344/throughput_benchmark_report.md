# Throughput Benchmark Performance Report

- **Model Evaluated**: `Qwen/Qwen3.8-27B-FP8`
- **Target Hardware**: AMD Radeon™ AI PRO R9700 (`gfx1201`, 32 GB GDDR6 VRAM)
- **Benchmark Suite**: vLLM Throughput Matrix (`vllm bench`)
- **Reference**: [vLLM Throughput CLI Docs](https://docs.vllm.ai/en/latest/cli/bench/throughput/)
- **Timestamp**: Tue Sep 22 06:49:04 PM EDT 2026
- **Prompts per Configuration**: 1

## Throughput & Latency Matrix

| Input Tokens (ISL) | Output Tokens (OSL) | Total Tokens | Output Throughput | Total Throughput | Mean TTFT | Mean TPOT | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2048** | **512** | 2560 | **10.56 tok/s** | **53.91 tok/s** | 314.99 ms | 94.27 ms | 48.49 s |
| **2048** | **2048** | 4096 | **10.37 tok/s** | **21.01 tok/s** | 315.81 ms | 96.33 ms | 197.50 s |
| **128** | **2048** | 2176 | **10.85 tok/s** | **11.81 tok/s** | 129.37 ms | 92.15 ms | 188.77 s |
| **1024** | **1024** | 2048 | **10.74 tok/s** | **22.06 tok/s** | 577.65 ms | 92.59 ms | 95.30 s |
| **8192** | **1024** | 9216 | **9.01 tok/s** | **81.61 tok/s** | 4398.71 ms | 106.74 ms | 113.59 s |
| **1024** | **8192** | 9216 | **10.00 tok/s** | **11.32 tok/s** | 577.54 ms | 99.94 ms | 819.21 s |

## Metric Definitions
- **Input Tokens (ISL)**: Number of prompt context tokens fed into the model.
- **Output Tokens (OSL)**: Number of generative completion tokens sampled.
- **Output Throughput**: Speed of generated tokens (`completion_tokens / duration`).
- **Total Throughput**: Combined prefill and decode token processing speed (`(input_tokens + output_tokens) / duration`).
- **Mean TTFT (Time to First Token)**: Prefill latency before the first token is emitted.
- **Mean TPOT (Time per Output Token)**: Average decode step time per subsequent token.
