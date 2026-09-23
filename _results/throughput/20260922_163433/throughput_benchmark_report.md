# Throughput Benchmark Performance Report

- **Model Evaluated**: `Qwen/Qwen3-0.6B`
- **Target Hardware**: AMD Radeon™ AI PRO R9700 (`gfx1201`, 32 GB GDDR6 VRAM)
- **Benchmark Suite**: vLLM Throughput Matrix (`vllm bench`)
- **Reference**: [vLLM Throughput CLI Docs](https://docs.vllm.ai/en/latest/cli/bench/throughput/)
- **Timestamp**: Tue Sep 22 04:38:28 PM EDT 2026
- **Prompts per Configuration**: 2

## Throughput & Latency Matrix

| Input Tokens (ISL) | Output Tokens (OSL) | Total Tokens | Output Throughput | Total Throughput | Mean TTFT | Mean TPOT | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2048** | **512** | 2560 | **287.50 tok/s** | **1442.00 tok/s** | 115.24 ms | 6.74 ms | 3.56 s |
| **2048** | **2048** | 4096 | **267.48 tok/s** | **536.00 tok/s** | 63.60 ms | 7.45 ms | 15.31 s |
| **128** | **2048** | 2176 | **418.77 tok/s** | **446.58 tok/s** | 24.00 ms | 4.76 ms | 9.78 s |
| **1024** | **1024** | 2048 | **389.23 tok/s** | **781.51 tok/s** | 32.44 ms | 5.11 ms | 5.26 s |
| **8192** | **1024** | 9216 | **127.90 tok/s** | **1152.11 tok/s** | 926.45 ms | 14.72 ms | 16.01 s |
| **1024** | **8192** | 9216 | **201.06 tok/s** | **226.39 tok/s** | 32.59 ms | 9.94 ms | 81.49 s |

## Metric Definitions
- **Input Tokens (ISL)**: Number of prompt context tokens fed into the model.
- **Output Tokens (OSL)**: Number of generative completion tokens sampled.
- **Output Throughput**: Speed of generated tokens (`completion_tokens / duration`).
- **Total Throughput**: Combined prefill and decode token processing speed (`(input_tokens + output_tokens) / duration`).
- **Mean TTFT (Time to First Token)**: Prefill latency before the first token is emitted.
- **Mean TPOT (Time per Output Token)**: Average decode step time per subsequent token.
