# Multi-Engine Throughput Benchmark Performance Report

- **Target Hardware**: AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6)
- **GPU Profile**: `r9700`
- **ROCm ISA**: `gfx1201`
- **Engines Tested**: `vllm`
- **Model**: `Qwen3.8-27B-Quark-AWQ-MXFP4`
- **Server**: `http://127.0.0.1:8000`
- **Benchmark Suite**: vLLM Throughput Matrix (`vllm bench serve`)
- **Reference**: [vLLM Throughput CLI Docs](https://docs.vllm.ai/en/latest/cli/bench/throughput/)
- **Timestamp**: Wed Oct  7 08:49:47 PM EDT 2026
- **Concurrency (CONC)**: 1
- **Prompt Sizing Strategy**: Dynamic by OSL (OSL=8192 -> 20, OSL!=8192 -> 50)

## Comparative Throughput & Latency Matrix

| Engine | Input Tokens (ISL) | Output Tokens (OSL) | Prompts | Total Tokens | Output Throughput | TTFT | TPOT | Power (W / % TDP) | Mem Bandwidth (% Peak) | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **vllm** | **8192** | **1024** | 50 | 460800 | **29.50 tok/s** | 3129.0 ms | 30.87 ms | 206.77 W (68.92%) | 567.3 GB/s (59.1%) | 1735.69 s |
| **vllm** | **1024** | **8192** | 20 | 184320 | **32.56 tok/s** | 481.1 ms | 30.66 ms | 208.74 W (69.58%) | 626.2 GB/s (65.2%) | 5032.11 s |
| **vllm** | **1024** | **1024** | 50 | 102400 | **32.35 tok/s** | 482.8 ms | 30.48 ms | 207.02 W (69.01%) | 622.1 GB/s (64.8%) | 1582.55 s |

## Metric Definitions
- **Input Tokens (ISL)**: Number of prompt context tokens fed into the model.
- **Output Tokens (OSL)**: Number of generative completion tokens sampled.
- **Prompts**: Number of requests executed in this test slice.
- **Output Throughput**: Speed of generated tokens (`completion_tokens / duration`).
- **Total Throughput**: Combined prefill and decode token processing speed (`(input_tokens + output_tokens) / duration`).
- **TTFT (Time to First Token)**: Median latency before the first token is emitted.
- **TPOT (Time per Output Token)**: Average decode step time per subsequent token.
- **Power (W / % TDP)**: Active board power consumption in Watts and percentage of device TDP.
- **Mem Bandwidth (% Peak)**: Inferred weight memory streaming traffic and percentage of device peak memory bandwidth.
