# MI350P GPT-OSS-120B AIM sweep

**Date:** 29 September 2026  
**SKU:** 1× AMD Instinct MI350P (`gfx950`, PCI `0x75a8`, 144 GB HBM3E), GPU 0. GPU 1 stayed idle.  
**Image:** `amdenterpriseai/liqid-aim-instinct-openai-gpt-oss-120b:0.12.0-rc14` (vLLM 0.19.1, ROCm 7.13)  
**Checkpoint:** local `openai/gpt-oss-120b` MXFP4 snapshot, served offline  
**Profile:** `vllm-mi350p-mxfp4-tp1-throughput` (manual selection, TP=1, `max-model-len` 131072, `max-num-batched-tokens` 2048, `max-num-seqs` 512, `ROCM_AITER_UNIFIED_ATTN`, prefix caching off, async scheduling off)

This page is the AIM tech-preview record. It is separate from the Quark Qwen campaign in [MI350P.md](MI350P.md) and from the 20B R9700 recipe in [GPT-OSS.md](GPT-OSS.md). Llama 3.3 70B, Mistral Small 3.2 24B, and Gemma 3 27B AIM images were not run here.

## Method

`vllm bench serve` inside the serving container, random dataset, exact input and output lengths, `--ignore-eos`, `--percentile-metrics ttft,tpot,itl,e2el`, percentiles 75/90/99. Prompt count is `10 × concurrency`. Warmup count is `2 × concurrency`. One run per cell. JSON: `_results/aim_mi350p/`. Runner: `scripts/bench_aim_gptoss_mi350p.sh`.

Three cells were still in progress when the sweep was stopped: 65,536/1,024 at concurrency 16, and 122,880/1,024 at concurrency 1 and 4.

At 32,768/1,024 concurrency 32, one of 320 requests failed. The tok/s below is the client’s completed-token rate (319 finished).

## Output tok/s

| Shape (in/out) | C1 | C4 | C8 | C16 | C32 | C64 | C128 | C256 | C512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/2048 | 230.52 | 679.49 | 1,086.71 | 1,771.17 | 2,671.44 | 3,817.16 | 5,459.07 | 7,556.15 | |
| 256/256 | 225.95 | 655.26 | 1,033.20 | 1,666.17 | 2,535.89 | 3,664.10 | 5,211.13 | 7,244.94 | |
| 1024/1024 | 229.43 | 669.61 | 1,047.02 | 1,655.99 | 2,428.83 | 3,392.96 | 4,795.34 | 6,377.00 | 8,459.24 |
| 2048/128 | 207.66 | 537.72 | 761.26 | 1,054.71 | 1,379.30 | 1,654.06 | 1,902.30 | 1,894.67 | |
| 8192/1024 | 213.36 | 558.73 | 796.45 | 1,127.85 | 1,449.02 | 1,737.42 | 1,989.80 | | |
| 8192/3072 | 220.15 | 606.61 | 899.24 | 1,334.69 | 1,804.53 | 2,275.34 | 2,704.29 | 2,799.02 | |
| 32768/1024 | 163.67 | 301.84 | 360.68 | 417.07 | 453.41 | | | | |
| 65536/1024 | 111.02 | 152.24 | 165.83 | | | | | | |

## Median TTFT (ms)

| Shape (in/out) | C1 | C4 | C8 | C16 | C32 | C64 | C128 | C256 | C512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/2048 | 29.07 | 44.79 | 47.29 | 66.61 | 85.10 | 109.07 | 117.77 | 135.69 | |
| 256/256 | 24.74 | 50.98 | 65.87 | 78.61 | 95.08 | 101.99 | 110.43 | 124.30 | |
| 1024/1024 | 40.74 | 100.20 | 132.31 | 150.03 | 108.65 | 151.77 | 180.69 | 204.67 | 251.70 |
| 2048/128 | 60.32 | 156.32 | 200.50 | 170.93 | 160.93 | 171.78 | 562.07 | 9,158.32 | |
| 8192/1024 | 252.88 | 483.78 | 479.90 | 501.92 | 490.31 | 531.85 | 537.20 | | |
| 8192/3072 | 255.44 | 527.62 | 543.83 | 510.92 | 492.83 | 542.43 | 560.95 | 42,328.15 | |
| 32768/1024 | 1,407.80 | 2,568.11 | 2,518.58 | 2,507.58 | 2,353.53 | | | | |
| 65536/1024 | 3,926.82 | 7,435.15 | 6,905.88 | | | | | | |

Short-output cells flatten once prefill dominates. 2,048/128 stops rising after C128 (1,902 tok/s at C128, 1,895 at C256) and median TTFT at C256 is 9.2 s. 8,192/3,072 reaches 2,799 tok/s at C256 with median TTFT 42.3 s.

## Sheet anchors

The pasted MI350P AIMS sheet quoted 3-run medians. This table is one run on the same profile family. Only the cells whose sheet figures were in that paste are listed.

| Cell | This run tok/s | Sheet median tok/s | This median TTFT (ms) | Sheet median TTFT (ms) |
|---|---:|---:|---:|---:|
| 128/2048 C1 | 230.52 | 221.07 | 29.07 | 20.55 |
| 128/2048 C256 | 7,556.15 | 7,895.49 | 135.69 | |
| 256/256 C1 | 225.95 | 218.61 | 24.74 | |
| 1024/1024 C1 | 229.43 | 220.81 | 40.74 | |
| 1024/1024 C512 | 8,459.24 | 8,082.98 | 251.70 | |
