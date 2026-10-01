---
type: Technical Report
title: GPT-OSS-20B native MXFP4
description: openai/gpt-oss-20b ships native MXFP4 and fits one 32 GB card. That is
  the Radeon AI PRO R9700S (and the R9700, the same 64 CU / 32 GB / 300 W GPU). openai/gpt-oss-120b
  does not fit that card. The 120B AIM sweep on...
tags:
- technical-report
- gpt
- oss
status: stable
---

# GPT-OSS-20B native MXFP4

`openai/gpt-oss-20b` ships native MXFP4 and fits one 32 GB card. That is the Radeon AI PRO R9700S (and the R9700, the same 64 CU / 32 GB / 300 W GPU). `openai/gpt-oss-120b` does not fit that card. The 120B AIM sweep on one MI350P is [MI350P-AIMS.md](MI350P-AIMS.md).

This is stock vLLM, not the Quark Radiance compose used for Qwen3.8-27B. The Qwen FP8 KV pin, architecture override, and tool-parser flags are not applied.

No tok/s from this recipe is in [TCO.md](TCO.md) yet. The fit row there is a capacity statement, not a weighed checkpoint.

## Serve

```bash
VLLM_ROCM_USE_AITER=1 vllm serve openai/gpt-oss-20b \
  --dtype auto -tp 1 --no-enable-prefix-caching --disable-uvicorn-access-log
```

`VLLM_ROCM_USE_AITER=1` is set on the 32 GB card and on MI350P. Unified attention stays off on gfx1201. The stock AITER unified kernel asks for 66,048 bytes of LDS, and RDNA 4 caps a workgroup at 65,536 bytes. That crash is recorded in [R9700.md](R9700.md). MI350P keeps unified attention because CDNA 4 has the larger LDS.

## Bench

```bash
vllm bench serve \
  --model openai/gpt-oss-20b \
  --percentile-metrics tpot,ttft,itl,e2el \
  --dataset-name random \
  --ignore-eos \
  --temperature 0 \
  --max-concurrency 1 \
  --num-prompts 10 \
  --random-input-len 1024 \
  --random-output-len 1024
```

One command does both steps and writes `_results/gpt_oss_20b/`:

```bash
./scripts/bench_gpt_oss_20b.sh
```

`--bench-only` skips launch when port 8000 is already healthy. The launcher stops `rocm-inference-server` if that container is holding the port. Restore the Qwen server with `./setup.sh` when this track is idle.

The same serve line is what `./setup.sh -m gpt-oss-20b` and `./test.sh --models gpt-oss-20b` start. The test dispatcher uses 1,024/1,024 and 10 prompts unless `--test-cases` or `--num-prompts` is passed. `-q` stays the 128:64 smoke test.
