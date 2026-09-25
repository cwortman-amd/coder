# MXFP4 vs FP8/BF16 production-sampling comparison

## Status

**Blocked until an exclusive-GPU window.** The frozen MXFP4 control is live on
`127.0.0.1:8000` and was not restarted or displaced for this work. Running a
second 27B server is not a valid side-by-side setup on this host.

## Model availability

- **MXFP4:** local, complete, and serving as `awq` from
  `models/Qwen3.8-27B-Quark-AWQ-MXFP4-sharded`.
- **FP8:** Hugging Face revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` is complete in the local cache.
  Its index references 66 safetensor files; all 66 targets exist and total
  30,866,866,928 bytes. No download is needed.
- **BF16:** no local full-weight Qwen3.8-27B BF16 model was found. It was not
  downloaded.

## Controlled comparison recipe

1. Preserve `_results/quality/sampling_mxfp4.json`.
2. During an approved exclusive-GPU window, stop the MXFP4 container manually.
3. Launch FP8 with `scripts/launch_vllm_fp8.sh`. The script refuses to act while
   port 8000 is healthy and does not set `VLLM_ROCM_USE_AITER`.
4. Run the identical suite:

   ```bash
   python3 scripts/eval_sampling_tasks.py \
     --model fp8 \
     --profile fp8-production-sampling \
     --out _results/quality/sampling_fp8.json
   ```

5. Compare overall and per-category accuracy/validity. This suite records one
   seeded sampled response per item, so differences are indicative rather than
   a statistical confidence bound. For a stronger comparison, repeat with
   multiple seeds and aggregate.

No DFLASH speculative configuration belongs in either target-quantization run.

## Results

| Model | Overall | Math | JSON exact / valid | Code static score / valid | Multilingual | Reasoning |
|---|---:|---:|---:|---:|---:|---:|
| MXFP4 | 30/32 (93.75%) | 6/8 | 8/8 / 8/8 | 8/8 / 8/8 | 4/4 | 4/4 |
| FP8 | Blocked: exclusive GPU needed | — | — | — | — | — |
| BF16 | Blocked: weights unavailable | — | — | — | — | — |
