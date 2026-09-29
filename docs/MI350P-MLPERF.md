# MI350P GPT-OSS-120B MLPerf Inference v6.1

**Date:** 29 September 2026  
**SKU:** 1× AMD Instinct MI350P (`gfx950`, 128 CUs), GPU 0, TP=1  
**Image:** `rocm/amd-mlperf:mi355x_gptoss_120b_inference_6.1` (`sha256:4f17b38a81f735274caa8792c57ed884c7164fd79c41238e02f249f005d881ae`)  
**Harness configs:** `offline_mi350x` and `server_mi350x`, with `harness_config.device_count=1`  
**User conf:** `_results/mlperf_gptoss/user_mi350p.conf`  
**Tune table:** `_results/mlperf_gptoss/gptoss_fp8fp4_tuned_fmoe_cu128.csv`, selected with `AITER_CONFIG_FMOE`

This is a local performance-run record. It is not a closed-division submission, and it is not interchangeable with the AIM `vllm bench serve` numbers in [MI350P-AIMS.md](MI350P-AIMS.md). AMD’s published MI350P results are eight-GPU scores: 48,971.10 tokens/s Offline and 40,867.90 tokens/s Server.

## Claim

On one MI350P, we reproduced a VALID GPT-OSS-120B MLPerf Inference v6.1 Offline performance run at 5,736 tokens/s, using the AMD container and MI350X configuration with a documented 128-CU AITER tuning-table adaptation. The 5-QPS Server run completed 5,925 tokens/s but was INVALID on p99 TTFT and TPOT. A valid single-GPU Server score is not established yet.

This is the performance-run verdict only. A closed-division claim still needs the image digest, the table diff, and the separate GPT-OSS accuracy run.

## Offline

LoadGen marked the run **VALID**. Samples/s 4.205, tokens/s **5,736**. `target_qps` 6, `min_duration` 600000 ms. Summary: `_results/mlperf_gptoss/Offline/performance/run_1/mlperf_log_summary.txt`.

Eight times 5,736 is 45,888 tokens/s, about 93.7% of the published eight-GPU Offline score. That is a scale-out sanity check, not a measured eight-GPU reproduction.

## Server at 5 QPS

LoadGen marked the run **INVALID**. Completed samples/s **4.34** against scheduled samples/s **5.05**. Completed tokens/s **5,925.21**. p99 TTFT **361.9 s** (limit 3 s). p99 TPOT **103.6 ms** (limit 80 ms). `min_duration` 1,800,000 ms was satisfied. Early stopping was not. Summary: `_results/mlperf_gptoss/Server/performance/run_1/mlperf_log_summary.txt`.

The sample-rate gap is the queue. It explains the TTFT. It does not by itself explain the TPOT miss, so TPOT stays a second gate after the queue is gone. Do not quote 5,925 tokens/s as a Server score.

## 128-CU tuning-table adaptation

The image ships an AITER MoE table keyed for 256 CUs. On this 128-CU MI350P that lookup missed, and the CK-tile path then failed warmup with `Out dtype only support BFloat16/Float16!` when `split_k>1`. Copying the shipped table and setting `cu_num` to 128 selected the FlyDSL MXFP4 kernels (`flydsl_moe1_afp8_wfp4_bf16_*` / `flydsl_moe2_afp8_wfp4_bf16_*`) and let warmup finish. The unmodified image did not reproduce the run. AMD’s guide says to pick the GPU configuration; it does not describe this table edit. Precision was left at the submission MXFP4 path.

## Server QPS sweep

Keep the cu128 table, TP=1, GPU 0, checkpoint, performance parquet, image, and `server_mi350x` fixed. Change only Server `target_qps`, starting at 3.0 so the baseline is under the 4.34 samples/s completion rate seen at 5 QPS. Confirm `target_qps` in each `mlperf_log_summary.txt`. A run is a Server score only when LoadGen prints VALID, offered and completed samples/s stay together, p99 TTFT is under 3 s, and p99 TPOT is under 80 ms, for the full 30-minute duration.

A 3.0 QPS attempt on 29 Sep confirmed the override (`Overriding default QPS with 3.0`, LoadGen `effective_target_qps` 3, `effective_min_duration_ms` 1800000) and reached warmup on the cu128 table without the BF16/FP16 output-dtype failure. LoadGen did not write a verdict. The container was stopped with SIGTERM during warmup, and `rocm-inference-server` is on the GPU again. Runner: `_results/mlperf_gptoss/run_server_qps_sweep.sh`. User conf for the 3.0 candidate: `_results/mlperf_gptoss/user_server_qps_3p0.conf`.
