#!/bin/bash
# Sequential Server target-QPS sweep. One GPU, cu128 FlyDSL table, server_mi350x.
# Each candidate uses the 30-minute min_duration from its user conf.
set -u
ROOT=/home/amd/workspace/coder/_results/mlperf_gptoss
LOG=$ROOT/qps_sweep.log
QPS_LIST=(3.0 3.5 4.0 4.2)

echo "SWEEP $(date -Is) begin" | tee -a "$LOG"
for QPS in "${QPS_LIST[@]}"; do
  tag=${QPS/./p}
  conf=$ROOT/user_server_qps_${tag}.conf
  outdir=/results/Server/performance/qps_${tag}
  cat > "$conf" <<EOF
gpt-oss-120b.Offline.target_qps = 6
gpt-oss-120b.Offline.min_duration = 600000
gpt-oss-120b.Server.target_qps = ${QPS}
gpt-oss-120b.Server.min_duration = 1800000
EOF
  docker cp "$conf" gptoss_test:/lab-mlperf-inference/code/gpt-oss-120b/user_server_qps_${tag}.conf
  echo "START $(date -Is) qps=$QPS" | tee -a "$LOG"
  docker exec gptoss_test bash -c "unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES; export AITER_CONFIG_FMOE=/results/gptoss_fp8fp4_tuned_fmoe_cu128.csv; python /lab-mlperf-inference/code/main.py --config-path /lab-mlperf-inference/code/gpt-oss-120b/ --config-name server_mi350x test_mode=performance harness_config.user_conf_path=/lab-mlperf-inference/code/gpt-oss-120b/user_server_qps_${tag}.conf harness_config.target_qps=${QPS} harness_config.device_count=1 harness_config.output_log_dir=${outdir} > /results/server_qps_${tag}.log 2>&1"
  rc=$?
  summary=$ROOT/Server/performance/qps_${tag}/mlperf_log_summary.txt
  if [[ -f $summary ]]; then
    python3 - "$summary" "$QPS" "$rc" <<'PY' | tee -a "$LOG"
import sys
text=open(sys.argv[1]).read().splitlines()
def grab(label):
    for line in text:
        if label in line:
            return line.split(":",1)[-1].strip()
    return ""
def ns_ms(label):
    raw=grab(label)
    try:
        return f"{int(raw)/1e6:.1f}"
    except ValueError:
        return raw
print(
    f"DONE qps={sys.argv[2]} rc={sys.argv[3]} "
    f"verdict={grab('Result is')} "
    f"completed_samples_s={grab('Completed samples per second')} "
    f"scheduled_samples_s={grab('Scheduled samples per second')} "
    f"completed_tokens_s={grab('Completed tokens per second')} "
    f"p99_ttft_ms={ns_ms('99.00 percentile first token latency')} "
    f"p99_tpot_ms={ns_ms('99.00 percentile time to output token')} "
    f"target_qps={grab('target_qps')}"
)
PY
  else
    echo "FAIL $(date -Is) qps=$QPS rc=$rc no summary" | tee -a "$LOG"
  fi
done
echo "SWEEP $(date -Is) complete" | tee -a "$LOG"
