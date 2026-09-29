#!/usr/bin/env bash
# Warmed C1/C4/C8 streaming latency at 1K, 8K, and near-16K context.
set -euo pipefail

PROFILE="${1:?profile name required}"
OUT_ROOT="${2:?output directory required}"
mkdir -p "${OUT_ROOT}"

run_shape() {
  local label="$1" input="$2" output="$3" concurrency="$4"
  python3 scripts/bench_openai_stream.py \
    --model awq --input-len "${input}" --output-len 32 \
    --num-prompts "${concurrency}" --concurrency "${concurrency}" \
    --out "${OUT_ROOT}/${label}_warmup.json" >/dev/null
  python3 scripts/bench_openai_stream.py \
    --model awq --input-len "${input}" --output-len "${output}" \
    --num-prompts "${concurrency}" --concurrency "${concurrency}" \
    --out "${OUT_ROOT}/${label}.json"
}

for concurrency in 1 4 8; do
  run_shape "c${concurrency}_1k_512" 1024 512 "${concurrency}"
  run_shape "c${concurrency}_8k_128" 8192 128 "${concurrency}"
  run_shape "c${concurrency}_15k_128" 15300 128 "${concurrency}"
done

python3 - "${PROFILE}" "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

profile, root = sys.argv[1], Path(sys.argv[2])
print(profile)
for path in sorted(root.glob("c*.json")):
    if path.name.endswith("_warmup.json"):
        continue
    row = json.loads(path.read_text())
    print(
        f"{path.stem:16} ok={row['successful']}/{row['num_prompts']} "
        f"tok/s={row['output_throughput']:.2f} "
        f"ttft={row['ttft_p50_ms']:.2f}/{row['ttft_p95_ms']:.2f}ms "
        f"itl={row['itl_p50_ms']:.2f}/{row['itl_p95_ms']:.2f}ms"
    )
PY
