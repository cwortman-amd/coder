#!/bin/bash
# Rebuild every evaluation and experiment report from the published copies.
# Inputs are docs/results and docs/profiling.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

RESULTS="${ROOT}/docs/results"
PROFILING="${ROOT}/docs/profiling"
FIGURES="${ROOT}/docs/figures"

if [ -x "${ROOT}/.venv/bin/python" ]; then
    PY="${ROOT}/.venv/bin/python"
else
    PY="python3"
fi

FAILED=0

usage() {
    cat <<EOF
Usage: $(basename "$0")

Rebuild reports and figures from the latest published measurements.

Inputs:
  docs/results     AgentX campaigns, agent-chain manifests, latency samples, KV and P/D records
  docs/profiling   Qwen latency index, GPU utilization, R9700 phase profiles

Writes analysis next to each campaign and figures under docs/figures/.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi
if [ $# -gt 0 ]; then
    echo "Unknown option: $1" >&2
    usage >&2
    exit 1
fi

run() {
    echo "------------------------------------------------------------------------"
    echo "+ $*"
    if "$@"; then
        return 0
    fi
    echo "FAILED: $*" >&2
    FAILED=1
}

agentx_has_exports() {
    "$PY" - "$1" <<'PY'
import json
import sys
from pathlib import Path

campaign = Path(sys.argv[1])
root = Path.cwd()
manifest = json.loads((campaign / "manifest.json").read_text())
for run in manifest.get("runs") or []:
    if run.get("status") != "completed":
        continue
    local = campaign / f"c{run.get('concurrency')}"
    recorded = Path(str(run.get("artifact_dir") or ""))
    if not recorded.is_absolute():
        recorded = root / recorded
    for candidate in (local, recorded):
        if (candidate / "profile_export.jsonl").is_file():
            raise SystemExit(0)
raise SystemExit(1)
PY
}

echo "Published results:  ${RESULTS}"
echo "Published profiles: ${PROFILING}"

while IFS= read -r manifest; do
    [ -n "$manifest" ] || continue
    campaign="$(dirname "$manifest")"
    if agentx_has_exports "$campaign"; then
        run "$PY" "${ROOT}/scripts/analyze_agentx_tail_sweep.py" "$campaign"
    else
        echo "------------------------------------------------------------------------"
        echo "AgentX campaign ${campaign} has no request exports beside the manifest. Keeping its existing analysis."
    fi
    if [ -f "${campaign}/analysis.json" ] && [ -f "${campaign}/request_samples.csv" ]; then
        if [[ "$campaign" == "${RESULTS}/tail_study/"* ]]; then
            figure_dir="${FIGURES}/agentx/${campaign#${RESULTS}/}"
            run "$PY" "${ROOT}/scripts/plot_agentx_tail_sweep.py" \
                --results-dir "$campaign" \
                --output-dir "$figure_dir"
        fi
    fi
done < <(find "$RESULTS" -name manifest.json -print | sort | while IFS= read -r manifest; do
    if grep -q 'agentx_concurrency_tail_pilot' "$manifest"; then
        printf '%s\n' "$manifest"
    fi
done)

# Generate publication figures keyed by device identifier (r9700, mi350p) instead of date.
for dev in r9700 mi350p; do
    best_campaign="$("$PY" - "$RESULTS" "$dev" <<'PY'
import json, sys
from pathlib import Path
results, target_dev = Path(sys.argv[1]), sys.argv[2].lower()
candidates = []
for manifest_path in results.rglob("manifest.json"):
    try:
        data = json.loads(manifest_path.read_text())
    except Exception:
        continue
    if data.get("kind") != "agentx_concurrency_tail_pilot":
        continue
    c = manifest_path.parent
    if not (c / "analysis.json").is_file() or not (c / "request_samples.csv").is_file():
        continue
    prof = data.get("gpu_profile")
    if not prof:
        s = str(c).lower()
        prof = "mi350p" if ("mi350" in s or "gfx950" in s) else "r9700"
    if prof.lower() != target_dev:
        continue
    runs = [r for r in data.get("runs", []) if r.get("status") == "completed"]
    started = data.get("started_utc", "")
    date_str = started[:10] if started else "1970-01-01"
    candidates.append((date_str, len(runs), started, str(c)))
if candidates:
    candidates.sort()
    print(candidates[-1][3])
PY
)"
    if [ -n "$best_campaign" ] && [ -d "$best_campaign" ]; then
        run "$PY" "${ROOT}/scripts/plot_agentx_tail_sweep.py" \
            --results-dir "$best_campaign" \
            --output-dir "${FIGURES}/agentx/${dev}" \
            --device "$dev"
        if [ "$dev" = "r9700" ]; then
            run "$PY" "${ROOT}/scripts/plot_agentx_tail_sweep.py" \
                --results-dir "$best_campaign" \
                --output-dir "${FIGURES}/agentx" \
                --device "$dev"
        fi
    fi
done

newest_chain=""
while IFS= read -r manifest; do
    [ -n "$manifest" ] || continue
    newest_chain="$manifest"
    report_dir="$(dirname "$manifest")"
    run "$PY" "${ROOT}/scripts/analyze_tail_metrics.py" --input "$manifest" \
        | tee "${report_dir}/agent_chain_analysis.txt"
    run "$PY" "${ROOT}/scripts/plot_tail_distributions.py" \
        --input "$manifest" \
        --out "${report_dir}/agent_chain_dashboard.png"
done < <(find "$RESULTS" -name agent_chain_manifest.json -printf '%T@ %p\n' | sort -n | cut -d' ' -f2-)

if [ -n "$newest_chain" ]; then
    run "$PY" "${ROOT}/scripts/plot_tail_distributions.py" \
        --input "$newest_chain" \
        --out "${FIGURES}/tail_study_dashboard.png"
fi

for profile in r9700 mi350p; do
    root="${RESULTS}/qwen3.8-27b-mxfp4/latency/${profile}"
    if [ -d "${root}/1k64" ] || [ -d "${root}/8k64" ]; then
        run "$PY" "${ROOT}/scripts/generate_latency_histogram_plots.py" \
            --results-root "$root" \
            --output-dir "${FIGURES}/latency/${profile}"
    fi
done

run "$PY" "${ROOT}/scripts/generate_kv_plots.py"
run "$PY" "${ROOT}/scripts/generate_pd_plots.py"
run "$PY" "${ROOT}/scripts/generate_tco_plots.py"
if [ -f "${PROFILING}/gpu_metrics.json" ]; then
    run "$PY" "${ROOT}/scripts/plot_gpu_utilization.py"
fi

if [ "$FAILED" -ne 0 ]; then
    echo "One or more report generators failed." >&2
    exit 1
fi
echo "Reports regenerated from docs/results and docs/profiling."
