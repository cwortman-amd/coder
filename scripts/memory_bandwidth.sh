#!/usr/bin/env bash
# Practical GPU global-memory streaming bandwidth via RVS BABEL (BabelStream HIP kernels).
#
# This is the empirical HBM/GDDR roof for a *conditional* decode tok/s estimate:
#   R_decode,conditional = B_BABEL_Read / assumed_weight_bytes_per_token
# BABEL is not vLLM decode DRAM traffic. Do not use RVS pebb (host↔GPU PCIe) or
# pbqt (GPU↔GPU) for this roof.
#
# Usage:
#   ./scripts/memory_bandwidth.sh
#   ./scripts/memory_bandwidth.sh --quick --gpu 0
#   ./scripts/memory_bandwidth.sh --keep-server --weight-gib 17.91
#
# Needs render/kfd access (sudo -n rvs / amd-smi on this host). Stops the local
# vLLM container by default so arrays are not fighting a 27B resident.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVS="${RVS_BIN:-/opt/rocm/bin/rvs}"
RVS_CONF_ROOT="/opt/rocm/share/rocm-validation-suite/conf"
CONTAINER="${VLLM_CONTAINER_NAME:-rocm-inference-server}"

GPU_ORD=0
REPEATS=3
QUICK=0
KEEP_SERVER=0
RESTORE=0
WEIGHT_GIB="${WEIGHT_GIB:-17.91}"
SPEC_GBS=""
OUT=""
CONF_OVERRIDE=""
NUM_ITER=""
ARRAY_SIZE=""

usage() {
  cat <<'EOF'
memory_bandwidth.sh — RVS babel / BabelStream GPU global-memory roof

  --gpu N            0-based card among amd-smi list (default 0)
  --repeats N        independent RVS process repeats; report median/range (default 3)
  --quick            fewer kernel iterations (smoke / CI)
  --keep-server      do not stop vLLM (contaminated VRAM; not a clean roof)
  --restore          relaunch ./scripts/launch_vllm_mxfp4.sh after the run
  --weight-gib F     assumed full-weight stream per decode token (default 17.91)
  --spec-gbs F       advertised peak GB/s (default: from amd-smi MAX_BANDWIDTH)
  --array-size N     BabelStream element count (not bytes)
  --num-iter N       kernels per action
  --config PATH      explicit RVS yaml (still rewritten for GPU pin / repeats)
  --out DIR          output directory
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpu) GPU_ORD="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --keep-server) KEEP_SERVER=1; shift ;;
    --restore) RESTORE=1; shift ;;
    --weight-gib) WEIGHT_GIB="$2"; shift 2 ;;
    --spec-gbs) SPEC_GBS="$2"; shift 2 ;;
    --array-size) ARRAY_SIZE="$2"; shift 2 ;;
    --num-iter) NUM_ITER="$2"; shift 2 ;;
    --config) CONF_OVERRIDE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

as_root() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    sudo -n "$@"
  fi
}

if [[ ! -x "${RVS}" ]]; then
  echo "missing ${RVS}" >&2
  exit 1
fi
if ! as_root true >/dev/null 2>&1; then
  echo "need passwordless sudo for rvs/amd-smi (render/kfd)." >&2
  exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-${ROOT}/_results/profiling/babel/${STAMP}}"
mkdir -p "${OUT}"

{
  echo "utc=${STAMP}"
  echo "host=$(hostname)"
  echo "rvs=$(as_root "${RVS}" --version 2>/dev/null | tr -d '\n')"
  echo "rocm=$(cat /opt/rocm/.info/version 2>/dev/null || true)"
  uname -a
} | tee "${OUT}/meta.txt"

as_root "${RVS}" -g >"${OUT}/rvs_gpus.txt" 2>&1 || true
as_root amd-smi list >"${OUT}/amd_smi_list.txt" 2>&1 || true
as_root amd-smi static -g "${GPU_ORD}" >"${OUT}/amd_smi_static.txt" 2>&1 || true

python3 - "${OUT}" "${GPU_ORD}" <<'PY' >"${OUT}/gpu_select.json"
import json, re, sys
from pathlib import Path
out, ord_s = Path(sys.argv[1]), int(sys.argv[2])
listing = (out / "rvs_gpus.txt").read_text(errors="replace")
static = (out / "amd_smi_static.txt").read_text(errors="replace")
gpus = []
for line in listing.splitlines():
    m = re.search(
        r"([0-9a-fA-F:.]+)\s+-\s+GPU\[\s*(\d+)\s*-\s*(\d+)\]\s+(.*?)\s+\(Device\s+(\d+)\)",
        line,
    )
    if m:
        gpus.append({
            "bdf": m.group(1),
            "rvs_index": int(m.group(2)),
            "kfd_id": int(m.group(3)),
            "name": m.group(4).strip(),
            "device_code": m.group(5),
        })
if not gpus:
    sys.stderr.write("RVS listed no supported GPUs. See rvs_gpus.txt\n")
    sys.exit(1)
if ord_s < 0 or ord_s >= len(gpus):
    sys.stderr.write(f"GPU ordinal {ord_s} out of range 0..{len(gpus)-1}\n")
    sys.exit(1)
g = gpus[ord_s]
name = g["name"].upper()
pci = ""
m = re.search(r"DEVICE_ID:\s*0x([0-9a-fA-F]+)", static)
if m:
    pci = m.group(1).lower()
arch = "unknown"
sku_conf = None
spec_gbs = None
bw = re.search(r"MAX_BANDWIDTH:\s*(\d+)\s*GB/s", static)
if bw:
    spec_gbs = float(bw.group(1))
if "MI350P" in name or pci == "75a8":
    arch = "gfx950"
    sku_conf = "MI350P-600W/babel_single.conf"
    spec_gbs = spec_gbs or 4096.0
elif "MI350X" in name:
    arch = "gfx950"
    sku_conf = "MI350X/babel.conf"
    spec_gbs = spec_gbs or 8000.0
elif "R9700" in name or pci in {"7550", "7551"} or "gfx1201" in static.lower():
    arch = "gfx1201"
    sku_conf = None  # no packaged babel SKU; generated conf
    spec_gbs = spec_gbs or 640.0
g.update({
    "ordinal": ord_s,
    "pci_device_id": pci,
    "arch": arch,
    "sku_rel_conf": sku_conf,
    "spec_gbs": spec_gbs,
    "vram_type": (re.search(r"TYPE:\s+(\S+)", static) or [None, None])[1],
})
print(json.dumps({"selected": g, "all": gpus}, indent=2))
PY

SEL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["kfd_id"])' "${OUT}/gpu_select.json")"
IDX="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["ordinal"])' "${OUT}/gpu_select.json")"
ARCH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["arch"])' "${OUT}/gpu_select.json")"
NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["name"])' "${OUT}/gpu_select.json")"
SKU_REL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"].get("sku_rel_conf") or "")' "${OUT}/gpu_select.json")"
if [[ -z "${SPEC_GBS}" ]]; then
  SPEC_GBS="$(python3 -c 'import json,sys; v=json.load(open(sys.argv[1]))["selected"].get("spec_gbs"); print(v if v is not None else "")' "${OUT}/gpu_select.json")"
fi

if [[ -n "${CONF_OVERRIDE}" ]]; then
  SRC_CONF="${CONF_OVERRIDE}"
elif [[ -n "${SKU_REL}" && -f "${RVS_CONF_ROOT}/${SKU_REL}" ]]; then
  SRC_CONF="${RVS_CONF_ROOT}/${SKU_REL}"
elif [[ -f "${RVS_CONF_ROOT}/babel.conf" ]]; then
  echo "warning: no SKU babel conf for ${NAME} (${ARCH}); using generic babel.conf + large array" | tee -a "${OUT}/meta.txt"
  SRC_CONF="${RVS_CONF_ROOT}/babel.conf"
else
  echo "no babel configuration found under ${RVS_CONF_ROOT}" >&2
  exit 1
fi
echo "src_conf=${SRC_CONF}" | tee -a "${OUT}/meta.txt"

# Default working set: SKU file's array_size, else 2^28 elements (~2 GiB doubles) so
# the stream misses on-chip cache. gfx1201 32 GB can still hold 3×2 GiB triad buffers.
# Never 4 GiB/array on a ~32 GiB host: RVS host-side alloc hangs/OOM after "Using HIP device".
HOST_GIB="$(awk '/MemTotal:/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)"
MAX_ARRAY_ELEMS=268435456  # 2 GiB doubles
if [[ "${HOST_GIB}" -ge 64 ]]; then
  MAX_ARRAY_ELEMS=536870912
fi
if [[ "${QUICK}" -eq 1 ]]; then
  NUM_ITER="${NUM_ITER:-50}"
  REPEATS=1
  ARRAY_SIZE="${ARRAY_SIZE:-134217728}"  # 128M doubles ~1 GiB; still > L2
fi
RVS_TIMEOUT="${RVS_TIMEOUT:-900}"
if [[ "${QUICK}" -eq 1 ]]; then
  RVS_TIMEOUT=180
fi
if [[ -z "${NUM_ITER}" ]]; then
  NUM_ITER="$(awk '/^[[:space:]]*num_iter:/{print $2; exit}' "${SRC_CONF}" || true)"
  NUM_ITER="${NUM_ITER:-1000}"
fi
if [[ -z "${ARRAY_SIZE}" ]]; then
  ARRAY_SIZE="$(awk '/^[[:space:]]*array_size:/{print $2; exit}' "${SRC_CONF}" || true)"
  if [[ -z "${ARRAY_SIZE}" || "${ARRAY_SIZE}" -lt 67108864 ]]; then
    ARRAY_SIZE=268435456
  fi
fi
if [[ "${ARRAY_SIZE}" -gt "${MAX_ARRAY_ELEMS}" ]]; then
  echo "capping BABEL array_size ${ARRAY_SIZE} -> ${MAX_ARRAY_ELEMS} (host ${HOST_GIB} GiB; never 4 GiB arrays on 32 GiB hosts)" | tee -a "${OUT}/meta.txt"
  ARRAY_SIZE="${MAX_ARRAY_ELEMS}"
fi
ARR_GIB="$(python3 -c "print(${ARRAY_SIZE}*8/(1024**3))")"
if awk "BEGIN {exit !(${ARR_GIB} >= 3.9)}"; then
  echo "refusing BABEL array_size=${ARRAY_SIZE} (~${ARR_GIB} GiB/array). This 32 GiB-class host OOMs at 4 GiB." | tee -a "${OUT}/meta.txt"
  exit 2
fi
echo "host_gib=${HOST_GIB} array_size=${ARRAY_SIZE} array_gib=${ARR_GIB}" | tee -a "${OUT}/meta.txt"

RUN_CONF="${OUT}/babel_run.conf"
python3 - "${SRC_CONF}" "${RUN_CONF}" "${SEL}" "${IDX}" "${NUM_ITER}" "${ARRAY_SIZE}" "${REPEATS}" <<'PY'
import sys
from pathlib import Path
src, dst, gpu, idx, niter, arr, repeats = sys.argv[1:8]
text = Path(src).read_text()
# Pin one GPU; serial repeats are done by re-invoking rvs, not YAML count.
# device = KFD GPU ID from `rvs -g` (GPU[ node - ID ]); device_index = 0-based card.
repl = {
    "device:": f"  device: {gpu}",
    "device_index:": f"  device_index: {idx}",
    "parallel:": "  parallel: false",
    "count:": "  count: 1",
    "num_iter:": f"  num_iter: {niter}",
    "array_size:": f"  array_size: {arr}",
    "mibibytes:": "  mibibytes: true",
    "o/p_csv:": "  o/p_csv: false",
    "read:": "  read: true",
    "write:": "  write: true",
    "copy:": "  copy: true",
    "triad:": "  triad: true",
    "test_type:": "  test_type: 2",
}
out_lines = []
seen = set()
for line in text.splitlines():
    key = line.split(":", 1)[0].strip() + (":" if ":" in line else "")
    stripped = line.lstrip()
    hit = None
    for prefix, new in repl.items():
        if stripped.startswith(prefix):
            hit = new
            seen.add(prefix)
            break
    out_lines.append(hit if hit is not None else line)
# Ensure required keys exist on the first action.
inject = [k for k in ("device_index:", "read:", "write:", "copy:", "triad:", "mibibytes:") if k not in seen]
if inject:
    rebuilt = []
    injected = False
    for line in out_lines:
        rebuilt.append(line)
        if (not injected) and line.lstrip().startswith("module:"):
            for k in inject:
                rebuilt.append(repl[k])
            injected = True
    out_lines = rebuilt
header = [
    f"# generated from {src}",
    f"# rvs device/index {gpu}; array_size is ELEMENT count; test_type 2 = double",
    f"# mibibytes true => RVS reports GiB/s (convert to decimal GB/s for spec compare)",
    f"# yaml count=1; process repeats={repeats}",
    "",
]
Path(dst).write_text("\n".join(header + out_lines) + "\n")
PY

STOPPED=0
if [[ "${KEEP_SERVER}" -eq 0 ]]; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    echo "stopping ${CONTAINER} so BABEL owns GPU HBM/GDDR" | tee -a "${OUT}/meta.txt"
    docker stop "${CONTAINER}" >/dev/null
    STOPPED=1
  fi
else
  echo "WARNING: --keep-server; resident vLLM will reduce attainable BABEL bandwidth" | tee -a "${OUT}/meta.txt"
fi

telemetry() {
  local tag=$1
  timeout 15 as_root amd-smi metric -g "${GPU_ORD}" >"${OUT}/metric_${tag}.txt" 2>&1 || true
  timeout 8 as_root amd-smi static -g "${GPU_ORD}" >"${OUT}/static_${tag}.txt" 2>&1 || true
}

telemetry before
echo "running BABEL on ${NAME} kfd_id=${SEL} device_index=${IDX} array_size=${ARRAY_SIZE} num_iter=${NUM_ITER} repeats=${REPEATS}" | tee -a "${OUT}/meta.txt"

for i in $(seq 1 "${REPEATS}"); do
  echo "===== babel repeat ${i}/${REPEATS} =====" | tee -a "${OUT}/rvs.log"
  telemetry "rep${i}_start"
  # Do not pass -i: a mismatched index vs device: KFD id made RVS spin on CPU with GPU idle.
  # HIP_VISIBLE_DEVICES pins the physical card; conf device/device_index select inside RVS.
  if ! as_root env HIP_VISIBLE_DEVICES="${GPU_ORD}" \
      stdbuf -oL -eL timeout "${RVS_TIMEOUT}" \
      "${RVS}" -c "${RUN_CONF}" -p false -d 3 \
      -l "${OUT}/rvs_rep${i}.debug.log" \
      -j "${OUT}/rvs_rep${i}.json" \
      >>"${OUT}/rvs.log" 2>&1; then
    echo "rvs repeat ${i} exited nonzero or timed out after ${RVS_TIMEOUT}s" | tee -a "${OUT}/rvs.log"
  fi
  telemetry "rep${i}_end"
done
telemetry after

python3 "${ROOT}/scripts/parse_rvs_babel.py" "${OUT}" \
  --out-dir "${OUT}" \
  --weight-gib "${WEIGHT_GIB}" \
  --spec-gbs "${SPEC_GBS:-4096}" \
  --array-size "${ARRAY_SIZE}" \
  --name "${NAME}" \
  --arch "${ARCH}" \
  --num-iter "${NUM_ITER}" \
  --repeats "${REPEATS}"

if [[ "${RESTORE}" -eq 1 && "${STOPPED}" -eq 1 ]]; then
  echo "restoring vLLM HF MXFP4 control"
  "${ROOT}/scripts/launch_vllm_mxfp4.sh"
fi

echo "artifacts: ${OUT}"
echo "Do not treat pebb (PCIe) or pbqt (P2P) as this HBM/GDDR roof."
