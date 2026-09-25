#!/usr/bin/env bash
# Target VLLM::EngineCore only. Never wrap `vllm serve`.
#
# Live path: docker exec rocprofv3 --pid <EngineCore> (SDK 1.3.2 attach).
# That succeeds only if the EngineCore process already has a thread named
# `rocp-bg-attach` (ROCP_TOOL_ATTACH consumed in *this* PID, not the API parent).
#
# Exec path: --emit-exec-wrapper writes a spawn executable that rocprofv3-wraps
# the EngineCore Python child. Installing it requires a new EngineCore spawn
# (server restart). Do not use it to wrap the parent `vllm serve` PID.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${VLLM_CONTAINER_NAME:-rocm-inference-server}"
OUT="${ROOT}/_results/profiling/enginecore_attach"
DURATION_MS=2500
FORCE=0
EMIT=""
TINY=1

usage() {
  cat <<'EOF'
profile_enginecore_attach.sh — EngineCore-only rocprofv3 attach (not vllm serve)

  --out DIR                 default _results/profiling/enginecore_attach
  --container NAME          default rocm-inference-server
  --attach-duration-msec N  default 2500
  --no-tiny-decode          skip the max_tokens=8 request
  --force                   attempt attach even without rocp-bg-attach thread
  --emit-exec-wrapper PATH  write EngineCore spawn wrapper (restart required)
  --identify-only           print PIDs / kfd / threads and exit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --out) OUT="$2"; shift 2 ;;
    --container) NAME="$2"; shift 2 ;;
    --attach-duration-msec) DURATION_MS="$2"; shift 2 ;;
    --no-tiny-decode) TINY=0; shift ;;
    --force) FORCE=1; shift ;;
    --emit-exec-wrapper) EMIT="$2"; shift 2 ;;
    --identify-only) IDENTIFY=1; shift ;;
    *) echo "unknown arg $1" >&2; usage; exit 2 ;;
  esac
done
IDENTIFY="${IDENTIFY:-0}"

emit_exec_wrapper() {
  local dest="$1"
  mkdir -p "$(dirname "${dest}")"
  cat > "${dest}" <<'WRAP'
#!/usr/bin/env bash
# multiprocessing spawn executable: rocprofv3 is parent of EngineCore Python only.
# Do not set this as the docker --entrypoint (that wraps vllm serve).
# Usage (next launch, not live attach):
#   bind-mount this file and, in the API parent before EngineCore start,
#   multiprocessing.spawn.set_executable("/path/to/enginecore_rocprof_exec.sh")
set -euo pipefail
OUT_DIR="${ENGINECORE_ROCPROF_DIR:-/results/profiling/enginecore_exec}"
mkdir -p "${OUT_DIR}"
exec /opt/rocm/bin/rocprofv3 \
  --kernel-trace --hip-runtime-trace --memory-copy-trace \
  --process-sync --stats --summary \
  --output-format csv pftrace \
  --output-directory "${OUT_DIR}" \
  --output-file enginecore \
  -- \
  "$@"
WRAP
  chmod +x "${dest}"
  echo "wrote EngineCore spawn wrapper ${dest}"
}

if [[ -n "${EMIT}" ]]; then
  emit_exec_wrapper "${EMIT}"
  exit 0
fi

mkdir -p "${OUT}/rocprof"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "container ${NAME} is not running" >&2
  exit 1
fi

identify() {
  docker exec "${NAME}" bash -lc '
set -euo pipefail
echo "===== ps ====="
ps -eo pid,ppid,pgid,stat,comm,args | grep -Ei "vllm|Engine|python|rocprof" | grep -v grep || true
echo "===== kfd/dri ====="
for pid in $(ls /proc | grep -E "^[0-9]+$"); do
  comm=$(cat /proc/$pid/comm 2>/dev/null || true)
  if ls -l /proc/$pid/fd 2>/dev/null | grep -q /dev/kfd; then
    echo "PID $pid comm=$comm kfd=yes"
    ls -l /proc/$pid/fd 2>/dev/null | grep -E "/dev/kfd|/dev/dri" || true
  fi
done
echo "===== EngineCore env (rocprof) ====="
ec=$(ps -eo pid,comm | awk "\$2 ~ /EngineCor|EngineCore/ {print \$1; exit}")
if [[ -n "${ec:-}" ]]; then
  echo "enginecore_nspid=$ec"
  tr "\0" "\n" < /proc/$ec/environ | grep -Ei "ROCP|ROCPROF|VLLM_ENABLE|AITER" | sort || true
  echo "===== threads named attach/rocprof ====="
  found=0
  for t in /proc/$ec/task/*/comm; do
    n=$(cat "$t" 2>/dev/null || true)
    case "$n" in
      *rocp*|*attach*|*prof*) echo "$(basename "$(dirname "$t")") $n"; found=1 ;;
    esac
  done
  if [[ "$found" -eq 0 ]]; then echo "(no rocprof/attach thread names)"; fi
  echo "===== maps attach libs ====="
  grep -i attach /proc/$ec/maps || echo "(librocprofiler-sdk-attach not mapped)"
else
  echo "EngineCore PID not found"
fi
'
  echo "===== docker top (host PIDs) ====="
  docker top "${NAME}" -eo pid,ppid,stat,cmd || docker top "${NAME}"
}

{
  echo "utc=${STAMP} container=${NAME} duration_ms=${DURATION_MS}"
  docker exec "${NAME}" rocprofv3 --version || true
} | tee "${OUT}/meta.txt"

identify | tee "${OUT}/identity.txt"

EC_NS="$(awk -F= '/^enginecore_nspid=/{print $2}' "${OUT}/identity.txt" | tail -1 || true)"
if [[ -z "${EC_NS}" ]]; then
  EC_NS="$(docker exec "${NAME}" bash -lc 'ps -eo pid,comm | awk "\$2 ~ /EngineCor/ {print \$1; exit}"')"
fi
echo "enginecore_nspid=${EC_NS}" | tee -a "${OUT}/meta.txt"

HAS_BG=0
if [[ -n "${EC_NS}" ]]; then
  if docker exec "${NAME}" bash -lc "grep -qx rocp-bg-attach /proc/${EC_NS}/task/*/comm 2>/dev/null"; then
    HAS_BG=1
  fi
fi
echo "rocp_bg_attach_thread=${HAS_BG}" | tee -a "${OUT}/meta.txt"

if [[ "${IDENTIFY}" -eq 1 ]]; then
  exit 0
fi

if [[ "${HAS_BG}" -eq 0 && "${FORCE}" -eq 0 ]]; then
  cat <<EOF | tee -a "${OUT}/meta.txt"
BLOCKED: EngineCore nsPID ${EC_NS} has no rocp-bg-attach thread.
docker exec rocprofv3 --pid is safe (refuses before ptrace) but cannot emit KERNEL_DISPATCH.
Parent ROCP_TOOL_ATTACH=1 is not present in EngineCore environ on this spawn path.
Use --force to reproduce the refusal, or a future EngineCore spawn exec wrapper
(scripts/profile_enginecore_attach.sh --emit-exec-wrapper ...). Do not wrap vllm serve.
EOF
  exit 3
fi

if [[ "${TINY}" -eq 1 ]]; then
  python3 "${ROOT}/scripts/bench_openai_chat.py" \
    --base-url "http://127.0.0.1:8000/v1" --model awq \
    --input-len 32 --output-len 8 --num-prompts 1 --concurrency 1 --timeout 60 \
    --out "${OUT}/tiny_decode.json" > "${OUT}/tiny_decode.log" 2>&1 &
  BENCH_PID=$!
else
  BENCH_PID=""
fi

set +e
docker exec "${NAME}" rocprofv3 \
  --pid "${EC_NS}" \
  --attach-children=false \
  --attach-duration-msec "${DURATION_MS}" \
  --attach-sync-output \
  --kernel-trace --hip-runtime-trace \
  --output-format csv pftrace \
  --output-directory /results/profiling/enginecore_attach/rocprof \
  --output-file attach_try \
  > "${OUT}/rocprofv3_attach.stdout" 2> "${OUT}/rocprofv3_attach.stderr"
ATT_RC=$?
set -e
echo "rocprofv3_attach_exit=${ATT_RC}" | tee -a "${OUT}/meta.txt"

if [[ -n "${BENCH_PID}" ]]; then
  wait "${BENCH_PID}" || true
fi

curl -sS -o /dev/null -w "health_http=%{http_code}\n" http://127.0.0.1:8000/health | tee -a "${OUT}/meta.txt"
docker exec "${NAME}" ps -eo pid,ppid,stat,comm | tee "${OUT}/ps_after.txt"

if grep -Rql "KERNEL_DISPATCH" "${OUT}/rocprof" 2>/dev/null; then
  echo "KERNEL_DISPATCH=yes" | tee -a "${OUT}/meta.txt"
else
  echo "KERNEL_DISPATCH=no" | tee -a "${OUT}/meta.txt"
fi
exit "${ATT_RC}"
