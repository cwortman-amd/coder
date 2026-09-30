#!/usr/bin/env bash
# Read-only dual-GPU PCIe / RCCL triage.
# Requires Open MPI for OMPI_COMM_WORLD_LOCAL_RANK.
#
# bash scripts/debug_tp.sh
# RUNS=3 bash scripts/debug_tp.sh
# SKIP_BENCH=1 bash scripts/debug_tp.sh

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/_results/tp_debug/$(date +%Y%m%d-%H%M%S)}"
RUNS="${RUNS:-1}"
SKIP_BENCH="${SKIP_BENCH:-0}"

GPU0_BDF="${GPU0_BDF:-0000:8b:00.0}"
GPU1_BDF="${GPU1_BDF:-0001:c7:00.0}"
GPU0_INDEX="${GPU0_INDEX:-0}"
GPU1_INDEX="${GPU1_INDEX:-1}"

mkdir -p "$OUT"
export PATH="/opt/rocm/bin:/opt/rocm/lib/llvm/bin:${PATH:-}"

[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] ||
    { echo "RUNS must be a positive integer" >&2; exit 2; }
[[ "$SKIP_BENCH" == 0 || "$SKIP_BENCH" == 1 ]] ||
    { echo "SKIP_BENCH must be 0 or 1" >&2; exit 2; }
[[ "$GPU0_BDF" != "$GPU1_BDF" ]] ||
    { echo "GPU BDFs must be distinct" >&2; exit 2; }
[[ "$GPU0_INDEX" != "$GPU1_INDEX" ]] ||
    { echo "GPU indices must be distinct" >&2; exit 2; }

for bdf in "$GPU0_BDF" "$GPU1_BDF"; do
    [[ -d "/sys/bus/pci/devices/$bdf" ]] ||
        { echo "Missing PCI device $bdf" >&2; exit 2; }
done

# Never silently accept a failed capture as a successful measurement.
capture() {
    local dest=$1 rc
    shift
    if "$@" >"$dest" 2>&1; then
        printf 'exit=0\n' >>"$dest"
    else
        rc=$?
        printf 'exit=%d\n' "$rc" >>"$dest"
        printf 'WARN: capture failed (%d): %s\n' "$rc" "$dest" >&2
    fi
}

FAILURES=0
run_logged() {
    local log=$1 rc
    shift
    printf 'COMMAND:' >"$log"
    printf ' %q' "$@" >>"$log"
    printf '\n' >>"$log"

    if "$@" >>"$log" 2>&1; then
        rc=0
    else
        rc=$?
        ((FAILURES+=1))
        printf 'FAIL (%d): %s\n' "$rc" "$log" >&2
    fi

    printf 'exit=%d\n' "$rc" >>"$log"
}

discover_rbt() {
    local candidate
    if [[ -n "${RBT:-}" ]]; then
        [[ -x "$RBT" ]] && printf '%s\n' "$RBT"
        return 0
    fi
    for candidate in \
        /opt/rocm/bin/rocm-bandwidth-test \
        /opt/rocm-7.14.0/bin/rocm-bandwidth-test \
        /tmp/rbt-deb/opt/rocm-6.3.3/bin/rocm-bandwidth-test
    do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 0
}

mpi_linked() {
    command -v ldd >/dev/null &&
        ldd "$1" 2>/dev/null | grep -q 'libmpi\.so'
}

discover_allreduce() {
    local candidate
    local -a candidates=()

    [[ -n "${RCCL_ALLREDUCE:-}" ]] &&
        candidates+=("$RCCL_ALLREDUCE")
    [[ -n "${RCCL_TESTS:-}" ]] &&
        candidates+=("$RCCL_TESTS/all_reduce_perf")
    candidates+=(
        /tmp/rccl-tests/build/all_reduce_perf
        "$ROOT/rccl-tests/build/all_reduce_perf"
        /opt/rocm/bin/all_reduce_perf
    )

    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate" ]] && mpi_linked "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 0
}

RBT_BIN="$(discover_rbt)"
ALLREDUCE_BIN="$(discover_allreduce)"

{
    date -Is
    uname -a
    printf 'OUT=%s RUNS=%s SKIP_BENCH=%s\n' \
        "$OUT" "$RUNS" "$SKIP_BENCH"
    printf 'GPU0_INDEX=%s GPU0_BDF=%s\n' "$GPU0_INDEX" "$GPU0_BDF"
    printf 'GPU1_INDEX=%s GPU1_BDF=%s\n' "$GPU1_INDEX" "$GPU1_BDF"
    printf 'RBT=%s\n' "${RBT_BIN:-<missing>}"
    printf 'MPI_ALLREDUCE=%s\n' "${ALLREDUCE_BIN:-<missing>}"
    printf 'HSA_FORCE_FINE_GRAIN_PCIE=%s\n' \
        "${HSA_FORCE_FINE_GRAIN_PCIE-<unset>}"
    printf 'ROCR_VISIBLE_DEVICES=%s\n' \
        "${ROCR_VISIBLE_DEVICES-<unset>}"
    printf 'HIP_VISIBLE_DEVICES=%s\n' \
        "${HIP_VISIBLE_DEVICES-<unset>}"
    printf 'NCCL_P2P_DISABLE=%s\n' \
        "${NCCL_P2P_DISABLE-<unset>}"
    printf 'NCCL_CUMEM_ENABLE=%s\n' \
        "${NCCL_CUMEM_ENABLE-<unset>}"
    printf 'NCCL_WIN_ENABLE=%s\n' \
        "${NCCL_WIN_ENABLE-<unset>}"
    printf '%s\n' '--- kernel command line ---'
    cat /proc/cmdline
} >"$OUT/manifest.txt"

capture "$OUT/amd-smi-list.txt" amd-smi list
capture "$OUT/amd-smi-static.txt" amd-smi static
capture "$OUT/amd-smi-metric-before.txt" amd-smi metric
capture "$OUT/topology-access.txt" amd-smi topology --access
capture "$OUT/topology-link-type.txt" amd-smi topology --link-type
capture "$OUT/topology-hops.txt" amd-smi topology --hops
capture "$OUT/pci-tree.txt" lspci -Dtv

if command -v lstopo-no-graphics >/dev/null; then
    capture "$OUT/lstopo-with-io.txt" \
        lstopo-no-graphics --of console
fi

if command -v numactl >/dev/null; then
    capture "$OUT/numa.txt" numactl --hardware
fi

{
    hipconfig --version 2>/dev/null || true
    mpirun --version 2>/dev/null || true
    if [[ -n "$ALLREDUCE_BIN" ]]; then
        printf '\nSelected test linkage:\n'
        ldd "$ALLREDUCE_BIN" 2>/dev/null || true
    fi
    if [[ -x /opt/rocm/bin/all_reduce_perf ]]; then
        printf '\nStock test linkage:\n'
        ldd /opt/rocm/bin/all_reduce_perf 2>/dev/null || true
    fi
} >"$OUT/versions-and-linkage.txt"

# Timestamped reset/error history. Do not reduce it to the last 120 lines
# of dmesg, which can hide the cause preceding a GPU reset.
if command -v journalctl >/dev/null; then
    capture "$OUT/kernel-gpu-events.txt" bash -c \
      'journalctl -k -b -o short-iso --no-pager 2>&1 |
       grep -Ei "amdgpu|GPU reset|wedged|VM fault|AER|PCIe Bus Error|RAS"'
fi

snapshot_pcie() {
    local tag=$1 bdf
    for bdf in "$GPU0_BDF" "$GPU1_BDF"; do
        capture "$OUT/pci-${tag}-${bdf//[:.]/_}.txt" \
            lspci -Dvv -s "$bdf"
    done

    python3 - "$OUT/pcie-${tag}.txt" \
        "$GPU0_BDF" "$GPU1_BDF" <<'PY'
import os
import re
import sys

dest, *seeds = sys.argv[1:]
pattern = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
devices = []

for seed in seeds:
    path = os.path.realpath(f"/sys/bus/pci/devices/{seed}")
    while path.startswith("/sys/devices/"):
        name = os.path.basename(path)
        if pattern.fullmatch(name) and name not in devices:
            devices.append(name)
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent

with open(dest, "w") as output:
    for bdf in devices:
        base = f"/sys/bus/pci/devices/{bdf}"
        output.write(f"BDF {bdf}\n")
        for field in (
            "current_link_speed",
            "current_link_width",
            "aer_dev_correctable",
            "aer_dev_nonfatal",
            "aer_dev_fatal",
        ):
            try:
                with open(f"{base}/{field}") as source:
                    value = source.read().strip()
            except OSError:
                value = "<unavailable>"
            output.write(f"{field}:\n{value}\n")
        output.write("\n")
PY
}

snapshot_pcie before

if [[ -n "$RBT_BIN" ]]; then
    capture "$OUT/rbt-devices.txt" "$RBT_BIN" -e
    capture "$OUT/rbt-topology.txt" "$RBT_BIN" -t
else
    printf '%s\n' 'RBT binary not found' >"$OUT/rbt-devices.txt"
fi

# Override these explicitly if the installed RBT -e format differs.
# Require unambiguous BDF matches; do not infer IDs from amd-smi order.
if [[ -n "$RBT_BIN" ]]; then
    python3 - "$OUT/rbt-devices.txt" \
        "$GPU0_BDF" "$GPU1_BDF" "$OUT/rbt-id-map.txt" <<'PY'
import re
import sys

source, gpu0, gpu1, dest = sys.argv[1:]
text = open(source, errors="replace").read()
blocks = re.split(r"\n\s*Device Index:\s*", text)
matches = {gpu0: set(), gpu1: set()}

for block in blocks[1:]:
    index = re.match(r"\s*(\d+)", block)
    bdf = re.search(r"Device\s+BDF:\s*([0-9a-fA-F:.]+)", block)
    if not index or not bdf:
        continue

    actual = bdf.group(1).lower()
    for wanted in matches:
        domain, bus, slot = wanted.lower().split(":")
        short = f"{bus}:{int(slot.split('.')[0], 16)}.{slot.split('.')[1]}"
        if actual in (wanted.lower(), short):
            matches[wanted].add(index.group(1))

with open(dest, "w") as output:
    for wanted, indices in matches.items():
        result = next(iter(indices)) if len(indices) == 1 else "UNRESOLVED"
        output.write(f"{wanted} RBT_INDEX={result}\n")
PY
else
    printf '%s\n' 'RBT unavailable' >"$OUT/rbt-id-map.txt"
fi

lookup_rbt_index() {
    local bdf=$1
    awk -v b="$bdf" '$1 == b {
        split($2, x, "=");
        print x[2]
    }' "$OUT/rbt-id-map.txt"
}

RBT_GPU0="${RBT_GPU0:-$(lookup_rbt_index "$GPU0_BDF")}"
RBT_GPU1="${RBT_GPU1:-$(lookup_rbt_index "$GPU1_BDF")}"

printf 'RBT_GPU0=%s RBT_GPU1=%s\n' \
    "${RBT_GPU0:-<missing>}" "${RBT_GPU1:-<missing>}" \
    >>"$OUT/manifest.txt"

if [[ "$SKIP_BENCH" == 0 ]]; then
    export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"

    # Make the timed environment explicit. Preserve the original values
    # in manifest.txt above, but do not inherit external visibility masks.
    unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
    unset NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE

    if [[ -n "$RBT_BIN" &&
          "$RBT_GPU0" =~ ^[0-9]+$ &&
          "$RBT_GPU1" =~ ^[0-9]+$ &&
          "$RBT_GPU0" != "$RBT_GPU1" ]]; then
        for ((run=1; run<=RUNS; run++)); do
            run_logged "$OUT/rbt-0to1-small-run${run}.log" \
                "$RBT_BIN" -s "$RBT_GPU0" -d "$RBT_GPU1" -l
            run_logged "$OUT/rbt-1to0-small-run${run}.log" \
                "$RBT_BIN" -s "$RBT_GPU1" -d "$RBT_GPU0" -l
            run_logged "$OUT/rbt-0to1-bandwidth-run${run}.log" \
                "$RBT_BIN" -s "$RBT_GPU0" -d "$RBT_GPU1"
            run_logged "$OUT/rbt-1to0-bandwidth-run${run}.log" \
                "$RBT_BIN" -s "$RBT_GPU1" -d "$RBT_GPU0"
        done
    else
        printf '%s\n' \
          'SKIPPED: RBT missing or BDF-to-RBT-ID map unresolved' \
          >"$OUT/rbt-skipped.txt"
    fi

    if [[ -n "$ALLREDUCE_BIN" ]] && command -v mpirun >/dev/null; then
        # In each rank, expose exactly its assigned physical GPU. This is
        # Open MPI-specific; rank 0/1 mapping is logged before exec.
        MPI=(
            mpirun -np 2 --bind-to numa
            -x HSA_FORCE_FINE_GRAIN_PCIE
            -x GPU0_INDEX -x GPU1_INDEX
            -x GPU0_BDF -x GPU1_BDF
        )

        export GPU0_INDEX GPU1_INDEX GPU0_BDF GPU1_BDF

        RANK_CMD='
          set -eu
          case "${OMPI_COMM_WORLD_LOCAL_RANK:?}" in
            0)
              export ROCR_VISIBLE_DEVICES="$GPU0_INDEX"
              expected_bdf="$GPU0_BDF"
              ;;
            1)
              export ROCR_VISIBLE_DEVICES="$GPU1_INDEX"
              expected_bdf="$GPU1_BDF"
              ;;
            *)
              echo "Unexpected local rank" >&2
              exit 2
              ;;
          esac
          unset HIP_VISIBLE_DEVICES
          printf "PLACEMENT world_rank=%s local_rank=%s ROCR_VISIBLE_DEVICES=%s expected_bdf=%s\n" \
            "$OMPI_COMM_WORLD_RANK" "$OMPI_COMM_WORLD_LOCAL_RANK" \
            "$ROCR_VISIBLE_DEVICES" "$expected_bdf"
          exec "$@"
        '

        for ((run=1; run<=RUNS; run++)); do
            for spec in \
                "small:16:1M" \
                "decode:10240:10240" \
                "large:1M:256M"
            do
                IFS=: read -r label min_size max_size <<<"$spec"
                run_logged "$OUT/rccl-${label}-run${run}.log" \
                    "${MPI[@]}" bash -c "$RANK_CMD" _ \
                    "$ALLREDUCE_BIN" \
                    -b "$min_size" -e "$max_size" -f 2 \
                    -g 1 -d bfloat16 -o sum \
                    -w 20 -n 100 -c 1
            done
        done
    else
        printf '%s\n' \
          'SKIPPED: no MPI-linked all_reduce_perf or mpirun' \
          >"$OUT/rccl-skipped.txt"
    fi

    capture "$OUT/amd-smi-metric-after.txt" amd-smi metric
    snapshot_pcie after
fi

{
    echo "TP PCIe triage"
    echo
    echo "Physical GPUs: $GPU0_BDF and $GPU1_BDF"
    echo "RBT indices: ${RBT_GPU0:-unresolved} and ${RBT_GPU1:-unresolved}"
    echo "MPI-linked RCCL test: ${ALLREDUCE_BIN:-missing}"
    echo "Benchmarks skipped: $SKIP_BENCH"
    echo "Timed-command failures: $FAILURES"
    echo
    echo "Read manifest.txt and rbt-id-map.txt before interpreting benchmarks."
    echo "Check each rccl-*.log for two PLACEMENT lines, distinct expected BDFs,"
    echo "valid table rows, correctness (#wrong 0), and exit=0."
    echo "RBT small-size times measure directed copies, not remote loads."
    echo "RCCL time measures a collective, not raw PCIe one-way latency."
    echo "Treat RBT rates above a single-link physical ceiling as unverified"
    echo "until transfer type, device mapping, and byte/time accounting are checked."
    echo "Compare pcie-before.txt with pcie-after.txt when benchmarks ran."
    echo "Check kernel-gpu-events.txt for reset timing on $GPU1_BDF."
    echo
    echo "Results: $OUT"
} >"$OUT/TRIAGE.txt"

cat "$OUT/TRIAGE.txt"
((FAILURES == 0))
