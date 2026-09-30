#!/usr/bin/env bash
set -Eeuo pipefail

: "${RBT_GPU0:?Set RBT_GPU0 from rocm-bandwidth-test -e}"
: "${RBT_GPU1:?Set RBT_GPU1 from rocm-bandwidth-test -e}"

RBT="${RBT:-/opt/rocm/bin/rocm-bandwidth-test}"
RCCL_TESTS="${RCCL_TESTS:-./rccl-tests/build}"
VISIBLE_GPUS="${VISIBLE_GPUS:-0,1}"
RUNS="${RUNS:-3}"
OUT="${OUT:-./p2p-characterization-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$OUT"
command -v mpirun >/dev/null
command -v amd-smi >/dev/null
[[ -x "$RBT" ]] || { echo "Not executable: $RBT" >&2; exit 1; }
[[ -x "$RCCL_TESTS/all_reduce_perf" ]] ||
    { echo "Not executable: $RCCL_TESTS/all_reduce_perf" >&2; exit 1; }
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] ||
    { echo "RUNS must be a positive integer" >&2; exit 1; }

export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
export ROCR_VISIBLE_DEVICES="$VISIBLE_GPUS"

{
    date -Is
    uname -a
    printf 'RBT_GPU0=%s RBT_GPU1=%s\n' "$RBT_GPU0" "$RBT_GPU1"
    printf 'VISIBLE_GPUS=%s HSA_FORCE_FINE_GRAIN_PCIE=%s\n' \
        "$VISIBLE_GPUS" "$HSA_FORCE_FINE_GRAIN_PCIE"
    printf 'RCCL_TESTS=%s RUNS=%s\n' "$RCCL_TESTS" "$RUNS"
    printf 'NCCL_P2P_DISABLE=%s\n' "${NCCL_P2P_DISABLE-<unset>}"
    printf 'NCCL_CUMEM_ENABLE=%s\n' "${NCCL_CUMEM_ENABLE-<unset>}"
    printf 'NCCL_WIN_ENABLE=%s\n' "${NCCL_WIN_ENABLE-<unset>}"
    cat /proc/cmdline
} > "$OUT/manifest.txt"

amd-smi list > "$OUT/amd-smi-list.txt" 2>&1
amd-smi topology --access > "$OUT/peer-access.txt" 2>&1
amd-smi topology --link-type > "$OUT/link-type.txt" 2>&1
lspci -Dtv > "$OUT/pci-tree.txt" 2>&1
"$RBT" -e > "$OUT/rbt-devices.txt" 2>&1
"$RBT" -t > "$OUT/rbt-topology.txt" 2>&1

for bdf in 0000:8b:00.0 0001:c7:00.0; do
    {
        echo "BDF=$bdf"
        cat "/sys/bus/pci/devices/$bdf/current_link_speed"
        cat "/sys/bus/pci/devices/$bdf/current_link_width"
        lspci -Dvv -s "$bdf"
    } > "$OUT/pci-${bdf//:/_}.txt" 2>&1
done

run_logged() {
    local log=$1
    shift
    echo "Running: $*" | tee "$log"
    "$@" 2>&1 | tee -a "$log"
}

for run in $(seq 1 "$RUNS"); do
    for direction in "0to1:$RBT_GPU0:$RBT_GPU1" \
                     "1to0:$RBT_GPU1:$RBT_GPU0"; do
        IFS=: read -r label src dst <<< "$direction"

        run_logged "$OUT/rbt-${label}-small-run${run}.log" \
            "$RBT" -s "$src" -d "$dst" -l

        run_logged "$OUT/rbt-${label}-bandwidth-run${run}.log" \
            "$RBT" -s "$src" -d "$dst"
    done
done

# Open MPI: pass environment explicitly to both local ranks.
MPI=(mpirun -np 2 --bind-to numa
     -x ROCR_VISIBLE_DEVICES
     -x HSA_FORCE_FINE_GRAIN_PCIE)

# The small sweep emphasizes operation time; the large sweep emphasizes bandwidth.
for run in $(seq 1 "$RUNS"); do
    for spec in "small:16:1M" "large:1M:256M"; do
        IFS=: read -r label min_size max_size <<< "$spec"

        run_logged "$OUT/rccl-allreduce-${label}-run${run}.log" \
            "${MPI[@]}" "$RCCL_TESTS/all_reduce_perf" \
            -b "$min_size" -e "$max_size" -f 2 \
            -g 1 -d bfloat16 -o sum \
            -w 20 -n 100 -c 1
    done
done

echo "Completed. Results in: $OUT"
