#!/usr/bin/env bash
# Apply this repo's NIXL HIP-IPC patches onto a NIXL source tree.
#
# The patches in patches/nixl/ are git diffs against NIXL v1.4.0
# (commit c0a1102b94d173049a5478c23e765ba37681e2ca). Apply them before
# configuring Meson. A host-built plugin needs GLIBC_2.38 and will not
# load in vllm/vllm-openai-rocm; build inside that image when the
# consumer is vLLM. See docs/KV_CONNECTOR.md section 5.4.3.
#
# Usage:
#   scripts/apply_nixl_hip_ipc_patch.sh /path/to/nixl
#   scripts/apply_nixl_hip_ipc_patch.sh --check /path/to/nixl
#   scripts/apply_nixl_hip_ipc_patch.sh --reverse /path/to/nixl
#
# NIXL_SRC may be set instead of the positional path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="${ROOT}/patches/nixl"
mode="apply"
src="${NIXL_SRC:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--check|--reverse] /path/to/nixl

  --check    Report whether each patch applies. Do not modify the tree.
  --reverse  Remove patches that are already applied, last file first.

Patches are applied in lexical order from:
  ${PATCH_DIR}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) mode="check"; shift ;;
    --reverse) mode="reverse"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -n "${src}" ]]; then
        echo "unexpected extra argument: $1" >&2
        exit 2
      fi
      src="$1"
      shift
      ;;
  esac
done

if [[ -z "${src}" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -d "${src}" ]]; then
  echo "NIXL source directory not found: ${src}" >&2
  exit 1
fi
if [[ ! -f "${src}/meson.build" ]]; then
  echo "not a NIXL tree (meson.build missing): ${src}" >&2
  exit 1
fi

shopt -s nullglob
patches=("${PATCH_DIR}"/*.patch)
shopt -u nullglob
if [[ ${#patches[@]} -eq 0 ]]; then
  echo "no patches in ${PATCH_DIR}" >&2
  exit 1
fi

if [[ "${mode}" == "reverse" ]]; then
  reversed=()
  for ((i=${#patches[@]}-1; i>=0; i--)); do
    reversed+=("${patches[$i]}")
  done
  patches=("${reversed[@]}")
fi

if [[ -d "${src}/.git" ]]; then
  desc="$(git -C "${src}" describe --tags --always --dirty 2>/dev/null || true)"
  echo "NIXL tree: ${src} (${desc:-unknown})"
else
  echo "NIXL tree: ${src} (not a git checkout)"
fi
echo "Expected base: v1.4.0 (c0a1102b94d173049a5478c23e765ba37681e2ca)"

apply_status() {
  local patch="$1"
  local forward_args=(--check --whitespace=nowarn)
  local reverse_args=(--reverse --check --whitespace=nowarn)
  if git -C "${src}" apply "${forward_args[@]}" "${patch}" >/dev/null 2>&1; then
    echo "forward"
  elif git -C "${src}" apply "${reverse_args[@]}" "${patch}" >/dev/null 2>&1; then
    echo "applied"
  else
    echo "conflict"
  fi
}

failed=0
for patch in "${patches[@]}"; do
  name="$(basename "${patch}")"
  status="$(apply_status "${patch}")"
  case "${mode}:${status}" in
    check:forward)
      echo "ok: ${name} would apply"
      ;;
    check:applied)
      echo "ok: ${name} is already applied"
      ;;
    apply:forward)
      git -C "${src}" apply --whitespace=nowarn "${patch}"
      echo "applied: ${name}"
      ;;
    apply:applied)
      echo "skip: ${name} is already applied"
      ;;
    reverse:applied)
      git -C "${src}" apply --reverse --whitespace=nowarn "${patch}"
      echo "reversed: ${name}"
      ;;
    reverse:forward)
      echo "skip: ${name} is not applied"
      ;;
    *)
      echo "failed: ${name} does not apply cleanly to ${src}" >&2
      git -C "${src}" apply --check --whitespace=nowarn "${patch}" >&2 || true
      failed=1
      ;;
  esac
done

if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

if [[ "${mode}" == "apply" ]]; then
  cat <<EOF

Next, from ${src}, build only the HIP_IPC plugin. Inside the vLLM image:

  pip install 'meson==1.3.2' ninja
  meson setup build-hip-ipc \\
    --prefix=/opt/nixl-hip-ipc \\
    -Denable_plugins=HIP_IPC \\
    -Dhip_path=/opt/rocm \\
    -Dwheel_variant=rocm \\
    -Dbuild_tests=false \\
    -Dbuild_examples=false
  meson compile -C build-hip-ipc
  meson install -C build-hip-ipc
EOF
fi
