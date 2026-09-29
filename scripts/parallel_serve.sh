#!/usr/bin/env bash
# One server per GPU.
# Different models share the host for a comparison. A kernel variant
# (aiter or mxfp4) is never placed on the GPU that serves the baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/lib/gpu_profile.sh"
# shellcheck disable=SC1091
source "${ROOT}/lib/model_profile.sh"
# shellcheck disable=SC1091
source "${ROOT}/lib/serve.sh"

if [ -f "$HOME/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$HOME/.env"
    set +a
fi
if [ -f "${ROOT}/.env" ]; then
    PREV_HF_TOKEN="${HF_TOKEN:-}"
    set -a
    # shellcheck disable=SC1091
    source "${ROOT}/.env"
    set +a
    if [ -z "${HF_TOKEN:-}" ] && [ -n "${PREV_HF_TOKEN:-}" ]; then
        export HF_TOKEN="$PREV_HF_TOKEN"
    fi
fi

GPU_PROFILE="${GPU_PROFILE:-auto}"
SLOTS=""
GPU_LIST=""
PRINT_ONLY=0
ACTION="up"

usage() {
    cat <<'EOF'
Usage: scripts/parallel_serve.sh --slots SPEC[,SPEC...] [--gpus 0,1] [--print]

Each SPEC is profile[@kernel][:gpu[:port]]

  qwen3.8,gpt-oss-20b
      First free GPU gets Qwen3.8. The next GPU gets GPT-OSS.

  qwen3.8@rocm_attn,qwen3.8@aiter
      Stock ROCM_ATTN on one GPU. AITER unified attention on another GPU.

  qwen3.8@rocm_attn:0:8000,qwen3.8@mxfp4:1:8001
      Baseline FP8 on GPU 0. Quark MXFP4 on GPU 1.

Kernels: rocm_attn (stock), aiter (ROCM_AITER_UNIFIED_ATTN), mxfp4 (Qwen Quark only).
HIP_VISIBLE_DEVICES selects the card. ROCR_VISIBLE_DEVICES is not set.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slots)
            SLOTS="$2"; shift 2 ;;
        --gpus)
            GPU_LIST="$2"; shift 2 ;;
        -g|--gpu-profile)
            GPU_PROFILE="$2"; shift 2 ;;
        --print)
            PRINT_ONLY=1; shift ;;
        --down)
            ACTION="down"; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1 ;;
    esac
done

apply_gpu_profile

detect_gpu_ids() {
    local ids=()
    if [ -n "$GPU_LIST" ]; then
        IFS=',' read -ra ids <<< "$GPU_LIST"
        printf '%s\n' "${ids[@]}"
        return
    fi
    mapfile -t ids < <(list_target_gpu_ids "$GPU_PROFILE" || true)
    if [ "${#ids[@]}" -eq 0 ]; then
        # One discrete slot. Do not scan every amd-smi ordinal; that list
        # includes the integrated APU.
        ids=(0)
    fi
    printf '%s\n' "${ids[@]}"
}

if [ "$ACTION" = "down" ]; then
    docker ps -a --format '{{.Names}}' | grep -E '^rocm-parallel-g[0-9]+$' | while read -r name; do
        docker rm -f "$name" >/dev/null
        echo "Removed ${name}"
    done
    exit 0
fi

if [ -z "$SLOTS" ]; then
    echo "Error: --slots is required." >&2
    usage >&2
    exit 1
fi

mapfile -t GPU_IDS < <(detect_gpu_ids)
declare -a SLOT_PROFILE=() SLOT_KERNEL=() SLOT_GPU=() SLOT_PORT=()
IFS=',' read -ra RAW_SLOTS <<< "$SLOTS"
next_gpu=0
next_port=8000

for raw in "${RAW_SLOTS[@]}"; do
    raw="${raw#"${raw%%[![:space:]]*}"}"
    raw="${raw%"${raw##*[![:space:]]}"}"
    [ -n "$raw" ] || continue
    head="${raw%%:*}"
    rest=""
    if [[ "$raw" == *:* ]]; then
        rest="${raw#*:}"
    fi
    profile="$head"
    kernel=""
    if [[ "$head" == *@* ]]; then
        profile="${head%%@*}"
        kernel="${head#*@}"
    fi
    gpu=""
    port=""
    if [ -n "$rest" ]; then
        gpu="${rest%%:*}"
        if [[ "$rest" == *:* ]]; then
            port="${rest#*:}"
        fi
    fi
    if [ -z "$gpu" ]; then
        if [ "$next_gpu" -ge "${#GPU_IDS[@]}" ]; then
            echo "Not enough GPUs for '${raw}'. Detected: ${GPU_IDS[*]}. Pass --gpus 0,1 when detection is short." >&2
            exit 1
        fi
        gpu="${GPU_IDS[$next_gpu]}"
        next_gpu=$((next_gpu + 1))
    fi
    if [ -z "$port" ]; then
        port="$next_port"
        next_port=$((next_port + 1))
    else
        if [ "$port" -ge "$next_port" ]; then
            next_port=$((port + 1))
        fi
    fi
    SLOT_PROFILE+=("$profile")
    SLOT_KERNEL+=("$kernel")
    SLOT_GPU+=("$gpu")
    SLOT_PORT+=("$port")
done

if [ "${#SLOT_PROFILE[@]}" -eq 0 ]; then
    echo "No slots parsed from --slots." >&2
    exit 1
fi

declare -A GPU_SEEN=()
for i in "${!SLOT_GPU[@]}"; do
    g="${SLOT_GPU[$i]}"
    if [ -n "${GPU_SEEN[$g]:-}" ]; then
        echo "GPU ${g} is assigned twice (${GPU_SEEN[$g]} and ${SLOT_PROFILE[$i]}@${SLOT_KERNEL[$i]:-default})." >&2
        echo "Put the optimized kernel on a different GPU." >&2
        exit 1
    fi
    GPU_SEEN[$g]="${SLOT_PROFILE[$i]}@${SLOT_KERNEL[$i]:-default}"
done

gpu_holder() {
    local gpu="$1"
    local id hip name
    while read -r id; do
        [ -n "$id" ] || continue
        hip="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$id" 2>/dev/null | sed -n 's/^HIP_VISIBLE_DEVICES=//p' | head -n1)"
        name="$(docker inspect -f '{{.Name}}' "$id" 2>/dev/null | sed 's#^/##')"
        if [ "$name" = "rocm-parallel-g${gpu}" ]; then
            continue
        fi
        if [[ ",${hip}," == *",${gpu},"* ]]; then
            echo "$name"
            return 0
        fi
    done < <(docker ps -q)
    return 1
}

mkdir -p "${ROOT}/_results/parallel"
MANIFEST="${ROOT}/_results/parallel/slots.tsv"
: > "$MANIFEST"
exec 3>>"$MANIFEST"

for i in "${!SLOT_PROFILE[@]}"; do
    profile="${SLOT_PROFILE[$i]}"
    kernel="${SLOT_KERNEL[$i]}"
    gpu="${SLOT_GPU[$i]}"
    port="${SLOT_PORT[$i]}"
    if [ "$PRINT_ONLY" -eq 0 ]; then
        holder="$(gpu_holder "$gpu" || true)"
        if [ -n "$holder" ]; then
            echo "GPU ${gpu} is already used by container ${holder}. Stop it before a parallel launch." >&2
            exit 1
        fi
    fi

    (
        apply_model_profile "$profile" vllm single
        apply_kernel_variant "$kernel"
        name="rocm-parallel-g${gpu}"
        image="${VLLM_IMAGE}"
        entrypoint=(vllm serve)
        if [ "${VLLM_SERVE_RECIPE:-}" = "gpt-oss-mxfp4" ]; then
            # Native MXFP4. gpt-oss-20b fits a 32 GB R9700S; 120b does not.
            serve=(
                "$MODEL_NAME"
                --host "${INFERENCE_BIND_HOST:-127.0.0.1}"
                --port "$port"
                --dtype auto
                --tensor-parallel-size 1
                --no-enable-prefix-caching
                --disable-uvicorn-access-log
            )
        else
            serve=(
                "$MODEL_NAME"
                --host "${INFERENCE_BIND_HOST:-127.0.0.1}"
                --port "$port"
                --served-model-name "$SERVED_MODEL_NAME"
                --max-model-len "${MAX_MODEL_LEN}"
                --gpu-memory-utilization "${GPU_MEM_UTIL}"
                --enable-auto-tool-choice
                --tool-call-parser "$TOOL_PARSER"
            )
            if [ -n "${KV_CACHE_MEMORY_BYTES:-}" ]; then
                serve+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
            fi
            if [ -n "${VLLM_COMPILATION_CONFIG:-}" ]; then
                serve+=(--compilation-config "$VLLM_COMPILATION_CONFIG")
            fi
            if [ -n "${VLLM_HF_OVERRIDES:-}" ]; then
                serve+=(--hf-overrides "$VLLM_HF_OVERRIDES")
            fi
            if [ -n "${ATTENTION_BACKEND:-}" ]; then
                serve+=(--attention-backend "$ATTENTION_BACKEND")
            fi
        fi
        mounts=(
            -v "${HF_HOME}:/root/.cache/huggingface"
            -v "${ROOT}/_results:/results"
            -v "${MODELS_DIR}:/models"
        )
        if [ "${MODEL_PROFILE}" = "qwen3.8" ] && [ "${KERNEL_VARIANT}" != "mxfp4" ]; then
            mounts+=(
                -v "${ROOT}/patches/qwen3_5_vllm030.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py:ro"
                -v "${ROOT}/scripts/qwen3_5_rocm10.py:/opt/python/lib/python3.14/site-packages/vllm/model_executor/models/qwen3_5.py:ro"
            )
        fi
        if [ "${KERNEL_VARIANT}" = "mxfp4" ]; then
            image="${MXFP4_IMAGE}"
            entrypoint=(/opt/radiance_entrypoint.sh)
            serve=(
                "$MODEL_PATH"
                --served-model-name "$SERVED_MODEL_NAME"
                --host "${INFERENCE_BIND_HOST:-127.0.0.1}"
                --port "$port"
                --quantization quark
                --dtype auto
                --kv-cache-dtype fp8
                --max-model-len "${MAX_MODEL_LEN}"
                --gpu-memory-utilization "${GPU_MEM_UTIL}"
            )
        fi
        env_args=(
            -e "HIP_VISIBLE_DEVICES=${gpu}"
            -e "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
            -e "GPU_ARCHS=${PYTORCH_ROCM_ARCH}"
            -e "GPU_PROFILE=${GPU_PROFILE}"
            -e "HF_HOME=/root/.cache/huggingface"
            -e "HF_TOKEN=${HF_TOKEN:-}"
            -e "SAFETENSORS_FAST_GPU=1"
            -e "HIP_FORCE_DEV_KERNARG=1"
            -e "TOKENIZERS_PARALLELISM=false"
            -e "PYTHONUNBUFFERED=1"
            -e "VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0}"
            -e "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=${VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:-0}"
        )
        if [ -n "${HSA_NO_SCRATCH_RECLAIM:-}" ]; then
            env_args+=(-e "HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM}")
        fi
        if [ -n "${AMDGCN_USE_BUFFER_OPS:-}" ]; then
            env_args+=(-e "AMDGCN_USE_BUFFER_OPS=${AMDGCN_USE_BUFFER_OPS}")
        fi
        if [ -n "${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-}" ]; then
            env_args+=(-e "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION}")
        fi
        if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
            env_args+=(-e "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}")
        fi
        serve_gpu_flags "$gpu"
        docker_cmd=(
            docker run -d --name "$name" --restart no
            --network host --ipc host --shm-size 64g
            "${SERVE_GPU_FLAGS[@]}"
            "${env_args[@]}"
            "${mounts[@]}"
            --entrypoint "${entrypoint[0]}"
            "$image"
        )
        if [ "${#entrypoint[@]}" -gt 1 ]; then
            docker_cmd+=("${entrypoint[@]:1}")
        fi
        docker_cmd+=("${serve[@]}")
        if [ "$PRINT_ONLY" -eq 1 ]; then
            printf 'GPU %s port %s profile %s kernel %s model %s\n' \
                "$gpu" "$port" "$MODEL_PROFILE" "${KERNEL_VARIANT:-default}" "$SERVED_MODEL_NAME"
            printf ' %q' "${docker_cmd[@]}"
            printf '\n'
        else
            docker rm -f "$name" >/dev/null 2>&1 || true
            "${docker_cmd[@]}" >/dev/null
            echo "Started ${name}: ${MODEL_PROFILE} ${KERNEL_VARIANT:-default} on GPU ${gpu} port ${port} (${SERVED_MODEL_NAME})"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$gpu" "$port" "$MODEL_PROFILE" "${KERNEL_VARIANT:-default}" \
            "http://127.0.0.1:${port}" "$SERVED_MODEL_NAME" "$TOKENIZER_NAME" >&3
    )
done
exec 3>&-

if [ "$PRINT_ONLY" -eq 1 ]; then
    exit 0
fi

echo "Waiting for parallel servers..."
fail=0
while read -r gpu port profile kernel url served tokenizer; do
    [ -n "${port:-}" ] || continue
    ready=0
    for _ in $(seq 1 120); do
        if curl -sf --connect-timeout 2 "${url}/health" >/dev/null 2>&1 \
            || curl -sf --connect-timeout 2 "${url}/v1/models" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 5
    done
    if [ "$ready" -ne 1 ]; then
        echo "GPU ${gpu} ${profile} on ${url} did not become healthy. Logs: docker logs rocm-parallel-g${gpu}" >&2
        fail=1
    else
        echo "Ready ${profile} (${kernel}) ${url} model ${served}"
    fi
done < "$MANIFEST"

echo "Slot table: ${MANIFEST}"
exit "$fail"
