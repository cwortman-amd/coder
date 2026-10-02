#!/bin/bash
# Named checkpoints for setup.sh and test.sh.
# qwen3.8 keeps the current Qwen3.5 serve flags. gpt-oss-20b is native MXFP4
# on stock vLLM (AITER on, dtype auto, TP=1, prefix caching off) and fits a
# 32 GB R9700S. It does not receive the Qwen architecture override.
# Source after lib/gpu_profile.sh so GPU_PROFILE is already set.

_load_model_catalog() {
    local profile="$1"
    local _catalog_py
    _catalog_py="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/catalog.py"
    if [ -f "${_catalog_py}" ]; then
        # shellcheck disable=SC2046
        eval "$(python3 "${_catalog_py}" exports-model "${profile}")"
    fi
}

apply_model_profile() {
    local raw="${1:-}"
    local engine="${2:-${INFERENCE_ENGINE:-vllm}}"
    local topology="${3:-single}"
    local key family alias=0

    if [ -z "$raw" ]; then
        raw="${MODEL_PROFILE:-${MODEL_NAME:-qwen3.8}}"
    fi
    key="${raw,,}"

    case "$key" in
        qwen3.8|qwen|qwen3.8-27b|qwen3.8-27b-fp8)
            family="qwen3.8"; alias=1 ;;
        gpt-oss|gpt-oss-20b)
            family="gpt-oss-20b"; alias=1 ;;
        gpt-oss-120b)
            family="gpt-oss-120b"; alias=1 ;;
        *flash-next*|*flash_next*)
            family="generic" ;;
        *gpt-oss-120b*)
            family="gpt-oss-120b" ;;
        *gpt-oss*)
            family="gpt-oss-20b" ;;
        *qwen3.8*|*qwen3_5*|*quark-awq-mxfp4*)
            family="qwen3.8" ;;
        *)
            family="generic" ;;
    esac

    export VLLM_ROCM_USE_AITER=0
    export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
    export HSA_NO_SCRATCH_RECLAIM=""
    export AMDGCN_USE_BUFFER_OPS=""
    export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="NONE"
    export VLLM_REASONING_PARSER=""
    export VLLM_HF_OVERRIDES=""
    export VLLM_SKIP_HF_OVERRIDES=0
    export VLLM_SERVE_RECIPE=""
    export MODEL_PROFILE_ENGINE="$engine"

    case "$family" in
        qwen3.8)
            _load_model_catalog qwen3.8
            export MODEL_PROFILE="qwen3.8"
            export TOOL_PARSER="hermes"
            export TOKENIZER_NAME="${MODEL_CATALOG_ID}"
            export VLLM_HF_OVERRIDES='{"architectures": ["Qwen3_5ForCausalLM"]}'
            export VLLM_SKIP_HF_OVERRIDES=0
            export SGLANG_TOOL_PARSER="${SGLANG_TOOL_PARSER:-qwen3_coder}"
            export SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-qwen3}"
            if [ "$SGLANG_TOOL_PARSER" = "none" ]; then
                export SGLANG_TOOL_PARSER="qwen3_coder"
            fi
            if [ "$SGLANG_REASONING_PARSER" = "none" ]; then
                export SGLANG_REASONING_PARSER="qwen3"
            fi
            if [ "$alias" -eq 1 ]; then
                case "$engine" in
                    llama.cpp)
                        export MODEL_NAME="Qwen3.8-27B"
                        export SERVED_MODEL_NAME="Qwen3.8-27B"
                        ;;
                    mxfp4)
                        export MODEL_PATH="${MODEL_PATH:-/models/${MODEL_MXFP4_ID}}"
                        export MXFP4_MODEL_NAME="${MXFP4_MODEL_NAME:-${MODEL_MXFP4_ID}}"
                        export MODEL_NAME="$MXFP4_MODEL_NAME"
                        export SERVED_MODEL_NAME="$MXFP4_MODEL_NAME"
                        export MODEL_CONTAINER="${MODEL_MXFP4_CONTAINER}"
                        ;;
                    *)
                        export MODEL_NAME="${MODEL_CATALOG_ID}"
                        export SERVED_MODEL_NAME="$MODEL_NAME"
                        ;;
                esac
            else
                export MODEL_NAME="$raw"
                export SERVED_MODEL_NAME="$raw"
                if [ "$engine" = "mxfp4" ]; then
                    export MXFP4_MODEL_NAME="$raw"
                    case "$raw" in
                        /*) export MODEL_PATH="$raw" ;;
                    esac
                fi
            fi
            ;;
        gpt-oss-20b|gpt-oss-120b)
            if [ "$topology" != "single" ]; then
                echo "GPT-OSS profiles launch on a single GPU. Topology '${topology}' is not supported." >&2
                return 1
            fi
            if [ "$engine" = "mxfp4" ] || [ "$engine" = "llama.cpp" ]; then
                echo "GPT-OSS ships native MXFP4. Serving it with stock vLLM, not the Quark or GGUF compose file." >&2
                engine="vllm"
                export MODEL_PROFILE_ENGINE="vllm"
            fi
            _load_model_catalog "$family"
            if [ "$family" = "gpt-oss-120b" ]; then
                export MODEL_PROFILE="gpt-oss-120b"
                export MODEL_NAME="${MODEL_CATALOG_ID}"
                if [ "${GPU_PROFILE:-r9700}" = "r9700" ]; then
                    echo "gpt-oss-120b does not fit a 32 GB R9700 or R9700S. Use gpt-oss-20b on that card, or run 120b on MI350P." >&2
                fi
            else
                export MODEL_PROFILE="gpt-oss-20b"
                export MODEL_NAME="${MODEL_CATALOG_ID}"
            fi
            if [ "$alias" -eq 0 ]; then
                export MODEL_NAME="$raw"
            fi
            export SERVED_MODEL_NAME="$MODEL_NAME"
            export TOKENIZER_NAME="$MODEL_NAME"
            export TOOL_PARSER="openai"
            export VLLM_HF_OVERRIDES=""
            export VLLM_SKIP_HF_OVERRIDES=1
            export VLLM_SERVE_RECIPE="gpt-oss-mxfp4"
            export SGLANG_TOOL_PARSER="none"
            export SGLANG_REASONING_PARSER="none"
            export SGLANG_MAMBA_RADIX="none"
            # Native MXFP4. AITER is required on the 32 GB R9700S as well as MI350P.
            # Unified attention stays off on RDNA 4: its LDS tile exceeds 64 KB.
            export VLLM_ROCM_USE_AITER=1
            if [ "${GPU_PROFILE:-}" = "mi350p" ]; then
                export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
                export HSA_NO_SCRATCH_RECLAIM=1
                export AMDGCN_USE_BUFFER_OPS=0
                export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
            fi
            ;;
        *)
            export MODEL_PROFILE="custom"
            export MODEL_NAME="$raw"
            export SERVED_MODEL_NAME="$raw"
            export TOOL_PARSER="${TOOL_PARSER:-hermes}"
            export VLLM_HF_OVERRIDES=""
            export VLLM_SKIP_HF_OVERRIDES=1
            export SGLANG_TOOL_PARSER="none"
            export SGLANG_REASONING_PARSER="none"
            export SGLANG_MAMBA_RADIX="none"
            if [[ "$raw" == */* ]]; then
                export TOKENIZER_NAME="$raw"
            else
                export TOKENIZER_NAME=""
            fi
            ;;
    esac

    export INFERENCE_ENGINE="$engine"
}

# Kernel variants are served on their own GPU. rocm_attn is the stock
# decoder attention path. aiter and mxfp4 are the optimized alternatives.
apply_kernel_variant() {
    local kernel="${1:-}"
    kernel="${kernel,,}"
    case "$kernel" in
        ""|default)
            export KERNEL_VARIANT=""
            export ATTENTION_BACKEND=""
            ;;
        rocm_attn|stock|baseline)
            export KERNEL_VARIANT="rocm_attn"
            export ATTENTION_BACKEND="ROCM_ATTN"
            export VLLM_ROCM_USE_AITER=0
            export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
            ;;
        aiter|aiter_unified)
            if [[ "${MODEL_PROFILE:-}" == gpt-oss* ]] && [ "${GPU_PROFILE:-r9700}" != "mi350p" ]; then
                echo "GPT-OSS on a 32 GB R9700S already sets VLLM_ROCM_USE_AITER=1. Unified attention exceeds the 64 KB LDS limit on gfx1201." >&2
                return 1
            fi
            export KERNEL_VARIANT="aiter"
            export ATTENTION_BACKEND="ROCM_AITER_UNIFIED_ATTN"
            export VLLM_ROCM_USE_AITER=1
            export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
            ;;
        mxfp4|radiance)
            if [ "${MODEL_PROFILE:-}" != "qwen3.8" ]; then
                echo "The mxfp4 kernel variant is the Quark Qwen checkpoint. GPT-OSS already serves native MXFP4 from stock vLLM." >&2
                return 1
            fi
            export KERNEL_VARIANT="mxfp4"
            export ATTENTION_BACKEND=""
            if [ -z "${MODEL_MXFP4_ID:-}" ]; then
                _load_model_catalog qwen3.8
            fi
            export MODEL_PATH="${MODEL_PATH:-/models/${MODEL_MXFP4_ID}}"
            export MXFP4_MODEL_NAME="${MXFP4_MODEL_NAME:-${MODEL_MXFP4_ID}}"
            export MODEL_NAME="$MXFP4_MODEL_NAME"
            export SERVED_MODEL_NAME="$MXFP4_MODEL_NAME"
            export MODEL_PROFILE_ENGINE="mxfp4"
            ;;
        *)
            echo "Unknown kernel variant '${kernel}'. Use rocm_attn, aiter, or mxfp4." >&2
            return 1
            ;;
    esac
}
