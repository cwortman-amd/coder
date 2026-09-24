#!/bin/bash
# ==============================================================================
# GPU profile presets for Radeon AI PRO R9700 (gfx1201) and Instinct MI350P
# (gfx950). Source after loading .env. Override with GPU_PROFILE=r9700|mi350p|auto.
# ==============================================================================

_gpu_profile_detect() {
    local info=""
    if command -v rocminfo >/dev/null 2>&1; then
        info="$(rocminfo 2>/dev/null || true)"
    fi
    if command -v rocm-smi >/dev/null 2>&1; then
        info="${info}$(rocm-smi --showproductname --showid 2>/dev/null || true)"
    fi
    if command -v lspci >/dev/null 2>&1; then
        info="${info}$(lspci -nn 2>/dev/null || true)"
    fi
    if echo "$info" | grep -Eqi 'gfx950|MI350P|MI350 X|MI350X|MI355|1002:75a'; then
        echo "mi350p"
        return
    fi
    if echo "$info" | grep -Eqi 'gfx1201|R9700'; then
        echo "r9700"
        return
    fi
    echo "${GPU_PROFILE:-r9700}"
}

apply_gpu_profile() {
    local requested="${GPU_PROFILE:-auto}"
    requested="${requested,,}"
    case "$requested" in
        r9700|gfx1201|radeon) GPU_PROFILE="r9700" ;;
        mi350p|mi350|gfx950|instinct) GPU_PROFILE="mi350p" ;;
        auto|"") GPU_PROFILE="$(_gpu_profile_detect)" ;;
        *)
            echo "Unknown GPU_PROFILE='$requested' (use r9700, mi350p, or auto)" >&2
            return 1
            ;;
    esac

    VIDEO_GID="$(getent group video 2>/dev/null | cut -d: -f3 || true)"
    RENDER_GID="$(getent group render 2>/dev/null | cut -d: -f3 || true)"
    export VIDEO_GID="${VIDEO_GID:-44}"
    export RENDER_GID="${RENDER_GID:-109}"

    case "$GPU_PROFILE" in
        mi350p)
            export GPU_PROFILE="mi350p"
            export GPU_FAMILY="instinct"
            export GPU_MARKETING_NAME="AMD Instinct MI350P"
            export PYTORCH_ROCM_ARCH="gfx950"
            export GPU_ISA="gfx950"
            export GPU_VRAM_GB="144"
            export GPU_TDP_W="600"
            export GPU_LABEL="AMD Instinct™ MI350P (gfx950, 144 GB HBM3E)"
            unset HSA_OVERRIDE_GFX_VERSION || true
            export VLLM_ROCM_FP8_PADDING=""
            export VLLM_COMPILATION_CONFIG="{\"cudagraph_mode\": \"PIECEWISE\"}"
            export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-34359738368}"
            export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
            export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
            export RADIANCE_USE_R4D="0"
            export RADIANCE_R4D_ATTN_FP8="0"
            export MXFP4_IMAGE="local/vllm-mxfp4:gfx950"
            export GGUF_IMAGE="local/llama.cpp:rocm10-gfx950"
            export VLLM_IMAGE="${VLLM_IMAGE:-rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0}"
            # .env may still pin the R9700 coder-sglang tag; gfx950 needs the ROCm image.
            export SGLANG_IMAGE="rocm/sgl-dev:v0.5.15.post1-ubuntu24.04-py3.14-rocm10.0.0"
            export SGLANG_TOOL_PARSER="${SGLANG_TOOL_PARSER:-qwen3_coder}"
            export SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-qwen3}"
            export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-aiter}"
            export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-131072}"
            export SGLANG_TOKENIZER_PATH="${SGLANG_TOKENIZER_PATH:-}"
            export ROCM_USE_AITER="${ROCM_USE_AITER:-1}"
            export SGLANG_MAMBA_RADIX="${SGLANG_MAMBA_RADIX:-no_buffer}"
            ;;
        *)
            export GPU_PROFILE="r9700"
            export GPU_FAMILY="radeon"
            export GPU_MARKETING_NAME="AMD Radeon AI PRO R9700"
            export PYTORCH_ROCM_ARCH="gfx1201"
            export GPU_ISA="gfx1201"
            export GPU_VRAM_GB="32"
            export GPU_TDP_W="300"
            export GPU_LABEL="AMD Radeon™ AI PRO R9700 (gfx1201, 32 GB GDDR6)"
            export HSA_OVERRIDE_GFX_VERSION="12.0.1"
            export VLLM_ROCM_FP8_PADDING="0"
            export VLLM_COMPILATION_CONFIG="{\"cudagraph_mode\": \"NONE\"}"
            export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1073741824}"
            export MAX_MODEL_LEN="${MAX_MODEL_LEN:-9600}"
            export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
            export RADIANCE_USE_R4D="1"
            export RADIANCE_R4D_ATTN_FP8="3"
            export MXFP4_IMAGE="local/vllm-mxfp4:gfx1201"
            export GGUF_IMAGE="local/llama.cpp:rocm10-gfx1201"
            export VLLM_IMAGE="${VLLM_IMAGE:-rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0}"
            ;;
    esac

    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
    export INFERENCE_ENGINE="${INFERENCE_ENGINE:-vllm}"
    export INFERENCE_PORT="${INFERENCE_PORT:-8000}"
    export OPENCODE_PORT="${OPENCODE_PORT:-4096}"
    export ROUTER_BIND_HOST="${ROUTER_BIND_HOST:-127.0.0.1}"
    export MODELS_DIR="${MODELS_DIR:-${SCRIPT_DIR:-.}/models}"
    export HF_HOME="${HF_HOME:-${HF_CACHE_DIR:-$HOME/.cache/huggingface}}"
    export HF_CACHE_DIR="${HF_CACHE_DIR:-$HF_HOME}"
}

compose_file_for_engine() {
    local engine="${1:-vllm}"
    engine="${engine,,}"
    case "$engine" in
        llama.cpp|llamacpp|gguf) echo "docker-compose.gguf.yml" ;;
        sglang) echo "docker-compose.sglang.yml" ;;
        mxfp4) echo "docker-compose.mxfp4.yml" ;;
        tp2) echo "docker-compose.tp2.yml" ;;
        dp2) echo "docker-compose.dp2.yml" ;;
        pd) echo "docker-compose.pd.yml" ;;
        vllm|*) echo "docker-compose.yml" ;;
    esac
}

docker_compose() {
    local file="${COMPOSE_FILE:-docker-compose.yml}"
    docker compose -f "$file" "$@"
}

upsert_env_var() {
    local key="$1"
    local val="$2"
    local file="${3:-${SCRIPT_DIR}/.env}"
    python3 - "$key" "$val" "$file" <<'PY'
import os, re, sys
key, val, path = sys.argv[1], sys.argv[2], sys.argv[3]
lines = []
if os.path.exists(path):
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
pat = re.compile(r"^\s*" + re.escape(key) + r"\s*=")
out, found = [], False
for line in lines:
    if pat.match(line) and not found:
        out.append(f"{key}={val}")
        found = True
    else:
        out.append(line)
if not found:
    if out and out[-1] != "":
        out.append("")
    out.append(f"{key}={val}")
os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
with open(path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
PY
}

fill_empty_env_var() {
    local key="$1"
    local val="$2"
    local file="${3:-${SCRIPT_DIR}/.env}"
    python3 - "$key" "$val" "$file" <<'PY'
import os, re, sys
key, val, path = sys.argv[1], sys.argv[2], sys.argv[3]
if not val or not os.path.exists(path):
    raise SystemExit(0)
lines = open(path, encoding="utf-8").read().splitlines()
pat = re.compile(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$")
out, found, changed = [], False, False
for line in lines:
    m = pat.match(line)
    if m and not found:
        found = True
        current = m.group(1).strip().strip("'\"")
        if current == "":
            out.append(f"{key}={val}")
            changed = True
        else:
            out.append(line)
    else:
        out.append(line)
if not found:
    out.append(f"{key}={val}")
    changed = True
if changed:
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
PY
}
