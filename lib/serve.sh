#!/bin/bash
# Shared GPU access flags for a serve or profile container.
# Source after lib/gpu_profile.sh when that file has already set the GIDs.
# HIP_VISIBLE_DEVICES is the GPU index, or a comma list such as 0,1.

serve_refresh_gids() {
    if [ -n "${VIDEO_GID:-}" ] && [ -n "${RENDER_GID:-}" ]; then
        return 0
    fi
    local video render
    video="$(getent group video 2>/dev/null | cut -d: -f3 || true)"
    render="$(getent group render 2>/dev/null | cut -d: -f3 || true)"
    export VIDEO_GID="${video:-44}"
    export RENDER_GID="${render:-109}"
}

# Fills SERVE_GPU_FLAGS. Pass the HIP ordinal or list as $1.
serve_gpu_flags() {
    local gpu="${1:-${HIP_VISIBLE_DEVICES:-0}}"
    serve_refresh_gids
    SERVE_GPU_FLAGS=(
        --device /dev/kfd
        --device /dev/dri
        --group-add "${VIDEO_GID}"
        --group-add "${RENDER_GID}"
        --security-opt seccomp=unconfined
        --security-opt apparmor=unconfined
        --security-opt label=disable
        -e "HIP_VISIBLE_DEVICES=${gpu}"
    )
}
